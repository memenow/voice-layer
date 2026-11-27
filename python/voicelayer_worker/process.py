"""Runtime directories, stale-lock reclamation, and background server autostart."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path

import psutil

ProbeFn = Callable[[], tuple[bool, str | None]]

_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


def cleanup_background_processes() -> None:
    """Reap completed background provider child processes."""

    remaining: list[subprocess.Popen[bytes]] = []
    for process in _BACKGROUND_PROCESSES:
        if process.poll() is None:
            remaining.append(process)
        else:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                remaining.append(process)
    _BACKGROUND_PROCESSES[:] = remaining


atexit.register(cleanup_background_processes)


def provider_runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the runtime directory used for provider locks, logs, and temporary WAVs."""

    source = environ or os.environ
    base = source.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    runtime_dir = Path(base) / "voicelayer" / "providers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _read_pid_from_lock(lock_path: Path) -> int | None:
    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except OSError:
        return None
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_runs_binary(pid: int, expected_binary: str) -> bool:
    if pid <= 0:
        return False
    try:
        cmdline = psutil.Process(pid).cmdline()
    except (psutil.Error, OSError):
        return False
    if not cmdline or not cmdline[0]:
        return False
    return Path(cmdline[0]).name == Path(expected_binary).name


def reclaim_stale_lock(lock_path: Path, expected_binary: str) -> bool:
    """Delete ``lock_path`` if its recorded PID is no longer running ``expected_binary``.

    Leaves ambiguous or partially written lock files untouched so a concurrent launcher
    can complete safely.
    """

    if not lock_path.exists():
        return False
    pid = _read_pid_from_lock(lock_path)
    if pid is None or _pid_runs_binary(pid, expected_binary):
        return False
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
    return True


def wait_for_probe(
    probe: ProbeFn,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, str | None]:
    """Poll ``probe`` until it succeeds or ``timeout_seconds`` elapses."""

    deadline = time.monotonic() + timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        reachable, error = probe()
        if reachable:
            return True, None
        last_error = error
        time.sleep(poll_interval_seconds)
    return False, last_error


def autostart_process(
    *,
    key: str,
    command: list[str],
    endpoint: str,
    probe: ProbeFn,
    launch_timeout_seconds: float,
    poll_interval_seconds: float,
    runtime_dir: Path | None = None,
) -> tuple[bool, str | None]:
    """Spawn a detached local server under an exclusive lock file and wait for readiness."""

    target_dir = runtime_dir if runtime_dir is not None else provider_runtime_dir()
    lock_path = target_dir / f"{key}.lock"
    log_path = target_dir / f"{key}.log"
    state_path = target_dir / f"{key}.json"

    # Lock files hold the launching Python worker's PID, so check against sys.executable.
    reclaim_stale_lock(lock_path, sys.executable)

    lock_fd: int | None = None
    acquired = False
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        acquired = True
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
            lock_fd = None
            lock_file.write(str(os.getpid()))

        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                start_new_session=True,
            )
        _BACKGROUND_PROCESSES.append(process)
        state_path.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "endpoint": endpoint,
                    "started_at_epoch_seconds": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return wait_for_probe(probe, launch_timeout_seconds, poll_interval_seconds)
    except FileExistsError:
        return wait_for_probe(probe, launch_timeout_seconds, poll_interval_seconds)
    except FileNotFoundError as exc:
        return False, f"Unable to find `{command[0]}`: {exc}"
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if acquired:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()
