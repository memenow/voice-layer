"""Provider adapters for the VoiceLayer Python worker."""

from __future__ import annotations

import contextlib
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import (
    load_llm_provider_config,
    load_whisper_provider_config,
)


class ProviderInvocationError(RuntimeError):
    """Raised when the configured provider cannot satisfy a request."""


def provider_runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the runtime directory used for provider state files."""

    source = environ or os.environ
    base = source.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    runtime_dir = Path(base) / "voicelayer" / "providers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def _read_pid_from_lock(lock_path: Path) -> int | None:
    """Read the PID written by the lock owner.

    Returns ``None`` when the file is missing, unreadable, partially
    written (empty), non-integer, or contains a non-positive value. The
    latter is treated as "unknown" so a malformed file never routes into
    the liveness probe against ``/proc/0`` or negative PIDs.
    """

    try:
        raw = lock_path.read_text(encoding="utf-8").strip()
    except (FileNotFoundError, PermissionError, OSError):
        return None
    if not raw:
        return None
    try:
        pid = int(raw)
    except ValueError:
        return None
    return pid if pid > 0 else None


def _pid_runs_binary(pid: int, expected_binary: str) -> bool:
    """Best-effort Linux-only check that ``pid`` still runs ``expected_binary``.

    Reads ``/proc/<pid>/cmdline`` (NUL-separated argv) and compares the
    basename of argv[0] to the basename of ``expected_binary`` so both
    absolute paths and bare names resolve correctly. Returns False on any
    failure (process gone, non-Linux platform, binary mismatch).

    TODO(portable): on non-Linux targets ``/proc`` is absent and this
    function always returns False, so :func:`reclaim_stale_lock` would
    evict every lock regardless of owner liveness. When VoiceLayer grows
    a macOS or BSD target, derive the cmdline check from ``ps``,
    ``sysctl``, or ``psutil`` instead.
    """

    if pid <= 0:
        return False
    cmdline_path = Path(f"/proc/{pid}/cmdline")
    try:
        raw = cmdline_path.read_bytes()
    except (FileNotFoundError, PermissionError, OSError):
        return False
    argv0_bytes = raw.split(b"\x00", 1)[0]
    if not argv0_bytes:
        return False
    try:
        argv0 = argv0_bytes.decode("utf-8")
    except UnicodeDecodeError:
        return False
    return Path(argv0).name == Path(expected_binary).name


def reclaim_stale_lock(lock_path: Path, expected_binary: str) -> bool:
    """Remove ``lock_path`` if its recorded owner is no longer running.

    Returns True when a stale lock was deleted so the caller can retry
    the ``os.open(..., O_EXCL)`` happy path. Returns False when the lock
    owner is still alive (the caller must wait) or when the state is
    ambiguous (partially-written PID, /proc unreadable) so we err on the
    safe side and leave the lock intact.
    """

    if not lock_path.exists():
        return False
    pid = _read_pid_from_lock(lock_path)
    if pid is None:
        # The owner may have just created the lock and not yet written
        # its PID. Leave the lock alone; the caller will fall through to
        # the shared `wait_for_endpoint` path.
        return False
    if _pid_runs_binary(pid, expected_binary):
        return False
    with contextlib.suppress(FileNotFoundError):
        lock_path.unlink()
    return True


def supported_providers(
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return provider descriptors for the Python worker boundary."""

    from voicelayer_orchestrator.providers.llm_openai_compatible import (
        configured_llm_descriptor,
    )

    whisper_config = load_whisper_provider_config(environ)
    providers: list[dict[str, Any]] = [
        {
            "id": "whisper_cpp",
            "kind": "asr",
            "transport": "whisper_cli" if whisper_config is not None else "stdio_worker",
            "local": True,
            "default_enabled": True,
            "experimental": False,
            "license": "MIT",
        },
        {
            "id": "voxtral_realtime",
            "kind": "asr",
            "transport": "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "Apache-2.0",
        },
    ]

    configured = configured_llm_descriptor(load_llm_provider_config(environ))
    if configured is not None:
        providers.append(configured)
    else:
        providers.append(
            {
                "id": "gemma_4_local",
                "kind": "llm",
                "transport": "stdio_worker",
                "local": True,
                "default_enabled": True,
                "experimental": False,
                "license": "Apache-2.0",
            }
        )

    return providers
