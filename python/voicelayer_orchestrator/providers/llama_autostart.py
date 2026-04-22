"""Auto-start lifecycle management for local `llama-server` endpoints."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import subprocess
import time
import urllib.parse
from collections.abc import Mapping

from voicelayer_orchestrator.config import (
    LlamaServerLaunchConfig,
    OpenAICompatibleConfig,
    load_llama_server_launch_config,
)
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    provider_runtime_dir,
    reclaim_stale_lock,
)
from voicelayer_orchestrator.providers.llm_openai_compatible import (
    is_local_endpoint,
    probe_llm_endpoint,
)

_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


def cleanup_background_processes() -> None:
    """Reap completed background provider processes."""

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


def endpoint_key(endpoint: str) -> str:
    """Build a filesystem-safe key from the configured endpoint."""

    parsed = urllib.parse.urlparse(endpoint)
    host = (parsed.hostname or "localhost").replace(":", "_")
    port = parsed.port or 8080
    return f"{host}-{port}"


def build_llama_server_command(
    config: OpenAICompatibleConfig,
    launch: LlamaServerLaunchConfig,
) -> list[str]:
    """Build the `llama-server` command for the configured endpoint."""

    if not is_local_endpoint(config.endpoint):
        raise ProviderInvocationError(
            "Automatic llama-server startup only supports local endpoints."
        )

    parsed = urllib.parse.urlparse(config.endpoint)
    port = parsed.port or 8080
    if launch.model_path:
        source_args = ["-m", launch.model_path]
    elif launch.hf_repo:
        source_args = ["-hf", launch.hf_repo]
    else:
        raise ProviderInvocationError(
            "Automatic llama-server startup requires "
            "`VOICELAYER_LLAMA_MODEL_PATH` or `VOICELAYER_LLAMA_HF_REPO`."
        )

    return [
        launch.server_bin,
        *source_args,
        "--port",
        str(port),
        *launch.extra_args,
    ]


def wait_for_llm_endpoint(
    config: OpenAICompatibleConfig,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, str | None]:
    """Wait for the configured LLM endpoint to become reachable."""

    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        reachable, error = probe_llm_endpoint(config)
        if reachable:
            return True, None
        last_error = error
        time.sleep(poll_interval_seconds)
    return False, last_error


def autostart_llama_server(
    config: OpenAICompatibleConfig,
    launch: LlamaServerLaunchConfig,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Launch `llama-server` when the endpoint is local and currently unavailable."""

    runtime_dir = provider_runtime_dir(environ)
    key = endpoint_key(config.endpoint)
    lock_path = runtime_dir / f"{key}.lock"
    log_path = runtime_dir / f"{key}.log"
    state_path = runtime_dir / f"{key}.json"
    source = dict(os.environ)
    if environ:
        source.update(environ)

    try:
        command = build_llama_server_command(config, launch)
    except ProviderInvocationError as exc:
        return False, str(exc)

    # If a previous owner crashed without cleanup, remove its orphaned lock
    # so we can retake the launch path instead of waiting 30s for a dead
    # server. Benign TOCTOU: if another launcher creates the lock between
    # this call and O_EXCL below, we fall into the FileExistsError branch
    # and wait like before.
    reclaim_stale_lock(lock_path, command[0])

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
                env=source,
                start_new_session=True,
            )
        _BACKGROUND_PROCESSES.append(process)
        state_path.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "endpoint": config.endpoint,
                    "started_at_epoch_seconds": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return wait_for_llm_endpoint(
            config,
            timeout_seconds=launch.launch_timeout_seconds,
            poll_interval_seconds=launch.poll_interval_seconds,
        )
    except FileExistsError:
        # Another process owns the lock and is either launching or has
        # launched the server. Wait for its endpoint without touching the
        # lock file -- removing it would let a third caller racing with us
        # re-enter the launch path and spawn a duplicate server.
        return wait_for_llm_endpoint(
            config,
            timeout_seconds=launch.launch_timeout_seconds,
            poll_interval_seconds=launch.poll_interval_seconds,
        )
    except FileNotFoundError as exc:
        return False, f"Unable to find `{launch.server_bin}`: {exc}"
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if acquired:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()


def ensure_llm_endpoint(
    config: OpenAICompatibleConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return endpoint readiness, launching `llama-server` when configured to do so."""

    if config is None:
        return False, None

    reachable, error = probe_llm_endpoint(config)
    if reachable:
        return True, None

    launch = load_llama_server_launch_config(environ)
    if launch is None:
        return False, error

    return autostart_llama_server(config, launch, environ)
