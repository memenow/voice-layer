"""Persistent whisper-server HTTP provider for VoiceLayer.

The upstream ``whisper-server`` binary (from whisper.cpp) keeps the ggml model
mmapped across requests and answers ``POST /inference`` with a multipart
payload. Reusing a live server avoids the cold-start cost that rules out
naive fixed-segment transcription with ``whisper-cli``.

This module provides:

- :func:`probe_whisper_server` — cheap readiness check over HTTP.
- :func:`ensure_whisper_server` — optional autostart of a local
  ``whisper-server`` subprocess, following the same lazy-launch pattern as
  :mod:`voicelayer_orchestrator.providers.llama_autostart`.
- :func:`transcribe_with_whisper_server` — send an audio file and parse the
  transcript JSON.
"""

from __future__ import annotations

import atexit
import contextlib
import json
import mimetypes
import os
import subprocess
import time
import urllib.error
import urllib.request
import uuid
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import WhisperServerConfig
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    provider_runtime_dir,
    reclaim_stale_lock,
)

_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


def cleanup_background_processes() -> None:
    """Reap completed background whisper-server processes."""

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


def probe_whisper_server(
    config: WhisperServerConfig, timeout_seconds: float | None = None
) -> tuple[bool, str | None]:
    """Probe ``<base_url>/`` for readiness. Returns (reachable, error_message)."""

    probe_timeout = (
        timeout_seconds if timeout_seconds is not None else min(config.timeout_seconds, 5.0)
    )
    request = urllib.request.Request(config.base_url, method="GET")
    try:
        with urllib.request.urlopen(request, timeout=probe_timeout) as response:
            status = response.status
    except urllib.error.HTTPError as exc:
        # An HTTP error means the server is up but the path returned non-2xx; still reachable.
        return True, f"HTTP {exc.code}"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {exc.reason}"
    except TimeoutError:
        return False, "timeout"
    except ConnectionError as exc:
        return False, f"connection error: {exc}"

    if 200 <= status < 500:
        return True, None
    return False, f"unexpected HTTP status {status}"


def _server_endpoint_key(config: WhisperServerConfig) -> str:
    host = config.host.replace(":", "_")
    return f"whisper-server-{host}-{config.port}"


def _build_whisper_server_command(config: WhisperServerConfig) -> list[str]:
    if not config.server_bin:
        raise ProviderInvocationError(
            "Autostart requires VOICELAYER_WHISPER_SERVER_BIN to point at a whisper-server binary."
        )
    if not config.model_path:
        raise ProviderInvocationError(
            "Autostart requires VOICELAYER_WHISPER_MODEL_PATH to point at a ggml model."
        )

    return [
        config.server_bin,
        "-m",
        config.model_path,
        "--host",
        config.host,
        "--port",
        str(config.port),
        *config.extra_args,
    ]


def autostart_whisper_server(
    config: WhisperServerConfig,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Launch ``whisper-server`` in the background and wait for readiness."""

    runtime_dir = provider_runtime_dir(environ)
    key = _server_endpoint_key(config)
    lock_path = runtime_dir / f"{key}.lock"
    log_path = runtime_dir / f"{key}.log"
    state_path = runtime_dir / f"{key}.json"
    source = dict(os.environ)
    if environ:
        source.update(environ)

    try:
        command = _build_whisper_server_command(config)
    except ProviderInvocationError as exc:
        return False, str(exc)

    # If a previous owner crashed without cleanup, remove its orphaned lock
    # so we can retake the launch path instead of waiting for a dead
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
                    "endpoint": config.base_url,
                    "started_at_epoch_seconds": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return _wait_for_whisper_server(config)
    except FileExistsError:
        # Another process owns the lock and is either launching or has
        # launched the server. Wait for its endpoint without touching the
        # lock file -- removing it would let a third caller racing with us
        # re-enter the launch path and spawn a duplicate server.
        return _wait_for_whisper_server(config)
    except FileNotFoundError as exc:
        return False, f"Unable to find `{config.server_bin}`: {exc}"
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        if acquired:
            with contextlib.suppress(FileNotFoundError):
                lock_path.unlink()


def _wait_for_whisper_server(config: WhisperServerConfig) -> tuple[bool, str | None]:
    deadline = time.monotonic() + config.launch_timeout_seconds
    last_error: str | None = None
    while time.monotonic() < deadline:
        reachable, error = probe_whisper_server(config)
        if reachable:
            return True, None
        last_error = error
        time.sleep(config.poll_interval_seconds)
    return False, last_error


def ensure_whisper_server(
    config: WhisperServerConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return (reachable, error) for the configured server, auto-starting when asked."""

    if config is None:
        return False, None

    reachable, error = probe_whisper_server(config)
    if reachable:
        return True, None

    if not config.auto_start:
        return False, error

    return autostart_whisper_server(config, environ)


def _encode_multipart(
    file_path: str,
    file_bytes: bytes,
    fields: Mapping[str, str],
) -> tuple[bytes, str]:
    """Encode a multipart/form-data body with a file upload and plain fields."""

    boundary = f"----voicelayer{uuid.uuid4().hex}"
    crlf = b"\r\n"
    pieces: list[bytes] = []
    for name, value in fields.items():
        pieces.append(f"--{boundary}".encode())
        pieces.append(f'Content-Disposition: form-data; name="{name}"'.encode())
        pieces.append(b"")
        pieces.append(value.encode("utf-8"))

    filename = Path(file_path).name
    mime_type, _ = mimetypes.guess_type(filename)
    mime_type = mime_type or "application/octet-stream"
    pieces.append(f"--{boundary}".encode())
    pieces.append(f'Content-Disposition: form-data; name="file"; filename="{filename}"'.encode())
    pieces.append(f"Content-Type: {mime_type}".encode())
    pieces.append(b"")
    pieces.append(file_bytes)
    pieces.append(f"--{boundary}--".encode())
    pieces.append(b"")

    body = crlf.join(pieces)
    content_type = f"multipart/form-data; boundary={boundary}"
    return body, content_type


def transcribe_with_whisper_server(
    params: Mapping[str, Any],
    config: WhisperServerConfig,
) -> dict[str, Any]:
    """Send ``audio_file`` to ``whisper-server`` and return the VoiceLayer transcription shape."""

    audio_file = str(params.get("audio_file", "")).strip()
    language = str(params.get("language") or "auto").strip() or "auto"
    translate_to_english = bool(params.get("translate_to_english", False))

    if not audio_file:
        raise ProviderInvocationError("Transcribe requests require `audio_file`.")
    file_path = Path(audio_file)
    if not file_path.is_file():
        raise ProviderInvocationError(f"Audio file does not exist: {audio_file}")

    try:
        file_bytes = file_path.read_bytes()
    except OSError as exc:
        raise ProviderInvocationError(f"Unable to read audio file: {exc}") from exc

    body, content_type = _encode_multipart(
        audio_file,
        file_bytes,
        {
            "response_format": "json",
            "language": language,
            "translate": "true" if translate_to_english else "false",
        },
    )

    url = f"{config.base_url}/inference"
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": content_type},
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            raw = response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ProviderInvocationError(f"whisper-server returned HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise ProviderInvocationError(f"whisper-server is unreachable: {exc.reason}") from exc
    except TimeoutError as exc:
        raise ProviderInvocationError("whisper-server timed out.") from exc

    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderInvocationError(
            f"whisper-server returned non-JSON body: {raw[:200]!r}"
        ) from exc

    text_raw = payload.get("text", "") if isinstance(payload, dict) else ""
    text = str(text_raw).strip()
    if not text or text == "[BLANK_AUDIO]":
        # Keep empty text rather than raising: callers decide whether silence is an error.
        text = ""

    detected_language: str | None
    if language == "auto":
        detected_language = payload.get("language") if isinstance(payload, dict) else None
    else:
        detected_language = language

    return {
        "text": text,
        "detected_language": detected_language,
        "notes": [
            f"Transcribed by whisper-server at {config.base_url} using the configured ggml model.",
        ],
    }
