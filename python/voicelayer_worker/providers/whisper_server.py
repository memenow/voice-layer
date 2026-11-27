"""Persistent ``whisper-server`` HTTP provider."""

from __future__ import annotations

import mimetypes
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from voicelayer_worker.config import WhisperServerConfig
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.process import autostart_process
from voicelayer_worker.transcript import collapse_nonspeech_transcript


def probe_whisper_server(
    config: WhisperServerConfig,
    timeout_seconds: float | None = None,
) -> tuple[bool, str | None]:
    """Probe ``<base_url>/`` for HTTP readiness."""

    probe_timeout = (
        timeout_seconds if timeout_seconds is not None else min(config.timeout_seconds, 5.0)
    )
    try:
        response = httpx.get(config.base_url, timeout=probe_timeout)
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.HTTPError as exc:
        return False, f"unreachable: {exc}"

    if response.status_code < 500:
        return True, None
    return False, f"unexpected HTTP status {response.status_code}"


def build_whisper_server_command(config: WhisperServerConfig) -> list[str]:
    """Build the argv for launching ``whisper-server``."""

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


def validate_autostart_prerequisites(config: WhisperServerConfig) -> tuple[bool, str | None]:
    """Validate that autostart has the required binary and model path."""

    try:
        build_whisper_server_command(config)
    except ProviderInvocationError as exc:
        return False, str(exc)
    return True, None


def autostart_whisper_server(config: WhisperServerConfig) -> tuple[bool, str | None]:
    """Launch ``whisper-server`` in the background and wait until reachable."""

    try:
        command = build_whisper_server_command(config)
    except ProviderInvocationError as exc:
        return False, str(exc)

    key = f"whisper-server-{config.host.replace(':', '_')}-{config.port}"
    return autostart_process(
        key=key,
        command=command,
        endpoint=config.base_url,
        probe=lambda: probe_whisper_server(config),
        launch_timeout_seconds=config.launch_timeout_seconds,
        poll_interval_seconds=config.poll_interval_seconds,
    )


def ensure_whisper_server(config: WhisperServerConfig | None) -> tuple[bool, str | None]:
    """Return ``(reachable, error)`` for ``whisper-server``, auto-starting if configured."""

    if config is None:
        return False, None

    reachable, error = probe_whisper_server(config)
    if reachable:
        return True, None
    if not config.auto_start:
        return False, error
    return autostart_whisper_server(config)


def transcribe_with_whisper_server(
    params: Mapping[str, Any],
    config: WhisperServerConfig,
) -> dict[str, Any]:
    """POST ``audio_file`` to ``whisper-server`` and return the normalized transcript."""

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

    mime_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"

    try:
        response = httpx.post(
            f"{config.base_url}/inference",
            data={
                "response_format": "json",
                "language": language,
                "translate": "true" if translate_to_english else "false",
            },
            files={"file": (file_path.name, file_bytes, mime_type)},
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise ProviderInvocationError(
            f"whisper-server returned HTTP {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ProviderInvocationError("whisper-server timed out.") from exc
    except httpx.HTTPError as exc:
        raise ProviderInvocationError(f"whisper-server is unreachable: {exc}") from exc
    except ValueError as exc:
        raise ProviderInvocationError("whisper-server returned a non-JSON body.") from exc

    raw_text = str(payload.get("text", "")).strip() if isinstance(payload, dict) else ""
    text = collapse_nonspeech_transcript(raw_text)

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
