"""Provider orchestration for the VoiceLayer worker.

Owns everything between the JSON-RPC method surface and the provider
adapters: the silero-vad pre-pass, whisper server/cli dispatch with
fallback, LLM endpoint readiness, and health assembly.
"""

from __future__ import annotations

from typing import Any

from voicelayer_orchestrator.config import (
    llm_config,
    vad_config,
    whisper_config,
    whisper_server_config,
)
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    ProviderUnavailableError,
    provider_runtime_dir,
)
from voicelayer_orchestrator.providers.llama_autostart import ensure_llm_endpoint
from voicelayer_orchestrator.providers.llm_openai_compatible import (
    build_compose_payload,
    build_rewrite_payload,
    build_translate_payload,
)
from voicelayer_orchestrator.providers.vad_segmenter import apply_vad_prepass
from voicelayer_orchestrator.providers.whisper_cli import (
    transcribe_with_whisper_cli,
    validate_whisper_provider,
)
from voicelayer_orchestrator.providers.whisper_server import (
    ensure_whisper_server,
    transcribe_with_whisper_server,
)


def transcribe(params: dict[str, Any]) -> dict[str, Any]:
    """Transcribe ``audio_file``, preferring whisper-server over whisper-cli."""
    effective_params, extra_notes, short_circuit = _apply_vad_prepass(params)
    if short_circuit is not None:
        return short_circuit

    server_config = whisper_server_config()
    server_error: str | None = None
    if server_config is not None:
        reachable, probe_error = ensure_whisper_server(server_config)
        if reachable:
            try:
                result = transcribe_with_whisper_server(effective_params, server_config)
                return _with_notes(result, extra_notes)
            except ProviderInvocationError as exc:
                server_error = str(exc)
        else:
            server_error = probe_error or "whisper-server unreachable"

    cli_config = whisper_config()
    if cli_config is None:
        if server_error is not None:
            raise ProviderInvocationError(
                f"whisper-server failed ({server_error}) and no whisper-cli fallback is configured."
            )
        raise ProviderUnavailableError(
            "No transcription provider is configured for the requested workflow."
        )

    try:
        result = transcribe_with_whisper_cli(effective_params, cli_config)
    except ProviderInvocationError as exc:
        detail = str(exc)
        if server_error is not None:
            detail = f"{detail} (whisper-server also failed: {server_error})"
        raise ProviderInvocationError(detail) from exc
    return _with_notes(result, extra_notes)


def compose(params: dict[str, Any]) -> dict[str, Any]:
    config = _ready_llm_config()
    return build_compose_payload(params, config)


def rewrite(params: dict[str, Any]) -> dict[str, Any]:
    config = _ready_llm_config()
    return build_rewrite_payload(params, config)


def translate(params: dict[str, Any]) -> dict[str, Any]:
    config = _ready_llm_config()
    return build_translate_payload(params, config)


def health_report() -> dict[str, Any]:
    """Assemble the worker health payload for the daemon's health cache."""
    llm = llm_config()
    llm_reachable, llm_error = ensure_llm_endpoint(llm)
    whisper = whisper_config()
    asr_configured, asr_error = validate_whisper_provider(whisper)
    return {
        "status": "ok",
        "worker": "voicelayer_orchestrator",
        "protocol": "2.0",
        "asr_configured": asr_configured,
        "asr_binary": None if whisper is None else whisper.binary,
        "asr_model_path": None if whisper is None else whisper.model_path,
        "asr_error": asr_error,
        "llm_configured": llm is not None,
        "llm_model": None if llm is None else llm.model,
        "llm_endpoint": None if llm is None else llm.endpoint,
        "llm_reachable": llm_reachable,
        "llm_error": llm_error,
    }


def _ready_llm_config():
    config = llm_config()
    if config is None:
        raise ProviderUnavailableError(
            "No model provider is configured for the requested workflow."
        )
    reachable, error = ensure_llm_endpoint(config)
    if not reachable:
        raise ProviderInvocationError(f"Configured LLM endpoint is not ready: {error}")
    return config


def _with_notes(result: dict[str, Any], extra_notes: list[str]) -> dict[str, Any]:
    if not extra_notes:
        return result
    return {**result, "notes": [*extra_notes, *result.get("notes", [])]}


def _apply_vad_prepass(
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Run the optional silero-vad pre-pass.

    Returns ``(effective_params, extra_notes, short_circuit_result)``. When
    ``short_circuit_result`` is not ``None`` the caller returns it directly
    without invoking whisper (VAD detected no speech). Otherwise the caller
    proceeds with ``effective_params`` and appends ``extra_notes``.
    """
    config = vad_config()
    if config is None:
        return params, [], None

    audio_file = str(params.get("audio_file", "")).strip()
    if not audio_file:
        return params, [], None

    try:
        runtime_dir = provider_runtime_dir() / "vad"
        trimmed_path, regions = apply_vad_prepass(audio_file, config, runtime_dir)
    except ProviderInvocationError as exc:
        return params, [f"VAD pre-pass failed, transcribing raw audio: {exc}"], None

    if not regions:
        return (
            params,
            [],
            {
                "text": "",
                "detected_language": None,
                "notes": ["VAD detected no speech; whisper inference was skipped."],
            },
        )

    new_params = dict(params)
    new_params["audio_file"] = trimmed_path
    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) from the original capture."
    )
    return new_params, [note], None
