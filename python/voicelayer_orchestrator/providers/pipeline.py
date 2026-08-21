"""Provider orchestration for the VoiceLayer worker.

Owns everything between the JSON-RPC method surface and the provider
adapters: the silero-vad pre-pass, whisper server/cli dispatch with
fallback, LLM endpoint readiness, and health assembly.
"""

from __future__ import annotations

from typing import Any

from voicelayer_orchestrator.config import (
    llm_config,
    mimo_asr_config,
    qwen3_asr_config,
    vad_config,
    whisper_config,
    whisper_server_config,
)
from voicelayer_orchestrator.providers import (
    InvalidProviderParamsError,
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
from voicelayer_orchestrator.providers.mimo_asr import (
    transcribe_with_mimo,
    validate_mimo_provider,
)
from voicelayer_orchestrator.providers.qwen3_asr import (
    transcribe_with_qwen3_asr,
    validate_qwen3_asr_provider,
)
from voicelayer_orchestrator.providers.vad_segmenter import (
    apply_vad_prepass,
    probe_audio_file,
)
from voicelayer_orchestrator.providers.whisper_cli import (
    transcribe_with_whisper_cli,
    validate_whisper_provider,
)
from voicelayer_orchestrator.providers.whisper_server import (
    ensure_whisper_server,
    probe_whisper_server,
    transcribe_with_whisper_server,
    validate_autostart_prerequisites,
)

WHISPER_PROVIDER_ID = "whisper_cpp"
MIMO_PROVIDER_ID = "mimo_v2_5_asr"
QWEN3_ASR_PROVIDER_ID = "qwen3_asr_1_7b"

_SUPPORTED_ASR_IDS = {WHISPER_PROVIDER_ID, MIMO_PROVIDER_ID, QWEN3_ASR_PROVIDER_ID}


def segment_probe(params: dict[str, Any]) -> dict[str, Any]:
    """Classify a short WAV probe as speech/silence via silero-vad."""

    audio_file = str(params.get("audio_file", "")).strip()
    if not audio_file:
        raise ProviderInvocationError("segment_probe requires params.audio_file.")

    config = vad_config()
    if config is None:
        raise ProviderUnavailableError(
            "VAD is not configured; segment_probe requires the `[vad]` config section."
        )
    return probe_audio_file(audio_file, config)


def transcribe(params: dict[str, Any]) -> dict[str, Any]:
    """Transcribe ``audio_file``, honoring an explicit `provider_id`.

    An explicit id selects a single backend with no fallback; the default
    (absent or `whisper_cpp`) walks the whisper-server → whisper-cli chain.
    """
    request_params = dict(params)
    raw_provider_id = request_params.pop("provider_id", None)
    if raw_provider_id is not None and not isinstance(raw_provider_id, str):
        raise InvalidProviderParamsError(
            "transcribe params.provider_id must be a string when present."
        )
    provider_id = raw_provider_id.strip() if raw_provider_id else None
    if provider_id is not None and provider_id not in _SUPPORTED_ASR_IDS:
        raise ProviderUnavailableError(
            f"Unknown ASR provider id `{provider_id}`. Supported ids: "
            f"`{WHISPER_PROVIDER_ID}`, `{MIMO_PROVIDER_ID}`, `{QWEN3_ASR_PROVIDER_ID}`."
        )

    if provider_id == MIMO_PROVIDER_ID:
        config = mimo_asr_config()
        if config is None:
            raise ProviderUnavailableError(
                "MiMo-V2.5-ASR is not configured. Set VOICELAYER_MIMO_MODEL_PATH and "
                "VOICELAYER_MIMO_TOKENIZER_PATH (or the `[mimo_asr]` config section)."
            )
        return transcribe_with_mimo(request_params, config)

    if provider_id == QWEN3_ASR_PROVIDER_ID:
        config = qwen3_asr_config()
        if config is None:
            raise ProviderUnavailableError(
                "Qwen3-ASR-1.7B is not configured. Set VOICELAYER_QWEN3_ASR_MODEL_PATH "
                "(or the `[qwen3_asr]` config section)."
            )
        return transcribe_with_qwen3_asr(request_params, config)

    effective_params, extra_notes, short_circuit = _apply_vad_prepass(request_params)
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
    server = whisper_server_config()
    mimo = mimo_asr_config()
    mimo_configured, mimo_error = validate_mimo_provider(mimo)
    qwen3 = qwen3_asr_config()
    qwen3_configured, qwen3_error = validate_qwen3_asr_provider(qwen3)
    asr_configured, asr_error = validate_whisper_provider(whisper)

    if server is not None:
        whisper_mode = "server"
    elif whisper is not None:
        whisper_mode = "cli"
    else:
        whisper_mode = "unconfigured"

    # A server-only configuration is legitimate — don't require the CLI
    # binary + model to also be set.
    if whisper_mode == "server" and not asr_configured:
        reachable, probe_error = probe_whisper_server(server)
        if reachable:
            asr_configured = True
            asr_error = None
        elif server.auto_start:
            # Autostart requested but transcribe would immediately fail if
            # launcher prereqs (server binary, model) are missing — surface
            # that so /health and `vl doctor` don't report a false positive.
            prereq_ok, prereq_error = validate_autostart_prerequisites(server)
            if prereq_ok:
                asr_configured = True
                asr_error = None
            else:
                asr_error = prereq_error
        else:
            asr_error = probe_error or "whisper-server is not reachable"

    return {
        "status": "ok",
        "worker": "voicelayer_orchestrator",
        "protocol": "2.0",
        "asr_configured": asr_configured,
        "asr_binary": None if whisper is None else whisper.binary,
        "asr_model_path": None if whisper is None else whisper.model_path,
        "asr_error": asr_error,
        "whisper_mode": whisper_mode,
        "whisper_server_url": None if server is None else server.base_url,
        "mimo_configured": mimo_configured,
        "mimo_model_path": None if mimo is None else mimo.model_path,
        "mimo_error": mimo_error,
        "qwen3_asr_configured": qwen3_configured,
        "qwen3_asr_model_path": None if qwen3 is None else qwen3.model_path,
        "qwen3_asr_error": qwen3_error,
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
