"""Provider orchestration for VoiceLayer worker requests.

Owns the single Silero VAD pre-pass, ASR provider selection and whisper-server to
whisper-cli fallback, LLM readiness checks, and worker health assembly.
"""

from __future__ import annotations

import contextlib
from pathlib import Path
from typing import Any

from voicelayer_worker.audio import stitch_wav_segments as _stitch_wav_segments
from voicelayer_worker.config import (
    OpenAICompatibleConfig,
    llama_launch_config,
    llm_config,
    mimo_asr_config,
    qwen3_asr_config,
    vad_config,
    whisper_config,
    whisper_server_config,
)
from voicelayer_worker.errors import (
    InvalidProviderParamsError,
    ProviderInvocationError,
    ProviderUnavailableError,
)
from voicelayer_worker.process import provider_runtime_dir
from voicelayer_worker.providers.mimo_asr import transcribe_with_mimo, validate_mimo_provider
from voicelayer_worker.providers.openai_compatible import (
    build_compose_payload,
    build_rewrite_payload,
    build_translate_payload,
    ensure_llm_endpoint,
)
from voicelayer_worker.providers.qwen3_asr import (
    transcribe_with_qwen3_asr,
    validate_qwen3_asr_provider,
)
from voicelayer_worker.providers.whisper_cli import (
    transcribe_with_whisper_cli,
    validate_whisper_provider,
)
from voicelayer_worker.providers.whisper_server import (
    ensure_whisper_server,
    probe_whisper_server,
    transcribe_with_whisper_server,
    validate_autostart_prerequisites,
)
from voicelayer_worker.vad import apply_vad_prepass, probe_audio_file

WHISPER_PROVIDER_ID = "whisper_cpp"
MIMO_PROVIDER_ID = "mimo_v2_5_asr"
QWEN3_ASR_PROVIDER_ID = "qwen3_asr_1_7b"

_SUPPORTED_ASR_IDS = frozenset({WHISPER_PROVIDER_ID, MIMO_PROVIDER_ID, QWEN3_ASR_PROVIDER_ID})


def segment_probe(params: dict[str, Any]) -> dict[str, Any]:
    """Classify a short WAV clip as speech or silence using Silero VAD."""

    raw_audio_file = params.get("audio_file")
    if not isinstance(raw_audio_file, str) or not raw_audio_file.strip():
        raise InvalidProviderParamsError(
            "segment_probe requires params.audio_file (non-empty string)."
        )

    config = vad_config()
    if config is None:
        raise ProviderUnavailableError(
            "VAD is not configured; set VOICELAYER_WHISPER_VAD_ENABLED=true and "
            "VOICELAYER_WHISPER_VAD_MODEL_PATH (or the `[vad]` config section)."
        )
    return probe_audio_file(raw_audio_file.strip(), config)


def stitch_wav_segments(params: dict[str, Any]) -> dict[str, Any]:
    """Validate parameters and concatenate WAV segment files."""

    audio_files = params.get("audio_files")
    out_file = params.get("out_file")
    if (
        not isinstance(audio_files, list)
        or not audio_files
        or not all(isinstance(path, str) and path for path in audio_files)
    ):
        raise InvalidProviderParamsError(
            "stitch_wav_segments requires params.audio_files (list of strings)."
        )
    if not isinstance(out_file, str) or not out_file:
        raise InvalidProviderParamsError(
            "stitch_wav_segments requires params.out_file (non-empty string)."
        )
    return _stitch_wav_segments(audio_files, out_file)


def transcribe(params: dict[str, Any]) -> dict[str, Any]:
    """Transcribe ``audio_file`` using the selected ASR backend with a single VAD pre-pass."""

    request_params = dict(params)
    provider_id = _resolve_provider_id(request_params.pop("provider_id", None))

    # Validate provider configuration before running VAD so unconfigured backends fail fast.
    target = _resolve_asr_target(provider_id, request_params)

    effective_params, extra_notes, short_circuit, trimmed_path = _run_vad_prepass(
        request_params,
        provider_label=provider_id or WHISPER_PROVIDER_ID,
    )
    try:
        if short_circuit is not None:
            return short_circuit
        result = target(effective_params)
        return _with_notes(result, extra_notes)
    finally:
        if trimmed_path is not None:
            with contextlib.suppress(OSError):
                trimmed_path.unlink()


def compose(params: dict[str, Any]) -> dict[str, Any]:
    return build_compose_payload(params, _ready_llm_config())


def rewrite(params: dict[str, Any]) -> dict[str, Any]:
    return build_rewrite_payload(params, _ready_llm_config())


def translate(params: dict[str, Any]) -> dict[str, Any]:
    return build_translate_payload(params, _ready_llm_config())


def health_report() -> dict[str, Any]:
    """Assemble the worker health snapshot reported to ``voicelayerd``."""

    llm = llm_config()
    llm_reachable, llm_error = ensure_llm_endpoint(llm, llama_launch_config())
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

    if whisper_mode == "server" and server is not None and not asr_configured:
        reachable, probe_error = probe_whisper_server(server)
        if reachable:
            asr_configured = True
            asr_error = None
        elif server.auto_start:
            prereq_ok, prereq_error = validate_autostart_prerequisites(server)
            asr_configured = prereq_ok
            asr_error = None if prereq_ok else prereq_error
        else:
            asr_error = probe_error or "whisper-server is not reachable"

    return {
        "status": "ok",
        "worker": "voicelayer_worker",
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


def _resolve_provider_id(raw_provider_id: Any) -> str | None:
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
    return provider_id


def _resolve_asr_target(provider_id: str | None, request_params: dict[str, Any]) -> Any:
    if provider_id == MIMO_PROVIDER_ID:
        config = mimo_asr_config()
        if config is None:
            raise ProviderUnavailableError(
                "MiMo-V2.5-ASR is not configured. Set VOICELAYER_MIMO_MODEL_PATH and "
                "VOICELAYER_MIMO_TOKENIZER_PATH (or the `[mimo_asr]` config section)."
            )
        if bool(request_params.get("translate_to_english", False)):
            raise ProviderInvocationError(
                "MiMo-V2.5-ASR does not support `translate_to_english`; "
                "use whisper.cpp or follow up with the LLM `translate` workflow."
            )
        return lambda effective: transcribe_with_mimo(effective, config)

    if provider_id == QWEN3_ASR_PROVIDER_ID:
        config = qwen3_asr_config()
        if config is None:
            raise ProviderUnavailableError(
                "Qwen3-ASR-1.7B is not configured. Set VOICELAYER_QWEN3_ASR_MODEL_PATH "
                "(or the `[qwen3_asr]` config section)."
            )
        if bool(request_params.get("translate_to_english", False)):
            raise ProviderInvocationError(
                "Qwen3-ASR-1.7B does not support `translate_to_english`; "
                "use whisper.cpp or follow up with the LLM `translate` workflow."
            )
        return lambda effective: transcribe_with_qwen3_asr(effective, config)

    if whisper_server_config() is None and whisper_config() is None:
        raise ProviderUnavailableError(
            "No transcription provider is configured for the requested workflow."
        )
    return _transcribe_with_whisper


def _transcribe_with_whisper(effective_params: dict[str, Any]) -> dict[str, Any]:
    server_config = whisper_server_config()
    server_error: str | None = None
    if server_config is not None:
        reachable, probe_error = ensure_whisper_server(server_config)
        if reachable:
            try:
                return transcribe_with_whisper_server(effective_params, server_config)
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
        return transcribe_with_whisper_cli(effective_params, cli_config)
    except ProviderInvocationError as exc:
        detail = str(exc)
        if server_error is not None:
            detail = f"{detail} (whisper-server also failed: {server_error})"
        raise ProviderInvocationError(detail) from exc


def _ready_llm_config() -> OpenAICompatibleConfig:
    config = llm_config()
    if config is None:
        raise ProviderUnavailableError(
            "No model provider is configured for the requested workflow."
        )
    reachable, error = ensure_llm_endpoint(config, llama_launch_config())
    if not reachable:
        raise ProviderInvocationError(f"Configured LLM endpoint is not ready: {error}")
    return config


def _run_vad_prepass(
    params: dict[str, Any],
    *,
    provider_label: str,
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None, Path | None]:
    config = vad_config()
    if config is None:
        return params, [], None, None

    audio_file = str(params.get("audio_file", "")).strip()
    if not audio_file or not Path(audio_file).is_file():
        return params, [], None, None

    try:
        trimmed_path_str, regions = apply_vad_prepass(
            audio_file,
            config,
            provider_runtime_dir() / "vad",
        )
    except ProviderInvocationError as exc:
        return (
            params,
            [f"VAD pre-pass failed, transcribing raw audio: {exc}"],
            None,
            None,
        )

    trimmed_path = Path(trimmed_path_str)
    if not regions:
        return (
            params,
            [],
            {
                "text": "",
                "detected_language": None,
                "notes": [f"VAD detected no speech; {provider_label} inference was skipped."],
            },
            trimmed_path,
        )

    updated = dict(params)
    updated["audio_file"] = trimmed_path_str
    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) from the original capture."
    )
    return updated, [note], None, trimmed_path


def _with_notes(result: dict[str, Any], extra_notes: list[str]) -> dict[str, Any]:
    if not extra_notes:
        return result
    return {**result, "notes": [*extra_notes, *result.get("notes", [])]}
