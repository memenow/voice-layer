"""Optional Qwen3-ASR-1.7B provider."""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_worker.audio import split_wav_segments, wav_duration_seconds
from voicelayer_worker.config import Qwen3AsrConfig
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.process import provider_runtime_dir
from voicelayer_worker.transcript import collapse_nonspeech_transcript

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()
_CPU_WARNING_EMITTED = False

_ALLOWED_DTYPES = ("float16", "bfloat16", "float32")

_LANGUAGE_TO_QWEN3: dict[str, str] = {
    "zh": "Chinese",
    "zh-cn": "Chinese",
    "zh-tw": "Chinese",
    "zh-hk": "Chinese",
    "cmn": "Chinese",
    "en": "English",
    "en-us": "English",
    "en-gb": "English",
    "yue": "Cantonese",
    "ar": "Arabic",
    "de": "German",
    "fr": "French",
    "es": "Spanish",
    "pt": "Portuguese",
    "pt-br": "Portuguese",
    "pt-pt": "Portuguese",
    "id": "Indonesian",
    "it": "Italian",
    "ko": "Korean",
    "ru": "Russian",
    "th": "Thai",
    "vi": "Vietnamese",
    "ja": "Japanese",
    "tr": "Turkish",
    "hi": "Hindi",
    "ms": "Malay",
    "nl": "Dutch",
    "sv": "Swedish",
    "da": "Danish",
    "fi": "Finnish",
    "pl": "Polish",
    "cs": "Czech",
    "fil": "Filipino",
    "tl": "Filipino",
    "fa": "Persian",
    "el": "Greek",
    "ro": "Romanian",
    "hu": "Hungarian",
    "mk": "Macedonian",
}

_QWEN3_TO_LANGUAGE_CODE: dict[str, str] = {
    "Chinese": "zh",
    "English": "en",
    "Cantonese": "yue",
    "Arabic": "ar",
    "German": "de",
    "French": "fr",
    "Spanish": "es",
    "Portuguese": "pt",
    "Indonesian": "id",
    "Italian": "it",
    "Korean": "ko",
    "Russian": "ru",
    "Thai": "th",
    "Vietnamese": "vi",
    "Japanese": "ja",
    "Turkish": "tr",
    "Hindi": "hi",
    "Malay": "ms",
    "Dutch": "nl",
    "Swedish": "sv",
    "Danish": "da",
    "Finnish": "fi",
    "Polish": "pl",
    "Czech": "cs",
    "Filipino": "fil",
    "Persian": "fa",
    "Greek": "el",
    "Romanian": "ro",
    "Hungarian": "hu",
    "Macedonian": "mk",
}


def validate_qwen3_asr_provider(config: Qwen3AsrConfig | None) -> tuple[bool, str | None]:
    """Validate that the configured Qwen3-ASR model directory and dtype are valid."""

    if config is None:
        return False, "No Qwen3-ASR-1.7B model path is configured."

    if config.torch_dtype not in _ALLOWED_DTYPES:
        return False, (
            f"VOICELAYER_QWEN3_ASR_TORCH_DTYPE must be one of {_ALLOWED_DTYPES}; "
            f"got {config.torch_dtype!r}."
        )
    if not Path(config.model_path).is_dir():
        return False, (
            "VOICELAYER_QWEN3_ASR_MODEL_PATH does not exist or is not a directory: "
            f"{config.model_path}"
        )

    return True, None


def _resolve_language_hint(language: str | None) -> str | None:
    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return None
    return _LANGUAGE_TO_QWEN3.get(normalized)


def _resolve_torch_dtype(name: str) -> Any:
    import torch  # noqa: PLC0415

    return {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[name]


def _maybe_emit_cpu_warning(device: str) -> None:
    global _CPU_WARNING_EMITTED
    if _CPU_WARNING_EMITTED or not device.startswith("cpu"):
        return
    print(
        "VOICELAYER_QWEN3_ASR_DEVICE=cpu: expect 10-30x realtime latency on Qwen3-ASR-1.7B; "
        "set VOICELAYER_QWEN3_ASR_DEVICE to a `cuda:N` device if available.",
        file=sys.stderr,
        flush=True,
    )
    _CPU_WARNING_EMITTED = True


def _load_qwen3_asr_model(config: Qwen3AsrConfig) -> Any:
    cache_key = (config.model_path, config.device, config.torch_dtype)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        _maybe_emit_cpu_warning(config.device)

        try:
            import torch  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ProviderInvocationError(
                "Qwen3-ASR-1.7B requires `torch`. Install the `qwen3-asr` extra: "
                '`uv pip install -e ".[qwen3-asr]"`.'
            ) from exc

        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ProviderInvocationError(
                "Cannot import `qwen_asr.Qwen3ASRModel`. Install the `qwen3-asr` extra: "
                '`uv pip install -e ".[qwen3-asr]"`.'
            ) from exc

        try:
            torch_dtype = _resolve_torch_dtype(config.torch_dtype)
        except KeyError as exc:
            raise ProviderInvocationError(
                f"Unsupported torch_dtype={config.torch_dtype!r}; "
                f"allowed values: {_ALLOWED_DTYPES}."
            ) from exc

        try:
            model = Qwen3ASRModel.from_pretrained(
                config.model_path,
                dtype=torch_dtype,
                device_map=config.device,
            )
        except Exception as exc:
            raise ProviderInvocationError(
                f"Failed to load Qwen3-ASR-1.7B from {config.model_path}: {exc}"
            ) from exc

        _MODEL_CACHE[cache_key] = model
        return model


def _run_segment_inference(
    model: Any,
    segment_path: Path,
    language_hint: str | None,
) -> tuple[str, str | None]:
    try:
        results = model.transcribe(audio=str(segment_path), language=language_hint)
    except Exception as exc:
        raise ProviderInvocationError(
            f"Qwen3-ASR-1.7B inference failed on segment {segment_path.name}: {exc}"
        ) from exc

    if not results:
        return "", None

    first = results[0]
    text_attr = getattr(first, "text", None)
    language_attr = getattr(first, "language", None)
    text = str(text_attr).strip() if text_attr is not None else ""
    raw_language = str(language_attr) if language_attr else None
    return text, raw_language


def transcribe_with_qwen3_asr(
    params: Mapping[str, Any],
    config: Qwen3AsrConfig,
    *,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """Run Qwen3-ASR-1.7B on ``audio_file`` (VAD pre-pass is handled by the pipeline)."""

    audio_file = str(params.get("audio_file", "")).strip()
    raw_language = params.get("language")
    language: str | None = raw_language.strip() if isinstance(raw_language, str) else None
    translate_to_english = bool(params.get("translate_to_english", False))

    if not audio_file:
        raise ProviderInvocationError("Transcribe requests require `audio_file`.")
    audio_path = Path(audio_file)
    if not audio_path.is_file():
        raise ProviderInvocationError(f"Audio file does not exist: {audio_file}")

    ready, error = validate_qwen3_asr_provider(config)
    if not ready:
        raise ProviderInvocationError(error or "Qwen3-ASR-1.7B is not ready.")

    if translate_to_english:
        raise ProviderInvocationError(
            "Qwen3-ASR-1.7B does not support `translate_to_english`; "
            "use whisper.cpp or follow up with the LLM `translate` workflow."
        )

    work_dir = (runtime_dir if runtime_dir is not None else provider_runtime_dir()) / "qwen3_asr"
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = wav_duration_seconds(audio_path)
    segments = split_wav_segments(
        audio_path,
        config.long_audio_split_seconds,
        work_dir,
        prefix="qwen3-segment",
    )
    owned_segments = [segment for segment in segments if segment != audio_path]

    try:
        language_hint = _resolve_language_hint(language)
        model = _load_qwen3_asr_model(config)
        transcripts: list[str] = []
        last_raw_language: str | None = None
        for segment_path in segments:
            text, raw_language_name = _run_segment_inference(model, segment_path, language_hint)
            transcripts.append(text)
            if raw_language_name is not None:
                last_raw_language = raw_language_name
    finally:
        for owned in owned_segments:
            with contextlib.suppress(OSError):
                owned.unlink()

    raw_text = " ".join(part for part in transcripts if part).strip()
    text = collapse_nonspeech_transcript(raw_text)
    notes = [
        f"Transcribed by Qwen3-ASR-1.7B (model `{config.model_path}`).",
        f"Device: {config.device}, dtype: {config.torch_dtype}.",
        (
            f"Language hint forwarded to qwen-asr: {language_hint}."
            if language_hint is not None
            else "Language hint omitted; qwen-asr auto-detected."
        ),
        f"Audio duration: {duration:.2f}s; segments processed: {len(segments)}.",
    ]
    if len(segments) > 1:
        notes.append("Long audio was split client-side before decoding.")
    if not text:
        notes.append("Qwen3-ASR-1.7B returned no speech for this audio.")

    if last_raw_language is not None:
        detected_language: str | None = _QWEN3_TO_LANGUAGE_CODE.get(
            last_raw_language, last_raw_language.lower() or None
        )
    else:
        detected_language = language or None

    return {
        "text": text,
        "detected_language": detected_language,
        "notes": notes,
    }
