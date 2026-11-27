"""Optional Xiaomi MiMo-V2.5-ASR provider."""

from __future__ import annotations

import contextlib
import sys
import threading
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_worker.audio import split_wav_segments, wav_duration_seconds
from voicelayer_worker.config import MimoAsrConfig
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.process import provider_runtime_dir
from voicelayer_worker.transcript import collapse_nonspeech_transcript

_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()

_LANGUAGE_TO_AUDIO_TAG: dict[str, str] = {
    "zh": "<chinese>",
    "zh-cn": "<chinese>",
    "zh-tw": "<chinese>",
    "zh-hk": "<chinese>",
    "yue": "<chinese>",
    "en": "<english>",
    "en-us": "<english>",
    "en-gb": "<english>",
}


def validate_mimo_provider(config: MimoAsrConfig | None) -> tuple[bool, str | None]:
    """Validate that configured MiMo model, tokenizer, and repo directories exist."""

    if config is None:
        return False, "No MiMo-V2.5-ASR model paths are configured."

    if not Path(config.model_path).is_dir():
        return False, (
            f"VOICELAYER_MIMO_MODEL_PATH does not exist or is not a directory: {config.model_path}"
        )
    if not Path(config.tokenizer_path).is_dir():
        return False, (
            "VOICELAYER_MIMO_TOKENIZER_PATH does not exist or is not a directory: "
            f"{config.tokenizer_path}"
        )
    if config.repo_path is not None and not Path(config.repo_path).is_dir():
        return False, (
            f"VOICELAYER_MIMO_REPO_PATH does not exist or is not a directory: {config.repo_path}"
        )

    return True, None


def _resolve_audio_tag(language: str | None, config: MimoAsrConfig) -> str | None:
    if language is None:
        return config.audio_tag
    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return config.audio_tag
    return _LANGUAGE_TO_AUDIO_TAG.get(normalized, config.audio_tag)


def _ensure_repo_on_path(repo_path: str | None) -> None:
    if not repo_path:
        return
    candidate = Path(repo_path)
    # Upstream uses PEP 420 `src.mimo_audio` plus relative imports into
    # `src.mimo_audio_tokenizer`, so the parent of `src/` must be on sys.path.
    if candidate.name == "src" and (candidate / "mimo_audio").is_dir():
        candidate = candidate.parent
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _load_mimo_model(config: MimoAsrConfig) -> Any:
    cache_key = (config.model_path, config.tokenizer_path, config.device)
    cached = _MODEL_CACHE.get(cache_key)
    if cached is not None:
        return cached

    with _MODEL_CACHE_LOCK:
        cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            return cached

        _ensure_repo_on_path(config.repo_path)
        try:
            import torch  # noqa: F401, PLC0415
        except ImportError as exc:
            raise ProviderInvocationError(
                "MiMo-V2.5-ASR requires `torch`. Install a CUDA build matching your GPU, "
                'then `uv pip install -e ".[mimo]"`.'
            ) from exc
        try:
            from src.mimo_audio.mimo_audio import (
                MimoAudio,  # type: ignore[import-not-found]  # noqa: PLC0415
            )
        except ImportError as exc:
            raise ProviderInvocationError(
                "Cannot import `src.mimo_audio.mimo_audio.MimoAudio`. Set "
                "VOICELAYER_MIMO_REPO_PATH to the root of a checkout of "
                "https://github.com/XiaomiMiMo/MiMo-V2.5-ASR."
            ) from exc

        try:
            model = MimoAudio(
                model_path=config.model_path,
                mimo_audio_tokenizer_path=config.tokenizer_path,
                device=config.device,
            )
        except Exception as exc:
            raise ProviderInvocationError(f"Failed to load MiMo-V2.5-ASR model: {exc}") from exc

        _MODEL_CACHE[cache_key] = model
        return model


def _load_wav_as_tensor(segment_path: Path) -> tuple[Any, int]:
    """Load a WAV into a mono float32 tensor via ``soundfile`` to bypass ``torchcodec``."""

    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415

    data, sample_rate = sf.read(str(segment_path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = data.mean(axis=1)
    return torch.from_numpy(data), int(sample_rate)


def _run_segment_inference(model: Any, segment_path: Path, audio_tag: str | None) -> str:
    wav_tensor, sample_rate = _load_wav_as_tensor(segment_path)
    if hasattr(model, "resample_audio_if_needed"):
        wav_tensor = model.resample_audio_if_needed(wav_tensor, sample_rate)

    kwargs: dict[str, Any] = {}
    if audio_tag is not None:
        kwargs["audio_tag"] = audio_tag

    try:
        result = model.asr_sft(wav_tensor, **kwargs)
    except TypeError as exc:
        message = str(exc)
        is_tag_arity = (
            audio_tag is not None
            and "audio_tag" in message
            and ("unexpected keyword argument" in message or "got an unexpected" in message)
        )
        if not is_tag_arity:
            raise ProviderInvocationError(
                f"MiMo-V2.5-ASR inference failed on segment {segment_path.name}: {exc}"
            ) from exc
        try:
            result = model.asr_sft(wav_tensor)
        except Exception as inner_exc:
            raise ProviderInvocationError(
                f"MiMo-V2.5-ASR inference failed on segment {segment_path.name}: {inner_exc}"
            ) from inner_exc
    except Exception as exc:
        raise ProviderInvocationError(
            f"MiMo-V2.5-ASR inference failed on segment {segment_path.name}: {exc}"
        ) from exc

    if isinstance(result, (list, tuple)):
        return " ".join(str(part) for part in result).strip()
    return str(result or "").strip()


def transcribe_with_mimo(
    params: Mapping[str, Any],
    config: MimoAsrConfig,
    *,
    runtime_dir: Path | None = None,
) -> dict[str, Any]:
    """Run MiMo-V2.5-ASR on ``audio_file`` (VAD pre-pass is handled by the pipeline)."""

    audio_file = str(params.get("audio_file", "")).strip()
    raw_language = params.get("language")
    language: str | None = raw_language.strip() if isinstance(raw_language, str) else None
    translate_to_english = bool(params.get("translate_to_english", False))

    if not audio_file:
        raise ProviderInvocationError("Transcribe requests require `audio_file`.")
    audio_path = Path(audio_file)
    if not audio_path.is_file():
        raise ProviderInvocationError(f"Audio file does not exist: {audio_file}")

    ready, error = validate_mimo_provider(config)
    if not ready:
        raise ProviderInvocationError(error or "MiMo-V2.5-ASR is not ready.")

    if translate_to_english:
        raise ProviderInvocationError(
            "MiMo-V2.5-ASR does not support `translate_to_english`; "
            "use whisper.cpp or follow up with the LLM `translate` workflow."
        )

    work_dir = (runtime_dir if runtime_dir is not None else provider_runtime_dir()) / "mimo"
    work_dir.mkdir(parents=True, exist_ok=True)
    duration = wav_duration_seconds(audio_path)
    segments = split_wav_segments(
        audio_path,
        config.long_audio_split_seconds,
        work_dir,
        prefix="mimo-segment",
    )
    owned_segments = [segment for segment in segments if segment != audio_path]

    try:
        audio_tag = _resolve_audio_tag(language, config)
        model = _load_mimo_model(config)
        transcripts = [_run_segment_inference(model, segment, audio_tag) for segment in segments]
    finally:
        for owned in owned_segments:
            with contextlib.suppress(OSError):
                owned.unlink()

    raw_text = " ".join(part for part in transcripts if part).strip()
    text = collapse_nonspeech_transcript(raw_text)
    notes = [
        f"Transcribed by MiMo-V2.5-ASR (model `{config.model_path}`).",
        f"Device: {config.device}, dtype: bfloat16 (wrapper default).",
        (
            f"Audio tag forwarded to the wrapper: {audio_tag}."
            if audio_tag is not None
            else "Audio tag left to MiMo auto-detect."
        ),
        f"Audio duration: {duration:.2f}s; segments processed: {len(segments)}.",
    ]
    if len(segments) > 1:
        notes.append("Long audio was split into bounded segments before decoding.")
    if not text:
        notes.append("MiMo-V2.5-ASR returned no speech for this audio.")

    if audio_tag == "<chinese>":
        detected_language: str | None = language or "zh"
    elif audio_tag == "<english>":
        detected_language = language or "en"
    else:
        detected_language = language or None

    return {
        "text": text,
        "detected_language": detected_language,
        "notes": notes,
    }
