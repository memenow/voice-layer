"""Qwen3-ASR-1.7B provider for VoiceLayer.

Loads ``Qwen/Qwen3-ASR-1.7B`` (Apache-2.0) through the official
``qwen-asr`` pip package, which wraps HuggingFace transformers with a
high-level ``Qwen3ASRModel.from_pretrained`` + ``model.transcribe``
API. The wrapper accepts an audio path or ``(numpy.ndarray, sr)``
tuple and returns a list of result objects whose elements expose
``.text`` and ``.language`` attributes; it also handles audio
preprocessing, resampling, and long-audio chunking internally, so the
worker hands the wrapper raw WAV paths and trusts it for the rest.

The model is loaded on the first transcribe call and kept warm in a
module-level cache for the lifetime of the worker process. Cold load
takes seconds on a configured GPU; per-call latency on a single
consumer CUDA accelerator with bf16 is sub-second on 5-15 s clips.

Optional. The whisper.cpp chain remains the default ASR provider.
Callers select Qwen3-ASR by setting ``TranscribeRequest.provider_id =
"qwen3_asr_1_7b"``.
"""

from __future__ import annotations

import contextlib
import sys
import threading
import time
import wave
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import Qwen3AsrConfig, load_whisper_vad_config
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    collapse_nonspeech_transcript,
    provider_runtime_dir,
)
from voicelayer_orchestrator.providers.vad_segmenter import apply_vad_prepass

# Model cache + lock. Keyed by the configuration tuple that determines
# weight identity (``model_path``), inference target (``device``), and
# load-time precision (``torch_dtype``). All three are operator-tunable
# so they belong in the cache key; multiple unique configurations are
# theoretically possible (operator switching between cuda:0 and cuda:1
# mid-session, or flipping bf16 to fp16 to debug a numerical issue) but
# the typical case is a single entry kept warm for the worker lifetime.
# The cache is intentionally unbounded; switching keys is operator-driven
# and infrequent.
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()

# CPU-mode latency warning fires once per worker process so a misconfigured
# device value does not look like a hung daemon. Mirrors the MiMo policy
# of accepting the operator's explicit choice rather than aborting.
_CPU_WARNING_EMITTED = False


# qwen-asr's wrapper expects the language as the English name from the
# model's ``support_languages`` list (e.g. ``"Chinese"``); ``None`` opts
# into auto-detect. Map the BCP-47-style codes VoiceLayer threads through
# its public surfaces into those names. Anything not in this table
# collapses to ``None`` so we never hand the wrapper a meaningless string.
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

# Reverse mapping: turn the wrapper's English language name into a short
# code on the response so callers see ``zh``/``en``/... in
# ``detected_language`` regardless of the backend (matches MiMo's
# behaviour). Languages absent from the table fall back to the lowercased
# upstream string so unrecognised entries surface verbatim rather than
# silently dropping to ``None``.
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

_ALLOWED_DTYPES = ("float16", "bfloat16", "float32")


# Module-level sentinel so callers can distinguish "VAD found no speech"
# from "VAD was not configured". A bare ``None`` already means "no
# trimmed file, run on the original audio"; the empty-speech case
# additionally requires short-circuiting before the model cold load.
_VAD_EMPTY_SPEECH: Any = object()


def validate_qwen3_asr_provider(
    config: Qwen3AsrConfig | None,
) -> tuple[bool, str | None]:
    """Return whether Qwen3-ASR-1.7B is ready to run.

    Cheap to call: only checks the configured model path exists on disk
    and that the dtype string is one we know how to map. The torch /
    transformers / qwen-asr import work happens on first transcribe so
    ``health`` and ``vl doctor`` stay fast.
    """

    if config is None:
        return False, "No Qwen3-ASR-1.7B model path is configured."

    if config.torch_dtype not in _ALLOWED_DTYPES:
        return False, (
            "VOICELAYER_QWEN3_ASR_DTYPE must be one of "
            f"{_ALLOWED_DTYPES}; got {config.torch_dtype!r}."
        )

    model_dir = Path(config.model_path)
    if not model_dir.is_dir():
        return False, (
            "VOICELAYER_QWEN3_ASR_MODEL_PATH does not exist or is not a "
            f"directory: {config.model_path}"
        )

    return True, None


def _resolve_language_hint(language: str | None) -> str | None:
    """Translate a request ``language`` into Qwen3-ASR's English name.

    Unknown codes (or ``None``/``auto``) collapse to ``None`` so the
    wrapper runs its own auto-detector rather than receiving a
    meaningless string.
    """

    if language is None:
        return None
    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return None
    return _LANGUAGE_TO_QWEN3.get(normalized)


def _wav_duration_seconds(audio_path: Path) -> float:
    """Read a WAV file header and return its duration in seconds.

    Mirrors the MiMo path: stdlib ``wave`` only, raises
    :class:`ProviderInvocationError` on malformed input so the caller
    surfaces a useful error rather than feeding a silent zero into the
    model.
    """

    try:
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
    except wave.Error as exc:
        raise ProviderInvocationError(f"Unable to read WAV header for {audio_path}: {exc}") from exc
    if rate <= 0:
        raise ProviderInvocationError(
            f"WAV at {audio_path} reports a non-positive sample rate ({rate})."
        )
    return frames / float(rate)


def _split_wav_into_segments(
    audio_path: Path,
    max_segment_seconds: float,
    runtime_dir: Path,
) -> list[Path]:
    """Split a WAV into ``max_segment_seconds`` chunks at frame boundaries.

    Default config sets ``long_audio_split_seconds=0`` which short-circuits
    here to ``[audio_path]`` because the upstream ``qwen-asr`` wrapper
    handles long audio internally. Operators who want to force
    worker-side chunking (e.g. to bound peak VRAM on extremely long
    captures) can set a positive value; the splitter then mirrors the
    MiMo path so the cleanup contract is identical for both providers.
    """

    if max_segment_seconds <= 0:
        return [audio_path]

    segments: list[Path] = []
    success = False
    try:
        with wave.open(str(audio_path), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / float(framerate) if framerate > 0 else 0.0
            if duration <= max_segment_seconds:
                success = True
                return [audio_path]

            frames_per_segment = int(max_segment_seconds * framerate)
            if frames_per_segment <= 0:
                success = True
                return [audio_path]

            timestamp_ms = int(time.time() * 1000)
            segment_index = 0
            wav.rewind()
            while True:
                chunk = wav.readframes(frames_per_segment)
                if not chunk:
                    break
                segment_path = runtime_dir / f"qwen3-segment-{timestamp_ms}-{segment_index:04d}.wav"
                with wave.open(str(segment_path), "wb") as out:
                    out.setnchannels(n_channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(chunk)
                segments.append(segment_path)
                segment_index += 1
            success = True
            return segments or [audio_path]
    except (wave.Error, OSError) as exc:
        raise ProviderInvocationError(f"Unable to split WAV at {audio_path}: {exc}") from exc
    finally:
        if not success:
            for partial in segments:
                with contextlib.suppress(OSError):
                    partial.unlink()


def _resolve_torch_dtype(name: str) -> Any:
    """Translate the configured dtype string into a ``torch`` dtype."""

    import torch  # noqa: PLC0415

    mapping = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    return mapping[name]


def _maybe_emit_cpu_warning(device: str) -> None:
    """Print a one-shot stderr warning when running on CPU.

    Qwen3-ASR-1.7B at fp32 on CPU is roughly 10-30x realtime; the
    warning saves operators from thinking the daemon is hung. Honors the
    operator's explicit choice (parity with MiMo): we never abort the
    request, only surface the latency expectation.
    """

    global _CPU_WARNING_EMITTED
    if _CPU_WARNING_EMITTED:
        return
    if not device.startswith("cpu"):
        return
    print(
        "VOICELAYER_QWEN3_ASR_DEVICE=cpu: expect 10-30x realtime latency on Qwen3-ASR-1.7B; "
        "set VOICELAYER_QWEN3_ASR_DEVICE to a `cuda:N` accelerator if available.",
        file=sys.stderr,
        flush=True,
    )
    _CPU_WARNING_EMITTED = True


def _load_qwen3_asr_model(config: Qwen3AsrConfig) -> Any:
    """Load (or return the cached) ``Qwen3ASRModel`` instance for ``config``.

    Synchronizes initialization across worker threads so a burst of
    concurrent transcribe requests during the cold-start window
    serializes on a single load and then races freely on inference.
    """

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
                "Qwen3-ASR-1.7B requires `torch`. Install the qwen3-asr extra "
                '(`uv pip install -e ".[qwen3-asr]"`) or pull a CUDA wheel from '
                "https://download.pytorch.org/whl that matches your GPU."
            ) from exc

        try:
            from qwen_asr import Qwen3ASRModel  # type: ignore[import-not-found]  # noqa: PLC0415
        except ImportError as exc:
            raise ProviderInvocationError(
                "Cannot import `qwen_asr.Qwen3ASRModel`. Install the qwen3-asr "
                'extra: `uv pip install -e ".[qwen3-asr]"`.'
            ) from exc

        try:
            torch_dtype = _resolve_torch_dtype(config.torch_dtype)
        except KeyError as exc:
            raise ProviderInvocationError(
                f"Unsupported VOICELAYER_QWEN3_ASR_DTYPE={config.torch_dtype!r}; "
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
    """Invoke ``model.transcribe`` on one segment.

    Returns ``(text, raw_language)`` where ``raw_language`` is the
    upstream English language name (``"Chinese"``, ``"English"``, ...)
    or ``None`` when the wrapper omits it. Translates upstream errors
    into :class:`ProviderInvocationError` so the worker turns them into
    a clean JSON-RPC ``-32005`` reply.
    """

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
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run Qwen3-ASR-1.7B against a local audio file.

    Honours an optional silero-vad pre-pass (shared with whisper / MiMo
    via ``VOICELAYER_WHISPER_VAD_*``), forwards the audio to the
    upstream wrapper, and returns the concatenated transcript along
    with backend-identifying notes. Worker-owned temporary files
    (VAD-trimmed WAV, optional client-side segment chunks) are cleaned
    up in a finally block whether the call succeeds, raises, or
    short-circuits on no-speech.

    Raises :class:`ProviderInvocationError` on any failure path so the
    worker turns it into a JSON-RPC ``-32005`` error without falling
    back to whisper (explicit selection means explicit failure).
    """

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
        # Qwen3-ASR-1.7B is transcription-only; the wrapper does not expose
        # a translation mode. Surface the limitation rather than silently
        # dropping the flag so callers can route translation through the
        # LLM workflow.
        raise ProviderInvocationError(
            "Qwen3-ASR-1.7B does not support `translate_to_english`; "
            "transcribe with the default whisper.cpp provider or run a "
            "follow-up `translate` request through the LLM workflow."
        )

    extra_notes, prepass_audio_path, vad_trimmed_path = _apply_vad_prepass_for_qwen3_asr(
        audio_file, environ
    )

    worker_owned_files: list[Path] = []
    if vad_trimmed_path is not None:
        worker_owned_files.append(vad_trimmed_path)

    try:
        if prepass_audio_path is _VAD_EMPTY_SPEECH:
            return {
                "text": "",
                "detected_language": None,
                "notes": [
                    "VAD detected no speech; Qwen3-ASR-1.7B inference was skipped.",
                    *extra_notes,
                ],
            }
        if prepass_audio_path is not None:
            audio_path = prepass_audio_path
            audio_file = str(prepass_audio_path)

        runtime_dir = provider_runtime_dir(environ) / "qwen3_asr"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        duration = _wav_duration_seconds(audio_path)
        segments = _split_wav_into_segments(
            audio_path, config.long_audio_split_seconds, runtime_dir
        )
        for segment in segments:
            if segment != audio_path:
                worker_owned_files.append(segment)

        language_hint = _resolve_language_hint(language)
        model = _load_qwen3_asr_model(config)

        transcripts: list[str] = []
        last_raw_language: str | None = None
        for segment_path in segments:
            text, raw_language_name = _run_segment_inference(model, segment_path, language_hint)
            transcripts.append(text)
            if raw_language_name is not None:
                last_raw_language = raw_language_name

        raw_text = " ".join(part for part in transcripts if part).strip()
        text = collapse_nonspeech_transcript(raw_text)
        notes = [
            f"Transcribed by Qwen3-ASR-1.7B (model `{config.model_path}`).",
            f"Device: {config.device}, dtype: {config.torch_dtype}.",
        ]
        if language_hint is not None:
            notes.append(f"Language hint forwarded to qwen-asr: {language_hint}.")
        else:
            notes.append("Language hint left empty; qwen-asr will auto-detect.")
        notes.append(f"Audio duration: {duration:.2f}s; segments processed: {len(segments)}.")
        if len(segments) > 1:
            notes.append(
                "Audio was split client-side because "
                "VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS is non-zero; "
                "transcripts were concatenated with a single space."
            )
        notes.extend(extra_notes)
        if not text:
            notes.append("Qwen3-ASR-1.7B returned no speech for this audio.")

        detected_language: str | None
        if last_raw_language is not None:
            detected_language = _QWEN3_TO_LANGUAGE_CODE.get(
                last_raw_language, last_raw_language.lower() or None
            )
        elif language:
            detected_language = language
        else:
            detected_language = None

        return {
            "text": text,
            "detected_language": detected_language,
            "notes": notes,
        }
    finally:
        # Best-effort cleanup. A leftover chunk in ``runtime_dir`` is not
        # worth surfacing as a transcribe failure; the daemon's restart
        # would sweep the runtime dir anyway. Without this loop, every
        # VAD-gated call leaks ~32 KB/s × trimmed-secs and every chunked
        # long-audio call leaks ~32 KB/s × split-window-secs of PCM,
        # which adds up over a long-running daemon.
        for owned_path in worker_owned_files:
            with contextlib.suppress(OSError):
                owned_path.unlink()


def _apply_vad_prepass_for_qwen3_asr(
    audio_file: str,
    environ: Mapping[str, str] | None,
) -> tuple[list[str], Any, Path | None]:
    """Run silero-vad on ``audio_file`` ahead of Qwen3-ASR inference.

    Returns ``(extra_notes, replacement_audio_path, trimmed_path)``.

    ``replacement_audio_path`` is one of:

    - ``None`` when VAD is unconfigured or fails (transcribe the raw WAV).
    - ``_VAD_EMPTY_SPEECH`` when VAD detected no speech (short-circuit).
    - A :class:`Path` pointing at the trimmed WAV that the caller should
      hand to Qwen3-ASR in place of the original.

    ``trimmed_path`` is the worker-owned WAV that the caller must
    unlink after the model finishes — set whenever
    :func:`apply_vad_prepass` actually wrote a file. ``None`` when VAD
    was unconfigured or raised before writing.

    Mirrors the MiMo wiring so operators see identical pre-pass
    behavior regardless of which backend they routed to.
    """

    vad_config = load_whisper_vad_config(environ)
    if vad_config is None:
        return [], None, None

    try:
        vad_dir = provider_runtime_dir(environ) / "vad"
        trimmed_path, regions = apply_vad_prepass(audio_file, vad_config, vad_dir)
    except ProviderInvocationError as exc:
        return (
            [f"VAD pre-pass failed, transcribing raw audio with Qwen3-ASR-1.7B: {exc}"],
            None,
            None,
        )

    trimmed_owned = Path(trimmed_path)
    if not regions:
        return [], _VAD_EMPTY_SPEECH, trimmed_owned

    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) before Qwen3-ASR-1.7B inference."
    )
    return [note], trimmed_owned, trimmed_owned
