"""Xiaomi MiMo-V2.5-ASR provider for VoiceLayer.

The provider runs through Xiaomi's `MimoAudio` Python wrapper, which
loads an 8B-parameter Qwen2-style decoder LM over the discrete RVQ
audio tokens emitted by the companion `MiMo-Audio-Tokenizer`. Both
weight directories must be downloaded locally; the wrapper class
lives in the upstream source tree (no wheel published today) so the
caller may need to point ``VOICELAYER_MIMO_REPO_PATH`` at a local
checkout to anchor it on ``sys.path``.

The model is loaded on the first transcribe call and kept warm in a
module-level cache for the lifetime of the worker process. Cold load
takes tens of seconds; per-call latency on a single consumer CUDA
GPU with bf16 + flash-attn is in the seconds range — significantly
slower than the whisper.cpp chain but with measurably better quality
and native multilingual support (zh / en / yue / wuu / nan / cmn-sc
/ zh-en code-switch). See `docs/guides/local-asr-provider.md` for
the hardware preconditions and the comparison table.

Optional. The whisper.cpp chain remains the default ASR provider.
Callers select MiMo by setting `TranscribeRequest.provider_id =
"mimo_v2_5_asr"`.
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

from voicelayer_orchestrator.config import MimoAsrConfig, load_whisper_vad_config
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    collapse_nonspeech_transcript,
    provider_runtime_dir,
)
from voicelayer_orchestrator.providers.vad_segmenter import apply_vad_prepass

# Model cache + lock. Keyed by the configuration tuple that determines
# weight identity (model + tokenizer paths) and inference target
# (device). Precision is fixed to bf16 inside the upstream wrapper, so
# it is intentionally not part of the cache key. Multiple unique
# configurations are theoretically possible (operator switches between
# cuda:0 and cuda:1 mid-session) but the typical case is a single entry
# kept warm for the worker lifetime. The cache is intentionally
# unbounded; switching keys is operator-driven and infrequent.
_MODEL_CACHE: dict[tuple[str, str, str], Any] = {}
_MODEL_CACHE_LOCK = threading.Lock()


# Whisper-style language codes that map cleanly to MiMo's `audio_tag`
# argument. The wrapper accepts `<chinese>` for any zh variant and
# `<english>` for en; everything else falls through to the operator's
# configured default and ultimately to MiMo's auto-detect.
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


def validate_mimo_provider(
    config: MimoAsrConfig | None,
) -> tuple[bool, str | None]:
    """Return whether MiMo-V2.5-ASR is ready to run.

    Cheap to call: only checks that the configured paths exist on
    disk so `health` and `vl doctor` stay fast. The torch /
    flash-attn / model-load work happens on first transcribe.
    """

    if config is None:
        return False, "No MiMo-V2.5-ASR model paths are configured."

    model_dir = Path(config.model_path)
    if not model_dir.is_dir():
        return False, (
            f"VOICELAYER_MIMO_MODEL_PATH does not exist or is not a directory: {config.model_path}"
        )

    tokenizer_dir = Path(config.tokenizer_path)
    if not tokenizer_dir.is_dir():
        return False, (
            "VOICELAYER_MIMO_TOKENIZER_PATH does not exist or is not "
            f"a directory: {config.tokenizer_path}"
        )

    if config.repo_path is not None:
        repo_dir = Path(config.repo_path)
        if not repo_dir.is_dir():
            return False, (
                "VOICELAYER_MIMO_REPO_PATH does not exist or is not "
                f"a directory: {config.repo_path}"
            )

    return True, None


def _resolve_audio_tag(
    language: str | None,
    config: MimoAsrConfig,
) -> str | None:
    """Translate the request `language` into MiMo's `audio_tag` string.

    Falls back to the operator-configured default
    (`config.audio_tag`) when the request omits a language or sets
    `auto`. Unknown languages also fall back so MiMo's wrapper does
    its own auto-detect rather than receiving a meaningless tag.
    """

    if language is None:
        return config.audio_tag
    normalized = language.strip().lower()
    if not normalized or normalized == "auto":
        return config.audio_tag
    return _LANGUAGE_TO_AUDIO_TAG.get(normalized, config.audio_tag)


def _wav_duration_seconds(audio_path: Path) -> float:
    """Read a WAV file header and return its duration in seconds.

    Uses the stdlib `wave` module; relies on the recorder writing
    PCM WAV files (the existing pw-record / arecord paths satisfy
    this). Non-WAV inputs raise ProviderInvocationError so the caller
    can surface a useful error rather than feeding a silent zero
    into MiMo.
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
    """Split a WAV into ≤ `max_segment_seconds` chunks and return them.

    The split is performed at frame boundaries via the stdlib `wave`
    module: no resampling, no decoding, no re-encoding. Mitigates
    Xiaomi MiMo issue #6 (decoder repetition past ~3 minutes of
    single-pass decoding) by keeping each `asr_sft` call below the
    empirically safe horizon.

    Returns the original path inside a single-element list when the
    input fits in a single chunk so the caller's iteration stays
    branch-free.
    """

    if max_segment_seconds <= 0:
        return [audio_path]

    try:
        with wave.open(str(audio_path), "rb") as wav:
            n_channels = wav.getnchannels()
            sampwidth = wav.getsampwidth()
            framerate = wav.getframerate()
            n_frames = wav.getnframes()
            duration = n_frames / float(framerate) if framerate > 0 else 0.0
            if duration <= max_segment_seconds:
                return [audio_path]

            frames_per_segment = int(max_segment_seconds * framerate)
            if frames_per_segment <= 0:
                return [audio_path]

            segments: list[Path] = []
            timestamp_ms = int(time.time() * 1000)
            segment_index = 0
            wav.rewind()
            while True:
                chunk = wav.readframes(frames_per_segment)
                if not chunk:
                    break
                segment_path = runtime_dir / f"mimo-segment-{timestamp_ms}-{segment_index:04d}.wav"
                with wave.open(str(segment_path), "wb") as out:
                    out.setnchannels(n_channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(chunk)
                segments.append(segment_path)
                segment_index += 1
            return segments or [audio_path]
    except wave.Error as exc:
        raise ProviderInvocationError(f"Unable to split WAV at {audio_path}: {exc}") from exc


def _ensure_repo_on_path(repo_path: str | None) -> None:
    """Prepend Xiaomi's MiMo-V2.5-ASR repo root to sys.path.

    The wrapper class lives at `<repo>/src/mimo_audio/mimo_audio.py`
    in the upstream source tree; the upstream README imports it as
    ``from src.mimo_audio.mimo_audio import MimoAudio``. Both
    ``src/`` and ``src/mimo_audio/`` ship without ``__init__.py`` and
    rely on PEP 420 namespace packages, while
    ``src/mimo_audio_tokenizer/`` is a regular package referenced via
    a relative import (``from ..mimo_audio_tokenizer import ...``).
    For that relative import to resolve, the **repo root** must be on
    ``sys.path`` so ``src`` itself is importable.

    Operators clone the repo and point ``VOICELAYER_MIMO_REPO_PATH``
    at the repo root. This helper tolerates the legacy convention of
    pointing at ``<repo>/src`` by walking one level up when it sees
    that layout. Skipped when ``repo_path`` is ``None`` so future
    pip-editable installs work without changing config.
    """

    if not repo_path:
        return
    candidate = Path(repo_path)
    # Legacy callers may have pointed at `<repo>/src` directly when an
    # earlier draft of this provider added the inner directory to
    # sys.path. The relative `..mimo_audio_tokenizer` import in the
    # wrapper requires `src` to be a child of an importable directory,
    # so promote to the repo root if we recognise the layout.
    if candidate.name == "src" and (candidate / "mimo_audio").is_dir():
        candidate = candidate.parent
    candidate_str = str(candidate)
    if candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)


def _load_mimo_model(config: MimoAsrConfig) -> Any:
    """Load (or return the cached) `MimoAudio` instance for `config`.

    Synchronizes initialization across worker threads so a burst of
    concurrent transcribe requests during the cold-start window
    serializes on a single load and then races freely on inference.
    """

    cache_key = (
        config.model_path,
        config.tokenizer_path,
        config.device,
    )
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
                "MiMo-V2.5-ASR requires `torch`. Install a CUDA build that "
                "matches your GPU compute capability — e.g., "
                "`uv pip install --index-url https://download.pytorch.org/whl/cu128 "
                '"torch>=2.8.0,<3.0" "torchaudio>=2.8.0,<3.0"` for Blackwell — '
                'then `uv pip install -e ".[mimo]"` for the rest of the stack.'
            ) from exc
        try:
            # Upstream import convention from the model card README:
            # `from src.mimo_audio.mimo_audio import MimoAudio`.
            # `src` is a PEP 420 namespace package; the relative
            # `..mimo_audio_tokenizer` import inside the wrapper only
            # resolves when the repo root (parent of `src`) is on
            # sys.path, so the import path keeps the `src.` prefix.
            from src.mimo_audio.mimo_audio import (
                MimoAudio,  # type: ignore[import-not-found]  # noqa: PLC0415
            )
        except ImportError as exc:
            raise ProviderInvocationError(
                "Cannot import `src.mimo_audio.mimo_audio.MimoAudio`. Set "
                "VOICELAYER_MIMO_REPO_PATH to the root of a checkout of "
                "https://github.com/XiaomiMiMo/MiMo-V2.5-ASR (the directory "
                "that contains `src/`)."
            ) from exc

        model = _instantiate_mimo_audio(MimoAudio, config)
        _MODEL_CACHE[cache_key] = model
        return model


def _instantiate_mimo_audio(
    mimo_audio_cls: Any,
    config: MimoAsrConfig,
) -> Any:
    """Construct a `MimoAudio` instance and convert errors uniformly.

    The upstream signature is
    ``MimoAudio(model_path, mimo_audio_tokenizer_path, device=None)``;
    precision is hardcoded to bfloat16 inside the wrapper. We pass our
    config keyword arguments through verbatim and surface any failure
    as :class:`ProviderInvocationError` so the worker can return a
    clean JSON-RPC error.
    """

    try:
        return mimo_audio_cls(
            model_path=config.model_path,
            mimo_audio_tokenizer_path=config.tokenizer_path,
            device=config.device,
        )
    except Exception as exc:
        raise ProviderInvocationError(f"Failed to load MiMo-V2.5-ASR model: {exc}") from exc


def transcribe_with_mimo(
    params: Mapping[str, Any],
    config: MimoAsrConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run MiMo-V2.5-ASR against a local audio file.

    Splits inputs longer than ``config.long_audio_split_seconds`` to
    work around upstream issue #6 (decoder repetition past ~3
    minutes), runs each segment through `MimoAudio.asr_sft`, and
    returns the concatenated text. The `notes` field surfaces the
    backend identity, device, audio tag, and segment count
    so operators can tell at a glance which path produced a
    transcript.

    Raises :class:`ProviderInvocationError` on any failure path so
    the worker can convert it into a JSON-RPC ``-32005`` error
    without falling back to whisper (explicit selection means
    explicit failure).
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

    ready, error = validate_mimo_provider(config)
    if not ready:
        raise ProviderInvocationError(error or "MiMo-V2.5-ASR is not ready.")

    if translate_to_english:
        # MiMo's `asr_sft` does not expose a translation mode today;
        # the operator must run translation via the LLM workflows
        # instead. Surface the limitation rather than silently
        # ignoring the flag.
        raise ProviderInvocationError(
            "MiMo-V2.5-ASR does not support `translate_to_english` today; "
            "transcribe with the default whisper.cpp provider or run a "
            "follow-up `translate` request through the LLM workflow."
        )

    # Optional silero-vad pre-pass. Shares the `VOICELAYER_WHISPER_VAD_*`
    # env vars with the whisper chain because silero-vad is ASR-backend
    # agnostic — operators configure it once and both transcribe paths
    # honor it. Behavior matches the whisper wiring exactly:
    #
    # - VAD unconfigured: pass the raw WAV to the model unchanged.
    # - VAD detects no speech: short-circuit to an empty transcript.
    #   Skips the (expensive) MiMo cold load and the per-segment
    #   inference loop, which is the whole reason VAD-on-MiMo is worth
    #   doing — MiMo can hallucinate transcripts on pure silence.
    # - VAD raises: log the failure note and fall back to the raw WAV
    #   so a transient VAD error never makes the transcribe path go
    #   dark.
    extra_notes, prepass_audio_path, vad_trimmed_path = _apply_vad_prepass_for_mimo(
        audio_file, environ
    )

    # Track every WAV the worker created so the finally block can
    # unlink them whether transcribe returns, raises, or short-circuits
    # on no-speech. Excludes the caller-supplied audio (the dictation
    # pipeline owns its lifetime via `keep_audio`).
    worker_owned_files: list[Path] = []
    if vad_trimmed_path is not None:
        worker_owned_files.append(vad_trimmed_path)

    try:
        if prepass_audio_path is _VAD_EMPTY_SPEECH:
            return {
                "text": "",
                "detected_language": None,
                "notes": [
                    "VAD detected no speech; MiMo inference was skipped.",
                    *extra_notes,
                ],
            }
        if prepass_audio_path is not None:
            audio_path = prepass_audio_path
            audio_file = str(prepass_audio_path)

        runtime_dir = provider_runtime_dir(environ) / "mimo"
        runtime_dir.mkdir(parents=True, exist_ok=True)
        duration = _wav_duration_seconds(audio_path)
        segments = _split_wav_into_segments(
            audio_path, config.long_audio_split_seconds, runtime_dir
        )
        # Per-chunk WAVs from `_split_wav_into_segments` are worker-owned
        # too. The splitter returns `[audio_path]` unchanged when the
        # input fits in a single chunk, so filter that case out — the
        # trimmed-VAD WAV (if any) is already tracked above, and the
        # caller's WAV must not be deleted here.
        for segment in segments:
            if segment != audio_path:
                worker_owned_files.append(segment)

        audio_tag = _resolve_audio_tag(language, config)
        model = _load_mimo_model(config)

        transcripts: list[str] = []
        for segment_path in segments:
            text = _run_segment_inference(model, segment_path, audio_tag)
            transcripts.append(text)

        raw_text = " ".join(part for part in transcripts if part).strip()
        text = collapse_nonspeech_transcript(raw_text)
        notes = [
            f"Transcribed by MiMo-V2.5-ASR (model `{config.model_path}`).",
            f"Device: {config.device}, dtype: bfloat16 (wrapper-default).",
        ]
        if audio_tag is not None:
            notes.append(f"Audio tag forwarded to the wrapper: {audio_tag}.")
        else:
            notes.append("Audio tag left to MiMo's auto-detect.")
        notes.append(f"Audio duration: {duration:.2f}s; segments processed: {len(segments)}.")
        if len(segments) > 1:
            notes.append(
                "Audio was split to mitigate MiMo issue #6 (decoder repetition past "
                "~3 minutes); transcripts were concatenated with a single space."
            )
        notes.extend(extra_notes)
        if not text:
            notes.append("MiMo-V2.5-ASR returned no speech for this audio.")

        detected_language: str | None = None
        if audio_tag == "<chinese>":
            detected_language = language or "zh"
        elif audio_tag == "<english>":
            detected_language = language or "en"
        elif language:
            detected_language = language

        return {
            "text": text,
            "detected_language": detected_language,
            "notes": notes,
        }
    finally:
        # Best-effort cleanup. A leftover chunk in `runtime_dir` is
        # not worth surfacing as a transcribe failure; the daemon's
        # restart will sweep the runtime dir anyway. Without this loop
        # each long-audio call leaks ~32 KB/s × split-window-secs of
        # PCM, and each VAD-trimmed run leaks ~32 KB/s × trimmed-secs;
        # both add up over a long-running daemon.
        for owned_path in worker_owned_files:
            with contextlib.suppress(OSError):
                owned_path.unlink()


# Module-level sentinel so callers can distinguish "VAD found no speech"
# from "VAD was not configured". A bare `None` already means "no
# trimmed file, run on the original audio"; the empty-speech case
# additionally requires short-circuiting before the MiMo cold load.
_VAD_EMPTY_SPEECH: Any = object()


def _apply_vad_prepass_for_mimo(
    audio_file: str,
    environ: Mapping[str, str] | None,
) -> tuple[list[str], Any, Path | None]:
    """Run silero-vad on ``audio_file`` ahead of MiMo inference.

    Returns ``(extra_notes, replacement_audio_path, trimmed_path)``.

    ``replacement_audio_path`` (the second member) is one of:

    - ``None`` when VAD is unconfigured or fails (transcribe the raw WAV).
    - ``_VAD_EMPTY_SPEECH`` when VAD detected no speech (short-circuit).
    - A :class:`Path` pointing at the trimmed WAV that the caller should
      hand to MiMo in place of the original.

    ``trimmed_path`` is the worker-owned WAV that the caller must
    unlink after MiMo finishes — set whenever ``apply_vad_prepass``
    actually wrote a file (both the trimmed-speech and empty-speech
    cases produce a sidecar WAV). ``None`` when VAD was unconfigured
    or raised before writing.

    Mirrors the whisper-side ``_apply_vad_prepass_if_configured`` so
    operators see identical pre-pass behavior regardless of which
    backend they routed to.
    """

    vad_config = load_whisper_vad_config(environ)
    if vad_config is None:
        return [], None, None

    try:
        vad_dir = provider_runtime_dir(environ) / "vad"
        trimmed_path, regions = apply_vad_prepass(audio_file, vad_config, vad_dir)
    except ProviderInvocationError as exc:
        return (
            [f"VAD pre-pass failed, transcribing raw audio with MiMo: {exc}"],
            None,
            None,
        )

    trimmed_owned = Path(trimmed_path)
    if not regions:
        return [], _VAD_EMPTY_SPEECH, trimmed_owned

    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) before MiMo inference."
    )
    return [note], trimmed_owned, trimmed_owned


def _load_wav_as_tensor(segment_path: Path) -> tuple[Any, int]:
    """Read a WAV file into a mono float32 ``torch.Tensor``.

    The upstream wrapper's default path for string inputs goes through
    ``torchaudio.load``, which on ``torchaudio>=2.10`` delegates to the
    ``torchcodec`` package and pulls in FFmpeg shared libraries. That
    chain is brittle in two ways: torchcodec built against
    ``torch>=2.11`` cannot load on the ``torch==2.10`` runtime we pin
    for the flash-attn 2.8.x wheel, and the FFmpeg loader path
    requires ``libavutil.so.5{6,7}`` which is not present in many
    minimal Ubuntu installs. Both issues disappear when we hand the
    wrapper a pre-loaded tensor (see ``preprocess_input`` in the
    wrapper, which short-circuits ``torchaudio.load`` on
    ``isinstance(input, torch.Tensor)``).

    Uses ``soundfile`` (pulled in transitively by ``librosa``) so we
    only depend on ``libsndfile``, which is part of the base Ubuntu
    install and already required by the rest of the audio stack.
    """

    import soundfile as sf  # noqa: PLC0415
    import torch  # noqa: PLC0415

    data, sr = sf.read(str(segment_path), dtype="float32", always_2d=False)
    if data.ndim == 2:
        # Reduce stereo (or any multi-channel input) to mono by mean —
        # the wrapper does the same when it loads via torchaudio, so
        # this preserves end-to-end behaviour.
        data = data.mean(axis=1)
    return torch.from_numpy(data), int(sr)


def _run_segment_inference(
    model: Any,
    segment_path: Path,
    audio_tag: str | None,
) -> str:
    """Invoke `model.asr_sft` on one segment, tolerating wrapper drift.

    Loads the WAV ourselves and hands the wrapper a tensor so the
    upstream ``torchaudio.load`` → ``torchcodec`` → FFmpeg dependency
    chain stays out of the runtime path; see ``_load_wav_as_tensor``
    for the rationale.
    """

    wav_tensor, sample_rate = _load_wav_as_tensor(segment_path)
    # The wrapper expects waveforms at the audio tokenizer's sample
    # rate. Reuse its own resample helper rather than duplicating the
    # logic so a future change in upstream's preprocessing flows
    # through automatically.
    if hasattr(model, "resample_audio_if_needed"):
        wav_tensor = model.resample_audio_if_needed(wav_tensor, sample_rate)

    kwargs: dict[str, Any] = {}
    if audio_tag is not None:
        kwargs["audio_tag"] = audio_tag
    try:
        result = model.asr_sft(wav_tensor, **kwargs)
    except TypeError as exc:
        # Older wrapper releases do not accept the `audio_tag` kwarg
        # and raise `TypeError: ... unexpected keyword argument
        # 'audio_tag'`. Retry without the kwarg so the call still
        # goes through. A `TypeError` from anywhere else (e.g., a
        # tensor-shape mismatch inside the wrapper) is real and must
        # surface as a provider failure rather than getting silently
        # swapped for a second untagged retry.
        message = str(exc)
        is_audio_tag_arity_error = (
            audio_tag is not None
            and "audio_tag" in message
            and ("unexpected keyword argument" in message or "got an unexpected" in message)
        )
        if not is_audio_tag_arity_error:
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
