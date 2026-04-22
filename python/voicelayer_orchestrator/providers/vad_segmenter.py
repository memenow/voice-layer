"""Silero-vad pre-pass for whisper transcription.

Runs silero-vad (ONNX, v4 or v5) over a WAV file to detect speech regions
and writes a trimmed WAV containing only the concatenated speech spans.
The whisper providers call :func:`apply_vad_prepass` inside the transcribe
flow so the daemon's view of the JSON-RPC ``transcribe`` contract is
unchanged — VAD is invisible to callers.

``onnxruntime`` and ``numpy`` are optional dependencies; install them via
the ``vad`` extra (``uv sync --extra vad``). Until both imports succeed,
the VAD entry points raise :class:`ProviderInvocationError` with an
installation hint so the transcribe flow can decide whether to disable
VAD or surface the error.
"""

from __future__ import annotations

import wave
from collections import OrderedDict
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import WhisperVadConfig
from voicelayer_orchestrator.providers import ProviderInvocationError

# Bounded LRU: the worker almost always uses a single silero model path, but
# we cap the cache so a misconfiguration that switches paths per request can
# never grow ONNX sessions without bound.
_SESSION_CACHE_MAX_ENTRIES = 2
_SESSION_CACHE: OrderedDict[str, Any] = OrderedDict()

# Hysteresis gap: once in-speech, the per-frame probability must drop at
# least this far below the enter threshold before we consider the speech
# region ended. Prevents chatter around the configured threshold without
# exposing another knob to operators.
_HYSTERESIS_MARGIN = 0.15


def _lazy_imports() -> tuple[Any, Any]:
    """Import onnxruntime and numpy lazily so the VAD extra is optional."""

    try:
        import numpy as np  # noqa: PLC0415
    except ImportError as exc:
        raise ProviderInvocationError(
            "VAD requires `numpy`. Install via `uv sync --extra vad`.",
        ) from exc
    try:
        import onnxruntime as ort  # noqa: PLC0415
    except ImportError as exc:
        raise ProviderInvocationError(
            "VAD requires `onnxruntime`. Install via `uv sync --extra vad`.",
        ) from exc
    return np, ort


def _load_wav_as_float32_mono(
    wav_path: Path,
    target_sample_rate: int,
) -> tuple[Any, int]:
    """Return (samples float32 mono @ target_sample_rate, original_sample_rate)."""

    np, _ = _lazy_imports()
    with wave.open(str(wav_path), "rb") as handle:
        n_channels = handle.getnchannels()
        sample_width = handle.getsampwidth()
        frame_rate = handle.getframerate()
        n_frames = handle.getnframes()
        raw = handle.readframes(n_frames)

    if sample_width == 1:
        pcm = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
    elif sample_width == 2:
        pcm = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    elif sample_width == 4:
        pcm = np.frombuffer(raw, dtype=np.int32).astype(np.float32) / 2147483648.0
    else:
        raise ProviderInvocationError(
            f"Unsupported WAV sample width {sample_width} bytes; expected 1, 2, or 4.",
        )

    if n_channels > 1:
        pcm = pcm.reshape(-1, n_channels).mean(axis=1)

    if frame_rate != target_sample_rate and pcm.size > 0:
        target_len = max(1, int(round(pcm.shape[0] * target_sample_rate / frame_rate)))
        source_positions = np.linspace(0, pcm.shape[0] - 1, num=target_len)
        source_indices = source_positions.astype(np.int64)
        pcm = pcm[source_indices]

    return pcm.astype(np.float32), frame_rate


def _load_vad_session(model_path: str) -> Any:
    """Cache silero-vad sessions by model path; ONNX graph load is expensive."""

    cached = _SESSION_CACHE.get(model_path)
    if cached is not None:
        _SESSION_CACHE.move_to_end(model_path)
        return cached

    _, ort = _lazy_imports()
    model_file = Path(model_path)
    if not model_file.is_file():
        raise ProviderInvocationError(f"Silero-vad model does not exist: {model_path}")

    options = ort.SessionOptions()
    options.intra_op_num_threads = 1
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(model_file),
        sess_options=options,
        providers=["CPUExecutionProvider"],
    )
    _SESSION_CACHE[model_path] = session
    while len(_SESSION_CACHE) > _SESSION_CACHE_MAX_ENTRIES:
        _SESSION_CACHE.popitem(last=False)
    return session


def _window_size_for(sample_rate: int) -> int:
    """Silero-vad window: 32 ms — 512 samples at 16 kHz, 256 at 8 kHz."""

    if sample_rate == 16000:
        return 512
    if sample_rate == 8000:
        return 256
    raise ProviderInvocationError(
        f"silero-vad supports 8000 Hz or 16000 Hz, not {sample_rate}.",
    )


def _speech_probabilities(
    session: Any,
    samples: Any,
    sample_rate: int,
) -> list[float]:
    """Run silero-vad sequentially over fixed-size windows, return per-frame probs."""

    np, _ = _lazy_imports()
    input_names = {node.name for node in session.get_inputs()}
    window = _window_size_for(sample_rate)
    sr_tensor = np.array(sample_rate, dtype=np.int64)
    probs: list[float] = []

    if "state" in input_names:
        state = np.zeros((2, 1, 128), dtype=np.float32)
        for offset in range(0, samples.shape[0] - window + 1, window):
            frame = samples[offset : offset + window].reshape(1, -1)
            output, state = session.run(None, {"input": frame, "state": state, "sr": sr_tensor})
            probs.append(float(output.ravel()[0]))
        return probs

    if "h" in input_names and "c" in input_names:
        h = np.zeros((2, 1, 64), dtype=np.float32)
        c = np.zeros((2, 1, 64), dtype=np.float32)
        for offset in range(0, samples.shape[0] - window + 1, window):
            frame = samples[offset : offset + window].reshape(1, -1)
            output, h, c = session.run(
                None,
                {"input": frame, "h": h, "c": c, "sr": sr_tensor},
            )
            probs.append(float(output.ravel()[0]))
        return probs

    raise ProviderInvocationError(
        f"Unrecognized silero-vad ONNX inputs: {sorted(input_names)}. Expected v4 or v5.",
    )


def _frames_to_regions(
    probs: list[float],
    config: WhisperVadConfig,
    frame_sec: float,
) -> list[tuple[int, int]]:
    """Convert per-frame speech probs into frame-index [start, end) regions.

    Uses two-level thresholds (enter at ``config.threshold``, leave at
    ``max(0, threshold - _HYSTERESIS_MARGIN)``) so borderline
    probabilities don't cause the speech state to chatter frame-by-frame.
    """

    enter_threshold = config.threshold
    # When `config.threshold` is small (e.g. 0.1) the fixed
    # `_HYSTERESIS_MARGIN` (0.15) would pin `leave_threshold` to 0 and an
    # active region could never exit. Cap the effective margin to half
    # the enter threshold so hysteresis scales with aggressive configs
    # while still suppressing chatter at the default 0.5.
    effective_margin = min(_HYSTERESIS_MARGIN, enter_threshold / 2.0)
    leave_threshold = max(0.0, enter_threshold - effective_margin)
    raw: list[tuple[int, int]] = []
    start: int | None = None
    for i, prob in enumerate(probs):
        if start is None:
            if prob >= enter_threshold:
                start = i
        else:
            if prob < leave_threshold:
                raw.append((start, i))
                start = None
    if start is not None:
        raw.append((start, len(probs)))

    min_gap = max(1, int(round((config.min_silence_ms / 1000.0) / frame_sec)))
    merged: list[tuple[int, int]] = []
    for region in raw:
        if merged and region[0] - merged[-1][1] <= min_gap:
            merged[-1] = (merged[-1][0], region[1])
        else:
            merged.append(region)

    min_len = max(1, int(round((config.min_speech_ms / 1000.0) / frame_sec)))
    merged = [r for r in merged if r[1] - r[0] >= min_len]

    pad = int(round((config.speech_pad_ms / 1000.0) / frame_sec))
    total = len(probs)
    padded: list[tuple[int, int]] = []
    for region in merged:
        lo = max(0, region[0] - pad)
        hi = min(total, region[1] + pad)
        padded.append((lo, hi))

    overlapped: list[tuple[int, int]] = []
    for region in padded:
        if overlapped and region[0] <= overlapped[-1][1]:
            overlapped[-1] = (overlapped[-1][0], max(overlapped[-1][1], region[1]))
        else:
            overlapped.append(region)

    max_frames = max(1, int(round(config.max_segment_secs / frame_sec)))
    split: list[tuple[int, int]] = []
    for lo, hi in overlapped:
        cursor = lo
        while hi - cursor > max_frames:
            split.append((cursor, cursor + max_frames))
            cursor += max_frames
        split.append((cursor, hi))
    return split


def detect_speech_regions(
    wav_path: Path,
    config: WhisperVadConfig,
) -> list[tuple[float, float]]:
    """Run silero-vad on ``wav_path`` and return (start_sec, end_sec) regions."""

    samples, _ = _load_wav_as_float32_mono(wav_path, config.sample_rate)
    if samples.size == 0:
        return []

    session = _load_vad_session(config.model_path)
    probs = _speech_probabilities(session, samples, config.sample_rate)
    frame_sec = _window_size_for(config.sample_rate) / config.sample_rate
    regions = _frames_to_regions(probs, config, frame_sec)
    return [(lo * frame_sec, hi * frame_sec) for lo, hi in regions]


def extract_speech_wav(
    wav_path: Path,
    regions: list[tuple[float, float]],
    out_dir: Path,
    config: WhisperVadConfig,
) -> Path:
    """Write a trimmed WAV (16-bit mono at ``config.sample_rate``) of speech spans."""

    np, _ = _lazy_imports()
    out_dir.mkdir(parents=True, exist_ok=True)
    samples, _ = _load_wav_as_float32_mono(wav_path, config.sample_rate)

    if regions:
        chunks: list[Any] = []
        for start_sec, end_sec in regions:
            start = max(0, int(round(start_sec * config.sample_rate)))
            end = min(samples.shape[0], int(round(end_sec * config.sample_rate)))
            if end > start:
                chunks.append(samples[start:end])
        stitched = np.concatenate(chunks) if chunks else np.zeros(0, dtype=np.float32)
    else:
        stitched = np.zeros(0, dtype=np.float32)

    suffix = ".vad-trimmed.wav" if stitched.size > 0 else ".vad-empty.wav"
    trimmed_path = out_dir / f"{wav_path.stem}{suffix}"
    pcm = np.clip(stitched * 32768.0, -32768.0, 32767.0).astype(np.int16)
    with wave.open(str(trimmed_path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(config.sample_rate)
        handle.writeframes(pcm.tobytes())
    return trimmed_path


def apply_vad_prepass(
    audio_file: str,
    config: WhisperVadConfig,
    out_dir: Path,
) -> tuple[str, list[tuple[float, float]]]:
    """Return ``(trimmed_wav_path, regions)`` after applying silero-vad.

    Raises :class:`ProviderInvocationError` when the input file is missing
    or ONNX/numpy are not importable.
    """

    wav_path = Path(audio_file)
    if not wav_path.is_file():
        raise ProviderInvocationError(f"VAD input does not exist: {audio_file}")
    regions = detect_speech_regions(wav_path, config)
    trimmed = extract_speech_wav(wav_path, regions, out_dir, config)
    return str(trimmed), regions


def reset_session_cache() -> None:
    """Drop cached ONNX sessions (used in tests)."""

    _SESSION_CACHE.clear()
