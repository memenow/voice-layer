"""WAV duration, splitting, and concatenation utilities."""

from __future__ import annotations

import contextlib
import os
import tempfile
import time
import wave
from pathlib import Path
from typing import Any

from voicelayer_worker.errors import ProviderInvocationError


def wav_duration_seconds(audio_path: Path) -> float:
    """Return the duration of a PCM WAV file in seconds."""

    try:
        with wave.open(str(audio_path), "rb") as wav:
            frames = wav.getnframes()
            rate = wav.getframerate()
    except (wave.Error, OSError) as exc:
        raise ProviderInvocationError(f"Unable to read WAV header for {audio_path}: {exc}") from exc
    if rate <= 0:
        raise ProviderInvocationError(
            f"WAV at {audio_path} reports a non-positive sample rate ({rate})."
        )
    return frames / float(rate)


def split_wav_segments(
    audio_path: Path,
    max_segment_seconds: float,
    runtime_dir: Path,
    *,
    prefix: str = "segment",
) -> list[Path]:
    """Split ``audio_path`` into frame-aligned WAV chunks at most ``max_segment_seconds`` long.

    Returns ``[audio_path]`` unchanged when splitting is disabled or the file already fits.
    Rolls back partial chunk files if an I/O or WAV error occurs mid-split.
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

            runtime_dir.mkdir(parents=True, exist_ok=True)
            timestamp_ms = int(time.time() * 1000)
            index = 0
            wav.rewind()
            while True:
                chunk = wav.readframes(frames_per_segment)
                if not chunk:
                    break
                segment_path = runtime_dir / f"{prefix}-{timestamp_ms}-{index:04d}.wav"
                with wave.open(str(segment_path), "wb") as out:
                    out.setnchannels(n_channels)
                    out.setsampwidth(sampwidth)
                    out.setframerate(framerate)
                    out.writeframes(chunk)
                segments.append(segment_path)
                index += 1

            success = True
            return segments or [audio_path]
    except (wave.Error, OSError) as exc:
        raise ProviderInvocationError(f"Unable to split WAV at {audio_path}: {exc}") from exc
    finally:
        if not success:
            for partial in segments:
                with contextlib.suppress(OSError):
                    partial.unlink()


def stitch_wav_segments(audio_files: list[str], out_file: str) -> dict[str, Any]:
    """Concatenate format-identical WAV files into ``out_file`` atomically."""

    if not audio_files:
        raise ProviderInvocationError("stitch_wav_segments requires at least one audio file.")

    out_path = Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.NamedTemporaryFile(
        dir=out_path.parent,
        suffix=".wav",
        delete=False,
    ) as tmp:
        tmp_path = Path(tmp.name)

    first_params: tuple[int, int, int] | None = None
    total_frames = 0
    try:
        with contextlib.ExitStack() as stack:
            writer: wave.Wave_write | None = None
            for idx, src in enumerate(audio_files):
                src_path = Path(src)
                if not src_path.is_file():
                    raise ProviderInvocationError(
                        f"stitch input at index {idx} does not exist: {src}"
                    )
                with wave.open(str(src_path), "rb") as reader:
                    params = (
                        reader.getnchannels(),
                        reader.getsampwidth(),
                        reader.getframerate(),
                    )
                    if writer is None:
                        writer = stack.enter_context(wave.open(str(tmp_path), "wb"))
                        first_params = params
                        writer.setnchannels(params[0])
                        writer.setsampwidth(params[1])
                        writer.setframerate(params[2])
                    elif params != first_params:
                        raise ProviderInvocationError(
                            f"stitch input at index {idx} has incompatible "
                            f"WAV parameters {params} vs first file's {first_params}"
                        )
                    writer.writeframes(reader.readframes(reader.getnframes()))
                    total_frames += reader.getnframes()
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    frame_rate = first_params[2] if first_params is not None else 0
    duration_secs = float(total_frames) / float(frame_rate) if frame_rate > 0 else 0.0
    return {
        "audio_file": str(out_path),
        "segment_count": len(audio_files),
        "duration_secs": duration_secs,
    }
