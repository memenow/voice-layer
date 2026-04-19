#!/usr/bin/env python3
"""Generate a deterministic silent WAV fixture for Phase 3 pre-flight benchmarking.

The output is 3 seconds of 16 kHz, 16-bit mono PCM silence written to
``tests/fixtures/small.wav``. It has no copyrighted audio content and is
reproducible bit-for-bit, which makes it safe to commit and to compare across
hosts when timing ``whisper-cli`` cold-start.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

DEFAULT_DURATION_SECONDS = 3.0
DEFAULT_SAMPLE_RATE_HZ = 16000
DEFAULT_OUTPUT = Path("tests/fixtures/small.wav")


def generate_silent_wav(
    output: Path,
    duration_seconds: float = DEFAULT_DURATION_SECONDS,
    sample_rate_hz: int = DEFAULT_SAMPLE_RATE_HZ,
) -> Path:
    """Write a mono 16-bit PCM silent WAV file to ``output`` and return its path."""

    output.parent.mkdir(parents=True, exist_ok=True)
    frame_count = int(duration_seconds * sample_rate_hz)
    with wave.open(str(output), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate_hz)
        handle.writeframes(b"\x00\x00" * frame_count)
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output path (default: {DEFAULT_OUTPUT})",
    )
    parser.add_argument(
        "--duration-seconds",
        type=float,
        default=DEFAULT_DURATION_SECONDS,
        help=f"Silence duration in seconds (default: {DEFAULT_DURATION_SECONDS})",
    )
    parser.add_argument(
        "--sample-rate-hz",
        type=int,
        default=DEFAULT_SAMPLE_RATE_HZ,
        help=f"Sample rate in Hz (default: {DEFAULT_SAMPLE_RATE_HZ})",
    )
    args = parser.parse_args()

    path = generate_silent_wav(
        args.output,
        duration_seconds=args.duration_seconds,
        sample_rate_hz=args.sample_rate_hz,
    )
    print(f"wrote {path} ({args.duration_seconds}s, {args.sample_rate_hz} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
