"""WAV probe concatenation helper.

The VAD-gated dictation orchestrator buffers 1-2 s probe WAV files and
flushes them as one logical speech unit to the transcribe provider. This
module exposes :func:`stitch_wav_segments`, the stdlib-based concatenator
backing the ``stitch_wav_segments`` JSON-RPC method.

Constraints:

- All input WAVs must share sample rate, sample width, and channel
  count. The first file's parameters become the output's parameters;
  a mismatch anywhere in the list raises
  :class:`ProviderInvocationError` (no silent re-sampling or coercion).
- Empty input lists are rejected — there is nothing meaningful to merge.
- The output is written to a temporary file in the same directory as
  ``out_file`` and then renamed via :func:`os.replace`. This keeps
  readers from observing a half-written WAV if the caller crashes
  mid-write.
"""

from __future__ import annotations

import contextlib
import os
import tempfile
import wave
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.providers import ProviderInvocationError


def stitch_wav_segments(audio_files: list[str], out_file: str) -> dict[str, Any]:
    """Concatenate ``audio_files`` into ``out_file`` and return metadata.

    The returned payload is the literal wire result of the
    ``stitch_wav_segments`` RPC::

        {"audio_file": <out_file>, "segment_count": <N>, "duration_secs": <float>}
    """

    if not audio_files:
        raise ProviderInvocationError(
            "stitch_wav_segments requires at least one audio file.",
        )

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
    # The wave writer must have nchannels/sampwidth/framerate set before
    # close() flushes the header; opening it eagerly and then erroring out
    # leaves us unable to close cleanly. Defer the writer's lifetime to
    # an ExitStack so it only opens after the first reader supplies the
    # parameters.
    try:
        with contextlib.ExitStack() as stack:
            writer: wave.Wave_write | None = None
            for idx, src in enumerate(audio_files):
                src_path = Path(src)
                if not src_path.is_file():
                    raise ProviderInvocationError(
                        f"stitch input at index {idx} does not exist: {src}",
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
                            f"WAV parameters {params} vs first file's {first_params}",
                        )
                    frames = reader.readframes(reader.getnframes())
                    writer.writeframes(frames)
                    total_frames += reader.getnframes()
        os.replace(tmp_path, out_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    assert first_params is not None  # noqa: S101 — caught above when list empty.
    frame_rate = first_params[2]
    duration_secs = float(total_frames) / float(frame_rate) if frame_rate > 0 else 0.0
    return {
        "audio_file": str(out_path),
        "segment_count": len(audio_files),
        "duration_secs": duration_secs,
    }
