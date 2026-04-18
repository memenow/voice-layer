"""whisper.cpp CLI provider for VoiceLayer."""

from __future__ import annotations

import os
import shutil
import subprocess
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import WhisperCppConfig
from voicelayer_orchestrator.providers import ProviderInvocationError, provider_runtime_dir


def validate_whisper_provider(
    config: WhisperCppConfig | None,
) -> tuple[bool, str | None]:
    """Return whether `whisper-cli` is ready to run."""

    if config is None:
        return False, "No whisper.cpp model path is configured."

    binary = shutil.which(config.binary) if "/" not in config.binary else config.binary
    if binary is None or not Path(binary).exists():
        return False, f"Unable to find `{config.binary}`."

    model_path = Path(config.model_path)
    if not model_path.is_file():
        return False, f"Configured whisper model does not exist: {config.model_path}"

    return True, None


def transcribe_with_whisper_cli(
    params: Mapping[str, Any],
    config: WhisperCppConfig,
    environ: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Run `whisper-cli` against a local audio file and return the transcript."""

    audio_file = str(params.get("audio_file", "")).strip()
    language = str(params.get("language") or "auto").strip()
    translate_to_english = bool(params.get("translate_to_english", False))

    if not audio_file:
        raise ProviderInvocationError("Transcribe requests require `audio_file`.")
    if not Path(audio_file).is_file():
        raise ProviderInvocationError(f"Audio file does not exist: {audio_file}")

    ready, error = validate_whisper_provider(config)
    if not ready:
        raise ProviderInvocationError(error or "whisper.cpp is not ready.")

    runtime_dir = provider_runtime_dir(environ)
    output_stem = runtime_dir / f"transcribe-{int(time.time() * 1000)}"
    command = [
        config.binary,
        "-m",
        config.model_path,
        "-f",
        audio_file,
        "-otxt",
        "-of",
        str(output_stem),
        "-np",
        "-l",
        language or "auto",
        *config.extra_args,
    ]
    if translate_to_english:
        command.append("-tr")
    if config.no_gpu:
        command.append("-ng")

    result = subprocess.run(  # noqa: S603
        command,
        capture_output=True,
        text=True,
        timeout=config.timeout_seconds,
        env=(dict(os.environ) | dict(environ or {})),
        check=False,
    )
    if result.returncode != 0:
        raise ProviderInvocationError(
            f"whisper.cpp failed with exit code {result.returncode}: {result.stderr.strip()}"
        )

    transcript_file = output_stem.with_suffix(".txt")
    if not transcript_file.is_file():
        raise ProviderInvocationError(
            f"whisper.cpp did not produce the expected transcript file: {transcript_file}"
        )

    text = transcript_file.read_text(encoding="utf-8").strip()
    if not text:
        raise ProviderInvocationError("whisper.cpp produced an empty transcript.")

    return {
        "text": text,
        "detected_language": None if language == "auto" else language,
        "notes": [
            f"Transcribed by `{config.binary}` using model `{config.model_path}`.",
            "Input audio must be supported by whisper.cpp CLI; "
            "the official example supports flac, mp3, ogg, and wav.",
        ],
    }
