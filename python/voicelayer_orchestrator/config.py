"""Environment-backed configuration for VoiceLayer provider backends."""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for a locally hosted OpenAI-compatible chat endpoint."""

    endpoint: str
    model: str
    api_key: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class LlamaServerLaunchConfig:
    """Configuration for automatically launching `llama-server`."""

    server_bin: str
    model_path: str | None
    hf_repo: str | None
    extra_args: tuple[str, ...]
    launch_timeout_seconds: float
    poll_interval_seconds: float


@dataclass(frozen=True)
class WhisperCppConfig:
    """Configuration for invoking `whisper-cli`."""

    binary: str
    model_path: str
    timeout_seconds: float
    no_gpu: bool
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class WhisperVadConfig:
    """Configuration for the silero-vad pre-pass applied to transcribe inputs.

    The VAD layer runs inside the Python worker before a WAV is handed to
    whisper. When it finds speech regions it concatenates them into a
    trimmed WAV and feeds that to the transcriber; the daemon and JSON-RPC
    contract never see the split, so callers behave identically whether
    VAD is enabled or not.
    """

    model_path: str
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    max_segment_secs: float
    sample_rate: int


@dataclass(frozen=True)
class WhisperServerConfig:
    """Configuration for talking to a persistent `whisper-server` HTTP endpoint."""

    host: str
    port: int
    timeout_seconds: float
    auto_start: bool
    server_bin: str | None
    model_path: str | None
    extra_args: tuple[str, ...]
    launch_timeout_seconds: float
    poll_interval_seconds: float

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"


def load_llm_provider_config(
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleConfig | None:
    """Load an OpenAI-compatible provider configuration from the environment."""

    source = environ or os.environ
    endpoint = source.get("VOICELAYER_LLM_ENDPOINT")
    model = source.get("VOICELAYER_LLM_MODEL")
    if not endpoint or not model:
        return None

    timeout_seconds = float(source.get("VOICELAYER_LLM_TIMEOUT_SECONDS", "60"))
    api_key = source.get("VOICELAYER_LLM_API_KEY") or None
    return OpenAICompatibleConfig(
        endpoint=endpoint.strip(),
        model=model.strip(),
        api_key=api_key,
        timeout_seconds=timeout_seconds,
    )


def load_llama_server_launch_config(
    environ: Mapping[str, str] | None = None,
) -> LlamaServerLaunchConfig | None:
    """Load optional auto-start configuration for `llama-server`."""

    source = environ or os.environ
    enabled = source.get("VOICELAYER_LLM_AUTO_START", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    return LlamaServerLaunchConfig(
        server_bin=source.get("VOICELAYER_LLAMA_SERVER_BIN", "llama-server"),
        model_path=source.get("VOICELAYER_LLAMA_MODEL_PATH"),
        hf_repo=source.get("VOICELAYER_LLAMA_HF_REPO"),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_LLAMA_SERVER_ARGS", ""))),
        launch_timeout_seconds=float(source.get("VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS", "45")),
        poll_interval_seconds=float(source.get("VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS", "0.5")),
    )


def load_whisper_provider_config(
    environ: Mapping[str, str] | None = None,
) -> WhisperCppConfig | None:
    """Load `whisper-cli` configuration from the environment."""

    source = environ or os.environ
    model_path = source.get("VOICELAYER_WHISPER_MODEL_PATH")
    if not model_path:
        return None

    return WhisperCppConfig(
        binary=source.get("VOICELAYER_WHISPER_BIN", "whisper-cli"),
        model_path=model_path.strip(),
        timeout_seconds=float(source.get("VOICELAYER_WHISPER_TIMEOUT_SECONDS", "300")),
        no_gpu=source.get("VOICELAYER_WHISPER_NO_GPU", "").strip().lower()
        in {"1", "true", "yes", "on"},
        extra_args=tuple(shlex.split(source.get("VOICELAYER_WHISPER_ARGS", ""))),
    )


def load_whisper_vad_config(
    environ: Mapping[str, str] | None = None,
) -> WhisperVadConfig | None:
    """Load silero-vad pre-pass configuration from the environment.

    Returns ``None`` when VAD is not explicitly enabled or the model path is
    missing. Callers treat a ``None`` config as "no VAD" and fall back to the
    raw WAV transcribe path.
    """

    source = environ or os.environ
    enabled = source.get("VOICELAYER_WHISPER_VAD_ENABLED", "").strip().lower()
    if enabled not in {"1", "true", "yes", "on"}:
        return None

    model_path = source.get("VOICELAYER_WHISPER_VAD_MODEL_PATH", "").strip()
    if not model_path:
        return None

    return WhisperVadConfig(
        model_path=model_path,
        threshold=float(source.get("VOICELAYER_WHISPER_VAD_THRESHOLD", "0.5")),
        min_speech_ms=int(source.get("VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS", "250")),
        min_silence_ms=int(source.get("VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS", "100")),
        speech_pad_ms=int(source.get("VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS", "30")),
        max_segment_secs=float(source.get("VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS", "30")),
        sample_rate=int(source.get("VOICELAYER_WHISPER_VAD_SAMPLE_RATE", "16000")),
    )


def load_whisper_server_config(
    environ: Mapping[str, str] | None = None,
) -> WhisperServerConfig | None:
    """Load persistent `whisper-server` configuration from the environment.

    Returns None when neither a host/port pair nor an autostart binary is
    configured, so callers can fall back to the one-shot ``whisper-cli``
    provider without handling an error.
    """

    source = environ or os.environ
    host = source.get("VOICELAYER_WHISPER_SERVER_HOST", "").strip()
    port_str = source.get("VOICELAYER_WHISPER_SERVER_PORT", "").strip()
    server_bin = source.get("VOICELAYER_WHISPER_SERVER_BIN")
    auto_start = source.get("VOICELAYER_WHISPER_SERVER_AUTO_START", "").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }

    if not host and not port_str and not server_bin and not auto_start:
        return None

    resolved_host = host or "127.0.0.1"
    resolved_port = int(port_str) if port_str else 8188

    return WhisperServerConfig(
        host=resolved_host,
        port=resolved_port,
        timeout_seconds=float(source.get("VOICELAYER_WHISPER_SERVER_TIMEOUT_SECONDS", "60")),
        auto_start=auto_start,
        server_bin=server_bin,
        model_path=source.get("VOICELAYER_WHISPER_MODEL_PATH"),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_WHISPER_SERVER_ARGS", ""))),
        launch_timeout_seconds=float(
            source.get("VOICELAYER_WHISPER_SERVER_LAUNCH_TIMEOUT_SECONDS", "30")
        ),
        poll_interval_seconds=float(
            source.get("VOICELAYER_WHISPER_SERVER_POLL_INTERVAL_SECONDS", "0.5")
        ),
    )
