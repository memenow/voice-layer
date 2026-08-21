"""Worker configuration received via the JSON-RPC ``initialize`` handshake.

The daemon owns configuration: it parses the unified TOML config file (with
``VOICELAYER_*`` environment overrides) and hands the provider-facing
sections to the worker in the ``initialize`` payload. The worker never
reads the process environment for provider settings.
"""

from __future__ import annotations

import shlex
from dataclasses import dataclass
from typing import Any


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


@dataclass(frozen=True)
class _WorkerConfig:
    llm: OpenAICompatibleConfig | None
    llama_launch: LlamaServerLaunchConfig | None
    whisper: WhisperCppConfig | None
    whisper_server: WhisperServerConfig | None
    vad: WhisperVadConfig | None


_CONFIG: _WorkerConfig | None = None


def configure(payload: dict[str, Any]) -> None:
    """Store the initialize payload as the worker's configuration."""
    global _CONFIG
    whisper = _parse_whisper(payload.get("whisper") or {})
    _CONFIG = _WorkerConfig(
        llm=_parse_llm(payload.get("llm") or {}),
        llama_launch=_parse_llama_launch(payload.get("llm") or {}),
        whisper=whisper,
        whisper_server=_parse_whisper_server(
            payload.get("whisper_server") or {},
            whisper.model_path if whisper else None,
        ),
        vad=_parse_vad(payload.get("vad") or {}),
    )


def is_configured() -> bool:
    return _CONFIG is not None


def _require_config() -> _WorkerConfig:
    if _CONFIG is None:
        raise RuntimeError("worker has not been initialized")
    return _CONFIG


def llm_config() -> OpenAICompatibleConfig | None:
    return _require_config().llm


def llama_launch_config() -> LlamaServerLaunchConfig | None:
    return _require_config().llama_launch


def whisper_config() -> WhisperCppConfig | None:
    return _require_config().whisper


def whisper_server_config() -> WhisperServerConfig | None:
    return _require_config().whisper_server


def vad_config() -> WhisperVadConfig | None:
    return _require_config().vad


def _text(section: dict[str, Any], key: str) -> str | None:
    value = section.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _number(section: dict[str, Any], key: str, default: float) -> float:
    value = section.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _integer(section: dict[str, Any], key: str, default: int) -> int:
    value = section.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _flag(section: dict[str, Any], key: str) -> bool:
    return bool(section.get(key, False))


def _args(section: dict[str, Any], key: str) -> tuple[str, ...]:
    return tuple(shlex.split(_text(section, key) or ""))


def _parse_llm(section: dict[str, Any]) -> OpenAICompatibleConfig | None:
    endpoint = _text(section, "endpoint")
    model = _text(section, "model")
    if not endpoint or not model:
        return None
    return OpenAICompatibleConfig(
        endpoint=endpoint,
        model=model,
        api_key=_text(section, "api_key"),
        timeout_seconds=_number(section, "timeout_seconds", 60.0),
    )


def _parse_llama_launch(section: dict[str, Any]) -> LlamaServerLaunchConfig | None:
    if not _flag(section, "auto_start"):
        return None
    return LlamaServerLaunchConfig(
        server_bin=_text(section, "server_bin") or "llama-server",
        model_path=_text(section, "model_path"),
        hf_repo=_text(section, "hf_repo"),
        extra_args=_args(section, "server_args"),
        launch_timeout_seconds=_number(section, "launch_timeout_seconds", 45.0),
        poll_interval_seconds=_number(section, "poll_interval_seconds", 0.5),
    )


def _parse_whisper(section: dict[str, Any]) -> WhisperCppConfig | None:
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return WhisperCppConfig(
        binary=_text(section, "binary") or "whisper-cli",
        model_path=model_path,
        timeout_seconds=_number(section, "timeout_seconds", 300.0),
        no_gpu=_flag(section, "no_gpu"),
        extra_args=_args(section, "extra_args"),
    )


def _parse_whisper_server(
    section: dict[str, Any],
    whisper_model_path: str | None,
) -> WhisperServerConfig | None:
    host = _text(section, "host")
    port = _integer(section, "port", 0)
    server_bin = _text(section, "server_bin")
    auto_start = _flag(section, "auto_start")
    if not host and not port and not server_bin and not auto_start:
        return None
    return WhisperServerConfig(
        host=host or "127.0.0.1",
        port=port or 8188,
        timeout_seconds=_number(section, "timeout_seconds", 60.0),
        auto_start=auto_start,
        server_bin=server_bin,
        # Autostart needs the ggml model path; it lives in the whisper
        # section of the shared config.
        model_path=whisper_model_path,
        extra_args=_args(section, "extra_args"),
        launch_timeout_seconds=_number(section, "launch_timeout_seconds", 30.0),
        poll_interval_seconds=_number(section, "poll_interval_seconds", 0.5),
    )


def _parse_vad(section: dict[str, Any]) -> WhisperVadConfig | None:
    if not _flag(section, "enabled"):
        return None
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return WhisperVadConfig(
        model_path=model_path,
        threshold=_number(section, "threshold", 0.5),
        min_speech_ms=_integer(section, "min_speech_ms", 250),
        min_silence_ms=_integer(section, "min_silence_ms", 100),
        speech_pad_ms=_integer(section, "speech_pad_ms", 30),
        max_segment_secs=_number(section, "max_segment_secs", 30.0),
        sample_rate=_integer(section, "sample_rate", 16000),
    )
