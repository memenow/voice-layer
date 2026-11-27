"""Worker configuration parsed from JSON-RPC ``initialize`` or ``VOICELAYER_*`` env vars.

The daemon parses ``config.toml`` with environment overrides and sends the provider
sections in the ``initialize`` handshake. Standalone invocations (CLI probing or unit
tests) can initialize through :func:`configure_from_environment` using the same parser.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for an OpenAI-compatible chat completions endpoint."""

    endpoint: str
    model: str
    api_key: str | None
    timeout_seconds: float


@dataclass(frozen=True)
class LlamaServerLaunchConfig:
    """Configuration for lazy ``llama-server`` autostart."""

    server_bin: str
    model_path: str | None
    hf_repo: str | None
    extra_args: tuple[str, ...]
    launch_timeout_seconds: float
    poll_interval_seconds: float


@dataclass(frozen=True)
class WhisperCppConfig:
    """Configuration for one-shot ``whisper-cli`` transcription."""

    binary: str
    model_path: str
    timeout_seconds: float
    no_gpu: bool
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class WhisperVadConfig:
    """Configuration for the Silero VAD pre-pass and speech probing."""

    model_path: str
    threshold: float
    min_speech_ms: int
    min_silence_ms: int
    speech_pad_ms: int
    max_segment_secs: float
    sample_rate: int


@dataclass(frozen=True)
class WhisperServerConfig:
    """Configuration for a persistent ``whisper-server`` HTTP endpoint."""

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
        """HTTP base URL of the endpoint."""
        return f"http://{self.host}:{self.port}"


@dataclass(frozen=True)
class MimoAsrConfig:
    """Configuration for the optional MiMo-V2.5-ASR provider."""

    model_path: str
    tokenizer_path: str
    repo_path: str | None
    device: str
    audio_tag: str | None
    timeout_seconds: float
    long_audio_split_seconds: float
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class Qwen3AsrConfig:
    """Configuration for the optional Qwen3-ASR-1.7B provider."""

    model_path: str
    device: str
    torch_dtype: str
    timeout_seconds: float
    long_audio_split_seconds: float
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class WorkerConfig:
    """Immutable configuration snapshot for all worker providers."""

    llm: OpenAICompatibleConfig | None = None
    llama_launch: LlamaServerLaunchConfig | None = None
    whisper: WhisperCppConfig | None = None
    whisper_server: WhisperServerConfig | None = None
    vad: WhisperVadConfig | None = None
    mimo_asr: MimoAsrConfig | None = None
    qwen3_asr: Qwen3AsrConfig | None = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> WorkerConfig:
        """Parse an ``initialize`` payload mapping into typed provider configs."""
        llm_section = _section(payload, "llm")
        whisper = _parse_whisper(_section(payload, "whisper"))
        return cls(
            llm=_parse_llm(llm_section),
            llama_launch=_parse_llama_launch(llm_section),
            whisper=whisper,
            whisper_server=_parse_whisper_server(
                _section(payload, "whisper_server"),
                whisper.model_path if whisper else None,
            ),
            vad=_parse_vad(_section(payload, "vad")),
            mimo_asr=_parse_mimo_asr(_section(payload, "mimo_asr")),
            qwen3_asr=_parse_qwen3_asr(_section(payload, "qwen3_asr")),
        )

    @classmethod
    def from_environ(cls, environ: Mapping[str, str] | None = None) -> WorkerConfig:
        """Parse ``VOICELAYER_*`` environment variables through the same payload schema."""
        return cls.from_payload(payload_from_environ(environ))


_CONFIG: WorkerConfig | None = None

_TRUTHY = frozenset({"1", "true", "yes", "on"})

_ENV_TO_PAYLOAD: dict[str, tuple[str, str, str]] = {
    "VOICELAYER_LLM_ENDPOINT": ("llm", "endpoint", "str"),
    "VOICELAYER_LLM_MODEL": ("llm", "model", "str"),
    "VOICELAYER_LLM_API_KEY": ("llm", "api_key", "str"),
    "VOICELAYER_LLM_TIMEOUT_SECONDS": ("llm", "timeout_seconds", "float"),
    "VOICELAYER_LLM_AUTO_START": ("llm", "auto_start", "bool"),
    "VOICELAYER_LLAMA_SERVER_BIN": ("llm", "server_bin", "str"),
    "VOICELAYER_LLAMA_MODEL_PATH": ("llm", "model_path", "str"),
    "VOICELAYER_LLAMA_HF_REPO": ("llm", "hf_repo", "str"),
    "VOICELAYER_LLAMA_SERVER_ARGS": ("llm", "server_args", "str"),
    "VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS": ("llm", "launch_timeout_seconds", "float"),
    "VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS": ("llm", "poll_interval_seconds", "float"),
    "VOICELAYER_WHISPER_BIN": ("whisper", "binary", "str"),
    "VOICELAYER_WHISPER_MODEL_PATH": ("whisper", "model_path", "str"),
    "VOICELAYER_WHISPER_TIMEOUT_SECONDS": ("whisper", "timeout_seconds", "float"),
    "VOICELAYER_WHISPER_NO_GPU": ("whisper", "no_gpu", "bool"),
    "VOICELAYER_WHISPER_ARGS": ("whisper", "extra_args", "str"),
    "VOICELAYER_WHISPER_SERVER_HOST": ("whisper_server", "host", "str"),
    "VOICELAYER_WHISPER_SERVER_PORT": ("whisper_server", "port", "int"),
    "VOICELAYER_WHISPER_SERVER_TIMEOUT_SECONDS": ("whisper_server", "timeout_seconds", "float"),
    "VOICELAYER_WHISPER_SERVER_AUTO_START": ("whisper_server", "auto_start", "bool"),
    "VOICELAYER_WHISPER_SERVER_BIN": ("whisper_server", "server_bin", "str"),
    "VOICELAYER_WHISPER_SERVER_ARGS": ("whisper_server", "extra_args", "str"),
    "VOICELAYER_WHISPER_SERVER_LAUNCH_TIMEOUT_SECONDS": (
        "whisper_server",
        "launch_timeout_seconds",
        "float",
    ),
    "VOICELAYER_WHISPER_SERVER_POLL_INTERVAL_SECONDS": (
        "whisper_server",
        "poll_interval_seconds",
        "float",
    ),
    "VOICELAYER_WHISPER_VAD_ENABLED": ("vad", "enabled", "bool"),
    "VOICELAYER_WHISPER_VAD_MODEL_PATH": ("vad", "model_path", "str"),
    "VOICELAYER_WHISPER_VAD_THRESHOLD": ("vad", "threshold", "float"),
    "VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS": ("vad", "min_speech_ms", "int"),
    "VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS": ("vad", "min_silence_ms", "int"),
    "VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS": ("vad", "speech_pad_ms", "int"),
    "VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS": ("vad", "max_segment_secs", "float"),
    "VOICELAYER_WHISPER_VAD_SAMPLE_RATE": ("vad", "sample_rate", "int"),
    "VOICELAYER_MIMO_MODEL_PATH": ("mimo_asr", "model_path", "str"),
    "VOICELAYER_MIMO_TOKENIZER_PATH": ("mimo_asr", "tokenizer_path", "str"),
    "VOICELAYER_MIMO_REPO_PATH": ("mimo_asr", "repo_path", "str"),
    "VOICELAYER_MIMO_DEVICE": ("mimo_asr", "device", "str"),
    "VOICELAYER_MIMO_AUDIO_TAG": ("mimo_asr", "audio_tag", "str"),
    "VOICELAYER_MIMO_TIMEOUT_SECONDS": ("mimo_asr", "timeout_seconds", "float"),
    "VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS": ("mimo_asr", "long_audio_split_seconds", "float"),
    "VOICELAYER_MIMO_ARGS": ("mimo_asr", "extra_args", "str"),
    "VOICELAYER_QWEN3_ASR_MODEL_PATH": ("qwen3_asr", "model_path", "str"),
    "VOICELAYER_QWEN3_ASR_DEVICE": ("qwen3_asr", "device", "str"),
    "VOICELAYER_QWEN3_ASR_DTYPE": ("qwen3_asr", "torch_dtype", "str"),
    "VOICELAYER_QWEN3_ASR_TORCH_DTYPE": ("qwen3_asr", "torch_dtype", "str"),
    "VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS": ("qwen3_asr", "timeout_seconds", "float"),
    "VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS": (
        "qwen3_asr",
        "long_audio_split_seconds",
        "float",
    ),
    "VOICELAYER_QWEN3_ASR_ARGS": ("qwen3_asr", "extra_args", "str"),
}


def payload_from_environ(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Convert ``VOICELAYER_*`` environment variables into an ``initialize`` payload."""

    source = environ if environ is not None else os.environ
    payload: dict[str, Any] = {}
    # Dict order places VOICELAYER_QWEN3_ASR_TORCH_DTYPE after VOICELAYER_QWEN3_ASR_DTYPE
    # so the canonical spelling wins when both are set.
    for env_key, (section, field, kind) in _ENV_TO_PAYLOAD.items():
        if env_key not in source:
            continue
        raw = source[env_key]
        if kind == "bool":
            value: Any = raw.strip().lower() in _TRUTHY
        elif kind == "int":
            if not raw.strip():
                continue
            try:
                value = int(raw)
            except ValueError:
                continue
        elif kind == "float":
            if not raw.strip():
                continue
            try:
                value = float(raw)
            except ValueError:
                continue
        else:
            if not raw.strip():
                continue
            value = raw
        payload.setdefault(section, {})[field] = value
    return payload


def configure(payload: Mapping[str, Any]) -> WorkerConfig:
    """Parse and store the active worker configuration from an ``initialize`` payload."""

    global _CONFIG
    _CONFIG = WorkerConfig.from_payload(payload)
    return _CONFIG


def configure_from_environment(environ: Mapping[str, str] | None = None) -> WorkerConfig:
    """Parse and store the active worker configuration from environment variables."""

    global _CONFIG
    _CONFIG = WorkerConfig.from_environ(environ)
    return _CONFIG


def is_configured() -> bool:
    """Return ``True`` when the worker has stored a configuration snapshot."""

    return _CONFIG is not None


def current_config() -> WorkerConfig:
    """Return the stored worker configuration or raise if uninitialized."""

    if _CONFIG is None:
        raise RuntimeError("worker has not been initialized")
    return _CONFIG


def reset_configuration_for_tests() -> None:
    """Clear the process-global configuration snapshot between test cases."""

    global _CONFIG
    _CONFIG = None


def llm_config() -> OpenAICompatibleConfig | None:
    return current_config().llm


def llama_launch_config() -> LlamaServerLaunchConfig | None:
    return current_config().llama_launch


def whisper_config() -> WhisperCppConfig | None:
    return current_config().whisper


def whisper_server_config() -> WhisperServerConfig | None:
    return current_config().whisper_server


def vad_config() -> WhisperVadConfig | None:
    return current_config().vad


def mimo_asr_config() -> MimoAsrConfig | None:
    return current_config().mimo_asr


def qwen3_asr_config() -> Qwen3AsrConfig | None:
    return current_config().qwen3_asr


def _section(payload: Mapping[str, Any], key: str) -> Mapping[str, Any]:
    raw = payload.get(key)
    return raw if isinstance(raw, Mapping) else {}


def _text(section: Mapping[str, Any], key: str) -> str | None:
    value = section.get(key)
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def _number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = section.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return float(value)
    return default


def _integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = section.get(key)
    if isinstance(value, bool):
        return default
    if isinstance(value, int):
        return value
    return default


def _positive_number(section: Mapping[str, Any], key: str, default: float) -> float:
    value = _number(section, key, default)
    return value if value > 0 else default


def _positive_integer(section: Mapping[str, Any], key: str, default: int) -> int:
    value = _integer(section, key, default)
    return value if value > 0 else default


def _flag(section: Mapping[str, Any], key: str) -> bool:
    return bool(section.get(key, False))


def _args(section: Mapping[str, Any], key: str) -> tuple[str, ...]:
    return tuple(shlex.split(_text(section, key) or ""))


def _parse_llm(section: Mapping[str, Any]) -> OpenAICompatibleConfig | None:
    endpoint = _text(section, "endpoint")
    model = _text(section, "model")
    if not endpoint or not model:
        return None
    return OpenAICompatibleConfig(
        endpoint=endpoint,
        model=model,
        api_key=_text(section, "api_key"),
        timeout_seconds=_positive_number(section, "timeout_seconds", 60.0),
    )


def _parse_llama_launch(section: Mapping[str, Any]) -> LlamaServerLaunchConfig | None:
    if not _flag(section, "auto_start"):
        return None
    return LlamaServerLaunchConfig(
        server_bin=_text(section, "server_bin") or "llama-server",
        model_path=_text(section, "model_path"),
        hf_repo=_text(section, "hf_repo"),
        extra_args=_args(section, "server_args"),
        launch_timeout_seconds=_positive_number(section, "launch_timeout_seconds", 45.0),
        poll_interval_seconds=_positive_number(section, "poll_interval_seconds", 0.5),
    )


def _parse_whisper(section: Mapping[str, Any]) -> WhisperCppConfig | None:
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return WhisperCppConfig(
        binary=_text(section, "binary") or "whisper-cli",
        model_path=model_path,
        timeout_seconds=_positive_number(section, "timeout_seconds", 300.0),
        no_gpu=_flag(section, "no_gpu"),
        extra_args=_args(section, "extra_args"),
    )


def _parse_whisper_server(
    section: Mapping[str, Any],
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
        port=port if port > 0 else 8188,
        timeout_seconds=_positive_number(section, "timeout_seconds", 60.0),
        auto_start=auto_start,
        server_bin=server_bin,
        model_path=whisper_model_path,
        extra_args=_args(section, "extra_args"),
        launch_timeout_seconds=_positive_number(section, "launch_timeout_seconds", 30.0),
        poll_interval_seconds=_positive_number(section, "poll_interval_seconds", 0.5),
    )


def _parse_vad(section: Mapping[str, Any]) -> WhisperVadConfig | None:
    if not _flag(section, "enabled"):
        return None
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return WhisperVadConfig(
        model_path=model_path,
        threshold=_positive_number(section, "threshold", 0.5),
        min_speech_ms=_positive_integer(section, "min_speech_ms", 250),
        min_silence_ms=_positive_integer(section, "min_silence_ms", 100),
        speech_pad_ms=max(0, _integer(section, "speech_pad_ms", 30)),
        max_segment_secs=_positive_number(section, "max_segment_secs", 30.0),
        sample_rate=_positive_integer(section, "sample_rate", 16000),
    )


def _parse_mimo_asr(section: Mapping[str, Any]) -> MimoAsrConfig | None:
    model_path = _text(section, "model_path")
    tokenizer_path = _text(section, "tokenizer_path")
    if not model_path or not tokenizer_path:
        return None
    return MimoAsrConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        repo_path=_text(section, "repo_path"),
        device=_text(section, "device") or "cuda",
        audio_tag=_text(section, "audio_tag"),
        timeout_seconds=_positive_number(section, "timeout_seconds", 600.0),
        long_audio_split_seconds=max(0.0, _number(section, "long_audio_split_seconds", 180.0)),
        extra_args=_args(section, "extra_args"),
    )


def _parse_qwen3_asr(section: Mapping[str, Any]) -> Qwen3AsrConfig | None:
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return Qwen3AsrConfig(
        model_path=model_path,
        device=_text(section, "device") or "cuda",
        torch_dtype=(_text(section, "torch_dtype") or "bfloat16").lower(),
        timeout_seconds=_positive_number(section, "timeout_seconds", 600.0),
        long_audio_split_seconds=max(0.0, _number(section, "long_audio_split_seconds", 0.0)),
        extra_args=_args(section, "extra_args"),
    )
