"""Worker configuration received via the JSON-RPC ``initialize`` handshake.

The daemon owns configuration: it parses the unified TOML config file (with
``VOICELAYER_*`` environment overrides) and hands the provider-facing
sections to the worker in the ``initialize`` payload. The worker never
reads the process environment for provider settings.
"""

from __future__ import annotations

import os
import shlex
from collections.abc import Mapping
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
class MimoAsrConfig:
    """Configuration for the optional Xiaomi MiMo-V2.5-ASR provider.

    The MiMo-V2.5-ASR model is an 8B-parameter audio-tokens-in /
    text-tokens-out causal LM that runs through Xiaomi's `MimoAudio`
    Python wrapper. Inference loads the model into the worker process
    on first use and the daemon's persistent-worker mode (see
    ``crates/voicelayerd/src/worker.rs::WorkerCommand::call``) keeps
    the same Python subprocess alive across every JSON-RPC request,
    so subsequent calls hit the warm cache in
    ``providers/mimo_asr.py`` with no reload cost. Operator env
    changes (``VOICELAYER_MIMO_*``) take effect on the next worker
    spawn — restart the daemon to apply them mid-session. Both the LM
    weights (``model_path``) and the companion MiMo-Audio-Tokenizer
    (``tokenizer_path``) are required. See
    ``docs/guides/local-asr-provider.html`` for the hardware envelope
    and the comparison table.

    The wrapper class lives in Xiaomi's source tree and is not
    distributed as a wheel today. ``repo_path`` is the local checkout
    of `XiaomiMiMo/MiMo-V2.5-ASR` whose **root** directory is prepended
    to ``sys.path`` so the upstream-canonical
    ``from src.mimo_audio.mimo_audio import MimoAudio`` import resolves
    (the wrapper relies on PEP 420 namespace packages and a relative
    import into ``src/mimo_audio_tokenizer``, so the parent of ``src``
    must be importable). Leave it ``None`` only if the wrapper is
    already importable (e.g., the operator installed it as a
    pip-editable package).

    Precision is intentionally not exposed: the upstream wrapper
    hardcodes ``torch_dtype=torch.bfloat16`` when loading the LM, so a
    ``dtype`` knob would lie about what actually happens. If a future
    upstream release accepts a dtype kwarg, expose a new env key
    rather than silently changing the meaning of an old one.

    The provider is opt-in (`TranscribeRequest.provider_id =
    "mimo_v2_5_asr"`). The whisper.cpp chain remains the default.
    """

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
    """Configuration for the optional Qwen3-ASR-1.7B provider.

    The provider runs through the official ``qwen-asr`` pip package's
    ``Qwen3ASRModel`` wrapper, which sits on top of HuggingFace
    transformers and exposes a high-level
    ``Qwen3ASRModel.from_pretrained`` + ``model.transcribe(audio,
    language)`` API. The wrapper handles audio preprocessing and
    long-audio chunking internally, so the worker passes paths through
    verbatim (the optional ``long_audio_split_seconds`` knob is
    available for operators who want to force client-side splitting).

    The model loads inside the worker process on the first transcribe
    call and the daemon's persistent-worker mode (see
    ``crates/voicelayerd/src/worker.rs::WorkerCommand::call``) keeps
    the same Python subprocess alive across every JSON-RPC request,
    so subsequent calls hit the warm cache in
    ``providers/qwen3_asr.py`` with no reload cost. Operator env
    changes (``VOICELAYER_QWEN3_ASR_*``) take effect on the next
    worker spawn — restart the daemon to apply them mid-session.
    Operators stage the Apache-2.0 weights via
    ``hf download Qwen/Qwen3-ASR-1.7B --local-dir <path>`` and point
    ``VOICELAYER_QWEN3_ASR_MODEL_PATH`` at that directory; the worker
    never triggers HuggingFace downloads on its own. See
    ``docs/guides/local-asr-provider.html`` for the hardware envelope
    and the comparison table.

    ``extra_args`` is reserved for future passthrough into
    ``model.transcribe`` keyword arguments. The current upstream
    ``qwen-asr`` wrapper exposes only ``audio`` and ``language``, so the
    field is parsed for forward compatibility but ignored at dispatch
    time. Mirrors the same reserved-knob pattern on
    :class:`MimoAsrConfig`.

    The provider is opt-in (`TranscribeRequest.provider_id =
    "qwen3_asr_1_7b"`). The whisper.cpp chain remains the default.
    """

    model_path: str
    device: str
    torch_dtype: str
    timeout_seconds: float
    long_audio_split_seconds: float
    extra_args: tuple[str, ...]


@dataclass(frozen=True)
class _WorkerConfig:
    llm: OpenAICompatibleConfig | None
    llama_launch: LlamaServerLaunchConfig | None
    whisper: WhisperCppConfig | None
    whisper_server: WhisperServerConfig | None
    vad: WhisperVadConfig | None
    mimo_asr: MimoAsrConfig | None
    qwen3_asr: Qwen3AsrConfig | None


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
        mimo_asr=_parse_mimo_asr(payload.get("mimo_asr") or {}),
        qwen3_asr=_parse_qwen3_asr(payload.get("qwen3_asr") or {}),
    )


def is_configured() -> bool:
    return _CONFIG is not None


def reset_configuration_for_tests() -> None:
    """Drop the stored configuration.

    Test-only: process-global state otherwise leaks across tests that patch
    different `VOICELAYER_*` environments. Production workers are configured
    exactly once by the daemon's `initialize` handshake.
    """
    global _CONFIG
    _CONFIG = None


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


def mimo_asr_config() -> MimoAsrConfig | None:
    return _require_config().mimo_asr


def qwen3_asr_config() -> Qwen3AsrConfig | None:
    return _require_config().qwen3_asr


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


# --- Environment bridge -------------------------------------------------
#
# `configure_from_environment()` maps the `VOICELAYER_*` override layer
# into the initialize payload shape. Production daemons always send an
# explicit `initialize`, so this path exists for (a) running the worker
# standalone during development and (b) the upstream test suites, which
# patch os.environ around worker calls.

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
    "VOICELAYER_QWEN3_ASR_TORCH_DTYPE": ("qwen3_asr", "torch_dtype", "str"),
    "VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS": ("qwen3_asr", "timeout_seconds", "float"),
    "VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS": (
        "qwen3_asr",
        "long_audio_split_seconds",
        "float",
    ),
    "VOICELAYER_QWEN3_ASR_ARGS": ("qwen3_asr", "extra_args", "str"),
}

_TRUTHY = {"1", "true", "yes", "on"}


def payload_from_environ(environ: Mapping[str, str] | None = None) -> dict[str, Any]:
    """Build an initialize payload from `VOICELAYER_*` environment variables."""

    source = environ if environ is not None else os.environ
    payload: dict[str, Any] = {}
    for env_key, (section, field, kind) in _ENV_TO_PAYLOAD.items():
        if env_key not in source:
            continue
        raw = source[env_key]
        if kind == "bool":
            value: Any = raw.strip().lower() in _TRUTHY
        elif kind == "int":
            if not raw.strip():
                continue
            value = int(raw)
        elif kind == "float":
            if not raw.strip():
                continue
            value = float(raw)
        else:
            if raw == "":
                continue
            value = raw
        payload.setdefault(section, {})[field] = value
    return payload


def configure_from_environment(environ: Mapping[str, str] | None = None) -> None:
    """Initialize worker configuration from the `VOICELAYER_*` environment."""

    configure(payload_from_environ(environ))


# --- Test-only environment loaders -------------------------------------
#
# Upstream test suites call these loaders directly with an injected
# environment map. They parse the `VOICELAYER_*` override layer into the
# same dataclasses the initialize handshake uses; production workers never
# call them (the daemon hands config via `initialize`).


def load_llm_provider_config(
    environ: Mapping[str, str] | None = None,
) -> OpenAICompatibleConfig | None:
    source = environ if environ is not None else os.environ
    endpoint = source.get("VOICELAYER_LLM_ENDPOINT")
    model = source.get("VOICELAYER_LLM_MODEL")
    if not endpoint or not model:
        return None
    return OpenAICompatibleConfig(
        endpoint=endpoint.strip(),
        model=model.strip(),
        api_key=source.get("VOICELAYER_LLM_API_KEY") or None,
        timeout_seconds=float(source.get("VOICELAYER_LLM_TIMEOUT_SECONDS", "60")),
    )


def load_llama_server_launch_config(
    environ: Mapping[str, str] | None = None,
) -> LlamaServerLaunchConfig | None:
    source = environ if environ is not None else os.environ
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
    source = environ if environ is not None else os.environ
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
    source = environ if environ is not None else os.environ
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
    source = environ if environ is not None else os.environ
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
    return WhisperServerConfig(
        host=host or "127.0.0.1",
        port=int(port_str) if port_str else 8188,
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


def load_mimo_asr_config(
    environ: Mapping[str, str] | None = None,
) -> MimoAsrConfig | None:
    source = environ if environ is not None else os.environ
    model_path = source.get("VOICELAYER_MIMO_MODEL_PATH")
    tokenizer_path = source.get("VOICELAYER_MIMO_TOKENIZER_PATH")
    if not model_path or not tokenizer_path:
        return None
    return MimoAsrConfig(
        model_path=model_path.strip(),
        tokenizer_path=tokenizer_path.strip(),
        repo_path=source.get("VOICELAYER_MIMO_REPO_PATH") or None,
        device=source.get("VOICELAYER_MIMO_DEVICE", "cuda"),
        audio_tag=(source.get("VOICELAYER_MIMO_AUDIO_TAG") or "").strip() or None,
        timeout_seconds=float(source.get("VOICELAYER_MIMO_TIMEOUT_SECONDS", "300")),
        long_audio_split_seconds=float(source.get("VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS", "0")),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_MIMO_ARGS", ""))),
    )


def load_qwen3_asr_config(
    environ: Mapping[str, str] | None = None,
) -> Qwen3AsrConfig | None:
    source = environ if environ is not None else os.environ
    model_path = (source.get("VOICELAYER_QWEN3_ASR_MODEL_PATH") or "").strip()
    if not model_path:
        return None
    return Qwen3AsrConfig(
        model_path=model_path,
        device=source.get("VOICELAYER_QWEN3_ASR_DEVICE", "cuda"),
        torch_dtype=(
            source.get(
                "VOICELAYER_QWEN3_ASR_TORCH_DTYPE",
                source.get("VOICELAYER_QWEN3_ASR_DTYPE", "bfloat16"),
            )
            or "bfloat16"
        ).lower(),
        timeout_seconds=float(source.get("VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS", "300")),
        long_audio_split_seconds=float(
            source.get("VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS", "0")
        ),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_QWEN3_ASR_ARGS", ""))),
    )


def _parse_mimo_asr(section: dict[str, Any]) -> MimoAsrConfig | None:
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
        timeout_seconds=_number(section, "timeout_seconds", 300.0),
        long_audio_split_seconds=_number(section, "long_audio_split_seconds", 0.0),
        extra_args=_args(section, "extra_args"),
    )


def _parse_qwen3_asr(section: dict[str, Any]) -> Qwen3AsrConfig | None:
    model_path = _text(section, "model_path")
    if not model_path:
        return None
    return Qwen3AsrConfig(
        model_path=model_path,
        device=_text(section, "device") or "cuda",
        torch_dtype=_text(section, "torch_dtype") or "bfloat16",
        timeout_seconds=_number(section, "timeout_seconds", 300.0),
        long_audio_split_seconds=_number(section, "long_audio_split_seconds", 0.0),
        extra_args=_args(section, "extra_args"),
    )
