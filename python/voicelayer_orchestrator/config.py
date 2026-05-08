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


@dataclass(frozen=True)
class MimoAsrConfig:
    """Configuration for the optional Xiaomi MiMo-V2.5-ASR provider.

    The MiMo-V2.5-ASR model is an 8B-parameter audio-tokens-in /
    text-tokens-out causal LM that runs through Xiaomi's `MimoAudio`
    Python wrapper. Inference loads the model into the worker process
    on first use. Under the current daemon every JSON-RPC request
    spawns a fresh ``python -m voicelayer_orchestrator.worker`` process
    and exits after the response (see
    ``crates/voicelayerd/src/worker.rs::WorkerCommand::call``), so the
    cold load is paid on every call; routed dictation sessions inherit
    the cost on every segment. The module-level cache in
    ``providers/mimo_asr.py`` is forward compatibility for a future
    persistent-worker mode. Both the LM weights (``model_path``) and
    the companion MiMo-Audio-Tokenizer (``tokenizer_path``) are
    required. See ``docs/guides/local-asr-provider.md`` for the
    cold-start trade-off.

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
    call. Under the current daemon every JSON-RPC request spawns a
    fresh ``python -m voicelayer_orchestrator.worker`` process and
    exits after the response (see
    ``crates/voicelayerd/src/worker.rs::WorkerCommand::call``), so the
    cold load is paid on every call; routed dictation sessions inherit
    the cost on every segment. Operators stage the Apache-2.0 weights
    via ``hf download Qwen/Qwen3-ASR-1.7B --local-dir <path>`` and
    point ``VOICELAYER_QWEN3_ASR_MODEL_PATH`` at that directory; the
    worker never triggers HuggingFace downloads on its own. See
    ``docs/guides/local-asr-provider.md`` for the cold-start trade-off.

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


def load_mimo_asr_config(
    environ: Mapping[str, str] | None = None,
) -> MimoAsrConfig | None:
    """Load Xiaomi MiMo-V2.5-ASR provider configuration from the environment.

    Returns ``None`` when either of the two required model paths is
    missing so callers can treat the provider as "not configured" the
    same way they treat whisper without surfacing an error.
    """

    source = environ or os.environ
    model_path = (source.get("VOICELAYER_MIMO_MODEL_PATH") or "").strip()
    tokenizer_path = (source.get("VOICELAYER_MIMO_TOKENIZER_PATH") or "").strip()
    if not model_path or not tokenizer_path:
        return None

    raw_audio_tag = (source.get("VOICELAYER_MIMO_AUDIO_TAG") or "").strip()
    audio_tag = raw_audio_tag or None

    raw_repo_path = (source.get("VOICELAYER_MIMO_REPO_PATH") or "").strip()
    repo_path = raw_repo_path or None

    return MimoAsrConfig(
        model_path=model_path,
        tokenizer_path=tokenizer_path,
        repo_path=repo_path,
        device=(source.get("VOICELAYER_MIMO_DEVICE") or "cuda:0").strip(),
        audio_tag=audio_tag,
        timeout_seconds=float(source.get("VOICELAYER_MIMO_TIMEOUT_SECONDS", "600")),
        long_audio_split_seconds=float(
            source.get("VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS", "180")
        ),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_MIMO_ARGS", ""))),
    )


def load_qwen3_asr_config(
    environ: Mapping[str, str] | None = None,
) -> Qwen3AsrConfig | None:
    """Load Qwen3-ASR-1.7B provider configuration from the environment.

    Returns ``None`` when ``VOICELAYER_QWEN3_ASR_MODEL_PATH`` is unset
    so callers treat the provider as "not configured" the same way they
    treat MiMo, without surfacing an error. Operators must pre-stage
    the weights themselves; the loader does not trigger HuggingFace
    downloads to keep cold-start cost predictable on the local-first
    runtime path.

    The default ``long_audio_split_seconds=0`` disables client-side
    splitting because the upstream ``qwen-asr`` wrapper handles long
    audio internally; operators can opt back into worker-side chunking
    by setting a positive value.
    """

    source = environ or os.environ
    model_path = (source.get("VOICELAYER_QWEN3_ASR_MODEL_PATH") or "").strip()
    if not model_path:
        return None

    return Qwen3AsrConfig(
        model_path=model_path,
        device=(source.get("VOICELAYER_QWEN3_ASR_DEVICE") or "cuda:0").strip(),
        torch_dtype=(source.get("VOICELAYER_QWEN3_ASR_DTYPE") or "bfloat16").strip().lower(),
        timeout_seconds=float(source.get("VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS", "600")),
        long_audio_split_seconds=float(
            source.get("VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS", "0")
        ),
        extra_args=tuple(shlex.split(source.get("VOICELAYER_QWEN3_ASR_ARGS", ""))),
    )
