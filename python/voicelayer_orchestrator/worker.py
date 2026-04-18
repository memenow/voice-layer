"""JSON-RPC stdio worker for VoiceLayer model orchestration."""

from __future__ import annotations

import atexit
import contextlib
import json
import os
import shlex
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO

from voicelayer_orchestrator.protocol import JSONRPC_VERSION, make_error, make_result

PROVIDER_UNAVAILABLE_CODE = -32004
PROVIDER_REQUEST_FAILED_CODE = -32005
INVALID_REQUEST_CODE = -32600
METHOD_NOT_FOUND_CODE = -32601
PARSE_ERROR_CODE = -32700
_BACKGROUND_PROCESSES: list[subprocess.Popen[bytes]] = []


@dataclass(frozen=True)
class OpenAICompatibleConfig:
    """Configuration for a locally hosted OpenAI-compatible chat endpoint."""

    endpoint: str
    model: str
    api_key: str | None
    timeout_seconds: float


class ProviderInvocationError(RuntimeError):
    """Raised when the configured provider cannot satisfy a request."""


def cleanup_background_processes() -> None:
    """Reap completed background provider processes."""

    remaining: list[subprocess.Popen[bytes]] = []
    for process in _BACKGROUND_PROCESSES:
        if process.poll() is None:
            remaining.append(process)
        else:
            try:
                process.wait(timeout=0)
            except subprocess.TimeoutExpired:
                remaining.append(process)
    _BACKGROUND_PROCESSES[:] = remaining


atexit.register(cleanup_background_processes)


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


def resolve_chat_completions_url(endpoint: str) -> str:
    """Normalize a configured endpoint into a chat completions URL."""

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def resolve_models_url(endpoint: str) -> str:
    """Normalize a configured endpoint into a models URL."""

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized.removesuffix("/chat/completions") + "/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def is_local_endpoint(endpoint: str) -> bool:
    """Return True when the configured endpoint points to a local host."""

    parsed = urllib.parse.urlparse(endpoint)
    host = (parsed.hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def configured_llm_descriptor(
    config: OpenAICompatibleConfig | None,
) -> dict[str, Any] | None:
    """Return the configured LLM descriptor when a real endpoint is available."""

    if config is None:
        return None

    provider_id = "gemma_4_local" if "gemma" in config.model.lower() else config.model
    return {
        "id": provider_id,
        "kind": "llm",
        "transport": "openai_chat_completions",
        "local": is_local_endpoint(config.endpoint),
        "default_enabled": True,
        "experimental": False,
        "license": "user_supplied",
    }


def supported_providers(
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return provider descriptors for the Python worker boundary."""

    whisper_config = load_whisper_provider_config(environ)
    providers = [
        {
            "id": "whisper_cpp",
            "kind": "asr",
            "transport": "whisper_cli" if whisper_config is not None else "stdio_worker",
            "local": True,
            "default_enabled": True,
            "experimental": False,
            "license": "MIT",
        },
        {
            "id": "voxtral_realtime",
            "kind": "asr",
            "transport": "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "Apache-2.0",
        },
    ]

    configured = configured_llm_descriptor(load_llm_provider_config(environ))
    if configured is not None:
        providers.append(configured)
    else:
        providers.append(
            {
                "id": "gemma_4_local",
                "kind": "llm",
                "transport": "stdio_worker",
                "local": True,
                "default_enabled": True,
                "experimental": False,
                "license": "Apache-2.0",
            }
        )

    return providers


def render_content_text(content: Any) -> str:
    """Normalize OpenAI-compatible content payloads into plain text."""

    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                text = block.get("text")
                if isinstance(text, str):
                    parts.append(text)
        return "\n".join(part.strip() for part in parts if part.strip())

    return ""


def invoke_chat_completion(
    system_prompt: str,
    user_prompt: str,
    config: OpenAICompatibleConfig,
) -> str:
    """Call the configured OpenAI-compatible chat completion endpoint."""

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    request = urllib.request.Request(
        resolve_chat_completions_url(config.endpoint),
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            **({"Authorization": f"Bearer {config.api_key}"} if config.api_key is not None else {}),
        },
        method="POST",
    )

    try:
        with urllib.request.urlopen(request, timeout=config.timeout_seconds) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        raise ProviderInvocationError(
            f"Configured LLM endpoint returned HTTP {exc.code}: {detail}"
        ) from exc
    except urllib.error.URLError as exc:
        raise ProviderInvocationError(
            f"Configured LLM endpoint is unreachable: {exc.reason}"
        ) from exc
    except TimeoutError as exc:
        raise ProviderInvocationError("Configured LLM endpoint timed out.") from exc

    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise ProviderInvocationError(
            "Configured LLM endpoint returned an unexpected response shape."
        ) from exc

    text = render_content_text(content)
    if not text:
        raise ProviderInvocationError("Configured LLM endpoint returned an empty completion.")
    return text


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


def provider_runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the runtime directory used for provider state files."""

    source = environ or os.environ
    base = source.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    runtime_dir = Path(base) / "voicelayer" / "providers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def endpoint_key(endpoint: str) -> str:
    """Build a filesystem-safe key from the configured endpoint."""

    parsed = urllib.parse.urlparse(endpoint)
    host = (parsed.hostname or "localhost").replace(":", "_")
    port = parsed.port or 8080
    return f"{host}-{port}"


def build_llama_server_command(
    config: OpenAICompatibleConfig,
    launch: LlamaServerLaunchConfig,
) -> list[str]:
    """Build the `llama-server` command for the configured endpoint."""

    if not is_local_endpoint(config.endpoint):
        raise ProviderInvocationError(
            "Automatic llama-server startup only supports local endpoints."
        )

    parsed = urllib.parse.urlparse(config.endpoint)
    port = parsed.port or 8080
    if launch.model_path:
        source_args = ["-m", launch.model_path]
    elif launch.hf_repo:
        source_args = ["-hf", launch.hf_repo]
    else:
        raise ProviderInvocationError(
            "Automatic llama-server startup requires "
            "`VOICELAYER_LLAMA_MODEL_PATH` or `VOICELAYER_LLAMA_HF_REPO`."
        )

    return [
        launch.server_bin,
        *source_args,
        "--port",
        str(port),
        *launch.extra_args,
    ]


def wait_for_llm_endpoint(
    config: OpenAICompatibleConfig,
    timeout_seconds: float,
    poll_interval_seconds: float,
) -> tuple[bool, str | None]:
    """Wait for the configured LLM endpoint to become reachable."""

    deadline = time.monotonic() + timeout_seconds
    last_error = None
    while time.monotonic() < deadline:
        reachable, error = probe_llm_endpoint(config)
        if reachable:
            return True, None
        last_error = error
        time.sleep(poll_interval_seconds)
    return False, last_error


def autostart_llama_server(
    config: OpenAICompatibleConfig,
    launch: LlamaServerLaunchConfig,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Launch `llama-server` when the endpoint is local and currently unavailable."""

    runtime_dir = provider_runtime_dir(environ)
    key = endpoint_key(config.endpoint)
    lock_path = runtime_dir / f"{key}.lock"
    log_path = runtime_dir / f"{key}.log"
    state_path = runtime_dir / f"{key}.json"
    source = dict(os.environ)
    if environ:
        source.update(environ)

    try:
        command = build_llama_server_command(config, launch)
    except ProviderInvocationError as exc:
        return False, str(exc)

    lock_fd: int | None = None
    try:
        lock_fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        with os.fdopen(lock_fd, "w", encoding="utf-8") as lock_file:
            lock_fd = None
            lock_file.write(str(os.getpid()))

        with log_path.open("ab") as log_handle:
            process = subprocess.Popen(  # noqa: S603
                command,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                stdin=subprocess.DEVNULL,
                env=source,
                start_new_session=True,
            )
        _BACKGROUND_PROCESSES.append(process)
        state_path.write_text(
            json.dumps(
                {
                    "pid": process.pid,
                    "command": command,
                    "endpoint": config.endpoint,
                    "started_at_epoch_seconds": time.time(),
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
        return wait_for_llm_endpoint(
            config,
            timeout_seconds=launch.launch_timeout_seconds,
            poll_interval_seconds=launch.poll_interval_seconds,
        )
    except FileExistsError:
        return wait_for_llm_endpoint(
            config,
            timeout_seconds=launch.launch_timeout_seconds,
            poll_interval_seconds=launch.poll_interval_seconds,
        )
    except FileNotFoundError as exc:
        return False, f"Unable to find `{launch.server_bin}`: {exc}"
    finally:
        if lock_fd is not None:
            os.close(lock_fd)
        with contextlib.suppress(FileNotFoundError):
            lock_path.unlink()


def probe_llm_endpoint(config: OpenAICompatibleConfig) -> tuple[bool, str | None]:
    """Probe the configured LLM endpoint for readiness."""

    request = urllib.request.Request(
        resolve_models_url(config.endpoint),
        headers=(
            {"Authorization": f"Bearer {config.api_key}"} if config.api_key is not None else {}
        ),
        method="GET",
    )

    try:
        with urllib.request.urlopen(request, timeout=min(config.timeout_seconds, 10.0)) as response:
            body = json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()
        return False, f"HTTP {exc.code}: {detail}"
    except urllib.error.URLError as exc:
        return False, f"unreachable: {exc.reason}"
    except TimeoutError:
        return False, "timeout"

    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return True, None
    return False, "unexpected models response shape"


def ensure_llm_endpoint(
    config: OpenAICompatibleConfig | None,
    environ: Mapping[str, str] | None = None,
) -> tuple[bool, str | None]:
    """Return endpoint readiness, launching `llama-server` when configured to do so."""

    if config is None:
        return False, None

    reachable, error = probe_llm_endpoint(config)
    if reachable:
        return True, None

    launch = load_llama_server_launch_config(environ)
    if launch is None:
        return False, error

    return autostart_llama_server(config, launch, environ)


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


def build_compose_payload(
    params: Mapping[str, Any], config: OpenAICompatibleConfig
) -> dict[str, Any]:
    """Build a composition preview artifact."""

    spoken_prompt = str(params.get("spoken_prompt", "")).strip()
    if not spoken_prompt:
        raise ProviderInvocationError("Compose requests require `spoken_prompt`.")

    archetype = str(params.get("archetype") or "custom").replace("_", " ")
    output_language = params.get("output_language")
    system_prompt = (
        "You are VoiceLayer Compose. Convert spoken intent into insertion-ready text. "
        "Return only the final text body with no commentary, labels, or Markdown fences."
    )
    user_prompt = (
        f"Task type: {archetype}\n"
        f"Preferred output language: {output_language or 'follow request context'}\n"
        f"User request:\n{spoken_prompt}"
    )
    generated_text = invoke_chat_completion(system_prompt, user_prompt, config)
    return {
        "title": f"Compose preview for {archetype}",
        "generated_text": generated_text,
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }


def build_rewrite_payload(
    params: Mapping[str, Any], config: OpenAICompatibleConfig
) -> dict[str, Any]:
    """Build a rewrite preview artifact."""

    source_text = str(params.get("source_text", "")).strip()
    style = str(params.get("style", "")).replace("_", " ").strip()
    output_language = params.get("output_language")
    if not source_text or not style:
        raise ProviderInvocationError("Rewrite requests require `source_text` and `style`.")

    system_prompt = (
        "You are VoiceLayer Rewrite. Rewrite the provided text according to the requested style. "
        "Return only the rewritten text."
    )
    language_hint = output_language or "keep source language unless the style requires translation"
    user_prompt = (
        f"Rewrite style: {style}\n"
        f"Preferred output language: {language_hint}\n"
        f"Source text:\n{source_text}"
    )
    generated_text = invoke_chat_completion(system_prompt, user_prompt, config)
    return {
        "title": f"Rewrite preview for {style}",
        "generated_text": generated_text,
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }


def build_translate_payload(
    params: Mapping[str, Any], config: OpenAICompatibleConfig
) -> dict[str, Any]:
    """Build a translation preview artifact."""

    source_text = str(params.get("source_text", "")).strip()
    target_language = str(params.get("target_language", "")).strip()
    if not source_text or not target_language:
        raise ProviderInvocationError(
            "Translate requests require `source_text` and `target_language`."
        )

    system_prompt = (
        "You are VoiceLayer Translate. Translate the input faithfully while "
        "preserving technical terms. Return only the translated text."
    )
    user_prompt = f"Target language: {target_language}\nSource text:\n{source_text}"
    generated_text = invoke_chat_completion(system_prompt, user_prompt, config)
    return {
        "title": f"Translate preview to {target_language}",
        "generated_text": generated_text,
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC request."""

    if request.get("jsonrpc") != JSONRPC_VERSION or "method" not in request:
        return make_error(None, INVALID_REQUEST_CODE, "Invalid JSON-RPC request.")

    identifier = request.get("id")
    method = request["method"]
    params = request.get("params")
    if params is not None and not isinstance(params, dict):
        return make_error(identifier, INVALID_REQUEST_CODE, "JSON-RPC params must be an object.")

    if method == "health":
        config = load_llm_provider_config()
        llm_reachable, llm_error = ensure_llm_endpoint(config)
        whisper_config = load_whisper_provider_config()
        asr_configured, asr_error = validate_whisper_provider(whisper_config)
        return make_result(
            identifier,
            {
                "status": "ok",
                "worker": "voicelayer_orchestrator",
                "protocol": JSONRPC_VERSION,
                "asr_configured": asr_configured,
                "asr_binary": None if whisper_config is None else whisper_config.binary,
                "asr_model_path": None if whisper_config is None else whisper_config.model_path,
                "asr_error": asr_error,
                "llm_configured": config is not None,
                "llm_model": None if config is None else config.model,
                "llm_endpoint": None if config is None else config.endpoint,
                "llm_reachable": llm_reachable,
                "llm_error": llm_error,
            },
        )

    if method == "list_providers":
        return make_result(identifier, {"providers": supported_providers()})

    if method == "transcribe":
        whisper_config = load_whisper_provider_config()
        if whisper_config is None:
            return make_error(
                identifier,
                PROVIDER_UNAVAILABLE_CODE,
                "No transcription provider is configured for the requested workflow.",
                {"method": method},
            )

        try:
            result = transcribe_with_whisper_cli(params or {}, whisper_config)
        except ProviderInvocationError as exc:
            return make_error(
                identifier,
                PROVIDER_REQUEST_FAILED_CODE,
                str(exc),
                {"method": method},
            )

        return make_result(identifier, result)

    if method in {"compose", "rewrite", "translate"}:
        config = load_llm_provider_config()
        if config is None:
            return make_error(
                identifier,
                PROVIDER_UNAVAILABLE_CODE,
                "No model provider is configured for the requested workflow.",
                {"method": method},
            )

        llm_reachable, llm_error = ensure_llm_endpoint(config)
        if not llm_reachable:
            return make_error(
                identifier,
                PROVIDER_REQUEST_FAILED_CODE,
                f"Configured LLM endpoint is not ready: {llm_error}",
                {"method": method},
            )

        try:
            if method == "compose":
                result = build_compose_payload(params or {}, config)
            elif method == "rewrite":
                result = build_rewrite_payload(params or {}, config)
            else:
                result = build_translate_payload(params or {}, config)
        except ProviderInvocationError as exc:
            return make_error(
                identifier,
                PROVIDER_REQUEST_FAILED_CODE,
                str(exc),
                {"method": method},
            )

        return make_result(identifier, result)

    return make_error(
        identifier,
        METHOD_NOT_FOUND_CODE,
        f"Unsupported method: {method}",
    )


def serve(stdin: TextIO, stdout: TextIO) -> int:
    """Serve JSON-RPC requests over stdio."""

    for raw_line in stdin:
        line = raw_line.strip()
        if not line:
            continue

        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            response = make_error(None, PARSE_ERROR_CODE, "Unable to parse JSON input.")
        else:
            response = handle_request(request)

        if response is not None:
            stdout.write(json.dumps(response, sort_keys=True) + "\n")
            stdout.flush()

    return 0


def main() -> int:
    """Program entry point."""

    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
