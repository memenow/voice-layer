"""OpenAI-compatible chat completions provider and local ``llama-server`` autostart."""

from __future__ import annotations

import urllib.parse
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import httpx

from voicelayer_worker.config import LlamaServerLaunchConfig, OpenAICompatibleConfig
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.process import autostart_process


def resolve_chat_completions_url(endpoint: str) -> str:
    """Normalize a base endpoint into a ``/v1/chat/completions`` URL."""

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized
    if normalized.endswith("/v1"):
        return f"{normalized}/chat/completions"
    return f"{normalized}/v1/chat/completions"


def resolve_models_url(endpoint: str) -> str:
    """Normalize a base endpoint into a ``/v1/models`` URL."""

    normalized = endpoint.rstrip("/")
    if normalized.endswith("/v1/chat/completions"):
        return normalized.removesuffix("/chat/completions") + "/models"
    if normalized.endswith("/v1"):
        return f"{normalized}/models"
    return f"{normalized}/v1/models"


def is_local_endpoint(endpoint: str) -> bool:
    """Return ``True`` when ``endpoint`` targets a loopback address."""

    host = (urllib.parse.urlparse(endpoint).hostname or "").lower()
    return host in {"localhost", "127.0.0.1", "::1"}


def endpoint_key(endpoint: str) -> str:
    """Build a filesystem-safe lock/log stem for ``endpoint``."""

    parsed = urllib.parse.urlparse(endpoint)
    host = (parsed.hostname or "localhost").replace(":", "_")
    port = parsed.port or 8080
    return f"{host}-{port}"


def configured_llm_descriptor(config: OpenAICompatibleConfig | None) -> dict[str, Any] | None:
    """Return the provider catalog entry for a configured OpenAI-compatible endpoint."""

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


def render_content_text(content: Any) -> str:
    """Flatten string or content-block arrays returned by chat completions into plain text."""

    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts = [
            block["text"].strip()
            for block in content
            if isinstance(block, dict)
            and isinstance(block.get("text"), str)
            and block["text"].strip()
        ]
        return "\n".join(parts)
    return ""


def _auth_headers(config: OpenAICompatibleConfig) -> dict[str, str]:
    if config.api_key is None:
        return {}
    return {"Authorization": f"Bearer {config.api_key}"}


def invoke_chat_completion(
    system_prompt: str,
    user_prompt: str,
    config: OpenAICompatibleConfig,
) -> str:
    """Call the configured chat completions endpoint and return the generated text."""

    payload = {
        "model": config.model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.2,
        "stream": False,
    }
    try:
        response = httpx.post(
            resolve_chat_completions_url(config.endpoint),
            json=payload,
            headers={"Content-Type": "application/json", **_auth_headers(config)},
            timeout=config.timeout_seconds,
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        detail = exc.response.text.strip()
        raise ProviderInvocationError(
            f"Configured LLM endpoint returned HTTP {exc.response.status_code}: {detail}"
        ) from exc
    except httpx.TimeoutException as exc:
        raise ProviderInvocationError("Configured LLM endpoint timed out.") from exc
    except httpx.HTTPError as exc:
        raise ProviderInvocationError(f"Configured LLM endpoint is unreachable: {exc}") from exc
    except ValueError as exc:
        raise ProviderInvocationError("Configured LLM endpoint returned a non-JSON body.") from exc

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


def probe_llm_endpoint(config: OpenAICompatibleConfig) -> tuple[bool, str | None]:
    """Probe ``/v1/models`` for endpoint readiness."""

    try:
        response = httpx.get(
            resolve_models_url(config.endpoint),
            headers=_auth_headers(config),
            timeout=min(config.timeout_seconds, 10.0),
        )
        response.raise_for_status()
        body = response.json()
    except httpx.HTTPStatusError as exc:
        return False, f"HTTP {exc.response.status_code}: {exc.response.text.strip()}"
    except httpx.TimeoutException:
        return False, "timeout"
    except httpx.HTTPError as exc:
        return False, f"unreachable: {exc}"
    except ValueError:
        return False, "non-JSON models response"

    if isinstance(body, dict) and isinstance(body.get("data"), list):
        return True, None
    return False, "unexpected models response shape"


def build_llama_server_command(
    config: OpenAICompatibleConfig,
    launch: LlamaServerLaunchConfig,
) -> list[str]:
    """Build the argv for launching a local ``llama-server`` instance."""

    if not is_local_endpoint(config.endpoint):
        raise ProviderInvocationError(
            "Automatic llama-server startup only supports local endpoints."
        )
    port = urllib.parse.urlparse(config.endpoint).port or 8080
    if launch.model_path:
        source_args = ["-m", launch.model_path]
    elif launch.hf_repo:
        source_args = ["-hf", launch.hf_repo]
    else:
        raise ProviderInvocationError(
            "Automatic llama-server startup requires "
            "`VOICELAYER_LLAMA_MODEL_PATH` or `VOICELAYER_LLAMA_HF_REPO`."
        )
    return [launch.server_bin, *source_args, "--port", str(port), *launch.extra_args]


def ensure_llm_endpoint(
    config: OpenAICompatibleConfig | None,
    launch: LlamaServerLaunchConfig | None = None,
    *,
    runtime_dir: Path | None = None,
) -> tuple[bool, str | None]:
    """Return endpoint readiness, launching ``llama-server`` when ``launch`` is configured."""

    if config is None:
        return False, None

    reachable, error = probe_llm_endpoint(config)
    if reachable:
        return True, None
    if launch is None:
        return False, error

    try:
        command = build_llama_server_command(config, launch)
    except ProviderInvocationError as exc:
        return False, str(exc)

    return autostart_process(
        key=endpoint_key(config.endpoint),
        command=command,
        endpoint=config.endpoint,
        probe=lambda: probe_llm_endpoint(config),
        launch_timeout_seconds=launch.launch_timeout_seconds,
        poll_interval_seconds=launch.poll_interval_seconds,
        runtime_dir=runtime_dir,
    )


def build_compose_payload(
    params: Mapping[str, Any],
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    """Generate a structured composition preview."""

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
    return {
        "title": f"Compose preview for {archetype}",
        "generated_text": invoke_chat_completion(system_prompt, user_prompt, config),
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }


def build_rewrite_payload(
    params: Mapping[str, Any],
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    """Generate a rewrite preview."""

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
    return {
        "title": f"Rewrite preview for {style}",
        "generated_text": invoke_chat_completion(system_prompt, user_prompt, config),
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }


def build_translate_payload(
    params: Mapping[str, Any],
    config: OpenAICompatibleConfig,
) -> dict[str, Any]:
    """Generate a translation preview."""

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
    return {
        "title": f"Translate preview to {target_language}",
        "generated_text": invoke_chat_completion(system_prompt, user_prompt, config),
        "notes": [f"Generated by `{config.model}` via OpenAI-compatible chat completions."],
    }
