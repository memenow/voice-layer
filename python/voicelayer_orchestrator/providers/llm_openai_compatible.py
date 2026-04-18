"""OpenAI-compatible chat completion provider for VoiceLayer."""

from __future__ import annotations

import json
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Mapping
from typing import Any

from voicelayer_orchestrator.config import OpenAICompatibleConfig
from voicelayer_orchestrator.providers import ProviderInvocationError


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
