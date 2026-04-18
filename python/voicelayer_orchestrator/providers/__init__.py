"""Provider adapters for the VoiceLayer Python worker."""

from __future__ import annotations

import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from voicelayer_orchestrator.config import (
    load_llm_provider_config,
    load_whisper_provider_config,
)


class ProviderInvocationError(RuntimeError):
    """Raised when the configured provider cannot satisfy a request."""


def provider_runtime_dir(environ: Mapping[str, str] | None = None) -> Path:
    """Return the runtime directory used for provider state files."""

    source = environ or os.environ
    base = source.get("XDG_RUNTIME_DIR") or tempfile.gettempdir()
    runtime_dir = Path(base) / "voicelayer" / "providers"
    runtime_dir.mkdir(parents=True, exist_ok=True)
    return runtime_dir


def supported_providers(
    environ: Mapping[str, str] | None = None,
) -> list[dict[str, Any]]:
    """Return provider descriptors for the Python worker boundary."""

    from voicelayer_orchestrator.providers.llm_openai_compatible import (
        configured_llm_descriptor,
    )

    whisper_config = load_whisper_provider_config(environ)
    providers: list[dict[str, Any]] = [
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
