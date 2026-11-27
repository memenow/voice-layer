"""Provider catalog descriptors exposed over ``list_providers``."""

from __future__ import annotations

from typing import Any

from voicelayer_worker.config import WorkerConfig, current_config
from voicelayer_worker.providers.openai_compatible import configured_llm_descriptor


def supported_providers(config: WorkerConfig | None = None) -> list[dict[str, Any]]:
    """Return provider descriptors derived from the active worker configuration."""

    active = config if config is not None else current_config()
    providers: list[dict[str, Any]] = [
        {
            "id": "whisper_cpp",
            "kind": "asr",
            "transport": "whisper_cli" if active.whisper is not None else "stdio_worker",
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
        {
            "id": "mimo_v2_5_asr",
            "kind": "asr",
            "transport": "in_process_torch" if active.mimo_asr is not None else "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "MIT",
        },
        {
            "id": "qwen3_asr_1_7b",
            "kind": "asr",
            "transport": "in_process_torch" if active.qwen3_asr is not None else "stdio_worker",
            "local": True,
            "default_enabled": False,
            "experimental": True,
            "license": "Apache-2.0",
        },
    ]

    llm_descriptor = configured_llm_descriptor(active.llm)
    if llm_descriptor is not None:
        providers.append(llm_descriptor)
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
