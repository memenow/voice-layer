"""JSON-RPC stdio worker for VoiceLayer model orchestration."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from voicelayer_orchestrator.config import (
    load_llm_provider_config,
    load_whisper_provider_config,
    load_whisper_server_config,
    load_whisper_vad_config,
)
from voicelayer_orchestrator.protocol import JSONRPC_VERSION, make_error, make_result
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    provider_runtime_dir,
    supported_providers,
)
from voicelayer_orchestrator.providers.llama_autostart import ensure_llm_endpoint
from voicelayer_orchestrator.providers.llm_openai_compatible import (
    build_compose_payload,
    build_rewrite_payload,
    build_translate_payload,
)
from voicelayer_orchestrator.providers.vad_segmenter import apply_vad_prepass
from voicelayer_orchestrator.providers.whisper_cli import (
    transcribe_with_whisper_cli,
    validate_whisper_provider,
)
from voicelayer_orchestrator.providers.whisper_server import (
    ensure_whisper_server,
    transcribe_with_whisper_server,
)

PROVIDER_UNAVAILABLE_CODE = -32004
PROVIDER_REQUEST_FAILED_CODE = -32005
INVALID_REQUEST_CODE = -32600
METHOD_NOT_FOUND_CODE = -32601
PARSE_ERROR_CODE = -32700


def _apply_vad_prepass_if_configured(
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None]:
    """Run the optional silero-vad pre-pass.

    Returns ``(effective_params, extra_notes, short_circuit_result)``. When
    ``short_circuit_result`` is not ``None`` the caller should return it
    directly without invoking whisper (VAD detected no speech). When it is
    ``None`` the caller proceeds with ``effective_params`` and appends
    ``extra_notes`` to whisper's response.
    """

    vad_config = load_whisper_vad_config()
    if vad_config is None:
        return params, [], None

    audio_file = str(params.get("audio_file", "")).strip()
    if not audio_file:
        return params, [], None

    try:
        runtime_dir = provider_runtime_dir() / "vad"
        trimmed_path, regions = apply_vad_prepass(audio_file, vad_config, runtime_dir)
    except ProviderInvocationError as exc:
        return params, [f"VAD pre-pass failed, transcribing raw audio: {exc}"], None

    if not regions:
        return (
            params,
            [],
            {
                "text": "",
                "detected_language": None,
                "notes": ["VAD detected no speech; whisper inference was skipped."],
            },
        )

    new_params = dict(params)
    new_params["audio_file"] = trimmed_path
    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) from the original capture."
    )
    return new_params, [note], None


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
        effective_params, extra_notes, short_circuit = _apply_vad_prepass_if_configured(
            dict(params or {})
        )
        if short_circuit is not None:
            return make_result(identifier, short_circuit)

        server_config = load_whisper_server_config()
        server_error: str | None = None
        if server_config is not None:
            reachable, probe_error = ensure_whisper_server(server_config)
            if reachable:
                try:
                    result = transcribe_with_whisper_server(effective_params, server_config)
                    if extra_notes:
                        result = {**result, "notes": [*extra_notes, *result.get("notes", [])]}
                    return make_result(identifier, result)
                except ProviderInvocationError as exc:
                    server_error = str(exc)
            else:
                server_error = probe_error or "whisper-server unreachable"

        cli_config = load_whisper_provider_config()
        if cli_config is None:
            if server_error is not None:
                return make_error(
                    identifier,
                    PROVIDER_REQUEST_FAILED_CODE,
                    (
                        f"whisper-server failed ({server_error}) and no whisper-cli "
                        "fallback is configured."
                    ),
                    {"method": method},
                )
            return make_error(
                identifier,
                PROVIDER_UNAVAILABLE_CODE,
                "No transcription provider is configured for the requested workflow.",
                {"method": method},
            )

        try:
            result = transcribe_with_whisper_cli(effective_params, cli_config)
        except ProviderInvocationError as exc:
            detail = str(exc)
            if server_error is not None:
                detail = f"{detail} (whisper-server also failed: {server_error})"
            return make_error(
                identifier,
                PROVIDER_REQUEST_FAILED_CODE,
                detail,
                {"method": method},
            )

        if extra_notes:
            result = {**result, "notes": [*extra_notes, *result.get("notes", [])]}
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
