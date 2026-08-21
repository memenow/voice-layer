"""JSON-RPC stdio worker for VoiceLayer model orchestration.

Protocol-only: request validation, the ``initialize`` handshake, and method
dispatch. All provider orchestration lives in
:mod:`voicelayer_orchestrator.providers.pipeline`.

The daemon keeps this process alive and multiplexes requests over stdio;
``initialize`` must be the first request and carries the provider
configuration payload.
"""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from voicelayer_orchestrator.config import configure, is_configured
from voicelayer_orchestrator.protocol import JSONRPC_VERSION, make_error, make_result
from voicelayer_orchestrator.providers import (
    ProviderInvocationError,
    ProviderUnavailableError,
    pipeline,
    supported_providers,
)

PROVIDER_UNAVAILABLE_CODE = -32004
PROVIDER_REQUEST_FAILED_CODE = -32005
NOT_INITIALIZED_CODE = -32002
INVALID_REQUEST_CODE = -32600
METHOD_NOT_FOUND_CODE = -32601
PARSE_ERROR_CODE = -32700


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Handle a single JSON-RPC request."""

    if request.get("jsonrpc") != JSONRPC_VERSION or "method" not in request:
        return make_error(None, INVALID_REQUEST_CODE, "Invalid JSON-RPC request.")

    identifier = request.get("id")
    method = request["method"]
    params = request.get("params")
    if params is not None and not isinstance(params, dict):
        return make_error(identifier, INVALID_REQUEST_CODE, "JSON-RPC params must be an object.")

    if method == "initialize":
        configure(params or {})
        return make_result(
            identifier,
            {
                "status": "ok",
                "worker": "voicelayer_orchestrator",
                "protocol": JSONRPC_VERSION,
            },
        )

    if not is_configured():
        return make_error(
            identifier,
            NOT_INITIALIZED_CODE,
            "Worker has not been initialized; send `initialize` first.",
        )

    try:
        if method == "health":
            return make_result(identifier, pipeline.health_report())
        if method == "list_providers":
            return make_result(identifier, {"providers": supported_providers()})
        if method == "transcribe":
            return make_result(identifier, pipeline.transcribe(params or {}))
        if method == "compose":
            return make_result(identifier, pipeline.compose(params or {}))
        if method == "rewrite":
            return make_result(identifier, pipeline.rewrite(params or {}))
        if method == "translate":
            return make_result(identifier, pipeline.translate(params or {}))
    except ProviderUnavailableError as exc:
        return make_error(identifier, PROVIDER_UNAVAILABLE_CODE, str(exc), {"method": method})
    except ProviderInvocationError as exc:
        return make_error(identifier, PROVIDER_REQUEST_FAILED_CODE, str(exc), {"method": method})

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
