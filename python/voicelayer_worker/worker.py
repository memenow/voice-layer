"""Stdio JSON-RPC 2.0 server entrypoint for the VoiceLayer worker."""

from __future__ import annotations

import json
import sys
from typing import Any, TextIO

from voicelayer_worker import pipeline
from voicelayer_worker.config import configure, configure_from_environment, is_configured
from voicelayer_worker.errors import (
    INVALID_REQUEST_CODE,
    METHOD_NOT_FOUND_CODE,
    PARSE_ERROR_CODE,
    PROVIDER_REQUEST_FAILED_CODE,
    PROVIDER_UNAVAILABLE_CODE,
    InvalidProviderParamsError,
    ProviderInvocationError,
    ProviderUnavailableError,
)
from voicelayer_worker.protocol import JSONRPC_VERSION, make_error, make_result
from voicelayer_worker.providers.registry import supported_providers

__all__ = [
    "INVALID_REQUEST_CODE",
    "METHOD_NOT_FOUND_CODE",
    "PARSE_ERROR_CODE",
    "PROVIDER_REQUEST_FAILED_CODE",
    "PROVIDER_UNAVAILABLE_CODE",
    "handle_request",
    "main",
    "serve",
]


def handle_request(request: dict[str, Any]) -> dict[str, Any] | None:
    """Validate and dispatch a single JSON-RPC 2.0 request."""

    if request.get("jsonrpc") != JSONRPC_VERSION or "method" not in request:
        return make_error(None, INVALID_REQUEST_CODE, "Invalid JSON-RPC request.")

    identifier = request.get("id")
    method = request["method"]
    params = request.get("params")
    if params is not None and not isinstance(params, dict):
        return make_error(identifier, INVALID_REQUEST_CODE, "JSON-RPC params must be an object.")

    if method == "initialize":
        try:
            configure(params or {})
        except ValueError as exc:
            return make_error(identifier, INVALID_REQUEST_CODE, str(exc), {"method": method})
        return make_result(
            identifier,
            {
                "status": "ok",
                "worker": "voicelayer_worker",
                "protocol": JSONRPC_VERSION,
            },
        )

    if not is_configured():
        try:
            configure_from_environment()
        except ValueError as exc:
            return make_error(identifier, INVALID_REQUEST_CODE, str(exc), {"method": method})

    payload = params or {}
    try:
        if method == "health":
            return make_result(identifier, pipeline.health_report())
        if method == "list_providers":
            return make_result(identifier, {"providers": supported_providers()})
        if method == "transcribe":
            return make_result(identifier, pipeline.transcribe(payload))
        if method == "segment_probe":
            return make_result(identifier, pipeline.segment_probe(payload))
        if method == "stitch_wav_segments":
            return make_result(identifier, pipeline.stitch_wav_segments(payload))
        if method in {"compose", "rewrite", "translate"}:
            handler = getattr(pipeline, method)
            return make_result(identifier, handler(payload))
    except InvalidProviderParamsError as exc:
        return make_error(identifier, INVALID_REQUEST_CODE, str(exc), {"method": method})
    except ProviderUnavailableError as exc:
        return make_error(identifier, PROVIDER_UNAVAILABLE_CODE, str(exc), {"method": method})
    except ProviderInvocationError as exc:
        return make_error(identifier, PROVIDER_REQUEST_FAILED_CODE, str(exc), {"method": method})

    return make_error(identifier, METHOD_NOT_FOUND_CODE, f"Unsupported method: {method}")


def serve(stdin: TextIO, stdout: TextIO) -> int:
    """Serve newline-delimited JSON-RPC requests until EOF."""

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
    """CLI entrypoint for ``python -m voicelayer_worker.worker``."""

    return serve(sys.stdin, sys.stdout)


if __name__ == "__main__":
    raise SystemExit(main())
