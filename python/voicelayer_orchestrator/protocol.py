"""JSON-RPC protocol primitives for VoiceLayer workers."""

from __future__ import annotations

from typing import Any

JSONRPC_VERSION = "2.0"


def make_result(identifier: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 success envelope.

    ``identifier`` is the ``id`` from the incoming request and is passed
    through verbatim so the caller can correlate the response. ``result``
    is placed under the top-level ``result`` key; no ``error`` key is set.
    """

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": identifier,
        "result": result,
    }


def make_error(
    identifier: Any, code: int, message: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC 2.0 error envelope.

    ``code`` is a numeric error code; the worker uses the JSON-RPC
    server-error range (``-32000..-32099``) plus the reserved
    ``-32600`` / ``-32601`` / ``-32700`` codes. ``message`` is required;
    ``data`` is optional structured detail and is omitted when falsy.
    """

    payload: dict[str, Any] = {
        "code": code,
        "message": message,
    }
    if data:
        payload["data"] = data

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": identifier,
        "error": payload,
    }
