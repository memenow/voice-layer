"""JSON-RPC protocol primitives for VoiceLayer workers."""

from __future__ import annotations

from typing import Any

JSONRPC_VERSION = "2.0"


def make_result(identifier: Any, result: dict[str, Any]) -> dict[str, Any]:
    """Build a JSON-RPC success response."""

    return {
        "jsonrpc": JSONRPC_VERSION,
        "id": identifier,
        "result": result,
    }


def make_error(
    identifier: Any, code: int, message: str, data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Build a JSON-RPC error response."""

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
