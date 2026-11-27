"""Error types and JSON-RPC error codes for the VoiceLayer worker."""

from __future__ import annotations

PROVIDER_UNAVAILABLE_CODE = -32004
PROVIDER_REQUEST_FAILED_CODE = -32005
INVALID_REQUEST_CODE = -32600
METHOD_NOT_FOUND_CODE = -32601
PARSE_ERROR_CODE = -32700


class ProviderInvocationError(RuntimeError):
    """Raised when a configured provider fails to satisfy a request."""


class ProviderUnavailableError(ProviderInvocationError):
    """Raised when no provider is configured for the requested workflow."""


class InvalidProviderParamsError(ProviderInvocationError):
    """Raised when request parameters fail validation."""
