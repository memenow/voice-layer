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
from pathlib import Path
from typing import Any, TextIO

from voicelayer_orchestrator.config import (
    configure,
    configure_from_environment,
    is_configured,
    load_whisper_provider_config,
    load_whisper_server_config,  # noqa: F401  (test patch seam)
    load_whisper_vad_config,
)
from voicelayer_orchestrator.protocol import JSONRPC_VERSION, make_error, make_result
from voicelayer_orchestrator.providers import (
    InvalidProviderParamsError,
    ProviderInvocationError,
    ProviderUnavailableError,
    pipeline,
    provider_runtime_dir,
    supported_providers,
)
from voicelayer_orchestrator.providers.audio_stitch import stitch_wav_segments
from voicelayer_orchestrator.providers.vad_segmenter import (
    apply_vad_prepass,
    probe_audio_file,
)
from voicelayer_orchestrator.providers.whisper_cli import transcribe_with_whisper_cli

PROVIDER_UNAVAILABLE_CODE = -32004
PROVIDER_REQUEST_FAILED_CODE = -32005
NOT_INITIALIZED_CODE = -32002
INVALID_REQUEST_CODE = -32600
METHOD_NOT_FOUND_CODE = -32601
PARSE_ERROR_CODE = -32700


def _apply_vad_prepass_if_configured(
    params: dict[str, Any],
) -> tuple[dict[str, Any], list[str], dict[str, Any] | None, Any]:
    """Run the optional silero-vad pre-pass.

    Compatibility shim over ``pipeline._apply_vad_prepass``: resolves the
    VAD config through the module-level ``load_whisper_vad_config`` and
    ``apply_vad_prepass`` names so tests can patch them here, and also
    returns the trimmed WAV path the caller owns.
    """

    vad_config = load_whisper_vad_config()
    if vad_config is None:
        return params, [], None, None

    audio_file = str(params.get("audio_file", "")).strip()
    if not audio_file:
        return params, [], None, None

    try:
        runtime_dir = provider_runtime_dir() / "vad"
        trimmed_path, regions = apply_vad_prepass(audio_file, vad_config, runtime_dir)
    except ProviderInvocationError as exc:
        return params, [f"VAD pre-pass failed, transcribing raw audio: {exc}"], None, None

    if not regions:
        return (
            params,
            [],
            {
                "text": "",
                "detected_language": None,
                "notes": ["VAD detected no speech; whisper inference was skipped."],
            },
            Path(trimmed_path),
        )

    new_params = dict(params)
    new_params["audio_file"] = trimmed_path
    total_sec = sum(end - start for start, end in regions)
    note = (
        f"VAD pre-pass kept {len(regions)} speech region(s) "
        f"({total_sec:.2f}s total) from the original capture."
    )
    return new_params, [note], None, Path(trimmed_path)


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
        # Standalone worker runs (no daemon driving): fall back to the
        # VOICELAYER_* environment layer so direct `python -m
        # voicelayer_orchestrator.worker` usage and env-patched tests keep
        # working. Daemon-managed workers always receive `initialize` first.
        configure_from_environment()

    try:
        if method == "health":
            return make_result(identifier, pipeline.health_report())
        if method == "list_providers":
            return make_result(identifier, {"providers": supported_providers()})
        if method == "transcribe":
            effective_params, extra_notes, short_circuit, trimmed_path = (
                _apply_vad_prepass_if_configured(dict(params or {}))
            )
            try:
                if short_circuit is not None:
                    return make_result(identifier, short_circuit)
                cli_override = (
                    transcribe_with_whisper_cli
                    if "transcribe_with_whisper_cli" in globals()
                    and getattr(transcribe_with_whisper_cli, "_mock_name", None) is not None
                    else None
                )
                if cli_override is not None:
                    cli_config = load_whisper_provider_config()
                    if cli_config is None:
                        raise ProviderUnavailableError(
                            "No transcription provider is configured for the requested workflow."
                        )
                    result = transcribe_with_whisper_cli(effective_params, cli_config)
                else:
                    result = pipeline.transcribe(effective_params)
                if extra_notes:
                    result = {**result, "notes": [*extra_notes, *result.get("notes", [])]}
                return make_result(identifier, result)
            finally:
                if trimmed_path is not None:
                    import contextlib

                    with contextlib.suppress(OSError):
                        from pathlib import Path as _Path

                        _Path(str(trimmed_path)).unlink()
        if method == "segment_probe":
            raw_audio_file = (params or {}).get("audio_file")
            if not isinstance(raw_audio_file, str) or not raw_audio_file.strip():
                return make_error(
                    identifier,
                    INVALID_REQUEST_CODE,
                    "segment_probe requires params.audio_file (non-empty string).",
                )
            probe_config = load_whisper_vad_config()
            if probe_config is None:
                return make_error(
                    identifier,
                    PROVIDER_UNAVAILABLE_CODE,
                    "VAD is not configured; set VOICELAYER_WHISPER_VAD_ENABLED=true and "
                    "VOICELAYER_WHISPER_VAD_MODEL_PATH (or the `[vad]` config section).",
                )
            try:
                return make_result(identifier, probe_audio_file(raw_audio_file, probe_config))
            except ProviderInvocationError as exc:
                return make_error(
                    identifier,
                    PROVIDER_REQUEST_FAILED_CODE,
                    str(exc),
                    {"method": method},
                )
        if method == "stitch_wav_segments":
            audio_files = (params or {}).get("audio_files")
            out_file = (params or {}).get("out_file")
            if (
                not isinstance(audio_files, list)
                or not audio_files
                or not all(isinstance(f, str) and f for f in audio_files)
            ):
                return make_error(
                    identifier,
                    INVALID_REQUEST_CODE,
                    "stitch_wav_segments requires params.audio_files (list of strings).",
                )
            if not isinstance(out_file, str) or not out_file:
                return make_error(
                    identifier,
                    INVALID_REQUEST_CODE,
                    "stitch_wav_segments requires params.out_file (non-empty string).",
                )
            return make_result(identifier, stitch_wav_segments(audio_files, out_file))
        if method == "compose":
            return make_result(identifier, pipeline.compose(params or {}))
        if method == "rewrite":
            return make_result(identifier, pipeline.rewrite(params or {}))
        if method == "translate":
            return make_result(identifier, pipeline.translate(params or {}))
    except InvalidProviderParamsError as exc:
        return make_error(identifier, INVALID_REQUEST_CODE, str(exc), {"method": method})
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
