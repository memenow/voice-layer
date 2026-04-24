#!/usr/bin/env python3
"""Deterministic JSON-RPC worker for voicelayerd integration tests.

The real Python worker (``python/voicelayer_orchestrator/worker.py``)
spawns whisper-cli, reaches out to llama-server, and otherwise requires
an audio stack that CI does not have. This stub speaks the same
stdio JSON-RPC protocol so tests can drive ``WorkerCommand`` without
those dependencies:

- Reads exactly one JSON-RPC request from stdin.
- Writes exactly one JSON-RPC response to stdout.
- Exits with status 0 on success, 1 on unrecoverable error (matches
  what ``WorkerCommand::call`` expects — it calls ``child.wait()``
  after reading the reply).

The first CLI argument, if present, is a path to a JSON config file.
Using a file (instead of environment variables) keeps the stub safe
under parallel ``cargo test`` execution: every test writes its own
config into a private tempdir and passes that path through
``WorkerCommand.args``, so concurrent runs never share state.

Config schema::

    {
      "transcribe_map": {"<stem>": "<text>"},
      "fail_stems": ["<stem>", ...],
      "default_transcribe_text": "<text>",
      "compose_payload": {"title": "...", "generated_text": "...", "notes": [...]},
      "rewrite_payload": {"title": "...", "generated_text": "...", "notes": [...]},
      "translate_payload": {"title": "...", "generated_text": "...", "notes": [...]},
      "fail_methods": ["compose", "rewrite", "translate"]
    }

A ``transcribe`` request whose ``params.audio_file`` basename (without
extension) matches a map entry returns ``{"text": <text>,
"detected_language": "en", "notes": []}`` (or an empty
``detected_language`` when ``<text>`` is empty). A stem listed in
``fail_stems`` returns a JSON-RPC error instead of a result.

When the stem matches neither ``transcribe_map`` nor ``fail_stems``,
``default_transcribe_text`` fills in the reply — or an empty text if
the key is absent. This keeps the fixture useful for callers whose
audio paths are not under test control (e.g. the one-shot capture
path uses a UUID-named file in ``$TMPDIR``).

``compose``, ``rewrite``, and ``translate`` return the matching
``<method>_payload`` object from config, or a default preview when the
key is absent. The payload shape mirrors
``python/voicelayer_orchestrator/providers/llm_openai_compatible.py``:
``{"title": str, "generated_text": str, "notes": list[str]}``. Adding
``<method>`` to ``fail_methods`` forces a JSON-RPC error for that
method instead of a result, which lets tests exercise the
``worker_error_receipt`` branch of each handler.

``health`` and ``list_providers`` return static payloads. Tests do not
exercise them today, but keeping them here makes the script a drop-in
replacement for any future probe.
"""

from __future__ import annotations

import json
import pathlib
import sys


def _load_config() -> dict:
    if len(sys.argv) < 2:
        return {}
    try:
        return json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _transcribe_response(request_id: object, params: dict, config: dict) -> dict:
    audio_path = pathlib.Path(str(params.get("audio_file", "")))
    stem = audio_path.stem

    fail_stems = set(config.get("fail_stems", []))
    if stem in fail_stems:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": f"mock worker failing on request for stem {stem}",
            },
        }

    mapping = config.get("transcribe_map", {})
    default_text = str(config.get("default_transcribe_text", ""))
    text = mapping.get(stem, default_text)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "text": text,
            "detected_language": "en" if text else None,
            "notes": [],
        },
    }


def _health_response(request_id: object) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "status": "ok",
            "worker": "mock",
            "protocol": "2.0",
            "asr_configured": True,
            "asr_binary": "/usr/bin/mock-whisper",
            "asr_model_path": "/usr/share/mock/model.bin",
            "asr_error": None,
            "whisper_mode": "mock",
            "whisper_server_url": None,
            "llm_configured": False,
            "llm_model": None,
            "llm_endpoint": None,
            "llm_reachable": False,
            "llm_error": None,
        },
    }


def _list_providers_response(request_id: object) -> dict:
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {"providers": []},
    }


def _preview_response(
    request_id: object,
    method: str,
    config: dict,
    default_title: str,
    default_generated_text: str,
) -> dict:
    fail_methods = set(config.get("fail_methods", []))
    if method in fail_methods:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32000,
                "message": f"mock worker failing on request for method {method}",
            },
        }

    payload = config.get(f"{method}_payload") or {}
    result = {
        "title": str(payload.get("title", default_title)),
        "generated_text": str(payload.get("generated_text", default_generated_text)),
        "notes": list(payload.get("notes", [])),
    }
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": result,
    }


def main() -> int:
    line = sys.stdin.readline()
    if not line:
        return 1
    try:
        request = json.loads(line)
    except json.JSONDecodeError:
        return 1

    config = _load_config()
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "transcribe":
        response = _transcribe_response(request_id, params, config)
    elif method == "compose":
        response = _preview_response(
            request_id,
            "compose",
            config,
            default_title="Compose preview",
            default_generated_text="mock compose output",
        )
    elif method == "rewrite":
        response = _preview_response(
            request_id,
            "rewrite",
            config,
            default_title="Rewrite preview",
            default_generated_text="mock rewrite output",
        )
    elif method == "translate":
        response = _preview_response(
            request_id,
            "translate",
            config,
            default_title="Translate preview",
            default_generated_text="mock translate output",
        )
    elif method == "health":
        response = _health_response(request_id)
    elif method == "list_providers":
        response = _list_providers_response(request_id)
    else:
        response = {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32601,
                "message": f"mock worker does not implement method {method!r}",
            },
        }

    sys.stdout.write(json.dumps(response) + "\n")
    sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
