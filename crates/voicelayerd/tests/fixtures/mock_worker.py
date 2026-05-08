#!/usr/bin/env python3
"""Deterministic JSON-RPC worker for voicelayerd integration tests.

The real Python worker (``python/voicelayer_orchestrator/worker.py``)
spawns whisper-cli, reaches out to llama-server, and otherwise requires
an audio stack that CI does not have. This stub speaks the same
stdio JSON-RPC protocol so tests can drive ``WorkerCommand`` without
those dependencies:

- Reads JSON-RPC requests line-by-line from stdin.
- Writes one JSON-RPC response line per request to stdout.
- Exits with status 0 when stdin closes (the daemon's persistent
  worker drops the child on shutdown, on a deterministic crash, or
  when the test fixture goes out of scope and `kill_on_drop(true)`
  reaps the process). Returns 1 only on unrecoverable parse failures.

Looping on stdin is required by the persistent-worker mode in
``WorkerCommand::call`` (see ``crates/voicelayerd/src/worker.rs``):
the daemon keeps a single child alive across requests so that GPU
provider model caches actually warm, then drains the response one
line at a time. A one-shot mock would let the persistent connection
collapse after the first call and force every subsequent test
request through a respawn dance.

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

``segment_probe`` looks up ``params.audio_file``'s stem in
``segment_probe_map`` and returns the configured verdict shape
``{"has_speech": bool, "speech_ratio": float, "regions": [...], "notes": [...]}``.
An unmapped stem falls back to ``segment_probe_default``; listing a
stem in ``segment_probe_fail_stems`` forces a JSON-RPC error.

``stitch_wav_segments`` creates the requested ``out_file`` as an empty
file so the downstream ``transcribe`` call in the same test can look
up its stem via ``transcribe_map``, and echoes a synthetic payload
``{"audio_file": out_file, "segment_count": N, "duration_secs": <stub>}``.
``stitch_should_fail=true`` swaps the result for a JSON-RPC error.

``health`` and ``list_providers`` return static payloads. Tests do not
exercise them today, but keeping them here makes the script a drop-in
replacement for any future probe.

When ``transcribe_log_path`` is set in the config, every ``transcribe``
call appends a single JSON line to that file capturing the request's
``audio_file`` basename, the ``provider_id`` field (or null when
omitted), and the resolved ``language``. Tests use the log to assert
that the daemon threads ``provider_id`` from a dictation session into
every transcribe call it issues. The log is opened in append mode so
concurrent ``cargo test`` runs can each point at their own tempfile
without trampling each other.
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

    log_path = config.get("transcribe_log_path")
    if log_path:
        record = {
            "audio_file_stem": stem,
            "provider_id": params.get("provider_id"),
            "language": params.get("language"),
            "translate_to_english": bool(params.get("translate_to_english", False)),
        }
        with pathlib.Path(log_path).open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record) + "\n")

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
            "mimo_configured": False,
            "mimo_model_path": None,
            "mimo_error": None,
            "qwen3_asr_configured": False,
            "qwen3_asr_model_path": None,
            "qwen3_asr_error": None,
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


def _segment_probe_response(request_id: object, params: dict, config: dict) -> dict:
    audio_file = str(params.get("audio_file", ""))
    stem = pathlib.Path(audio_file).stem

    fail_stems = set(config.get("segment_probe_fail_stems", []))
    if stem in fail_stems:
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32005,
                "message": f"mock segment_probe failing for stem {stem}",
            },
        }

    default = config.get(
        "segment_probe_default",
        {"has_speech": True, "speech_ratio": 1.0, "regions": [], "notes": []},
    )
    verdict = config.get("segment_probe_map", {}).get(stem, default)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "has_speech": bool(verdict.get("has_speech", True)),
            "speech_ratio": float(verdict.get("speech_ratio", 1.0)),
            "regions": list(verdict.get("regions", [])),
            "notes": list(verdict.get("notes", [])),
        },
    }


def _stitch_wav_segments_response(request_id: object, params: dict, config: dict) -> dict:
    if config.get("stitch_should_fail"):
        return {
            "jsonrpc": "2.0",
            "id": request_id,
            "error": {
                "code": -32005,
                "message": "mock stitch_wav_segments forced failure",
            },
        }

    out_file = str(params.get("out_file", ""))
    audio_files = list(params.get("audio_files", []))
    # The real helper writes a concatenated WAV. The mock just makes the
    # path exist so the downstream transcribe call finds something; tests
    # that care about stitched text key their transcribe_map by the
    # output path's stem.
    out_path = pathlib.Path(out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.touch()
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "result": {
            "audio_file": out_file,
            "segment_count": len(audio_files),
            # Deterministic stub: 1.5 s per probe. Tests that care about
            # the exact duration should override this through a dedicated
            # knob if the need arises; for now only segment_count is
            # validated on the Rust side.
            "duration_secs": float(len(audio_files)) * 1.5,
        },
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


def _dispatch_one(request: dict, config: dict) -> dict:
    request_id = request.get("id")
    method = request.get("method")
    params = request.get("params") or {}

    if method == "transcribe":
        return _transcribe_response(request_id, params, config)
    if method == "segment_probe":
        return _segment_probe_response(request_id, params, config)
    if method == "stitch_wav_segments":
        return _stitch_wav_segments_response(request_id, params, config)
    if method == "compose":
        return _preview_response(
            request_id,
            "compose",
            config,
            default_title="Compose preview",
            default_generated_text="mock compose output",
        )
    if method == "rewrite":
        return _preview_response(
            request_id,
            "rewrite",
            config,
            default_title="Rewrite preview",
            default_generated_text="mock rewrite output",
        )
    if method == "translate":
        return _preview_response(
            request_id,
            "translate",
            config,
            default_title="Translate preview",
            default_generated_text="mock translate output",
        )
    if method == "health":
        return _health_response(request_id)
    if method == "list_providers":
        return _list_providers_response(request_id)
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "error": {
            "code": -32601,
            "message": f"mock worker does not implement method {method!r}",
        },
    }


def main() -> int:
    config = _load_config()
    for raw_line in sys.stdin:
        line = raw_line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError:
            return 1
        response = _dispatch_one(request, config)
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()
    return 0


if __name__ == "__main__":
    sys.exit(main())
