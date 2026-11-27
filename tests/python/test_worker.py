"""Tests for stdio JSON-RPC server dispatch, ``initialize``, and ASR routing."""

from __future__ import annotations

import io
import json
import unittest
from unittest.mock import patch

from voicelayer_worker import pipeline, worker
from voicelayer_worker.config import configure, reset_configuration_for_tests
from voicelayer_worker.errors import (
    INVALID_REQUEST_CODE,
    METHOD_NOT_FOUND_CODE,
    PARSE_ERROR_CODE,
    PROVIDER_UNAVAILABLE_CODE,
)


class WorkerDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()

    def tearDown(self) -> None:
        reset_configuration_for_tests()

    def test_initialize_configures_worker_and_controls_list_providers_and_health(self) -> None:
        init_resp = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": "init-1",
                "method": "initialize",
                "params": {
                    "whisper": {"model_path": "/models/ggml-base.bin"},
                    "llm": {"endpoint": "http://127.0.0.1:8080", "model": "gemma-4-local"},
                },
            }
        )
        assert init_resp is not None
        self.assertEqual(init_resp["result"]["worker"], "voicelayer_worker")

        providers_resp = worker.handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "list_providers"}
        )
        assert providers_resp is not None
        by_id = {entry["id"]: entry for entry in providers_resp["result"]["providers"]}
        self.assertEqual(by_id["whisper_cpp"]["transport"], "whisper_cli")
        self.assertEqual(by_id["gemma_4_local"]["transport"], "openai_chat_completions")

        with patch.object(pipeline, "ensure_llm_endpoint", return_value=(True, None)):
            health_resp = worker.handle_request({"jsonrpc": "2.0", "id": 3, "method": "health"})
        assert health_resp is not None
        self.assertEqual(health_resp["result"]["worker"], "voicelayer_worker")
        self.assertEqual(health_resp["result"]["whisper_mode"], "cli")
        self.assertTrue(health_resp["result"]["llm_configured"])
        self.assertTrue(health_resp["result"]["llm_reachable"])

    def test_invalid_requests_and_unknown_methods(self) -> None:
        bad_proto = worker.handle_request({"jsonrpc": "1.0", "id": 1, "method": "health"})
        assert bad_proto is not None
        self.assertEqual(bad_proto["error"]["code"], INVALID_REQUEST_CODE)

        bad_params = worker.handle_request(
            {"jsonrpc": "2.0", "id": 2, "method": "health", "params": ["not-an-object"]}
        )
        assert bad_params is not None
        self.assertEqual(bad_params["error"]["code"], INVALID_REQUEST_CODE)

        unknown = worker.handle_request({"jsonrpc": "2.0", "id": 3, "method": "does_not_exist"})
        assert unknown is not None
        self.assertEqual(unknown["error"]["code"], METHOD_NOT_FOUND_CODE)

        bad_quotes = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "initialize",
                "params": {"whisper": {"model_path": "/m.bin", "extra_args": '"unclosed'}},
            }
        )
        assert bad_quotes is not None
        self.assertEqual(bad_quotes["error"]["code"], INVALID_REQUEST_CODE)

    def test_transcribe_provider_id_routing_and_errors(self) -> None:
        configure({})

        non_str = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "transcribe",
                "params": {"audio_file": "/tmp/a.wav", "provider_id": 123},
            }
        )
        assert non_str is not None
        self.assertEqual(non_str["error"]["code"], INVALID_REQUEST_CODE)

        for provider_id in ("unknown_asr", "mimo_v2_5_asr", "qwen3_asr_1_7b"):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "transcribe",
                    "params": {"audio_file": "/tmp/a.wav", "provider_id": provider_id},
                }
            )
            assert resp is not None
            self.assertEqual(resp["error"]["code"], PROVIDER_UNAVAILABLE_CODE)

        configure({"whisper": {"model_path": "/models/base.bin"}})
        with patch.object(
            pipeline,
            "transcribe_with_whisper_cli",
            return_value={"text": "explicit whisper", "detected_language": None, "notes": []},
        ) as cli_mock:
            ok = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "transcribe",
                    "params": {"audio_file": "/tmp/a.wav", "provider_id": "whisper_cpp"},
                }
            )
        assert ok is not None
        self.assertEqual(ok["result"]["text"], "explicit whisper")
        cli_mock.assert_called_once()

    def test_serve_skips_blank_lines_handles_parse_error_and_writes_responses(self) -> None:
        stdin = io.StringIO(
            "\n"
            "{not valid json\n"
            + json.dumps({"jsonrpc": "2.0", "id": 9, "method": "initialize", "params": {}})
            + "\n"
        )
        stdout = io.StringIO()
        exit_code = worker.serve(stdin, stdout)
        self.assertEqual(exit_code, 0)

        lines = [json.loads(line) for line in stdout.getvalue().splitlines()]
        self.assertEqual(len(lines), 2)
        self.assertEqual(lines[0]["error"]["code"], PARSE_ERROR_CODE)
        self.assertEqual(lines[1]["id"], 9)
        self.assertEqual(lines[1]["result"]["status"], "ok")
