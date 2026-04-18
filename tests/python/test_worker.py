from __future__ import annotations

import io
import json
import os
import pathlib
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import sys
import tempfile
import textwrap
import unittest
from unittest.mock import patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.worker import (  # noqa: E402
    METHOD_NOT_FOUND_CODE,
    PARSE_ERROR_CODE,
    PROVIDER_UNAVAILABLE_CODE,
    ensure_llm_endpoint,
    handle_request,
    load_llama_server_launch_config,
    load_llm_provider_config,
    load_whisper_provider_config,
    resolve_models_url,
    resolve_chat_completions_url,
    serve,
    supported_providers,
    validate_whisper_provider,
)


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps({"data": [{"id": "gemma-3-1b-it"}]}).encode("utf-8")
            )
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/v1/chat/completions":
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(
                json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "Professional backend status update."
                                }
                            }
                        ]
                    }
                ).encode("utf-8")
            )
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class FakeOpenAIServerMixin:
    server: ThreadingHTTPServer
    server_thread: threading.Thread

    def setUp(self) -> None:
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeOpenAIHandler)
        self.server_thread = threading.Thread(
            target=self.server.serve_forever, daemon=True
        )
        self.server_thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        super().tearDown()


class WorkerProtocolTest(FakeOpenAIServerMixin, unittest.TestCase):
    def create_fake_llama_server_script(self) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp(prefix="voicelayer-llama-test-")
        script_path = pathlib.Path(temp_dir) / "fake_llama_server.py"
        model_path = pathlib.Path(temp_dir) / "model.gguf"
        model_path.write_text("placeholder", encoding="utf-8")
        script_path.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import argparse
                import json
                import threading
                from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

                parser = argparse.ArgumentParser()
                parser.add_argument("-m", dest="model_path", default=None)
                parser.add_argument("-hf", dest="hf_repo", default=None)
                parser.add_argument("--port", type=int, required=True)
                parser.add_argument("--request-limit", type=int, default=2)
                args, _ = parser.parse_known_args()

                class Handler(BaseHTTPRequestHandler):
                    def _count(self):
                        self.server.request_count += 1
                        if self.server.request_count >= args.request_limit:
                            threading.Thread(target=self.server.shutdown, daemon=True).start()

                    def do_GET(self):
                        if self.path == "/v1/models":
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({{"data": [{{"id": "gemma-3-1b-it"}}]}}).encode("utf-8"))
                            self._count()
                            return
                        self.send_response(404)
                        self.end_headers()

                    def do_POST(self):
                        if self.path == "/v1/chat/completions":
                            length = int(self.headers.get("Content-Length", "0"))
                            self.rfile.read(length)
                            self.send_response(200)
                            self.send_header("Content-Type", "application/json")
                            self.end_headers()
                            self.wfile.write(json.dumps({{"choices": [{{"message": {{"content": "Auto-started llama-server response."}}}}]}}).encode("utf-8"))
                            self._count()
                            return
                        self.send_response(404)
                        self.end_headers()

                    def log_message(self, format, *args):
                        return

                server = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
                server.request_count = 0
                server.serve_forever()
                """
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        return str(script_path), str(model_path)

    def create_fake_whisper_cli_script(self) -> tuple[str, str]:
        temp_dir = tempfile.mkdtemp(prefix="voicelayer-whisper-test-")
        script_path = pathlib.Path(temp_dir) / "fake_whisper_cli.py"
        model_path = pathlib.Path(temp_dir) / "model.bin"
        audio_path = pathlib.Path(temp_dir) / "sample.wav"
        model_path.write_text("placeholder", encoding="utf-8")
        audio_path.write_text("not-real-audio", encoding="utf-8")
        script_path.write_text(
            textwrap.dedent(
                f"""\
                #!{sys.executable}
                import argparse
                from pathlib import Path

                parser = argparse.ArgumentParser()
                parser.add_argument("-m", dest="model_path")
                parser.add_argument("-f", dest="audio_file")
                parser.add_argument("-of", dest="output_file")
                parser.add_argument("-l", dest="language", default="auto")
                parser.add_argument("-tr", dest="translate", action="store_true")
                parser.add_argument("-otxt", action="store_true")
                parser.add_argument("-np", action="store_true")
                parser.add_argument("-ng", action="store_true")
                parser.add_argument("rest", nargs="*")
                args, _ = parser.parse_known_args()

                text = "Translated transcript." if args.translate else "Recognized transcript."
                Path(args.output_file + ".txt").write_text(text, encoding="utf-8")
                """
            ),
            encoding="utf-8",
        )
        script_path.chmod(0o755)
        return str(script_path), str(model_path), str(audio_path)

    def test_health_request_returns_ok(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "health"})
        assert response is not None
        self.assertEqual(response["result"]["status"], "ok")

    def test_list_providers_returns_expected_defaults(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 2, "method": "list_providers"})
        assert response is not None
        provider_ids = {provider["id"] for provider in response["result"]["providers"]}
        self.assertIn("whisper_cpp", provider_ids)
        self.assertIn("gemma_4_local", provider_ids)

    def test_supported_providers_reflect_configured_llm_endpoint(self) -> None:
        providers = supported_providers(
            {
                "VOICELAYER_LLM_ENDPOINT": "http://127.0.0.1:8080",
                "VOICELAYER_LLM_MODEL": "gemma-3-1b-it",
            }
        )
        provider_ids = {provider["id"] for provider in providers}
        self.assertIn("gemma_4_local", provider_ids)

    def test_generation_methods_fail_without_provider(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 3, "method": "compose"})
        assert response is not None
        self.assertEqual(response["error"]["code"], PROVIDER_UNAVAILABLE_CODE)

    def test_provider_config_requires_endpoint_and_model(self) -> None:
        self.assertIsNone(
            load_llm_provider_config(
                {"VOICELAYER_LLM_ENDPOINT": "http://localhost:8080"}
            )
        )
        self.assertIsNone(load_llm_provider_config({"VOICELAYER_LLM_MODEL": "gemma"}))

    def test_chat_completions_url_is_normalized(self) -> None:
        self.assertEqual(
            resolve_chat_completions_url("http://localhost:8080"),
            "http://localhost:8080/v1/chat/completions",
        )
        self.assertEqual(
            resolve_chat_completions_url("http://localhost:8080/v1"),
            "http://localhost:8080/v1/chat/completions",
        )
        self.assertEqual(
            resolve_chat_completions_url("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080/v1/chat/completions",
        )

    def test_models_url_is_normalized(self) -> None:
        self.assertEqual(
            resolve_models_url("http://localhost:8080"),
            "http://localhost:8080/v1/models",
        )
        self.assertEqual(
            resolve_models_url("http://localhost:8080/v1"),
            "http://localhost:8080/v1/models",
        )
        self.assertEqual(
            resolve_models_url("http://localhost:8080/v1/chat/completions"),
            "http://localhost:8080/v1/models",
        )

    def test_whisper_provider_config_requires_model(self) -> None:
        self.assertIsNone(load_whisper_provider_config({}))
        self.assertIsNone(
            load_whisper_provider_config({"VOICELAYER_WHISPER_BIN": "whisper-cli"})
        )

    def test_validate_whisper_provider_detects_missing_binary(self) -> None:
        config = load_whisper_provider_config(
            {
                "VOICELAYER_WHISPER_BIN": "/does/not/exist",
                "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/model.bin",
            }
        )
        ready, error = validate_whisper_provider(config)
        self.assertFalse(ready)
        self.assertIn("Unable to find", error)

    def test_health_reports_reachable_llm_when_configured(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VOICELAYER_LLM_ENDPOINT": self.endpoint,
                "VOICELAYER_LLM_MODEL": "gemma-3-1b-it",
            },
            clear=False,
        ):
            response = handle_request({"jsonrpc": "2.0", "id": 5, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["llm_configured"])
        self.assertTrue(response["result"]["llm_reachable"])
        self.assertEqual(response["result"]["llm_model"], "gemma-3-1b-it")

    def test_compose_succeeds_when_openai_compatible_endpoint_is_configured(self) -> None:
        with patch.dict(
            "os.environ",
            {
                "VOICELAYER_LLM_ENDPOINT": self.endpoint,
                "VOICELAYER_LLM_MODEL": "gemma-3-1b-it",
            },
            clear=False,
        ):
            response = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "compose",
                    "params": {
                        "spoken_prompt": "Write a professional backend status update.",
                        "archetype": "daily_report",
                    },
                }
            )

        assert response is not None
        self.assertEqual(
            response["result"]["generated_text"],
            "Professional backend status update.",
        )

    def test_load_llama_server_launch_config_reads_autostart_values(self) -> None:
        config = load_llama_server_launch_config(
            {
                "VOICELAYER_LLM_AUTO_START": "true",
                "VOICELAYER_LLAMA_SERVER_BIN": "/usr/local/bin/llama-server",
                "VOICELAYER_LLAMA_MODEL_PATH": "/models/gemma.gguf",
                "VOICELAYER_LLAMA_SERVER_ARGS": "--ctx-size 8192 --threads 4",
            }
        )
        assert config is not None
        self.assertEqual(config.server_bin, "/usr/local/bin/llama-server")
        self.assertEqual(config.model_path, "/models/gemma.gguf")
        self.assertEqual(config.extra_args, ("--ctx-size", "8192", "--threads", "4"))

    def test_health_can_autostart_local_llama_server(self) -> None:
        script_path, model_path = self.create_fake_llama_server_script()
        endpoint = "http://127.0.0.1:18080"
        runtime_dir = tempfile.mkdtemp(prefix="voicelayer-runtime-")
        environ = {
            "VOICELAYER_LLM_ENDPOINT": endpoint,
            "VOICELAYER_LLM_MODEL": "gemma-3-1b-it",
            "VOICELAYER_LLM_AUTO_START": "true",
            "VOICELAYER_LLAMA_SERVER_BIN": script_path,
            "VOICELAYER_LLAMA_MODEL_PATH": model_path,
            "VOICELAYER_LLAMA_SERVER_ARGS": "--request-limit 1",
            "VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS": "10",
            "VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS": "0.1",
            "XDG_RUNTIME_DIR": runtime_dir,
        }
        with patch.dict("os.environ", environ, clear=False):
            response = handle_request({"jsonrpc": "2.0", "id": 7, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["llm_configured"])
        self.assertTrue(response["result"]["llm_reachable"])
        self.assertEqual(response["result"]["llm_endpoint"], endpoint)

    def test_compose_can_autostart_local_llama_server(self) -> None:
        script_path, model_path = self.create_fake_llama_server_script()
        endpoint = "http://127.0.0.1:18081"
        runtime_dir = tempfile.mkdtemp(prefix="voicelayer-runtime-")
        environ = {
            "VOICELAYER_LLM_ENDPOINT": endpoint,
            "VOICELAYER_LLM_MODEL": "gemma-3-1b-it",
            "VOICELAYER_LLM_AUTO_START": "true",
            "VOICELAYER_LLAMA_SERVER_BIN": script_path,
            "VOICELAYER_LLAMA_MODEL_PATH": model_path,
            "VOICELAYER_LLAMA_SERVER_ARGS": "--request-limit 2",
            "VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS": "10",
            "VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS": "0.1",
            "XDG_RUNTIME_DIR": runtime_dir,
        }
        with patch.dict("os.environ", environ, clear=False):
            response = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 8,
                    "method": "compose",
                    "params": {
                        "spoken_prompt": "Write a professional backend status update.",
                        "archetype": "daily_report",
                    },
                }
            )

        assert response is not None
        self.assertEqual(
            response["result"]["generated_text"],
            "Auto-started llama-server response.",
        )

    def test_health_reports_configured_whisper_provider(self) -> None:
        script_path, model_path, _audio_path = self.create_fake_whisper_cli_script()
        with patch.dict(
            "os.environ",
            {
                "VOICELAYER_WHISPER_BIN": script_path,
                "VOICELAYER_WHISPER_MODEL_PATH": model_path,
            },
            clear=False,
        ):
            response = handle_request({"jsonrpc": "2.0", "id": 9, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["asr_configured"])
        self.assertEqual(response["result"]["asr_binary"], script_path)

    def test_transcribe_succeeds_when_whisper_cli_is_configured(self) -> None:
        script_path, model_path, audio_path = self.create_fake_whisper_cli_script()
        with patch.dict(
            "os.environ",
            {
                "VOICELAYER_WHISPER_BIN": script_path,
                "VOICELAYER_WHISPER_MODEL_PATH": model_path,
            },
            clear=False,
        ):
            response = handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "transcribe",
                    "params": {
                        "audio_file": audio_path,
                        "language": "auto",
                        "translate_to_english": False,
                    },
                }
            )

        assert response is not None
        self.assertEqual(response["result"]["text"], "Recognized transcript.")

    def test_unknown_method_returns_method_not_found(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 4, "method": "unknown"})
        assert response is not None
        self.assertEqual(response["error"]["code"], METHOD_NOT_FOUND_CODE)

    def test_serve_reports_parse_error(self) -> None:
        stdin = io.StringIO("{not-json}\n")
        stdout = io.StringIO()

        exit_code = serve(stdin, stdout)

        self.assertEqual(exit_code, 0)
        response = json.loads(stdout.getvalue())
        self.assertEqual(response["error"]["code"], PARSE_ERROR_CODE)


if __name__ == "__main__":
    unittest.main()
