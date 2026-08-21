from __future__ import annotations

import io
import json
import pathlib
import sys
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any
from unittest.mock import patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.config import (  # noqa: E402
    llm_config,
    whisper_config,
    whisper_server_config,
)
from voicelayer_orchestrator.providers import supported_providers  # noqa: E402
from voicelayer_orchestrator.providers.llm_openai_compatible import (  # noqa: E402
    resolve_chat_completions_url,
    resolve_models_url,
)
from voicelayer_orchestrator.providers.whisper_cli import (  # noqa: E402
    validate_whisper_provider,
)
from voicelayer_orchestrator.providers.whisper_server import (  # noqa: E402
    probe_whisper_server,
    transcribe_with_whisper_server,
)
from voicelayer_orchestrator.worker import (  # noqa: E402
    METHOD_NOT_FOUND_CODE,
    NOT_INITIALIZED_CODE,
    PARSE_ERROR_CODE,
    PROVIDER_UNAVAILABLE_CODE,
    handle_request,
    serve,
)


def init_worker(payload: dict[str, Any] | None = None) -> None:
    """Run the initialize handshake with a provider config payload."""

    response = handle_request(
        {
            "jsonrpc": "2.0",
            "id": "init",
            "method": "initialize",
            "params": payload or {},
        }
    )
    assert response is not None
    assert response["result"]["status"] == "ok"


class ConfiguredTestCase(unittest.TestCase):
    """Baseline: every test starts from an initialized, unconfigured worker."""

    def setUp(self) -> None:
        super().setUp()
        init_worker()


class FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(json.dumps({"data": [{"id": "gemma-3-1b-it"}]}).encode("utf-8"))
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
                    {"choices": [{"message": {"content": "Professional backend status update."}}]}
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
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        super().tearDown()


def create_fake_llama_server_script() -> tuple[str, str]:
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
                        payload = json.dumps(
                            {{"data": [{{"id": "gemma-3-1b-it"}}]}}
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(payload)
                        self._count()
                        return
                    self.send_response(404)
                    self.end_headers()

                def do_POST(self):
                    if self.path == "/v1/chat/completions":
                        length = int(self.headers.get("Content-Length", "0"))
                        self.rfile.read(length)
                        payload = json.dumps(
                            {{
                                "choices": [
                                    {{
                                        "message": {{
                                            "content": "Auto-started llama-server response."
                                        }}
                                    }}
                                ]
                            }}
                        ).encode("utf-8")
                        self.send_response(200)
                        self.send_header("Content-Type", "application/json")
                        self.end_headers()
                        self.wfile.write(payload)
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


def create_fake_whisper_cli_script() -> tuple[str, str, str]:
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


def llama_payload(endpoint: str, script_path: str, model_path: str, request_limit: int) -> dict:
    return {
        "llm": {
            "endpoint": endpoint,
            "model": "gemma-3-1b-it",
            "auto_start": True,
            "server_bin": script_path,
            "model_path": model_path,
            "server_args": f"--request-limit {request_limit}",
            "launch_timeout_seconds": 10,
            "poll_interval_seconds": 0.1,
        }
    }


class InitializeHandshakeTest(unittest.TestCase):
    def test_methods_before_initialize_are_rejected(self) -> None:
        import voicelayer_orchestrator.config as config_module

        # Force the uninitialized state regardless of test ordering.
        config_module._CONFIG = None
        response = handle_request({"jsonrpc": "2.0", "id": 1, "method": "health"})
        assert response is not None
        self.assertEqual(response["error"]["code"], NOT_INITIALIZED_CODE)
        init_worker()

    def test_initialize_returns_ok(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 2, "method": "initialize", "params": {}})
        assert response is not None
        self.assertEqual(response["result"]["status"], "ok")
        self.assertEqual(response["result"]["protocol"], "2.0")


class WorkerProtocolTest(FakeOpenAIServerMixin, ConfiguredTestCase):
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
        init_worker(
            {
                "llm": {
                    "endpoint": "http://127.0.0.1:8080",
                    "model": "gemma-3-1b-it",
                }
            }
        )
        provider_ids = {provider["id"] for provider in supported_providers()}
        self.assertIn("gemma_4_local", provider_ids)
        self.assertEqual(llm_config().endpoint, "http://127.0.0.1:8080")  # type: ignore[union-attr]

    def test_generation_methods_fail_without_provider(self) -> None:
        response = handle_request({"jsonrpc": "2.0", "id": 3, "method": "compose"})
        assert response is not None
        self.assertEqual(response["error"]["code"], PROVIDER_UNAVAILABLE_CODE)

    def test_provider_config_requires_endpoint_and_model(self) -> None:
        init_worker({"llm": {"endpoint": "http://localhost:8080"}})
        self.assertIsNone(llm_config())
        init_worker({"llm": {"model": "gemma"}})
        self.assertIsNone(llm_config())

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
        init_worker()
        self.assertIsNone(whisper_config())
        init_worker({"whisper": {"binary": "whisper-cli"}})
        self.assertIsNone(whisper_config())

    def test_validate_whisper_provider_detects_missing_binary(self) -> None:
        init_worker({"whisper": {"binary": "/does/not/exist", "model_path": "/tmp/model.bin"}})
        ready, error = validate_whisper_provider(whisper_config())
        self.assertFalse(ready)
        self.assertIn("Unable to find", error)

    def test_health_reports_reachable_llm_when_configured(self) -> None:
        init_worker({"llm": {"endpoint": self.endpoint, "model": "gemma-3-1b-it"}})
        response = handle_request({"jsonrpc": "2.0", "id": 5, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["llm_configured"])
        self.assertTrue(response["result"]["llm_reachable"])
        self.assertEqual(response["result"]["llm_model"], "gemma-3-1b-it")

    def test_compose_succeeds_when_openai_compatible_endpoint_is_configured(self) -> None:
        init_worker({"llm": {"endpoint": self.endpoint, "model": "gemma-3-1b-it"}})
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

    def test_llama_launch_config_reads_autostart_values(self) -> None:
        from voicelayer_orchestrator.config import llama_launch_config

        init_worker(
            {
                "llm": {
                    "endpoint": "http://127.0.0.1:8080",
                    "model": "gemma",
                    "auto_start": True,
                    "server_bin": "/usr/local/bin/llama-server",
                    "model_path": "/models/gemma.gguf",
                    "server_args": "--ctx-size 8192 --threads 4",
                }
            }
        )
        config = llama_launch_config()
        assert config is not None
        self.assertEqual(config.server_bin, "/usr/local/bin/llama-server")
        self.assertEqual(config.model_path, "/models/gemma.gguf")
        self.assertEqual(config.extra_args, ("--ctx-size", "8192", "--threads", "4"))

    def test_health_can_autostart_local_llama_server(self) -> None:
        script_path, model_path = create_fake_llama_server_script()
        endpoint = "http://127.0.0.1:18080"
        runtime_dir = tempfile.mkdtemp(prefix="voicelayer-runtime-")
        init_worker(llama_payload(endpoint, script_path, model_path, request_limit=1))
        with patch.dict("os.environ", {"XDG_RUNTIME_DIR": runtime_dir}, clear=False):
            response = handle_request({"jsonrpc": "2.0", "id": 7, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["llm_configured"])
        self.assertTrue(response["result"]["llm_reachable"])
        self.assertEqual(response["result"]["llm_endpoint"], endpoint)

    def test_compose_can_autostart_local_llama_server(self) -> None:
        script_path, model_path = create_fake_llama_server_script()
        endpoint = "http://127.0.0.1:18081"
        runtime_dir = tempfile.mkdtemp(prefix="voicelayer-runtime-")
        init_worker(llama_payload(endpoint, script_path, model_path, request_limit=2))
        with patch.dict("os.environ", {"XDG_RUNTIME_DIR": runtime_dir}, clear=False):
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
        script_path, model_path, _audio_path = create_fake_whisper_cli_script()
        init_worker({"whisper": {"binary": script_path, "model_path": model_path}})
        response = handle_request({"jsonrpc": "2.0", "id": 9, "method": "health"})

        assert response is not None
        self.assertTrue(response["result"]["asr_configured"])
        self.assertEqual(response["result"]["asr_binary"], script_path)

    def test_transcribe_succeeds_when_whisper_cli_is_configured(self) -> None:
        script_path, model_path, audio_path = create_fake_whisper_cli_script()
        init_worker({"whisper": {"binary": script_path, "model_path": model_path}})
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


class FakeWhisperServerHandler(BaseHTTPRequestHandler):
    text_payload = " hello world\n"
    language_payload: str | None = "en"

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/":
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><body>whisper server</body></html>")
            return

        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        if self.path == "/inference":
            length = int(self.headers.get("Content-Length", "0"))
            self.rfile.read(length)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            body: dict[str, str] = {"text": self.text_payload}
            if self.language_payload is not None:
                body["language"] = self.language_payload
            self.wfile.write(json.dumps(body).encode("utf-8"))
            return

        self.send_response(404)
        self.end_headers()

    def log_message(self, format: str, *args: object) -> None:  # noqa: A003
        return


class WhisperServerProviderTest(ConfiguredTestCase):
    def setUp(self) -> None:
        super().setUp()
        self.server = ThreadingHTTPServer(("127.0.0.1", 0), FakeWhisperServerHandler)
        self.server_thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.server_thread.start()
        self.host = "127.0.0.1"
        self.port = self.server.server_port
        self.audio_file = (
            pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-whisper-server-test-")) / "sample.wav"
        )
        self.audio_file.write_bytes(b"RIFFmockdataWAVEfmt ")

    def tearDown(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.server_thread.join(timeout=2)
        super().tearDown()

    def _config(self) -> object:
        init_worker({"whisper_server": {"host": self.host, "port": self.port}})
        return whisper_server_config()

    def test_whisper_server_config_returns_none_when_nothing_configured(self) -> None:
        init_worker()
        self.assertIsNone(whisper_server_config())

    def test_whisper_server_config_reads_host_port_and_autostart(self) -> None:
        init_worker(
            {
                "whisper": {"model_path": "/tmp/ggml.bin"},
                "whisper_server": {
                    "host": "127.0.0.1",
                    "port": 8188,
                    "auto_start": True,
                    "server_bin": "/tmp/whisper-server",
                    "extra_args": "-t 2",
                },
            }
        )
        config = whisper_server_config()
        assert config is not None
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8188)
        self.assertTrue(config.auto_start)
        self.assertEqual(config.server_bin, "/tmp/whisper-server")
        # The ggml model path is shared from the whisper section.
        self.assertEqual(config.model_path, "/tmp/ggml.bin")
        self.assertEqual(config.extra_args, ("-t", "2"))
        self.assertEqual(config.base_url, "http://127.0.0.1:8188")

    def test_whisper_server_config_defaults_host_and_port_when_only_autostart_set(self) -> None:
        init_worker(
            {
                "whisper": {"model_path": "/tmp/ggml.bin"},
                "whisper_server": {"auto_start": True, "server_bin": "/tmp/whisper-server"},
            }
        )
        config = whisper_server_config()
        assert config is not None
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8188)

    def test_probe_whisper_server_returns_reachable_when_root_is_served(self) -> None:
        reachable, error = probe_whisper_server(self._config())
        self.assertTrue(reachable)
        self.assertIsNone(error)

    def test_probe_whisper_server_returns_unreachable_when_port_closed(self) -> None:
        init_worker({"whisper_server": {"host": "127.0.0.1", "port": 1}})
        reachable, error = probe_whisper_server(whisper_server_config(), timeout_seconds=1.0)
        self.assertFalse(reachable)
        self.assertIsNotNone(error)

    def test_transcribe_with_whisper_server_returns_text(self) -> None:
        FakeWhisperServerHandler.text_payload = " hello world\n"
        FakeWhisperServerHandler.language_payload = "en"
        result = transcribe_with_whisper_server(
            {"audio_file": str(self.audio_file), "language": "auto"},
            self._config(),
        )
        self.assertEqual(result["text"], "hello world")
        self.assertEqual(result["detected_language"], "en")
        self.assertTrue(result["notes"])

    def test_transcribe_with_whisper_server_blank_audio_returns_empty_text(self) -> None:
        FakeWhisperServerHandler.text_payload = " [BLANK_AUDIO]\n"
        FakeWhisperServerHandler.language_payload = None
        result = transcribe_with_whisper_server(
            {"audio_file": str(self.audio_file), "language": "auto"},
            self._config(),
        )
        self.assertEqual(result["text"], "")

    def test_handle_request_dispatches_transcribe_to_whisper_server(self) -> None:
        FakeWhisperServerHandler.text_payload = " server path\n"
        FakeWhisperServerHandler.language_payload = "en"
        init_worker({"whisper_server": {"host": self.host, "port": self.port}})
        response = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 20,
                "method": "transcribe",
                "params": {
                    "audio_file": str(self.audio_file),
                    "language": "auto",
                },
            }
        )
        assert response is not None
        self.assertEqual(response["result"]["text"], "server path")


class WhisperTranscribeFallbackTest(FakeOpenAIServerMixin, ConfiguredTestCase):
    """When whisper-server is unreachable, the dispatcher falls back to whisper-cli."""

    def test_transcribe_falls_back_to_whisper_cli_when_server_unreachable(self) -> None:
        script_path, model_path, audio_path = create_fake_whisper_cli_script()
        init_worker(
            {
                "whisper": {"binary": script_path, "model_path": model_path},
                "whisper_server": {"host": "127.0.0.1", "port": 1},
            }
        )
        response = handle_request(
            {
                "jsonrpc": "2.0",
                "id": 21,
                "method": "transcribe",
                "params": {
                    "audio_file": audio_path,
                    "language": "auto",
                },
            }
        )
        assert response is not None
        self.assertIn("result", response)
        self.assertEqual(response["result"]["text"], "Recognized transcript.")


if __name__ == "__main__":
    unittest.main()
