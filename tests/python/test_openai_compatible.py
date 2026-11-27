"""Tests for OpenAI-compatible LLM client and ``llama-server`` autostart."""

from __future__ import annotations

import json
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer

from voicelayer_worker import worker
from voicelayer_worker.config import (
    LlamaServerLaunchConfig,
    OpenAICompatibleConfig,
    configure,
    reset_configuration_for_tests,
)
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.providers.openai_compatible import (
    build_llama_server_command,
    configured_llm_descriptor,
    ensure_llm_endpoint,
    is_local_endpoint,
    render_content_text,
    resolve_chat_completions_url,
    resolve_models_url,
)


class _FakeOpenAIHandler(BaseHTTPRequestHandler):
    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/v1/models":
            body = json.dumps({"data": [{"id": "gemma-4-local"}]}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return
        self.send_response(404)
        self.end_headers()

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(length).decode("utf-8"))
        user_msg = payload["messages"][-1]["content"]
        body = json.dumps(
            {
                "choices": [
                    {
                        "message": {
                            "content": f"Generated: {user_msg.splitlines()[-1]}",
                        }
                    }
                ]
            }
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class OpenAICompatibleHelpersTest(unittest.TestCase):
    def test_url_normalization_and_local_detection(self) -> None:
        self.assertEqual(
            resolve_chat_completions_url("http://127.0.0.1:8080"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            resolve_chat_completions_url("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )
        self.assertEqual(
            resolve_models_url("http://127.0.0.1:8080/v1/chat/completions"),
            "http://127.0.0.1:8080/v1/models",
        )
        self.assertTrue(is_local_endpoint("http://localhost:8080"))
        self.assertTrue(is_local_endpoint("http://127.0.0.1:8080"))
        self.assertFalse(is_local_endpoint("https://api.example.com/v1"))

    def test_render_content_text_and_descriptor(self) -> None:
        self.assertEqual(render_content_text("  hello  "), "hello")
        self.assertEqual(
            render_content_text([{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]),
            "a\nb",
        )
        self.assertEqual(render_content_text(None), "")

        cfg = OpenAICompatibleConfig("http://127.0.0.1:8080", "gemma-4-it", None, 30.0)
        desc = configured_llm_descriptor(cfg)
        assert desc is not None
        self.assertEqual(desc["id"], "gemma_4_local")
        self.assertTrue(desc["local"])

    def test_build_llama_server_command_and_ensure_endpoint(self) -> None:
        local_cfg = OpenAICompatibleConfig("http://127.0.0.1:9090", "gemma", None, 5.0)
        launch = LlamaServerLaunchConfig(
            server_bin="/usr/bin/llama-server",
            model_path="/models/gemma.gguf",
            hf_repo=None,
            extra_args=("--ctx-size", "4096"),
            launch_timeout_seconds=1.0,
            poll_interval_seconds=0.1,
        )
        self.assertEqual(
            build_llama_server_command(local_cfg, launch),
            [
                "/usr/bin/llama-server",
                "-m",
                "/models/gemma.gguf",
                "--port",
                "9090",
                "--ctx-size",
                "4096",
            ],
        )

        hf_launch = LlamaServerLaunchConfig("llama-server", None, "org/model", (), 1.0, 0.1)
        self.assertEqual(
            build_llama_server_command(local_cfg, hf_launch),
            ["llama-server", "-hf", "org/model", "--port", "9090"],
        )

        remote_cfg = OpenAICompatibleConfig("https://remote.example/v1", "m", None, 5.0)
        with self.assertRaises(ProviderInvocationError):
            build_llama_server_command(remote_cfg, launch)

        missing_src = LlamaServerLaunchConfig("llama-server", None, None, (), 1.0, 0.1)
        with self.assertRaises(ProviderInvocationError):
            build_llama_server_command(local_cfg, missing_src)

        self.assertEqual(ensure_llm_endpoint(None), (False, None))


class OpenAICompatibleRpcTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.server = HTTPServer(("127.0.0.1", 0), _FakeOpenAIHandler)
        self.endpoint = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def test_compose_rewrite_and_translate_dispatch_through_initialize_config(self) -> None:
        configure({"llm": {"endpoint": self.endpoint, "model": "gemma-4-local"}})

        compose = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "compose",
                "params": {"spoken_prompt": "Write release notes", "archetype": "email"},
            }
        )
        assert compose is not None
        self.assertIn("Write release notes", compose["result"]["generated_text"])

        rewrite = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "rewrite",
                "params": {"source_text": "rough draft", "style": "shorter"},
            }
        )
        assert rewrite is not None
        self.assertIn("rough draft", rewrite["result"]["generated_text"])

        translate = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "translate",
                "params": {"source_text": "hello world", "target_language": "zh"},
            }
        )
        assert translate is not None
        self.assertIn("hello world", translate["result"]["generated_text"])
