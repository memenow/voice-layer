"""Tests for ``whisper-server`` HTTP client, CLI fallback, and wrapper script parity."""

from __future__ import annotations

import json
import os
import pathlib
import subprocess
import tempfile
import textwrap
import threading
import unittest
from http.server import BaseHTTPRequestHandler, HTTPServer
from unittest.mock import patch

from voicelayer_worker import pipeline, worker
from voicelayer_worker.config import (
    WhisperServerConfig,
    WorkerConfig,
    configure,
    reset_configuration_for_tests,
)
from voicelayer_worker.providers.whisper_server import (
    build_whisper_server_command,
    probe_whisper_server,
    transcribe_with_whisper_server,
    validate_autostart_prerequisites,
)

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
WRAPPER = PROJECT_ROOT / "scripts" / "voicelayer-whisper-server-run.sh"


class _FakeWhisperHandler(BaseHTTPRequestHandler):
    reply_text = "hello from whisper-server"

    def do_GET(self) -> None:  # noqa: N802
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b"ok")

    def do_POST(self) -> None:  # noqa: N802
        length = int(self.headers.get("Content-Length", "0"))
        self.rfile.read(length)
        body = json.dumps({"text": self.reply_text, "language": "en"}).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: object) -> None:  # noqa: A002
        pass


class WhisperServerProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-ws-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

        self.server = HTTPServer(("127.0.0.1", 0), _FakeWhisperHandler)
        self.port = self.server.server_port
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()
        self.addCleanup(self._stop_server)

    def _stop_server(self) -> None:
        self.server.shutdown()
        self.server.server_close()

    def _config(self, **overrides: object) -> WhisperServerConfig:
        base = {
            "host": "127.0.0.1",
            "port": self.port,
            "timeout_seconds": 5.0,
            "auto_start": False,
            "server_bin": "/usr/bin/whisper-server",
            "model_path": "/models/ggml-base.bin",
            "extra_args": (),
            "launch_timeout_seconds": 5.0,
            "poll_interval_seconds": 0.1,
        }
        base.update(overrides)
        return WhisperServerConfig(**base)  # type: ignore[arg-type]

    def test_probe_and_autostart_prerequisites(self) -> None:
        reachable, err = probe_whisper_server(self._config())
        self.assertTrue(reachable)
        self.assertIsNone(err)

        ok, err = validate_autostart_prerequisites(self._config())
        self.assertTrue(ok)
        self.assertIsNone(err)

        ok_no_bin, err_no_bin = validate_autostart_prerequisites(self._config(server_bin=None))
        self.assertFalse(ok_no_bin)
        assert err_no_bin is not None
        self.assertIn("VOICELAYER_WHISPER_SERVER_BIN", err_no_bin)

        ok_no_model, err_no_model = validate_autostart_prerequisites(self._config(model_path=None))
        self.assertFalse(ok_no_model)
        assert err_no_model is not None
        self.assertIn("VOICELAYER_WHISPER_MODEL_PATH", err_no_model)

    def test_transcribe_normalizes_blank_audio_and_dispatches_via_worker(self) -> None:
        audio = self.root / "sample.wav"
        audio.write_bytes(b"RIFF")

        _FakeWhisperHandler.reply_text = "hello from server"
        result = transcribe_with_whisper_server({"audio_file": str(audio)}, self._config())
        self.assertEqual(result["text"], "hello from server")
        self.assertEqual(result["detected_language"], "en")

        _FakeWhisperHandler.reply_text = "[BLANK_AUDIO]"
        blank = transcribe_with_whisper_server({"audio_file": str(audio)}, self._config())
        self.assertEqual(blank["text"], "")

        _FakeWhisperHandler.reply_text = "via worker"
        configure({"whisper_server": {"host": "127.0.0.1", "port": self.port}})
        resp = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "transcribe",
                "params": {"audio_file": str(audio)},
            }
        )
        assert resp is not None
        self.assertEqual(resp["result"]["text"], "via worker")

    def test_falls_back_to_whisper_cli_when_server_unreachable(self) -> None:
        audio = self.root / "sample.wav"
        audio.write_bytes(b"RIFF")
        configure(
            {
                "whisper": {"model_path": "/models/base.bin"},
                "whisper_server": {"host": "127.0.0.1", "port": 1},
            }
        )
        with (
            patch.object(pipeline, "ensure_whisper_server", return_value=(False, "refused")),
            patch.object(
                pipeline,
                "transcribe_with_whisper_cli",
                return_value={"text": "from cli", "detected_language": None, "notes": []},
            ) as cli_mock,
        ):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio)},
                }
            )
        assert resp is not None
        self.assertEqual(resp["result"]["text"], "from cli")
        cli_mock.assert_called_once()


class WrapperPythonParityTest(unittest.TestCase):
    """Pin ``scripts/voicelayer-whisper-server-run.sh`` argv to ``build_whisper_server_command``."""

    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-wrapper-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.fake = self.root / "fake-whisper-server"
        self.fake.write_text(
            textwrap.dedent(
                """\
                #!/usr/bin/env bash
                for arg in "$@"; do
                    printf '%s\\n' "$arg"
                done
                """
            )
        )
        self.fake.chmod(0o755)

    def _run_wrapper(self, env: dict[str, str], cwd: pathlib.Path | None = None) -> list[str]:
        result = subprocess.run(
            [str(WRAPPER)],
            env={**os.environ, **env},
            cwd=str(cwd) if cwd is not None else None,
            capture_output=True,
            text=True,
            check=True,
        )
        return [line for line in result.stdout.splitlines() if line]

    def test_wrapper_matches_python_argv_across_options_and_quoting(self) -> None:
        cases = [
            {},
            {"VOICELAYER_WHISPER_SERVER_HOST": "0.0.0.0", "VOICELAYER_WHISPER_SERVER_PORT": "9001"},
            {"VOICELAYER_WHISPER_SERVER_ARGS": '-t 4 --prompt "hello world"'},
        ]
        for overrides in cases:
            env = {
                "VOICELAYER_WHISPER_SERVER_BIN": str(self.fake),
                "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/ggml-base.en.bin",
                "VOICELAYER_WHISPER_SERVER_AUTO_START": "1",
                **overrides,
            }
            cfg = WorkerConfig.from_environ(env).whisper_server
            assert cfg is not None
            self.assertEqual(self._run_wrapper(env), build_whisper_server_command(cfg)[1:])

        # No pathname glob expansion on extra args.
        glob_dir = self.root / "glob"
        glob_dir.mkdir()
        (glob_dir / "a.txt").touch()
        env = {
            "VOICELAYER_WHISPER_SERVER_BIN": str(self.fake),
            "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/ggml-base.en.bin",
            "VOICELAYER_WHISPER_SERVER_AUTO_START": "1",
            "VOICELAYER_WHISPER_SERVER_ARGS": "-f *.txt",
        }
        cfg = WorkerConfig.from_environ(env).whisper_server
        assert cfg is not None
        argv = self._run_wrapper(env, cwd=glob_dir)
        self.assertEqual(argv, build_whisper_server_command(cfg)[1:])
        self.assertIn("*.txt", argv)

    def test_wrapper_rejects_missing_vars_and_bad_quoting(self) -> None:
        clean_env = {
            k: v
            for k, v in os.environ.items()
            if k not in {"VOICELAYER_WHISPER_SERVER_BIN", "VOICELAYER_WHISPER_MODEL_PATH"}
        }
        no_bin = subprocess.run(
            [str(WRAPPER)],
            env={**clean_env, "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/m.bin"},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(no_bin.returncode, 0)
        self.assertIn("VOICELAYER_WHISPER_SERVER_BIN", no_bin.stderr)

        no_model = subprocess.run(
            [str(WRAPPER)],
            env={**clean_env, "VOICELAYER_WHISPER_SERVER_BIN": str(self.fake)},
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(no_model.returncode, 0)
        self.assertIn("VOICELAYER_WHISPER_MODEL_PATH", no_model.stderr)

        bad_quote = subprocess.run(
            [str(WRAPPER)],
            env={
                **clean_env,
                "VOICELAYER_WHISPER_SERVER_BIN": str(self.fake),
                "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/m.bin",
                "VOICELAYER_WHISPER_SERVER_ARGS": '--prompt "unterminated',
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(bad_quote.returncode, 0)
        self.assertIn("VOICELAYER_WHISPER_SERVER_ARGS", bad_quote.stderr)
