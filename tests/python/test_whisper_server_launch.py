"""Regression tests for ``scripts/voicelayer-whisper-server-run.sh``.

The bash wrapper and
``voicelayer_orchestrator.providers.whisper_server._build_whisper_server_command``
produce two independent argv layouts for the same underlying
``whisper-server`` invocation: the wrapper is used by
``systemd/voicelayer-whisper-server.service`` while the Python function
runs inside the daemon's autostart path. These tests pin the two in
lockstep so adding a new argument on one side fails CI until the other
side is updated.
"""

from __future__ import annotations

import os
import pathlib
import subprocess
import sys
import textwrap
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"
WRAPPER = PROJECT_ROOT / "scripts" / "voicelayer-whisper-server-run.sh"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.config import (  # noqa: E402
    WhisperServerConfig,
    load_whisper_server_config,
)
from voicelayer_orchestrator.providers.whisper_server import (  # noqa: E402
    _build_whisper_server_command,
)


def _write_fake_server(tmp_path: pathlib.Path) -> pathlib.Path:
    """Drop a fake `whisper-server` that prints argv on one token per line."""

    fake = tmp_path / "fake-whisper-server"
    fake.write_text(
        textwrap.dedent(
            """\
            #!/usr/bin/env bash
            for arg in "$@"; do
                printf '%s\\n' "$arg"
            done
            """
        )
    )
    fake.chmod(0o755)
    return fake


def _wrapper_argv(env: dict[str, str]) -> list[str]:
    """Invoke the wrapper and return the argv it would have passed to whisper-server."""

    merged = {**os.environ, **env}
    result = subprocess.run(
        [str(WRAPPER)],
        env=merged,
        capture_output=True,
        text=True,
        check=True,
    )
    # Token-per-line so args with embedded whitespace survive (they
    # shouldn't appear in this test matrix, but the format is safer).
    return [line for line in result.stdout.splitlines() if line]


class WrapperPythonParityTest(unittest.TestCase):
    def setUp(self) -> None:
        self.assertTrue(
            WRAPPER.is_file(),
            f"wrapper missing at {WRAPPER}; install step drifted?",
        )
        self.assertTrue(
            os.access(WRAPPER, os.X_OK),
            f"wrapper at {WRAPPER} is not executable",
        )
        tmp_root = pathlib.Path(subprocess.check_output(["mktemp", "-d"]).decode().strip())
        self.addCleanup(lambda: subprocess.run(["rm", "-rf", str(tmp_root)], check=False))
        self.tmp_dir = tmp_root

    def _fake_env(
        self,
        *,
        extra_args: str = "",
        host: str | None = None,
        port: str | None = None,
    ) -> tuple[dict[str, str], WhisperServerConfig]:
        fake = _write_fake_server(self.tmp_dir)
        env: dict[str, str] = {
            "VOICELAYER_WHISPER_SERVER_BIN": str(fake),
            "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/ggml-base.en.bin",
            "VOICELAYER_WHISPER_SERVER_AUTO_START": "1",
        }
        if host is not None:
            env["VOICELAYER_WHISPER_SERVER_HOST"] = host
        if port is not None:
            env["VOICELAYER_WHISPER_SERVER_PORT"] = port
        if extra_args:
            env["VOICELAYER_WHISPER_SERVER_ARGS"] = extra_args
        config = load_whisper_server_config(env)
        assert config is not None
        return env, config

    def test_wrapper_argv_matches_python_builder_minimum_config(self) -> None:
        env, config = self._fake_env()
        wrapper_argv = _wrapper_argv(env)
        python_argv = _build_whisper_server_command(config)
        # python_argv[0] is the binary path; the wrapper prints only
        # what exec'd into the binary as argv, i.e. python_argv[1:].
        self.assertEqual(wrapper_argv, python_argv[1:])

    def test_wrapper_argv_matches_python_builder_with_host_port_override(self) -> None:
        env, config = self._fake_env(host="0.0.0.0", port="9001")
        wrapper_argv = _wrapper_argv(env)
        python_argv = _build_whisper_server_command(config)
        self.assertEqual(wrapper_argv, python_argv[1:])

    def test_wrapper_argv_matches_python_builder_with_extra_args(self) -> None:
        env, config = self._fake_env(extra_args="-t 4 --flash-attn")
        wrapper_argv = _wrapper_argv(env)
        python_argv = _build_whisper_server_command(config)
        self.assertEqual(wrapper_argv, python_argv[1:])

    def test_wrapper_fails_clearly_without_server_bin(self) -> None:
        env = {
            "VOICELAYER_WHISPER_MODEL_PATH": "/tmp/ggml-base.en.bin",
        }
        result = subprocess.run(
            [str(WRAPPER)],
            env={
                **{k: v for k, v in os.environ.items() if k != "VOICELAYER_WHISPER_SERVER_BIN"},
                **env,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(
            result.returncode,
            0,
            "wrapper must reject missing VOICELAYER_WHISPER_SERVER_BIN",
        )
        self.assertIn("VOICELAYER_WHISPER_SERVER_BIN", result.stderr)

    def test_wrapper_fails_clearly_without_model_path(self) -> None:
        fake = _write_fake_server(self.tmp_dir)
        env = {
            "VOICELAYER_WHISPER_SERVER_BIN": str(fake),
        }
        result = subprocess.run(
            [str(WRAPPER)],
            env={
                **{k: v for k, v in os.environ.items() if k != "VOICELAYER_WHISPER_MODEL_PATH"},
                **env,
            },
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(
            result.returncode,
            0,
            "wrapper must reject missing VOICELAYER_WHISPER_MODEL_PATH",
        )
        self.assertIn("VOICELAYER_WHISPER_MODEL_PATH", result.stderr)


if __name__ == "__main__":
    unittest.main()
