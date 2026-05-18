from __future__ import annotations

import pathlib
import subprocess
import sys
import unittest
from unittest.mock import patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.config import (  # noqa: E402
    LlamaServerLaunchConfig,
    OpenAICompatibleConfig,
)
from voicelayer_orchestrator.providers import (  # noqa: E402
    ProviderInvocationError,
    llama_autostart,
)
from voicelayer_orchestrator.providers.llama_autostart import (  # noqa: E402
    build_llama_server_command,
    ensure_llm_endpoint,
    wait_for_llm_endpoint,
)


def _llm_config(endpoint: str = "http://127.0.0.1:18099") -> OpenAICompatibleConfig:
    return OpenAICompatibleConfig(
        endpoint=endpoint,
        model="gemma-3-1b-it",
        api_key=None,
        timeout_seconds=5.0,
    )


def _launch_config(
    *,
    model_path: str | None = "/models/gemma.gguf",
    hf_repo: str | None = None,
    extra_args: tuple[str, ...] = (),
) -> LlamaServerLaunchConfig:
    return LlamaServerLaunchConfig(
        server_bin="/usr/local/bin/llama-server",
        model_path=model_path,
        hf_repo=hf_repo,
        extra_args=extra_args,
        launch_timeout_seconds=10.0,
        poll_interval_seconds=0.1,
    )


class BuildLlamaServerCommandTest(unittest.TestCase):
    """Pin the argv shape consumed by ``llama-server``. Drift here
    would either bind the wrong port or load the wrong model on
    autostart.
    """

    def test_includes_model_and_port(self) -> None:
        config = _llm_config("http://127.0.0.1:18099")
        launch = _launch_config(
            model_path="/models/gemma.gguf",
            extra_args=("--ctx-size", "8192"),
        )
        argv = build_llama_server_command(config, launch)

        self.assertEqual(argv[0], "/usr/local/bin/llama-server")
        self.assertIn("-m", argv)
        self.assertEqual(argv[argv.index("-m") + 1], "/models/gemma.gguf")
        self.assertIn("--port", argv)
        self.assertEqual(argv[argv.index("--port") + 1], "18099")
        # Extra args are appended after the synthesized flags.
        self.assertEqual(argv[-2:], ["--ctx-size", "8192"])

    def test_uses_hf_repo_when_model_path_absent(self) -> None:
        # The launcher accepts an HF repo as an alternative to a local
        # model path; pin so a future refactor doesn't drop the branch.
        config = _llm_config("http://127.0.0.1:18099")
        launch = _launch_config(model_path=None, hf_repo="google/gemma-3-1b-it")
        argv = build_llama_server_command(config, launch)
        self.assertIn("-hf", argv)
        self.assertEqual(argv[argv.index("-hf") + 1], "google/gemma-3-1b-it")
        self.assertNotIn("-m", argv)

    def test_rejects_non_local_endpoint(self) -> None:
        # Autostart is only safe for endpoints we own; remote URLs must
        # never reach :func:`subprocess.Popen`.
        config = _llm_config("https://api.example.com/v1")
        launch = _launch_config()
        with self.assertRaises(ProviderInvocationError):
            build_llama_server_command(config, launch)

    def test_requires_model_path_or_hf_repo(self) -> None:
        config = _llm_config("http://127.0.0.1:18099")
        launch = _launch_config(model_path=None, hf_repo=None)
        with self.assertRaises(ProviderInvocationError):
            build_llama_server_command(config, launch)


class EnsureLlmEndpointTest(unittest.TestCase):
    """Pin the early-exit behavior of the lifecycle entry point so the
    worker never spawns ``llama-server`` when the LLM provider is
    unconfigured or already reachable.
    """

    def test_returns_unreachable_when_config_is_none(self) -> None:
        # No subprocess must be spawned and no error string must be
        # synthesized — the caller decides how to render "not
        # configured" upstream.
        called: list[object] = []

        def fake_popen(*args: object, **kwargs: object) -> object:
            called.append((args, kwargs))
            raise AssertionError("Popen must not be invoked when config is None")

        with patch.object(subprocess, "Popen", fake_popen):
            reachable, error = ensure_llm_endpoint(None)

        self.assertFalse(reachable)
        self.assertIsNone(error)
        self.assertEqual(called, [])

    def test_skips_autostart_when_launch_config_disabled(self) -> None:
        # Endpoint is unreachable AND ``VOICELAYER_LLM_AUTO_START`` is
        # unset — :func:`load_llama_server_launch_config` returns None
        # and the helper must not invoke :func:`autostart_llama_server`.
        config = _llm_config("http://127.0.0.1:18999")

        def fake_probe(_: OpenAICompatibleConfig) -> tuple[bool, str | None]:
            return False, "connection refused"

        def fake_autostart(*args: object, **kwargs: object) -> object:
            raise AssertionError("autostart must not run when launch config is disabled")

        with (
            patch.object(llama_autostart, "probe_llm_endpoint", fake_probe),
            patch.object(llama_autostart, "autostart_llama_server", fake_autostart),
        ):
            reachable, error = ensure_llm_endpoint(config, environ={})

        self.assertFalse(reachable)
        self.assertEqual(error, "connection refused")


class WaitForLlmEndpointTest(unittest.TestCase):
    """Pin the polling loop's timeout contract so the worker never
    blocks indefinitely when the launched server fails to come up.
    """

    def test_returns_false_on_immediate_timeout(self) -> None:
        # ``timeout_seconds=0`` puts the deadline at or before "now", so
        # the loop must exit without spinning. The probe is replaced so
        # the test does not depend on local TCP behavior and asserts
        # only the observable contract: a False return without blocking
        # on real ``time.sleep``.
        config = _llm_config("http://127.0.0.1:18999")

        def fake_probe(_: OpenAICompatibleConfig) -> tuple[bool, str | None]:
            return False, "connection refused"

        sleep_calls: list[float] = []

        def fake_sleep(seconds: float) -> None:
            sleep_calls.append(seconds)

        with (
            patch.object(llama_autostart, "probe_llm_endpoint", fake_probe),
            patch.object(llama_autostart.time, "sleep", fake_sleep),
        ):
            reachable, _error = wait_for_llm_endpoint(
                config,
                timeout_seconds=0.0,
                poll_interval_seconds=0.1,
            )

        self.assertFalse(reachable)
        # A 0s budget must not park on real ``time.sleep`` waiting for
        # another retry tick — at most one fast probe is allowed.
        self.assertEqual(sleep_calls, [])

    def test_returns_true_when_probe_succeeds_immediately(self) -> None:
        config = _llm_config("http://127.0.0.1:18999")

        def fake_probe(_: OpenAICompatibleConfig) -> tuple[bool, str | None]:
            return True, None

        with patch.object(llama_autostart, "probe_llm_endpoint", fake_probe):
            reachable, error = wait_for_llm_endpoint(
                config,
                timeout_seconds=5.0,
                poll_interval_seconds=0.1,
            )

        self.assertTrue(reachable)
        self.assertIsNone(error)


if __name__ == "__main__":
    unittest.main()
