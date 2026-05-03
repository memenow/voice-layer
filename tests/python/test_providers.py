from __future__ import annotations

import os
import pathlib
import sys
import tempfile
import unittest

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.providers import (  # noqa: E402
    collapse_nonspeech_transcript,
    provider_runtime_dir,
    reclaim_stale_lock,
    supported_providers,
)


class ReclaimStaleLockTest(unittest.TestCase):
    """Covers :func:`reclaim_stale_lock` which both autostart paths rely on."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-lock-test-"))
        self.lock_path = self.tmp_dir / "endpoint.lock"

    def test_returns_false_when_lock_absent(self) -> None:
        assert not self.lock_path.exists()
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))

    def test_leaves_lock_intact_when_owner_still_runs(self) -> None:
        # The current test process is guaranteed to be alive; write its PID
        # and claim its own argv[0] as the expected binary. argv[0] is
        # usually /usr/bin/python or similar; reading /proc/self/cmdline
        # tells us what the file-name comparison should match.
        my_pid = os.getpid()
        cmdline = pathlib.Path(f"/proc/{my_pid}/cmdline").read_bytes()
        argv0 = cmdline.split(b"\x00", 1)[0].decode("utf-8")
        binary_name = pathlib.Path(argv0).name
        self.lock_path.write_text(str(my_pid), encoding="utf-8")

        self.assertFalse(reclaim_stale_lock(self.lock_path, binary_name))
        self.assertTrue(self.lock_path.exists())

    def test_removes_lock_when_owner_pid_is_long_gone(self) -> None:
        # PID 0x7FFFFFFE is almost certainly not a live process on Linux;
        # at minimum /proc/<pid>/cmdline is unreadable, so the helper
        # should treat the lock as stale and delete it.
        self.lock_path.write_text("2147483646", encoding="utf-8")

        self.assertTrue(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertFalse(self.lock_path.exists())

    def test_removes_lock_when_pid_runs_different_binary(self) -> None:
        # Pair our own PID with a binary name we're definitely not running.
        my_pid = os.getpid()
        self.lock_path.write_text(str(my_pid), encoding="utf-8")

        self.assertTrue(
            reclaim_stale_lock(self.lock_path, "definitely-not-our-binary-42"),
        )
        self.assertFalse(self.lock_path.exists())

    def test_keeps_lock_when_pid_is_unparseable(self) -> None:
        # Simulates a partial write by the owning process: lock file
        # exists but PID hasn't landed yet. The helper should err on the
        # safe side and leave the lock intact.
        self.lock_path.write_text("", encoding="utf-8")
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertTrue(self.lock_path.exists())

        self.lock_path.write_text("not-a-pid", encoding="utf-8")
        self.assertFalse(reclaim_stale_lock(self.lock_path, "llama-server"))
        self.assertTrue(self.lock_path.exists())


class CollapseNonSpeechTranscriptTest(unittest.TestCase):
    """Covers :func:`collapse_nonspeech_transcript` shared by both whisper adapters."""

    def test_collapses_bracket_blank_audio(self) -> None:
        self.assertEqual(collapse_nonspeech_transcript("[BLANK_AUDIO]"), "")

    def test_collapses_bracket_music(self) -> None:
        self.assertEqual(collapse_nonspeech_transcript("[MUSIC]"), "")

    def test_collapses_parenthetical_foreign_language(self) -> None:
        self.assertEqual(
            collapse_nonspeech_transcript("(speaking in foreign language)"),
            "",
        )
        self.assertEqual(
            collapse_nonspeech_transcript("(speaks in foreign language)"),
            "",
        )

    def test_collapses_parenthetical_inaudible(self) -> None:
        self.assertEqual(collapse_nonspeech_transcript("(inaudible)"), "")

    def test_is_case_insensitive_and_ignores_inner_whitespace(self) -> None:
        self.assertEqual(collapse_nonspeech_transcript("[ music ]"), "")
        self.assertEqual(collapse_nonspeech_transcript("[  Music  Playing  ]"), "")
        self.assertEqual(collapse_nonspeech_transcript("  (INAUDIBLE)  "), "")

    def test_collapses_multiline_repeated_annotations(self) -> None:
        # whisper on a music-only recording often emits the same token
        # repeatedly — collapse the whole block.
        self.assertEqual(
            collapse_nonspeech_transcript("[MUSIC]\n[MUSIC]\n[MUSIC]\n"),
            "",
        )
        self.assertEqual(
            collapse_nonspeech_transcript("[BLANK_AUDIO]\n(inaudible)\n(music playing)"),
            "",
        )

    def test_preserves_mixed_content(self) -> None:
        # A transcript that contains real speech alongside a decorative
        # line must be returned verbatim — silently dropping the real
        # speech would be worse than leaving the annotation in place.
        mixed = "[MUSIC]\nI love this song."
        self.assertEqual(collapse_nonspeech_transcript(mixed), mixed)

    def test_preserves_real_speech_that_mentions_music(self) -> None:
        # Spoken "music" without brackets is real content — must not be
        # touched by the filter.
        spoken = "We talked about music and art."
        self.assertEqual(collapse_nonspeech_transcript(spoken), spoken)

    def test_collapses_novel_bracket_and_paren_annotations(self) -> None:
        # The filter is structural: anything standing alone in `[...]`
        # or `(...)` is treated as a whisper decorative annotation,
        # because whisper reserves brackets and parens for meta content
        # rather than transcribed speech. This catches ad-hoc
        # annotations (`(dramatic music)`, `(phone rings)`,
        # `[DOG_BARKING]`) that an allowlist would miss.
        for annotation in (
            "(dramatic music)",
            "(soft music)",
            "(phone rings)",
            "(door creaks)",
            "[DOG_BARKING]",
            "[ENGINE]",
        ):
            with self.subTest(annotation=annotation):
                self.assertEqual(collapse_nonspeech_transcript(annotation), "")

    def test_returns_empty_for_pure_whitespace_input(self) -> None:
        self.assertEqual(collapse_nonspeech_transcript(""), "")
        self.assertEqual(collapse_nonspeech_transcript("   "), "")
        self.assertEqual(collapse_nonspeech_transcript("\n\n"), "")


class ProviderRuntimeDirTest(unittest.TestCase):
    """Covers :func:`provider_runtime_dir`, which both autostart adapters
    use to anchor lock and pid files.

    The Rust daemon exposes the same `XDG_RUNTIME_DIR`-driven contract
    via :func:`voicelayerd::default_socket_path`; both sides must land
    under `$XDG_RUNTIME_DIR/voicelayer/...` when the variable is set.
    A regression that ignored the env var would scatter Python state
    files into `/tmp` while the Rust socket lived under
    `/run/user/<uid>`, leaving stale locks behind across reboots.
    """

    def setUp(self) -> None:
        super().setUp()
        # Each test owns an isolated tempdir to play the role of
        # `XDG_RUNTIME_DIR`. The mapping passed to the helper is a
        # plain dict so we never mutate `os.environ` and dodge the
        # cross-test serialisation problem that ENV_LOCK solves on the
        # Rust side.
        self.tmp_root = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-runtime-"))

    def test_uses_xdg_runtime_dir_when_set(self) -> None:
        environ = {"XDG_RUNTIME_DIR": str(self.tmp_root)}
        path = provider_runtime_dir(environ)
        self.assertEqual(path, self.tmp_root / "voicelayer" / "providers")
        self.assertTrue(path.is_dir())

    def test_falls_back_to_system_temp_when_xdg_runtime_dir_unset(self) -> None:
        # The helper's `source = environ or os.environ` short-circuits
        # an empty mapping to `os.environ`, so we have to pass a
        # non-empty mapping that simply lacks `XDG_RUNTIME_DIR` to
        # exercise the fallback. The dummy key is ignored by the
        # helper but keeps the truthiness check happy.
        environ = {"_test_marker": "1"}
        path = provider_runtime_dir(environ)
        self.assertEqual(path, pathlib.Path(tempfile.gettempdir()) / "voicelayer" / "providers")

    def test_creates_runtime_dir_when_missing(self) -> None:
        # Brand-new XDG dir: helper must mkdir the nested structure
        # without raising. A regression that dropped `mkdir(parents=True)`
        # would crash autostart on first boot before any state file
        # ever lands.
        environ = {"XDG_RUNTIME_DIR": str(self.tmp_root / "fresh")}
        self.assertFalse((self.tmp_root / "fresh").exists())
        path = provider_runtime_dir(environ)
        self.assertTrue(path.is_dir())


class SupportedProvidersTest(unittest.TestCase):
    """Covers :func:`supported_providers`, which the worker emits in
    response to JSON-RPC `list_providers`. Pinning the catalog shape
    keeps the worker side aligned with the Rust default catalog and
    catches drift in either direction (extra entry, dropped entry,
    misclassified `kind`).
    """

    def test_includes_three_asr_providers_and_an_llm_when_no_endpoint_configured(
        self,
    ) -> None:
        # Non-empty mapping with no VoiceLayer keys. `supported_providers`
        # eventually calls helpers that short-circuit `environ or
        # os.environ`, so an empty mapping would leak the developer's
        # ambient env into the test. The dummy marker keeps the
        # mapping truthy without supplying any voicelayer config.
        catalog = supported_providers({"_test_marker": "1"})
        ids = {entry["id"] for entry in catalog}
        self.assertEqual(
            ids,
            {"whisper_cpp", "voxtral_realtime", "mimo_v2_5_asr", "gemma_4_local"},
        )
        kinds = {entry["id"]: entry["kind"] for entry in catalog}
        self.assertEqual(
            kinds,
            {
                "whisper_cpp": "asr",
                "voxtral_realtime": "asr",
                "mimo_v2_5_asr": "asr",
                "gemma_4_local": "llm",
            },
        )

    def test_mimo_descriptor_is_optional_and_experimental(self) -> None:
        # MiMo-V2.5-ASR ships as opt-in: the catalog must advertise it
        # so `vl providers` can list it, but the descriptor must keep
        # `default_enabled=false` and `experimental=true` so callers
        # know the whisper chain remains the production default.
        catalog = supported_providers({"_test_marker": "1"})
        mimo = next(entry for entry in catalog if entry["id"] == "mimo_v2_5_asr")
        self.assertEqual(mimo["kind"], "asr")
        self.assertEqual(mimo["license"], "MIT")
        self.assertFalse(mimo["default_enabled"])
        self.assertTrue(mimo["experimental"])
        # Without env vars set, the descriptor reports the generic
        # stdio_worker transport so operators can tell at a glance
        # whether their environment has the model paths configured.
        self.assertEqual(mimo["transport"], "stdio_worker")

    def test_mimo_transport_flips_to_in_process_when_paths_configured(
        self,
        tmp_path: pathlib.Path = pathlib.Path("."),
    ) -> None:
        # Operator wires both VOICELAYER_MIMO_MODEL_PATH and
        # VOICELAYER_MIMO_TOKENIZER_PATH to existing directories. The
        # descriptor's `transport` flips to `in_process_torch` so the
        # operator can confirm at a glance that the worker would route
        # to the in-process MimoAudio wrapper rather than the legacy
        # stdio bridge.
        with tempfile.TemporaryDirectory(prefix="voicelayer-mimo-test-") as tmp:
            model_dir = pathlib.Path(tmp) / "weights"
            tokenizer_dir = pathlib.Path(tmp) / "tokenizer"
            model_dir.mkdir()
            tokenizer_dir.mkdir()
            catalog = supported_providers(
                {
                    "VOICELAYER_MIMO_MODEL_PATH": str(model_dir),
                    "VOICELAYER_MIMO_TOKENIZER_PATH": str(tokenizer_dir),
                }
            )
            mimo = next(entry for entry in catalog if entry["id"] == "mimo_v2_5_asr")
            self.assertEqual(mimo["transport"], "in_process_torch")

    def test_replaces_default_llm_with_configured_descriptor_when_endpoint_set(
        self,
    ) -> None:
        # When the operator wires `VOICELAYER_LLM_ENDPOINT` /
        # `VOICELAYER_LLM_MODEL`, `configured_llm_descriptor` returns
        # a descriptor; the catalog substitutes it for `gemma_4_local`
        # so `vl providers` shows the live endpoint instead of a
        # confusing static placeholder.
        environ = {
            "VOICELAYER_LLM_ENDPOINT": "http://localhost:8080/v1",
            "VOICELAYER_LLM_MODEL": "gpt-oss-20b",
        }
        catalog = supported_providers(environ)
        ids = [entry["id"] for entry in catalog]
        self.assertNotIn("gemma_4_local", ids)
        # The configured descriptor uses a synthesized id but always
        # has kind=llm; the asr pair stays put.
        kinds = {entry["kind"] for entry in catalog}
        self.assertEqual(kinds, {"asr", "llm"})

    def test_whisper_transport_flips_to_whisper_cli_when_whisper_config_present(
        self,
    ) -> None:
        # `transport` reflects whether the worker would dispatch through
        # whisper-cli (configured) or fall back to the legacy stdio
        # bridge. `vl providers` surfaces this so the operator can tell
        # at a glance whether the binary was wired up.
        environ = {
            "VOICELAYER_WHISPER_BIN": "/usr/local/bin/whisper-cli",
            "VOICELAYER_WHISPER_MODEL_PATH": "/var/cache/voicelayer/ggml-base.en.bin",
        }
        catalog = supported_providers(environ)
        whisper = next(entry for entry in catalog if entry["id"] == "whisper_cpp")
        self.assertEqual(whisper["transport"], "whisper_cli")

        catalog = supported_providers({"_test_marker": "1"})
        whisper = next(entry for entry in catalog if entry["id"] == "whisper_cpp")
        self.assertEqual(whisper["transport"], "stdio_worker")


if __name__ == "__main__":
    unittest.main()
