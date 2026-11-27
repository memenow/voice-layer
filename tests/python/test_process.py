"""Tests for process lock reclamation, runtime directories, transcripts, and registry."""

from __future__ import annotations

import os
import pathlib
import tempfile
import unittest

import psutil

from voicelayer_worker.config import WorkerConfig, reset_configuration_for_tests
from voicelayer_worker.process import provider_runtime_dir, reclaim_stale_lock, wait_for_probe
from voicelayer_worker.providers.registry import supported_providers
from voicelayer_worker.transcript import collapse_nonspeech_transcript


class ProcessHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-process-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_reclaim_stale_lock_handles_live_dead_mismatch_and_empty(self) -> None:
        lock_path = self.root / "endpoint.lock"
        self.assertFalse(reclaim_stale_lock(lock_path, "llama-server"))

        my_pid = os.getpid()
        my_binary = pathlib.Path(psutil.Process(my_pid).cmdline()[0]).name
        lock_path.write_text(str(my_pid), encoding="utf-8")
        self.assertFalse(reclaim_stale_lock(lock_path, my_binary))
        self.assertTrue(lock_path.exists())

        self.assertTrue(reclaim_stale_lock(lock_path, "definitely-not-our-binary"))
        self.assertFalse(lock_path.exists())

        lock_path.write_text("2147483646", encoding="utf-8")
        self.assertTrue(reclaim_stale_lock(lock_path, "llama-server"))
        self.assertFalse(lock_path.exists())

        for raw in ("", "not-a-pid"):
            lock_path.write_text(raw, encoding="utf-8")
            self.assertFalse(reclaim_stale_lock(lock_path, "llama-server"))
            self.assertTrue(lock_path.exists())

    def test_provider_runtime_dir_honors_xdg_and_tempdir_fallback(self) -> None:
        xdg = self.root / "fresh"
        path = provider_runtime_dir({"XDG_RUNTIME_DIR": str(xdg)})
        self.assertEqual(path, xdg / "voicelayer" / "providers")
        self.assertTrue(path.is_dir())

        fallback = provider_runtime_dir({"_marker": "1"})
        self.assertEqual(fallback, pathlib.Path(tempfile.gettempdir()) / "voicelayer" / "providers")

    def test_wait_for_probe_returns_immediately_on_success_or_timeout(self) -> None:
        ok, err = wait_for_probe(
            lambda: (True, None), timeout_seconds=1.0, poll_interval_seconds=0.01
        )
        self.assertTrue(ok)
        self.assertIsNone(err)

        ok, err = wait_for_probe(
            lambda: (False, "connection refused"),
            timeout_seconds=0.0,
            poll_interval_seconds=0.01,
        )
        self.assertFalse(ok)
        self.assertIsNone(err)


class TranscriptCleanupTest(unittest.TestCase):
    def test_collapses_only_decorative_annotations(self) -> None:
        for empty_case in (
            "",
            "   \n\n",
            "[BLANK_AUDIO]",
            "[MUSIC]\n[MUSIC]",
            "  (INAUDIBLE)  ",
            "(speaking in foreign language)",
            "(dramatic music)",
            "[DOG_BARKING]",
        ):
            self.assertEqual(collapse_nonspeech_transcript(empty_case), "")

        mixed = "[MUSIC]\nI love this song."
        self.assertEqual(collapse_nonspeech_transcript(mixed), mixed)
        self.assertEqual(
            collapse_nonspeech_transcript("We talked about music."),
            "We talked about music.",
        )


class SupportedProvidersRegistryTest(unittest.TestCase):
    REQUIRED_KEYS = frozenset(
        {"id", "kind", "transport", "local", "default_enabled", "experimental", "license"}
    )

    def test_default_catalog_shape_and_invariants(self) -> None:
        catalog = supported_providers(WorkerConfig())
        ids = [entry["id"] for entry in catalog]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(
            set(ids),
            {
                "whisper_cpp",
                "voxtral_realtime",
                "mimo_v2_5_asr",
                "qwen3_asr_1_7b",
                "gemma_4_local",
            },
        )
        for entry in catalog:
            self.assertFalse(self.REQUIRED_KEYS - entry.keys())
            self.assertIn(entry["kind"], {"asr", "llm"})
        mimo = next(entry for entry in catalog if entry["id"] == "mimo_v2_5_asr")
        self.assertEqual(mimo["transport"], "stdio_worker")
        self.assertEqual(mimo["license"], "MIT")
        self.assertFalse(mimo["default_enabled"])
        self.assertTrue(mimo["experimental"])

        qwen3 = next(entry for entry in catalog if entry["id"] == "qwen3_asr_1_7b")
        self.assertEqual(qwen3["transport"], "stdio_worker")
        self.assertEqual(qwen3["license"], "Apache-2.0")
        self.assertFalse(qwen3["default_enabled"])
        self.assertTrue(qwen3["experimental"])

    def test_configured_providers_update_transports_and_replace_default_llm(self) -> None:
        cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_WHISPER_MODEL_PATH": "/models/base.bin",
                "VOICELAYER_MIMO_MODEL_PATH": "/models/mimo",
                "VOICELAYER_MIMO_TOKENIZER_PATH": "/models/mimo-tok",
                "VOICELAYER_QWEN3_ASR_MODEL_PATH": "/models/qwen3",
                "VOICELAYER_LLM_ENDPOINT": "http://127.0.0.1:8080/v1",
                "VOICELAYER_LLM_MODEL": "gpt-oss-20b",
            }
        )
        catalog = supported_providers(cfg)
        by_id = {entry["id"]: entry for entry in catalog}
        self.assertEqual(by_id["whisper_cpp"]["transport"], "whisper_cli")
        self.assertEqual(by_id["mimo_v2_5_asr"]["transport"], "in_process_torch")
        self.assertEqual(by_id["qwen3_asr_1_7b"]["transport"], "in_process_torch")
        self.assertNotIn("gemma_4_local", by_id)
        self.assertEqual(by_id["gpt-oss-20b"]["kind"], "llm")
        self.assertEqual(sum(1 for entry in catalog if entry["kind"] == "llm"), 1)
