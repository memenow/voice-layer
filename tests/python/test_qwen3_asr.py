"""Tests for the optional Qwen3-ASR-1.7B provider and pipeline integration."""

from __future__ import annotations

import pathlib
import tempfile
import types
import unittest
import wave
from unittest.mock import MagicMock, patch

from voicelayer_worker import pipeline, worker
from voicelayer_worker.config import Qwen3AsrConfig, configure, reset_configuration_for_tests
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.providers import qwen3_asr


def _write_wav(path: pathlib.Path, duration_secs: float = 1.0, rate: int = 16000) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(duration_secs * rate))
    return path


class Qwen3AsrProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-qwen3-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.model_dir = self.root / "qwen3-model"
        self.model_dir.mkdir()

    def _config(self, **overrides: object) -> Qwen3AsrConfig:
        base = {
            "model_path": str(self.model_dir),
            "device": "cuda:0",
            "torch_dtype": "bfloat16",
            "timeout_seconds": 300.0,
            "long_audio_split_seconds": 0.0,
            "extra_args": (),
        }
        base.update(overrides)
        return Qwen3AsrConfig(**base)  # type: ignore[arg-type]

    def test_validate_provider_and_language_hints(self) -> None:
        self.assertFalse(qwen3_asr.validate_qwen3_asr_provider(None)[0])
        self.assertFalse(qwen3_asr.validate_qwen3_asr_provider(self._config(torch_dtype="int8"))[0])
        self.assertFalse(
            qwen3_asr.validate_qwen3_asr_provider(
                self._config(model_path=str(self.root / "missing"))
            )[0]
        )
        self.assertEqual(qwen3_asr.validate_qwen3_asr_provider(self._config()), (True, None))

        self.assertEqual(qwen3_asr._resolve_language_hint("zh-CN"), "Chinese")
        self.assertEqual(qwen3_asr._resolve_language_hint("yue"), "Cantonese")
        self.assertEqual(qwen3_asr._resolve_language_hint("en"), "English")
        self.assertIsNone(qwen3_asr._resolve_language_hint("auto"))
        self.assertIsNone(qwen3_asr._resolve_language_hint("unknown-lang"))

    def test_transcribe_splits_segments_maps_language_and_cleans_chunks(self) -> None:
        cfg = self._config(long_audio_split_seconds=1.0)
        audio = _write_wav(self.root / "clip.wav", duration_secs=2.2)

        with self.assertRaises(ProviderInvocationError):
            qwen3_asr.transcribe_with_qwen3_asr(
                {"audio_file": str(audio), "translate_to_english": True}, cfg
            )

        fake_model = MagicMock()
        fake_model.transcribe.side_effect = [
            [types.SimpleNamespace(text="你好", language="Chinese")],
            [types.SimpleNamespace(text="世界", language="Chinese")],
            [types.SimpleNamespace(text="!", language="Chinese")],
        ]
        runtime_dir = self.root / "runtime"

        with patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=fake_model):
            result = qwen3_asr.transcribe_with_qwen3_asr(
                {"audio_file": str(audio)},
                cfg,
                runtime_dir=runtime_dir,
            )

        self.assertEqual(result["text"], "你好 世界 !")
        self.assertEqual(result["detected_language"], "zh")
        self.assertEqual(fake_model.transcribe.call_count, 3)
        self.assertEqual(list((runtime_dir / "qwen3_asr").glob("*.wav")), [])

    def test_pipeline_runs_single_vad_prepass_before_qwen3_cold_load(self) -> None:
        audio = _write_wav(self.root / "silent.wav")
        trimmed = self.root / "silent.vad-empty.wav"
        trimmed.write_bytes(b"")
        configure(
            {
                "qwen3_asr": {"model_path": str(self.model_dir)},
                "vad": {"enabled": True, "model_path": "/tmp/silero.onnx"},
            }
        )

        with (
            patch.object(pipeline, "apply_vad_prepass", return_value=(str(trimmed), [])) as vad_fn,
            patch.object(qwen3_asr, "_load_qwen3_asr_model") as load_model,
        ):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio), "provider_id": "qwen3_asr_1_7b"},
                }
            )

        assert resp is not None
        self.assertEqual(resp["result"]["text"], "")
        vad_fn.assert_called_once()
        load_model.assert_not_called()
        self.assertFalse(trimmed.exists())
