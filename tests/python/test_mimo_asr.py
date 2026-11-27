"""Tests for the optional MiMo-V2.5-ASR provider and pipeline integration."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import wave
from unittest.mock import MagicMock, patch

from voicelayer_worker import pipeline, worker
from voicelayer_worker.config import MimoAsrConfig, configure, reset_configuration_for_tests
from voicelayer_worker.errors import ProviderInvocationError
from voicelayer_worker.providers import mimo_asr


def _write_wav(path: pathlib.Path, duration_secs: float = 1.0, rate: int = 16000) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(rate)
        handle.writeframes(b"\x00\x00" * int(duration_secs * rate))
    return path


class MimoAsrProviderTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-mimo-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)
        self.model_dir = self.root / "model"
        self.tok_dir = self.root / "tokenizer"
        self.model_dir.mkdir()
        self.tok_dir.mkdir()

    def _config(self, **overrides: object) -> MimoAsrConfig:
        base = {
            "model_path": str(self.model_dir),
            "tokenizer_path": str(self.tok_dir),
            "repo_path": None,
            "device": "cuda:0",
            "audio_tag": None,
            "timeout_seconds": 300.0,
            "long_audio_split_seconds": 0.0,
            "extra_args": (),
        }
        base.update(overrides)
        return MimoAsrConfig(**base)  # type: ignore[arg-type]

    def test_validate_provider_and_audio_tag_resolution(self) -> None:
        self.assertFalse(mimo_asr.validate_mimo_provider(None)[0])
        self.assertFalse(
            mimo_asr.validate_mimo_provider(self._config(model_path=str(self.root / "missing")))[0]
        )
        self.assertFalse(
            mimo_asr.validate_mimo_provider(
                self._config(tokenizer_path=str(self.root / "missing"))
            )[0]
        )
        self.assertEqual(mimo_asr.validate_mimo_provider(self._config()), (True, None))

        cfg = self._config(audio_tag="<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag("zh-CN", cfg), "<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag("en-US", cfg), "<english>")
        self.assertEqual(mimo_asr._resolve_audio_tag("auto", cfg), "<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag("fr", cfg), "<chinese>")

    def test_transcribe_validates_inputs_splits_long_audio_and_cleans_chunks(self) -> None:
        cfg = self._config(long_audio_split_seconds=1.0)
        with self.assertRaises(ProviderInvocationError):
            mimo_asr.transcribe_with_mimo({"audio_file": str(self.root / "nope.wav")}, cfg)

        audio = _write_wav(self.root / "long.wav", duration_secs=2.5)
        with self.assertRaises(ProviderInvocationError):
            mimo_asr.transcribe_with_mimo(
                {"audio_file": str(audio), "translate_to_english": True}, cfg
            )

        fake_model = MagicMock()
        fake_model.asr_sft.side_effect = ["part one", "part two", "part three"]
        runtime_dir = self.root / "runtime"

        with (
            patch.object(mimo_asr, "_load_mimo_model", return_value=fake_model),
            patch.object(mimo_asr, "_load_wav_as_tensor", return_value=(object(), 16000)),
        ):
            result = mimo_asr.transcribe_with_mimo(
                {"audio_file": str(audio), "language": "zh"},
                cfg,
                runtime_dir=runtime_dir,
            )

        self.assertEqual(result["text"], "part one part two part three")
        self.assertEqual(result["detected_language"], "zh")
        self.assertEqual(fake_model.asr_sft.call_count, 3)
        self.assertEqual(list((runtime_dir / "mimo").glob("*.wav")), [])

    def test_pipeline_runs_single_vad_prepass_before_mimo_cold_load(self) -> None:
        audio = _write_wav(self.root / "silent.wav")
        trimmed = self.root / "silent.vad-empty.wav"
        trimmed.write_bytes(b"")
        configure(
            {
                "mimo_asr": {
                    "model_path": str(self.model_dir),
                    "tokenizer_path": str(self.tok_dir),
                },
                "vad": {"enabled": True, "model_path": "/tmp/silero.onnx"},
            }
        )

        with (
            patch.object(pipeline, "apply_vad_prepass", return_value=(str(trimmed), [])) as vad_fn,
            patch.object(mimo_asr, "_load_mimo_model") as load_model,
        ):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio), "provider_id": "mimo_v2_5_asr"},
                }
            )

        assert resp is not None
        self.assertEqual(resp["result"]["text"], "")
        vad_fn.assert_called_once()
        load_model.assert_not_called()
        self.assertFalse(trimmed.exists())
