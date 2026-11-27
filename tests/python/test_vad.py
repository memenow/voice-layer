"""Tests for Silero VAD region extraction, speech probing, and single-pass pre-pass."""

from __future__ import annotations

import importlib.util
import pathlib
import tempfile
import unittest
import wave
from unittest.mock import patch

from voicelayer_worker import pipeline, vad, worker
from voicelayer_worker.config import WhisperVadConfig, configure, reset_configuration_for_tests
from voicelayer_worker.errors import (
    INVALID_REQUEST_CODE,
    PROVIDER_REQUEST_FAILED_CODE,
    PROVIDER_UNAVAILABLE_CODE,
    ProviderInvocationError,
)


def _vad_config(
    *,
    threshold: float = 0.5,
    min_speech_ms: int = 250,
    min_silence_ms: int = 100,
    speech_pad_ms: int = 30,
    max_segment_secs: float = 30.0,
    sample_rate: int = 16000,
    model_path: str = "/tmp/silero-vad.onnx",
) -> WhisperVadConfig:
    return WhisperVadConfig(
        model_path=model_path,
        threshold=threshold,
        min_speech_ms=min_speech_ms,
        min_silence_ms=min_silence_ms,
        speech_pad_ms=speech_pad_ms,
        max_segment_secs=max_segment_secs,
        sample_rate=sample_rate,
    )


def _write_silent_wav(path: pathlib.Path, duration_sec: float = 1.0) -> pathlib.Path:
    sample_rate = 16000
    frames = int(sample_rate * duration_sec)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(b"\x00\x00" * frames)
    return path


class _FakeSamples:
    def __init__(self, n: int) -> None:
        self.size = n
        self.shape = (n,)


class VadRegionLogicTest(unittest.TestCase):
    FRAME_SEC = 0.032

    def test_window_sizes(self) -> None:
        self.assertEqual(vad._window_size_for(16000), 512)
        self.assertEqual(vad._window_size_for(8000), 256)
        with self.assertRaises(ProviderInvocationError):
            vad._window_size_for(44100)

    def test_frames_to_regions_merging_padding_hysteresis_and_split(self) -> None:
        self.assertEqual(vad._frames_to_regions([], _vad_config(), self.FRAME_SEC), [])
        self.assertEqual(vad._frames_to_regions([0.1] * 100, _vad_config(), self.FRAME_SEC), [])

        # Short speech dropped.
        self.assertEqual(
            vad._frames_to_regions(
                [0.1] * 10 + [0.9] * 3 + [0.1] * 10,
                _vad_config(speech_pad_ms=0),
                self.FRAME_SEC,
            ),
            [],
        )

        # Small gap merges; large gap stays split.
        self.assertEqual(
            vad._frames_to_regions(
                [0.9] * 10 + [0.1] * 2 + [0.9] * 10,
                _vad_config(min_silence_ms=200, min_speech_ms=100, speech_pad_ms=0),
                self.FRAME_SEC,
            ),
            [(0, 22)],
        )
        self.assertEqual(
            vad._frames_to_regions(
                [0.9] * 10 + [0.1] * 20 + [0.9] * 10,
                _vad_config(min_silence_ms=100, min_speech_ms=100, speech_pad_ms=0),
                self.FRAME_SEC,
            ),
            [(0, 10), (30, 40)],
        )

        # Hysteresis holds borderline dip (0.40 >= 0.35) and exits on low-threshold silence.
        self.assertEqual(
            vad._frames_to_regions(
                [0.9] * 10 + [0.40] * 5 + [0.9] * 10,
                _vad_config(threshold=0.5, min_silence_ms=10, min_speech_ms=100, speech_pad_ms=0),
                self.FRAME_SEC,
            ),
            [(0, 25)],
        )
        self.assertEqual(
            vad._frames_to_regions(
                [0.5] * 10 + [0.02] * 10 + [0.5] * 10,
                _vad_config(threshold=0.1, min_silence_ms=10, min_speech_ms=100, speech_pad_ms=0),
                self.FRAME_SEC,
            ),
            [(0, 10), (20, 30)],
        )

        # Hard split by max_segment_secs.
        self.assertEqual(
            vad._frames_to_regions(
                [0.9] * 100,
                _vad_config(
                    min_silence_ms=10,
                    min_speech_ms=50,
                    speech_pad_ms=0,
                    max_segment_secs=32 * self.FRAME_SEC,
                ),
                self.FRAME_SEC,
            ),
            [(0, 32), (32, 64), (64, 96), (96, 100)],
        )


class ProbeAudioAndDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-probe-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_probe_audio_file_empty_speech_and_clamping(self) -> None:
        with self.assertRaises(ProviderInvocationError):
            vad.probe_audio_file(str(self.root / "missing.wav"), _vad_config())

        audio = self.root / "clip.wav"
        audio.touch()

        with patch.object(vad, "_load_wav_as_float32_mono", return_value=(_FakeSamples(0), 16000)):
            empty = vad.probe_audio_file(str(audio), _vad_config())
        self.assertFalse(empty["has_speech"])
        self.assertEqual(empty["speech_ratio"], 0.0)

        with (
            patch.object(
                vad, "_load_wav_as_float32_mono", return_value=(_FakeSamples(4000), 16000)
            ),
            patch.object(vad, "_load_vad_session", return_value=object()),
            patch.object(vad, "_speech_probabilities", return_value=[0.9] * 8),
            patch.object(vad, "_frames_to_regions", return_value=[(0, 100)]),
        ):
            clamped = vad.probe_audio_file(str(audio), _vad_config(sample_rate=16000))
        self.assertTrue(clamped["has_speech"])
        self.assertEqual(clamped["speech_ratio"], 1.0)

    def test_segment_probe_rpc_dispatch(self) -> None:
        configure({})
        resp = worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "segment_probe",
                "params": {"audio_file": "/tmp/a.wav"},
            }
        )
        assert resp is not None
        self.assertEqual(resp["error"]["code"], PROVIDER_UNAVAILABLE_CODE)

        configure({"vad": {"enabled": True, "model_path": "/tmp/silero.onnx"}})
        for bad in ({}, {"audio_file": 42}):
            err = worker.handle_request(
                {"jsonrpc": "2.0", "id": 2, "method": "segment_probe", "params": bad}
            )
            assert err is not None
            self.assertEqual(err["error"]["code"], INVALID_REQUEST_CODE)

        with patch.object(
            pipeline, "probe_audio_file", side_effect=ProviderInvocationError("boom")
        ):
            fail = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "segment_probe",
                    "params": {"audio_file": "/tmp/x.wav"},
                }
            )
            assert fail is not None
            self.assertEqual(fail["error"]["code"], PROVIDER_REQUEST_FAILED_CODE)

        verdict = {"has_speech": True, "speech_ratio": 0.5, "regions": [], "notes": []}
        with patch.object(pipeline, "probe_audio_file", return_value=verdict):
            ok = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "segment_probe",
                    "params": {"audio_file": "/tmp/x.wav"},
                }
            )
            assert ok is not None
            self.assertEqual(ok["result"], verdict)


class SinglePassVadPrepassTest(unittest.TestCase):
    """Verify VAD runs exactly once in ``pipeline.transcribe`` and cleans up sidecar WAVs."""

    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-vad-pipeline-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_vad_runs_once_and_unlinks_trimmed_wav_on_success(self) -> None:
        audio = _write_silent_wav(self.root / "raw.wav")
        trimmed = _write_silent_wav(self.root / "raw.vad-trimmed.wav")
        configure(
            {
                "whisper": {"model_path": "/models/base.bin"},
                "vad": {"enabled": True, "model_path": "/tmp/silero.onnx"},
            }
        )

        with (
            patch.object(
                pipeline, "apply_vad_prepass", return_value=(str(trimmed), [(0.0, 1.0)])
            ) as vad_call,
            patch.object(
                pipeline,
                "transcribe_with_whisper_cli",
                return_value={"text": "hello", "detected_language": "en", "notes": []},
            ) as cli_call,
        ):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 10,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio)},
                }
            )

        assert resp is not None
        self.assertEqual(resp["result"]["text"], "hello")
        vad_call.assert_called_once()
        self.assertEqual(cli_call.call_args.args[0]["audio_file"], str(trimmed))
        self.assertFalse(trimmed.exists())
        self.assertTrue(audio.exists())

    def test_vad_short_circuits_and_cleans_up_on_empty_speech(self) -> None:
        audio = _write_silent_wav(self.root / "silent.wav")
        trimmed = self.root / "silent.vad-empty.wav"
        trimmed.write_bytes(b"")
        configure(
            {
                "whisper": {"model_path": "/models/base.bin"},
                "vad": {"enabled": True, "model_path": "/tmp/silero.onnx"},
            }
        )

        with (
            patch.object(pipeline, "apply_vad_prepass", return_value=(str(trimmed), [])),
            patch.object(pipeline, "transcribe_with_whisper_cli") as cli_call,
        ):
            resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 11,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio)},
                }
            )

        assert resp is not None
        self.assertEqual(resp["result"]["text"], "")
        cli_call.assert_not_called()
        self.assertFalse(trimmed.exists())
        self.assertTrue(audio.exists())

    def test_vad_unlinks_trimmed_wav_when_provider_raises_and_falls_back_on_vad_error(self) -> None:
        audio = _write_silent_wav(self.root / "raw.wav")
        trimmed = _write_silent_wav(self.root / "raw.vad-trimmed.wav")
        configure(
            {
                "whisper": {"model_path": "/models/base.bin"},
                "vad": {"enabled": True, "model_path": "/tmp/silero.onnx"},
            }
        )

        with (
            patch.object(pipeline, "apply_vad_prepass", return_value=(str(trimmed), [(0.0, 1.0)])),
            patch.object(
                pipeline,
                "transcribe_with_whisper_cli",
                side_effect=ProviderInvocationError("whisper failed"),
            ),
        ):
            err = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 12,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio)},
                }
            )
        assert err is not None
        self.assertIn("whisper failed", err["error"]["message"])
        self.assertFalse(trimmed.exists())

        with (
            patch.object(
                pipeline, "apply_vad_prepass", side_effect=ProviderInvocationError("ort missing")
            ),
            patch.object(
                pipeline,
                "transcribe_with_whisper_cli",
                return_value={"text": "raw ok", "detected_language": None, "notes": []},
            ) as cli_call,
        ):
            ok = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 13,
                    "method": "transcribe",
                    "params": {"audio_file": str(audio)},
                }
            )
        assert ok is not None
        self.assertEqual(ok["result"]["text"], "raw ok")
        self.assertEqual(cli_call.call_args.args[0]["audio_file"], str(audio))
        self.assertTrue(any("VAD pre-pass failed" in note for note in ok["result"]["notes"]))


@unittest.skipUnless(importlib.util.find_spec("numpy") is not None, "numpy extra not installed")
class VadNumpyHelpersTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-vad-np-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_resample_and_extract_speech_wav(self) -> None:
        import numpy as np  # noqa: PLC0415

        path = self.root / "tone.wav"
        sr = 22050
        t = np.linspace(0.0, 1.0, sr, endpoint=False, dtype=np.float32)
        pcm = np.clip(np.sin(2 * np.pi * 440 * t) * 32768.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sr)
            handle.writeframes(pcm.tobytes())

        samples, orig_sr = vad._load_wav_as_float32_mono(path, target_sample_rate=16000)
        self.assertEqual(orig_sr, sr)
        self.assertAlmostEqual(samples.shape[0], 16000, delta=2)

        trimmed = vad.extract_speech_wav(path, [(0.0, 0.5)], self.root / "out", _vad_config())
        with wave.open(str(trimmed), "rb") as handle:
            self.assertAlmostEqual(handle.getnframes(), 8000, delta=8)
