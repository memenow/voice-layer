"""Tests for WAV duration, splitting, concatenation, and stitch RPC dispatch."""

from __future__ import annotations

import pathlib
import tempfile
import unittest
import wave
from unittest.mock import patch

from voicelayer_worker import pipeline, worker
from voicelayer_worker.audio import split_wav_segments, stitch_wav_segments, wav_duration_seconds
from voicelayer_worker.config import reset_configuration_for_tests
from voicelayer_worker.errors import (
    INVALID_REQUEST_CODE,
    PROVIDER_REQUEST_FAILED_CODE,
    ProviderInvocationError,
)


def _write_wav(
    path: pathlib.Path,
    *,
    frames: int = 1600,
    channels: int = 1,
    sampwidth: int = 2,
    framerate: int = 16000,
    byte_value: int = 1,
) -> pathlib.Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sampwidth)
        handle.setframerate(framerate)
        handle.writeframes(bytes([byte_value]) * (frames * channels * sampwidth))
    return path


class AudioUtilsTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()
        self.tmp = tempfile.TemporaryDirectory(prefix="voicelayer-audio-test-")
        self.addCleanup(self.tmp.cleanup)
        self.root = pathlib.Path(self.tmp.name)

    def test_wav_duration_and_invalid_input(self) -> None:
        wav = _write_wav(self.root / "half.wav", frames=8000, framerate=16000)
        self.assertAlmostEqual(wav_duration_seconds(wav), 0.5)

        bad = self.root / "bad.wav"
        bad.write_bytes(b"not a wav")
        with self.assertRaises(ProviderInvocationError):
            wav_duration_seconds(bad)

    def test_split_wav_segments_splits_and_rolls_back_on_error(self) -> None:
        short = _write_wav(self.root / "short.wav", frames=16000, framerate=16000)
        self.assertEqual(split_wav_segments(short, 2.0, self.root / "out"), [short])
        self.assertEqual(split_wav_segments(short, 0.0, self.root / "out"), [short])

        long_wav = _write_wav(self.root / "long.wav", frames=48000, framerate=16000)
        chunks = split_wav_segments(long_wav, 1.0, self.root / "chunks", prefix="part")
        self.assertEqual(len(chunks), 3)
        for chunk in chunks:
            self.assertAlmostEqual(wav_duration_seconds(chunk), 1.0)

        out_dir = self.root / "rollback"
        real_open = wave.open
        write_calls = 0

        def fail_second_write(filename: str, mode: str = "rb"):
            nonlocal write_calls
            if mode == "wb":
                write_calls += 1
                if write_calls == 2:
                    raise OSError("disk full")
            return real_open(filename, mode)

        with (
            patch("voicelayer_worker.audio.wave.open", side_effect=fail_second_write),
            self.assertRaises(ProviderInvocationError),
        ):
            split_wav_segments(long_wav, 1.0, out_dir, prefix="part")
        self.assertEqual(list(out_dir.glob("*.wav")), [])

    def test_stitch_wav_segments_validates_and_concatenates(self) -> None:
        with self.assertRaises(ProviderInvocationError):
            stitch_wav_segments([], str(self.root / "out.wav"))

        with self.assertRaises(ProviderInvocationError):
            stitch_wav_segments([str(self.root / "missing.wav")], str(self.root / "out.wav"))

        seg1 = _write_wav(self.root / "a.wav", frames=8000, framerate=16000, byte_value=1)
        seg2 = _write_wav(self.root / "b.wav", frames=16000, framerate=16000, byte_value=2)
        out = self.root / "nested" / "stitched.wav"

        result = stitch_wav_segments([str(seg1), str(seg2)], str(out))
        self.assertEqual(result["audio_file"], str(out))
        self.assertEqual(result["segment_count"], 2)
        self.assertAlmostEqual(result["duration_secs"], 1.5)
        self.assertTrue(out.is_file())

        bad_rate = _write_wav(self.root / "rate.wav", frames=8000, framerate=8000)
        with self.assertRaises(ProviderInvocationError):
            stitch_wav_segments([str(seg1), str(bad_rate)], str(self.root / "fail.wav"))

        bad_chan = _write_wav(self.root / "stereo.wav", frames=8000, channels=2)
        with self.assertRaises(ProviderInvocationError):
            stitch_wav_segments([str(seg1), str(bad_chan)], str(self.root / "fail2.wav"))


class StitchRpcDispatchTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()

    def test_validates_params_and_forwards_results(self) -> None:
        for bad_params in (
            {"out_file": "/tmp/out.wav"},
            {"audio_files": [], "out_file": "/tmp/out.wav"},
            {"audio_files": [123], "out_file": "/tmp/out.wav"},
            {"audio_files": ["/tmp/a.wav"]},
        ):
            resp = worker.handle_request(
                {"jsonrpc": "2.0", "id": 1, "method": "stitch_wav_segments", "params": bad_params}
            )
            assert resp is not None
            self.assertEqual(resp["error"]["code"], INVALID_REQUEST_CODE)

        with patch.object(
            pipeline,
            "_stitch_wav_segments",
            side_effect=ProviderInvocationError("mismatched sample rate"),
        ):
            err_resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "stitch_wav_segments",
                    "params": {"audio_files": ["/tmp/a.wav"], "out_file": "/tmp/out.wav"},
                }
            )
            assert err_resp is not None
            self.assertEqual(err_resp["error"]["code"], PROVIDER_REQUEST_FAILED_CODE)

        payload = {"audio_file": "/tmp/out.wav", "segment_count": 2, "duration_secs": 3.0}
        with patch.object(pipeline, "_stitch_wav_segments", return_value=payload):
            ok_resp = worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "stitch_wav_segments",
                    "params": {
                        "audio_files": ["/tmp/a.wav", "/tmp/b.wav"],
                        "out_file": "/tmp/out.wav",
                    },
                }
            )
            assert ok_resp is not None
            self.assertEqual(ok_resp["result"], payload)
