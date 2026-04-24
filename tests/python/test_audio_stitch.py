from __future__ import annotations

import pathlib
import sys
import tempfile
import unittest
import wave
from unittest.mock import patch

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[2]
PYTHON_ROOT = PROJECT_ROOT / "python"

if str(PYTHON_ROOT) not in sys.path:
    sys.path.insert(0, str(PYTHON_ROOT))

from voicelayer_orchestrator.providers import ProviderInvocationError  # noqa: E402
from voicelayer_orchestrator.providers.audio_stitch import (  # noqa: E402
    stitch_wav_segments,
)


def _write_wav(
    path: pathlib.Path,
    *,
    duration_sec: float,
    sample_rate: int = 16000,
    sample_width: int = 2,
    channels: int = 1,
    byte_value: int = 0,
) -> None:
    """Write a deterministic WAV of ``duration_sec`` seconds."""

    path.parent.mkdir(parents=True, exist_ok=True)
    n_frames = int(sample_rate * duration_sec)
    frame = bytes([byte_value]) * (sample_width * channels)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(channels)
        handle.setsampwidth(sample_width)
        handle.setframerate(sample_rate)
        handle.writeframes(frame * n_frames)


class StitchWavSegmentsHelperTest(unittest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-stitch-"))

    def test_rejects_empty_input_list(self) -> None:
        out = self.tmp_dir / "out.wav"
        with self.assertRaises(ProviderInvocationError) as ctx:
            stitch_wav_segments([], str(out))
        self.assertIn("at least one audio file", str(ctx.exception))
        self.assertFalse(out.exists())

    def test_raises_when_input_path_missing(self) -> None:
        missing = self.tmp_dir / "does-not-exist.wav"
        out = self.tmp_dir / "out.wav"
        with self.assertRaises(ProviderInvocationError) as ctx:
            stitch_wav_segments([str(missing)], str(out))
        self.assertIn("does not exist", str(ctx.exception))
        self.assertIn("index 0", str(ctx.exception))

    def test_concatenates_frames_and_preserves_format(self) -> None:
        a = self.tmp_dir / "a.wav"
        b = self.tmp_dir / "b.wav"
        c = self.tmp_dir / "c.wav"
        _write_wav(a, duration_sec=0.5, sample_rate=16000, byte_value=0x11)
        _write_wav(b, duration_sec=0.3, sample_rate=16000, byte_value=0x22)
        _write_wav(c, duration_sec=0.2, sample_rate=16000, byte_value=0x33)
        out = self.tmp_dir / "out.wav"

        result = stitch_wav_segments([str(a), str(b), str(c)], str(out))

        self.assertEqual(result["audio_file"], str(out))
        self.assertEqual(result["segment_count"], 3)
        self.assertAlmostEqual(result["duration_secs"], 1.0, places=4)

        with wave.open(str(out), "rb") as handle:
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
            self.assertEqual(handle.getframerate(), 16000)
            # 1.0 s at 16 kHz mono 16-bit = 16000 frames.
            self.assertEqual(handle.getnframes(), 16000)

    def test_rejects_mismatched_sample_rate(self) -> None:
        a = self.tmp_dir / "a.wav"
        b = self.tmp_dir / "b.wav"
        _write_wav(a, duration_sec=0.2, sample_rate=16000)
        _write_wav(b, duration_sec=0.2, sample_rate=8000)
        out = self.tmp_dir / "out.wav"

        with self.assertRaises(ProviderInvocationError) as ctx:
            stitch_wav_segments([str(a), str(b)], str(out))
        self.assertIn("incompatible", str(ctx.exception))
        self.assertIn("index 1", str(ctx.exception))
        # Atomic write: the temp file must not leak as the final output.
        self.assertFalse(out.exists())

    def test_rejects_mismatched_channel_count(self) -> None:
        a = self.tmp_dir / "a.wav"
        b = self.tmp_dir / "b.wav"
        _write_wav(a, duration_sec=0.2, channels=1)
        _write_wav(b, duration_sec=0.2, channels=2)
        out = self.tmp_dir / "out.wav"

        with self.assertRaises(ProviderInvocationError):
            stitch_wav_segments([str(a), str(b)], str(out))
        self.assertFalse(out.exists())

    def test_creates_output_directory_when_missing(self) -> None:
        a = self.tmp_dir / "a.wav"
        _write_wav(a, duration_sec=0.1)
        out = self.tmp_dir / "nested" / "dir" / "out.wav"

        stitch_wav_segments([str(a)], str(out))
        self.assertTrue(out.is_file())

    def test_single_segment_round_trips_cleanly(self) -> None:
        # Degenerate case but exercised by the orchestrator: a single
        # speech probe flushed on its own still goes through stitch.
        a = self.tmp_dir / "solo.wav"
        _write_wav(a, duration_sec=0.4, byte_value=0x55)
        out = self.tmp_dir / "solo-out.wav"

        result = stitch_wav_segments([str(a)], str(out))
        self.assertEqual(result["segment_count"], 1)
        self.assertAlmostEqual(result["duration_secs"], 0.4, places=4)


class StitchWavSegmentsDispatchTest(unittest.TestCase):
    """Wire-level tests for the JSON-RPC ``stitch_wav_segments`` method."""

    def setUp(self) -> None:
        super().setUp()
        from voicelayer_orchestrator import worker

        self.worker = worker
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-stitch-dispatch-"))

    def test_returns_invalid_request_when_audio_files_missing(self) -> None:
        response = self.worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 1,
                "method": "stitch_wav_segments",
                "params": {"out_file": "/tmp/out.wav"},
            }
        )
        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)
        self.assertIn("audio_files", response["error"]["message"])

    def test_returns_invalid_request_when_audio_files_is_empty(self) -> None:
        response = self.worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 2,
                "method": "stitch_wav_segments",
                "params": {"audio_files": [], "out_file": "/tmp/out.wav"},
            }
        )
        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)

    def test_returns_invalid_request_when_audio_files_contains_non_string(self) -> None:
        response = self.worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 3,
                "method": "stitch_wav_segments",
                "params": {"audio_files": ["/tmp/a.wav", 42], "out_file": "/tmp/out.wav"},
            }
        )
        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)

    def test_returns_invalid_request_when_out_file_missing(self) -> None:
        response = self.worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 4,
                "method": "stitch_wav_segments",
                "params": {"audio_files": ["/tmp/a.wav"]},
            }
        )
        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)

    def test_returns_provider_request_failed_when_helper_raises(self) -> None:
        missing = self.tmp_dir / "missing.wav"
        out = self.tmp_dir / "out.wav"
        response = self.worker.handle_request(
            {
                "jsonrpc": "2.0",
                "id": 5,
                "method": "stitch_wav_segments",
                "params": {
                    "audio_files": [str(missing)],
                    "out_file": str(out),
                },
            }
        )
        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.PROVIDER_REQUEST_FAILED_CODE)
        self.assertIn("does not exist", response["error"]["message"])

    def test_forwards_helper_result_verbatim_on_success(self) -> None:
        payload = {"audio_file": "/tmp/out.wav", "segment_count": 2, "duration_secs": 1.25}
        with patch.object(self.worker, "stitch_wav_segments", return_value=payload) as stitch:
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 6,
                    "method": "stitch_wav_segments",
                    "params": {
                        "audio_files": ["/tmp/a.wav", "/tmp/b.wav"],
                        "out_file": "/tmp/out.wav",
                    },
                }
            )

        assert response is not None
        self.assertNotIn("error", response)
        self.assertEqual(response["result"], payload)
        stitch.assert_called_once_with(["/tmp/a.wav", "/tmp/b.wav"], "/tmp/out.wav")


if __name__ == "__main__":
    unittest.main()
