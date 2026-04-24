from __future__ import annotations

import importlib.util
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

from voicelayer_orchestrator.config import (  # noqa: E402
    WhisperVadConfig,
    load_whisper_vad_config,
)
from voicelayer_orchestrator.providers.vad_segmenter import (  # noqa: E402
    _frames_to_regions,
    _window_size_for,
    probe_audio_file,
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


class WhisperVadConfigTest(unittest.TestCase):
    def test_returns_none_when_not_enabled(self) -> None:
        self.assertIsNone(load_whisper_vad_config({}))
        self.assertIsNone(
            load_whisper_vad_config(
                {"VOICELAYER_WHISPER_VAD_MODEL_PATH": "/tmp/silero.onnx"},
            )
        )

    def test_returns_none_when_model_path_missing(self) -> None:
        self.assertIsNone(
            load_whisper_vad_config({"VOICELAYER_WHISPER_VAD_ENABLED": "true"}),
        )

    def test_reads_all_tunables(self) -> None:
        config = load_whisper_vad_config(
            {
                "VOICELAYER_WHISPER_VAD_ENABLED": "1",
                "VOICELAYER_WHISPER_VAD_MODEL_PATH": "/tmp/silero.onnx",
                "VOICELAYER_WHISPER_VAD_THRESHOLD": "0.42",
                "VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS": "200",
                "VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS": "150",
                "VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS": "50",
                "VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS": "15",
                "VOICELAYER_WHISPER_VAD_SAMPLE_RATE": "16000",
            }
        )
        assert config is not None
        self.assertEqual(config.model_path, "/tmp/silero.onnx")
        self.assertAlmostEqual(config.threshold, 0.42)
        self.assertEqual(config.min_speech_ms, 200)
        self.assertEqual(config.min_silence_ms, 150)
        self.assertEqual(config.speech_pad_ms, 50)
        self.assertAlmostEqual(config.max_segment_secs, 15.0)
        self.assertEqual(config.sample_rate, 16000)


class WindowSizeTest(unittest.TestCase):
    def test_16khz_returns_512(self) -> None:
        self.assertEqual(_window_size_for(16000), 512)

    def test_8khz_returns_256(self) -> None:
        self.assertEqual(_window_size_for(8000), 256)

    def test_unsupported_rate_raises(self) -> None:
        with self.assertRaises(Exception):  # noqa: B017
            _window_size_for(44100)


class FramesToRegionsTest(unittest.TestCase):
    FRAME_SEC = 0.032  # 512 samples at 16kHz

    def test_empty_probs_returns_no_regions(self) -> None:
        regions = _frames_to_regions([], _vad_config(), self.FRAME_SEC)
        self.assertEqual(regions, [])

    def test_all_silence_returns_no_regions(self) -> None:
        probs = [0.1] * 100
        regions = _frames_to_regions(probs, _vad_config(), self.FRAME_SEC)
        self.assertEqual(regions, [])

    def test_contiguous_speech_produces_single_region_with_padding(self) -> None:
        # 20 speech frames in the middle of 40 total; min_speech_ms=250ms ~ 8 frames
        probs = [0.1] * 10 + [0.9] * 20 + [0.1] * 10
        config = _vad_config(min_speech_ms=100, min_silence_ms=50, speech_pad_ms=0)
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(10, 30)])

    def test_short_speech_below_min_length_is_dropped(self) -> None:
        # Only 3 speech frames, min_speech_ms=250ms ~ 8 frames
        probs = [0.1] * 10 + [0.9] * 3 + [0.1] * 10
        regions = _frames_to_regions(probs, _vad_config(speech_pad_ms=0), self.FRAME_SEC)
        self.assertEqual(regions, [])

    def test_small_gap_merges_neighboring_speech(self) -> None:
        # Two speech blocks separated by 2 silent frames (~64ms)
        # min_silence_ms=200ms ~ 6 frames, so the gap must merge.
        probs = [0.9] * 10 + [0.1] * 2 + [0.9] * 10
        config = _vad_config(min_silence_ms=200, min_speech_ms=100, speech_pad_ms=0)
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 22)])

    def test_large_gap_keeps_regions_separate(self) -> None:
        # Two speech blocks separated by 20 silent frames (~640ms).
        # min_silence_ms=100ms ~ 3 frames, so the gap stays.
        probs = [0.9] * 10 + [0.1] * 20 + [0.9] * 10
        config = _vad_config(min_silence_ms=100, min_speech_ms=100, speech_pad_ms=0)
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 10), (30, 40)])

    def test_padding_extends_regions_and_merges_overlapping_pads(self) -> None:
        probs = [0.1] * 5 + [0.9] * 10 + [0.1] * 2 + [0.9] * 10 + [0.1] * 5
        config = _vad_config(
            min_silence_ms=100,  # ~3 frames; the 2-frame gap must merge.
            min_speech_ms=100,
            speech_pad_ms=64,  # ~2 frames on each side.
        )
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        # Merged region (5, 27) then padded by 2 frames on each side.
        self.assertEqual(regions, [(3, 29)])

    def test_max_segment_hard_cuts_long_speech(self) -> None:
        # 100 speech frames, max_segment_secs forces splits every 32 frames.
        probs = [0.9] * 100
        config = _vad_config(
            min_silence_ms=10,
            min_speech_ms=50,
            speech_pad_ms=0,
            max_segment_secs=32 * self.FRAME_SEC,
        )
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 32), (32, 64), (64, 96), (96, 100)])

    def test_hysteresis_holds_speech_through_borderline_dips(self) -> None:
        # With enter=0.5 and hysteresis=0.15, leave threshold is 0.35.
        # A 0.40 probability in the middle of an active speech span must
        # not close the region (it only exits when the prob drops below
        # leave_threshold for at least one frame).
        probs = [0.9] * 10 + [0.40] * 5 + [0.9] * 10
        config = _vad_config(
            threshold=0.5,
            min_silence_ms=10,
            min_speech_ms=100,
            speech_pad_ms=0,
        )
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 25)])

    def test_hysteresis_closes_on_true_silence(self) -> None:
        # A dip all the way below leave_threshold does close the region.
        probs = [0.9] * 10 + [0.1] * 20 + [0.9] * 10
        config = _vad_config(
            threshold=0.5,
            min_silence_ms=10,
            min_speech_ms=100,
            speech_pad_ms=0,
        )
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 10), (30, 40)])

    def test_low_threshold_still_exits_speech_through_silence(self) -> None:
        # With a very low enter threshold (0.1) the fixed 0.15 margin
        # would pin leave_threshold to 0; the scaling cap keeps it
        # effective, so a 0.02 dip still closes the region.
        probs = [0.5] * 10 + [0.02] * 10 + [0.5] * 10
        config = _vad_config(
            threshold=0.1,
            min_silence_ms=10,
            min_speech_ms=100,
            speech_pad_ms=0,
        )
        regions = _frames_to_regions(probs, config, self.FRAME_SEC)
        self.assertEqual(regions, [(0, 10), (20, 30)])


class VadWorkerIntegrationTest(unittest.TestCase):
    """VAD integration into the transcribe dispatch, using monkey-patched apply_vad_prepass."""

    def setUp(self) -> None:
        super().setUp()
        # Lazy import to keep the module reference fresh after patches.
        from voicelayer_orchestrator import worker

        self.worker = worker
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-vad-worker-"))

    def _write_silent_wav(self, path: pathlib.Path, duration_sec: float = 1.0) -> None:
        sample_rate = 16000
        frames = int(sample_rate * duration_sec)
        path.parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(b"\x00\x00" * frames)

    def test_vad_disabled_does_not_invoke_prepass(self) -> None:
        audio_file = self.tmp_dir / "raw.wav"
        self._write_silent_wav(audio_file)

        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=None),
            patch.object(self.worker, "apply_vad_prepass") as prepass,
        ):
            params, notes, short = self.worker._apply_vad_prepass_if_configured(
                {"audio_file": str(audio_file)},
            )

        prepass.assert_not_called()
        self.assertEqual(params, {"audio_file": str(audio_file)})
        self.assertEqual(notes, [])
        self.assertIsNone(short)

    def test_vad_short_circuits_when_no_speech_found(self) -> None:
        audio_file = self.tmp_dir / "silence.wav"
        self._write_silent_wav(audio_file)

        vad_config = _vad_config()
        trimmed = self.tmp_dir / "silence.vad-empty.wav"
        trimmed.write_bytes(b"")

        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=vad_config),
            patch.object(
                self.worker,
                "apply_vad_prepass",
                return_value=(str(trimmed), []),
            ),
        ):
            _, _, short = self.worker._apply_vad_prepass_if_configured(
                {"audio_file": str(audio_file)},
            )

        assert short is not None
        self.assertEqual(short["text"], "")
        self.assertIn(
            "VAD detected no speech; whisper inference was skipped.",
            short["notes"],
        )

    def test_vad_replaces_audio_and_annotates_notes(self) -> None:
        audio_file = self.tmp_dir / "raw.wav"
        self._write_silent_wav(audio_file, duration_sec=2.0)

        vad_config = _vad_config()
        trimmed = self.tmp_dir / "raw.vad-trimmed.wav"
        trimmed.write_bytes(b"")
        regions = [(0.5, 1.0), (1.2, 1.7)]

        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=vad_config),
            patch.object(
                self.worker,
                "apply_vad_prepass",
                return_value=(str(trimmed), regions),
            ),
        ):
            params, notes, short = self.worker._apply_vad_prepass_if_configured(
                {"audio_file": str(audio_file)},
            )

        self.assertIsNone(short)
        self.assertEqual(params["audio_file"], str(trimmed))
        self.assertEqual(len(notes), 1)
        self.assertIn("2 speech region(s)", notes[0])
        self.assertIn("1.00s total", notes[0])

    def test_vad_prepass_error_falls_back_to_raw_audio(self) -> None:
        from voicelayer_orchestrator.providers import ProviderInvocationError

        audio_file = self.tmp_dir / "raw.wav"
        self._write_silent_wav(audio_file)
        vad_config = _vad_config()

        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=vad_config),
            patch.object(
                self.worker,
                "apply_vad_prepass",
                side_effect=ProviderInvocationError("onnxruntime missing"),
            ),
        ):
            params, notes, short = self.worker._apply_vad_prepass_if_configured(
                {"audio_file": str(audio_file)},
            )

        self.assertIsNone(short)
        self.assertEqual(params["audio_file"], str(audio_file))
        self.assertEqual(len(notes), 1)
        self.assertIn("VAD pre-pass failed", notes[0])
        self.assertIn("onnxruntime missing", notes[0])


class _FakeSamples:
    """Duck-typed stand-in for a numpy array so ``probe_audio_file`` can run
    without the optional ``vad`` extra installed. The helper only touches
    ``.size`` and ``.shape[0]``.
    """

    def __init__(self, n: int) -> None:
        self.size = n
        self.shape = (n,)


class ProbeAudioFileTest(unittest.TestCase):
    """Direct tests for ``probe_audio_file`` covering its transformation
    layer: missing files raise, silent inputs short-circuit, region seconds
    derive from frame indices, and ``speech_ratio`` falls in ``[0.0, 1.0]``.

    Mocks the numpy-dependent I/O layer (`_load_wav_as_float32_mono`,
    `_load_vad_session`, `_speech_probabilities`) so these tests run
    without the ``vad`` extra, matching the pattern in
    :class:`VadWorkerIntegrationTest` above.
    """

    FRAME_SEC = 0.032  # 512 samples at 16 kHz, matches _window_size_for(16000)

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-probe-helper-"))

    def test_raises_when_audio_file_missing(self) -> None:
        from voicelayer_orchestrator.providers import ProviderInvocationError

        missing = self.tmp_dir / "does-not-exist.wav"
        with self.assertRaises(ProviderInvocationError) as ctx:
            probe_audio_file(str(missing), _vad_config())
        self.assertIn(str(missing), str(ctx.exception))

    def test_returns_empty_result_when_sample_loader_yields_zero_samples(self) -> None:
        # Simulate a 0-frame WAV: loader returns an empty samples array, so
        # the helper short-circuits before any VAD inference runs.
        from voicelayer_orchestrator.providers import vad_segmenter

        audio = self.tmp_dir / "empty.wav"
        audio.touch()

        with patch.object(
            vad_segmenter,
            "_load_wav_as_float32_mono",
            return_value=(_FakeSamples(0), 16000),
        ) as loader:
            result = probe_audio_file(str(audio), _vad_config())

        loader.assert_called_once()
        self.assertFalse(result["has_speech"])
        self.assertEqual(result["speech_ratio"], 0.0)
        self.assertEqual(result["regions"], [])
        self.assertEqual(result["notes"], [])

    def test_forwards_speech_regions_and_clamps_ratio(self) -> None:
        # 16 kHz / 512 = 31.25 → 31 frames over 1.0 s.
        # A speech block covering frames 0..14 produces one region that
        # maps to seconds 0.0 .. 15 * FRAME_SEC.
        from voicelayer_orchestrator.providers import vad_segmenter

        audio = self.tmp_dir / "fake.wav"
        audio.touch()
        sample_rate = 16000
        samples = _FakeSamples(sample_rate)  # 1.0 s at 16 kHz
        speech_frames = [0.9] * 15 + [0.1] * 16

        with (
            patch.object(
                vad_segmenter,
                "_load_wav_as_float32_mono",
                return_value=(samples, sample_rate),
            ),
            patch.object(vad_segmenter, "_load_vad_session", return_value=object()),
            patch.object(vad_segmenter, "_speech_probabilities", return_value=speech_frames),
        ):
            config = _vad_config(
                threshold=0.5,
                min_speech_ms=100,
                min_silence_ms=50,
                speech_pad_ms=0,
                sample_rate=sample_rate,
            )
            result = probe_audio_file(str(audio), config)

        self.assertTrue(result["has_speech"])
        self.assertEqual(len(result["regions"]), 1)
        region = result["regions"][0]
        self.assertAlmostEqual(region["start_secs"], 0.0, places=5)
        self.assertAlmostEqual(region["end_secs"], 15 * self.FRAME_SEC, places=5)
        # Ratio = region length / audio duration; always in [0, 1].
        self.assertGreaterEqual(result["speech_ratio"], 0.0)
        self.assertLessEqual(result["speech_ratio"], 1.0)

    def test_clamps_ratio_when_padding_would_overshoot_duration(self) -> None:
        # `_frames_to_regions` can emit regions whose padded end extends
        # past the audio duration (padding is applied in frame units and
        # not clamped to the file length). The helper must cap
        # ``speech_ratio`` at 1.0 so callers never see a ratio > 1.
        from voicelayer_orchestrator.providers import vad_segmenter

        audio = self.tmp_dir / "fake.wav"
        audio.touch()
        sample_rate = 16000
        samples = _FakeSamples(sample_rate // 4)  # 0.25 s

        # Fake `_frames_to_regions` to return a single region that
        # obviously exceeds the clip duration (end index > frame count).
        with (
            patch.object(
                vad_segmenter,
                "_load_wav_as_float32_mono",
                return_value=(samples, sample_rate),
            ),
            patch.object(vad_segmenter, "_load_vad_session", return_value=object()),
            patch.object(vad_segmenter, "_speech_probabilities", return_value=[0.9] * 8),
            patch.object(
                vad_segmenter,
                "_frames_to_regions",
                return_value=[(0, 100)],  # 100 * 32 ms = 3.2 s, audio is 0.25 s
            ),
        ):
            result = probe_audio_file(str(audio), _vad_config(sample_rate=sample_rate))

        self.assertTrue(result["has_speech"])
        self.assertEqual(result["speech_ratio"], 1.0)


class SegmentProbeDispatchTest(unittest.TestCase):
    """Dispatch-level tests for the JSON-RPC ``segment_probe`` method.

    The method is a thin wrapper around
    :func:`vad_segmenter.probe_audio_file`; these tests pin the wire-level
    error classification (provider-unavailable, invalid-request,
    request-failed) and that the happy path forwards the helper's result
    verbatim.
    """

    def setUp(self) -> None:
        super().setUp()
        from voicelayer_orchestrator import worker

        self.worker = worker
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-segment-probe-"))

    def test_returns_provider_unavailable_when_vad_not_configured(self) -> None:
        with patch.object(self.worker, "load_whisper_vad_config", return_value=None):
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "segment_probe",
                    "params": {"audio_file": "/tmp/anything.wav"},
                }
            )

        assert response is not None
        self.assertIn("error", response)
        self.assertEqual(response["error"]["code"], self.worker.PROVIDER_UNAVAILABLE_CODE)
        self.assertIn("VOICELAYER_WHISPER_VAD_ENABLED", response["error"]["message"])

    def test_returns_invalid_request_when_audio_file_missing(self) -> None:
        with patch.object(self.worker, "load_whisper_vad_config", return_value=_vad_config()):
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 2,
                    "method": "segment_probe",
                    "params": {},
                }
            )

        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)
        self.assertIn("audio_file", response["error"]["message"])

    def test_returns_invalid_request_when_audio_file_is_not_string(self) -> None:
        with patch.object(self.worker, "load_whisper_vad_config", return_value=_vad_config()):
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "segment_probe",
                    "params": {"audio_file": 42},
                }
            )

        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.INVALID_REQUEST_CODE)

    def test_returns_provider_request_failed_when_probe_raises(self) -> None:
        from voicelayer_orchestrator.providers import ProviderInvocationError

        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=_vad_config()),
            patch.object(
                self.worker,
                "probe_audio_file",
                side_effect=ProviderInvocationError("probe input does not exist: /tmp/x.wav"),
            ),
        ):
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 4,
                    "method": "segment_probe",
                    "params": {"audio_file": "/tmp/x.wav"},
                }
            )

        assert response is not None
        self.assertEqual(response["error"]["code"], self.worker.PROVIDER_REQUEST_FAILED_CODE)
        self.assertIn("probe input does not exist", response["error"]["message"])

    def test_forwards_probe_audio_file_result_verbatim(self) -> None:
        probe_payload = {
            "has_speech": True,
            "speech_ratio": 0.63,
            "regions": [
                {"start_secs": 0.25, "end_secs": 0.75},
                {"start_secs": 1.10, "end_secs": 1.40},
            ],
            "notes": [],
        }
        with (
            patch.object(self.worker, "load_whisper_vad_config", return_value=_vad_config()),
            patch.object(self.worker, "probe_audio_file", return_value=probe_payload) as probe,
        ):
            response = self.worker.handle_request(
                {
                    "jsonrpc": "2.0",
                    "id": 5,
                    "method": "segment_probe",
                    "params": {"audio_file": "/tmp/probe.wav"},
                }
            )

        assert response is not None
        self.assertNotIn("error", response)
        self.assertEqual(response["result"], probe_payload)
        probe.assert_called_once()
        call_args = probe.call_args
        self.assertEqual(call_args.args[0], "/tmp/probe.wav")


@unittest.skipUnless(
    importlib.util.find_spec("numpy") is not None,
    "numpy (vad extra) is not installed",
)
class VadWavHelpersTest(unittest.TestCase):
    """Exercise helpers that require numpy. Skipped when the vad extra is absent."""

    def setUp(self) -> None:
        super().setUp()
        self.tmp_dir = pathlib.Path(tempfile.mkdtemp(prefix="voicelayer-vad-wav-"))

    def _write_wav(self, path: pathlib.Path, samples: object, sample_rate: int = 16000) -> None:
        import numpy as np  # noqa: PLC0415

        pcm = np.clip(np.asarray(samples) * 32768.0, -32768.0, 32767.0).astype(np.int16)
        with wave.open(str(path), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(sample_rate)
            handle.writeframes(pcm.tobytes())

    def test_load_wav_resamples_to_target_rate(self) -> None:
        import numpy as np  # noqa: PLC0415

        from voicelayer_orchestrator.providers.vad_segmenter import (  # noqa: PLC0415
            _load_wav_as_float32_mono,
        )

        path = self.tmp_dir / "sine-22050.wav"
        sample_rate = 22050
        t = np.linspace(0.0, 1.0, sample_rate, endpoint=False, dtype=np.float32)
        self._write_wav(path, np.sin(2 * np.pi * 440 * t), sample_rate=sample_rate)

        samples, original_sr = _load_wav_as_float32_mono(path, target_sample_rate=16000)
        self.assertEqual(original_sr, sample_rate)
        self.assertEqual(samples.dtype, np.float32)
        # Linear resampling keeps the proportion of samples within one unit.
        self.assertAlmostEqual(samples.shape[0], 16000, delta=2)

    def test_extract_speech_wav_writes_only_selected_regions(self) -> None:
        import numpy as np  # noqa: PLC0415

        from voicelayer_orchestrator.providers.vad_segmenter import (  # noqa: PLC0415
            extract_speech_wav,
        )

        path = self.tmp_dir / "source.wav"
        sample_rate = 16000
        duration = 3.0
        t = np.linspace(0.0, duration, int(sample_rate * duration), endpoint=False)
        tone = np.sin(2 * np.pi * 200 * t).astype(np.float32)
        self._write_wav(path, tone, sample_rate=sample_rate)

        config = _vad_config(sample_rate=sample_rate)
        out_dir = self.tmp_dir / "out"
        trimmed = extract_speech_wav(path, [(0.0, 0.5), (1.0, 1.5)], out_dir, config)
        with wave.open(str(trimmed), "rb") as handle:
            frames = handle.getnframes()
            self.assertEqual(handle.getframerate(), sample_rate)
            self.assertEqual(handle.getnchannels(), 1)
            self.assertEqual(handle.getsampwidth(), 2)
        # Two regions of 0.5s each = 1.0s = 16000 frames (+/- rounding).
        self.assertAlmostEqual(frames, sample_rate, delta=8)


if __name__ == "__main__":
    unittest.main()
