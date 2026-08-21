"""Tests for the optional Qwen3-ASR-1.7B provider.

The real provider depends on a multi-GB GPU model and the official
``qwen-asr`` pip package, neither of which CI has. The tests therefore
stay on the CPU side: they exercise the configuration loader, the path
validators, the language hint mapping, the long-audio splitter (stdlib
``wave`` only), the VAD pre-pass wiring, and the dispatch surface that
converts errors to ``ProviderInvocationError``. The model load and
per-segment inference are mocked at the boundary so a fresh ``uv sync
--group dev`` install runs the suite without pulling torch / qwen-asr.
"""

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

from voicelayer_orchestrator.config import (  # noqa: E402
    Qwen3AsrConfig,
    WhisperVadConfig,
    load_qwen3_asr_config,  # noqa: E402
)
from voicelayer_orchestrator.providers import (  # noqa: E402
    ProviderInvocationError,
    qwen3_asr,
)


def _write_silent_wav(path: pathlib.Path, duration_seconds: float) -> None:
    """Write a 16 kHz mono PCM16 silent WAV at ``path``."""

    sample_rate = 16_000
    n_frames = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


def _qwen3_config(model_dir: pathlib.Path) -> Qwen3AsrConfig:
    """Build a Qwen3AsrConfig pointing at an existing on-disk directory.

    Tests share this helper rather than constructing the dataclass
    inline so a future field addition only needs an update here.
    """

    return Qwen3AsrConfig(
        model_path=str(model_dir),
        device="cuda:0",
        torch_dtype="bfloat16",
        timeout_seconds=600.0,
        long_audio_split_seconds=0.0,
        extra_args=(),
    )


def _vad_config_sentinel() -> WhisperVadConfig:
    """``WhisperVadConfig`` populated with documented defaults."""

    return WhisperVadConfig(
        model_path="/abs/silero.onnx",
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=30,
        max_segment_secs=30.0,
        sample_rate=16_000,
    )


class LoadQwen3AsrConfigTest(unittest.TestCase):
    def test_returns_none_when_model_path_is_unset(self) -> None:
        # Empty mapping → no env keys → loader returns None so callers
        # treat the provider as "not configured" without an error.
        self.assertIsNone(load_qwen3_asr_config({"_test_marker": "1"}))
        # Empty/whitespace MODEL_PATH must be normalized to "not set".
        self.assertIsNone(load_qwen3_asr_config({"VOICELAYER_QWEN3_ASR_MODEL_PATH": "   "}))

    def test_reads_full_environment_with_optional_overrides(self) -> None:
        # Every documented knob must round-trip into the dataclass so
        # operator overrides actually take effect on the running worker.
        # Note: `extra_args` is reserved for a future passthrough into
        # `model.transcribe`; the current `qwen-asr` wrapper exposes
        # only `audio` and `language`, so the field is parsed (and
        # asserted here) for forward compatibility but is not forwarded
        # at dispatch today. See `Qwen3AsrConfig` docstring.
        config = load_qwen3_asr_config(
            {
                "VOICELAYER_QWEN3_ASR_MODEL_PATH": "/abs/qwen3-asr",
                "VOICELAYER_QWEN3_ASR_DEVICE": "cuda:1",
                "VOICELAYER_QWEN3_ASR_DTYPE": "float16",
                "VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS": "120",
                "VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS": "30",
                "VOICELAYER_QWEN3_ASR_ARGS": "--foo bar",
            }
        )
        assert config is not None
        self.assertEqual(config.model_path, "/abs/qwen3-asr")
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(config.torch_dtype, "float16")
        self.assertEqual(config.timeout_seconds, 120.0)
        self.assertEqual(config.long_audio_split_seconds, 30.0)
        self.assertEqual(config.extra_args, ("--foo", "bar"))

    def test_dtype_is_normalized_to_lower_case(self) -> None:
        # Operators sometimes copy-paste `BFloat16`; the loader must
        # normalize so `validate_qwen3_asr_provider` accepts the value.
        config = load_qwen3_asr_config(
            {
                "VOICELAYER_QWEN3_ASR_MODEL_PATH": "/abs/qwen3-asr",
                "VOICELAYER_QWEN3_ASR_DTYPE": "BFloat16",
            }
        )
        assert config is not None
        self.assertEqual(config.torch_dtype, "bfloat16")

    def test_long_audio_split_seconds_default_is_zero(self) -> None:
        # The upstream `qwen-asr` wrapper handles long audio internally,
        # so the default should disable client-side splitting. A
        # regression that flipped the default to a positive value would
        # silently start chunking and concatenating transcripts.
        config = load_qwen3_asr_config({"VOICELAYER_QWEN3_ASR_MODEL_PATH": "/abs/qwen3-asr"})
        assert config is not None
        self.assertEqual(config.long_audio_split_seconds, 0.0)


class ValidateQwen3AsrProviderTest(unittest.TestCase):
    def test_returns_false_when_config_is_none(self) -> None:
        ready, error = qwen3_asr.validate_qwen3_asr_provider(None)
        self.assertFalse(ready)
        assert error is not None
        self.assertIn("Qwen3-ASR-1.7B", error)

    def test_returns_false_when_model_path_does_not_exist(self) -> None:
        config = Qwen3AsrConfig(
            model_path="/does/not/exist",
            device="cuda:0",
            torch_dtype="bfloat16",
            timeout_seconds=600.0,
            long_audio_split_seconds=0.0,
            extra_args=(),
        )
        ready, error = qwen3_asr.validate_qwen3_asr_provider(config)
        self.assertFalse(ready)
        assert error is not None
        self.assertIn("VOICELAYER_QWEN3_ASR_MODEL_PATH", error)

    def test_returns_false_for_unknown_dtype(self) -> None:
        # Unknown dtypes must reject with the allowed-values list so
        # operators can fix their env without trial and error.
        with tempfile.TemporaryDirectory() as tmp:
            config = Qwen3AsrConfig(
                model_path=str(pathlib.Path(tmp)),
                device="cuda:0",
                torch_dtype="int8",
                timeout_seconds=600.0,
                long_audio_split_seconds=0.0,
                extra_args=(),
            )
            ready, error = qwen3_asr.validate_qwen3_asr_provider(config)
            self.assertFalse(ready)
            assert error is not None
            self.assertIn("VOICELAYER_QWEN3_ASR_DTYPE", error)
            self.assertIn("bfloat16", error)

    def test_returns_true_when_path_and_dtype_are_valid(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _qwen3_config(pathlib.Path(tmp))
            ready, error = qwen3_asr.validate_qwen3_asr_provider(config)
            self.assertTrue(ready)
            self.assertIsNone(error)


class ResolveLanguageHintTest(unittest.TestCase):
    def test_chinese_codes_map_to_chinese(self) -> None:
        for code in ("zh", "zh-CN", "zh-cn", "ZH", "cmn"):
            with self.subTest(code=code):
                self.assertEqual(qwen3_asr._resolve_language_hint(code), "Chinese")

    def test_english_codes_map_to_english(self) -> None:
        for code in ("en", "en-US", "EN-GB"):
            with self.subTest(code=code):
                self.assertEqual(qwen3_asr._resolve_language_hint(code), "English")

    def test_cantonese_yue_maps_to_cantonese(self) -> None:
        # The model card lists Cantonese as a separate top-level entry;
        # the BCP-47 code is `yue`. Pin the mapping so a regression that
        # collapsed it back into Chinese surfaces here.
        self.assertEqual(qwen3_asr._resolve_language_hint("yue"), "Cantonese")

    def test_auto_and_none_collapse_to_none(self) -> None:
        self.assertIsNone(qwen3_asr._resolve_language_hint(None))
        self.assertIsNone(qwen3_asr._resolve_language_hint("auto"))
        self.assertIsNone(qwen3_asr._resolve_language_hint("  "))

    def test_unknown_code_collapses_to_none(self) -> None:
        # `xx` is not a real language; the wrapper would reject it. We
        # collapse to None instead so qwen-asr runs its own
        # auto-detector rather than receiving a meaningless string.
        self.assertIsNone(qwen3_asr._resolve_language_hint("xx"))


class SplitWavIntoSegmentsTest(unittest.TestCase):
    def test_zero_max_segment_is_noop(self) -> None:
        # The default config sets long_audio_split_seconds=0 because
        # qwen-asr handles long audio internally; the splitter must
        # short-circuit cleanly so the call passes through unchanged.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "any.wav"
            _write_silent_wav(audio, 4.0)
            segments = qwen3_asr._split_wav_into_segments(audio, 0.0, tmp_path)
            self.assertEqual(segments, [audio])

    def test_short_audio_is_returned_unsplit_when_split_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "short.wav"
            _write_silent_wav(audio, 1.0)
            segments = qwen3_asr._split_wav_into_segments(audio, 30.0, tmp_path)
            self.assertEqual(segments, [audio])

    def test_long_audio_is_chunked_when_split_enabled(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.5)
            segments = qwen3_asr._split_wav_into_segments(audio, 2.0, tmp_path)
            self.assertEqual(len(segments), 3)
            for segment in segments:
                self.assertTrue(segment.is_file())
                self.assertTrue(segment.name.startswith("qwen3-segment-"))


class TranscribeWithQwen3AsrTest(unittest.TestCase):
    def test_missing_audio_file_raises_invocation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = _qwen3_config(pathlib.Path(tmp))
            with self.assertRaises(ProviderInvocationError):
                qwen3_asr.transcribe_with_qwen3_asr({}, config)

    def test_translate_to_english_is_explicitly_rejected(self) -> None:
        # Qwen3-ASR-1.7B is transcription-only; the worker surfaces the
        # limitation rather than silently dropping the flag, so callers
        # can route translation through the LLM workflow.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "sample.wav"
            _write_silent_wav(audio, 1.0)
            config = _qwen3_config(tmp_path)
            with self.assertRaisesRegex(ProviderInvocationError, "translate_to_english"):
                qwen3_asr.transcribe_with_qwen3_asr(
                    {"audio_file": str(audio), "translate_to_english": True},
                    config,
                )

    def test_dispatches_each_segment_through_loaded_model(self) -> None:
        # Force long-audio split so the dispatcher exercises the full
        # multi-segment loop. Mock `_run_segment_inference` so we never
        # import torch / qwen-asr at test time. The transcribe function
        # must split, call the inference fn once per segment, concatenate
        # outputs with a single space, surface segment count on notes,
        # and reverse-map the upstream language string into a BCP-47
        # code.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.0)
            config = Qwen3AsrConfig(
                model_path=str(tmp_path),
                device="cuda:0",
                torch_dtype="bfloat16",
                timeout_seconds=600.0,
                long_audio_split_seconds=2.0,
                extra_args=(),
            )

            calls: list[tuple[pathlib.Path, str | None]] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                language_hint: str | None,
            ) -> tuple[str, str | None]:
                calls.append((segment_path, language_hint))
                return f"chunk-{len(calls)}", "Chinese"

            sentinel_model = object()
            with (
                patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=sentinel_model),
                patch.object(qwen3_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = qwen3_asr.transcribe_with_qwen3_asr(
                    {"audio_file": str(audio), "language": "zh"},
                    config,
                )

            self.assertEqual(len(calls), 3)
            for _segment_path, language_hint in calls:
                # `language=zh` must be translated into the wrapper's
                # English `Chinese` name.
                self.assertEqual(language_hint, "Chinese")
            self.assertEqual(result["text"], "chunk-1 chunk-2 chunk-3")
            # `Chinese` round-trips back into `zh` so the response
            # uses the BCP-47 short code shared with MiMo / whisper.
            self.assertEqual(result["detected_language"], "zh")
            joined_notes = " | ".join(result["notes"])
            self.assertIn("Device: cuda:0", joined_notes)
            self.assertIn("dtype: bfloat16", joined_notes)
            self.assertIn("segments processed: 3", joined_notes)
            self.assertIn("Language hint forwarded to qwen-asr: Chinese", joined_notes)


class VadPrepassForQwen3AsrTest(unittest.TestCase):
    """Pin the silero-vad pre-pass wiring on the Qwen3 path.

    The whisper chain has applied silero-vad before transcribe since
    `VOICELAYER_WHISPER_VAD_ENABLED=1` was introduced; MiMo and now
    Qwen3-ASR-1.7B honor the same configuration so an operator who
    turned the pre-pass on once gets it on every backend. Without
    these pins, a regression that bypassed
    ``_apply_vad_prepass_for_qwen3_asr`` would surface only as silent
    hallucinations on a silent recording — exactly the failure mode
    VAD-on-Qwen3 was added to prevent.
    """

    def test_vad_unconfigured_passes_raw_audio_to_qwen3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "raw.wav"
            _write_silent_wav(audio, 1.0)
            config = _qwen3_config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _language_hint: str | None,
            ) -> tuple[str, str | None]:
                inferred.append(segment_path)
                return "raw transcript", None

            with (
                patch.object(qwen3_asr, "load_whisper_vad_config", return_value=None),
                patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=object()),
                patch.object(qwen3_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = qwen3_asr.transcribe_with_qwen3_asr({"audio_file": str(audio)}, config)

            self.assertEqual(inferred, [audio])
            self.assertEqual(result["text"], "raw transcript")
            joined_notes = " | ".join(result["notes"])
            self.assertNotIn("VAD pre-pass", joined_notes)
            self.assertNotIn("VAD detected no speech", joined_notes)

    def test_vad_short_circuits_when_no_speech_detected(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "silent.wav"
            _write_silent_wav(audio, 1.0)
            trimmed_audio = tmp_path / "silent.vad-empty.wav"
            trimmed_audio.write_bytes(b"")
            config = _qwen3_config(tmp_path)

            with (
                patch.object(
                    qwen3_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    qwen3_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), []),
                ),
                patch.object(qwen3_asr, "_load_qwen3_asr_model") as load_model,
                patch.object(qwen3_asr, "_run_segment_inference") as run_inference,
            ):
                result = qwen3_asr.transcribe_with_qwen3_asr({"audio_file": str(audio)}, config)

                # The whole point of the short-circuit is to skip the
                # cold model load and the per-segment loop.
                load_model.assert_not_called()
                run_inference.assert_not_called()

            self.assertEqual(result["text"], "")
            self.assertIsNone(result["detected_language"])
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD detected no speech", joined_notes)
            self.assertIn("Qwen3-ASR-1.7B inference was skipped", joined_notes)
            # The VAD-empty sidecar must be cleaned up; otherwise every
            # silent capture leaks a file under `runtime_dir/vad/`.
            self.assertFalse(trimmed_audio.exists())

    def test_vad_trimmed_audio_replaces_input_for_qwen3(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 5.0)
            trimmed_audio = tmp_path / "raw.vad-trimmed.wav"
            _write_silent_wav(trimmed_audio, 2.0)
            config = _qwen3_config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _language_hint: str | None,
            ) -> tuple[str, str | None]:
                inferred.append(segment_path)
                return "trimmed transcript", "English"

            with (
                patch.object(
                    qwen3_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    qwen3_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), [(0.5, 1.5), (3.0, 4.5)]),
                ),
                patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=object()),
                patch.object(qwen3_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = qwen3_asr.transcribe_with_qwen3_asr({"audio_file": str(raw_audio)}, config)

            # The model is invoked on the trimmed audio, not the
            # original capture; long-audio split runs over the post-VAD
            # WAV so the segments live under runtime_dir.
            self.assertEqual(len(inferred), 1)
            self.assertEqual(inferred[0], trimmed_audio)
            self.assertEqual(result["text"], "trimmed transcript")
            self.assertEqual(result["detected_language"], "en")
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD pre-pass kept 2 speech region(s)", joined_notes)
            self.assertIn("2.50s total", joined_notes)
            # Trimmed WAV is worker-owned: cleaned up on the way out.
            self.assertFalse(trimmed_audio.exists())
            # The caller-supplied raw WAV is *not* worker-owned.
            self.assertTrue(raw_audio.exists())

    def test_vad_trimmed_audio_is_cleaned_up_when_inference_fails(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 1.0)
            trimmed_audio = tmp_path / "raw.vad-trimmed.wav"
            _write_silent_wav(trimmed_audio, 1.0)
            config = _qwen3_config(tmp_path)

            def failing_inference(*_args: object, **_kwargs: object) -> tuple[str, str | None]:
                raise ProviderInvocationError("simulated qwen-asr failure")

            with (
                patch.object(
                    qwen3_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    qwen3_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), [(0.0, 1.0)]),
                ),
                patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=object()),
                patch.object(qwen3_asr, "_run_segment_inference", side_effect=failing_inference),
                self.assertRaises(ProviderInvocationError),
            ):
                qwen3_asr.transcribe_with_qwen3_asr({"audio_file": str(raw_audio)}, config)

            # Even on failure the trimmed sidecar is cleaned up.
            self.assertFalse(trimmed_audio.exists())
            self.assertTrue(raw_audio.exists())

    def test_vad_failure_falls_back_to_raw_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 1.0)
            config = _qwen3_config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _language_hint: str | None,
            ) -> tuple[str, str | None]:
                inferred.append(segment_path)
                return "raw transcript after vad failure", None

            with (
                patch.object(
                    qwen3_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    qwen3_asr,
                    "apply_vad_prepass",
                    side_effect=ProviderInvocationError("silero-vad onnx import failed"),
                ),
                patch.object(qwen3_asr, "_load_qwen3_asr_model", return_value=object()),
                patch.object(qwen3_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = qwen3_asr.transcribe_with_qwen3_asr({"audio_file": str(raw_audio)}, config)

            self.assertEqual(inferred, [raw_audio])
            self.assertEqual(result["text"], "raw transcript after vad failure")
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD pre-pass failed", joined_notes)
            self.assertIn("silero-vad onnx import failed", joined_notes)


if __name__ == "__main__":
    unittest.main()
