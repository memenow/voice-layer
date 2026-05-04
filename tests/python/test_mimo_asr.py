"""Tests for the optional Xiaomi MiMo-V2.5-ASR provider.

The real provider depends on a 15+ GB GPU model and Xiaomi's wrapper
class that is not pip-installable today. These tests therefore stay
on the CPU side: they exercise the configuration loader, the path
validators, the language → audio_tag mapping, the long-audio splitter
(stdlib `wave` only), and the dispatch surface that converts errors
to `ProviderInvocationError`. The model load itself is mocked in the
one test that needs to assert the segment-level dispatch contract.
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
    MimoAsrConfig,
    WhisperVadConfig,
    load_mimo_asr_config,
)
from voicelayer_orchestrator.providers import (  # noqa: E402
    ProviderInvocationError,
    mimo_asr,
)


def _write_silent_wav(path: pathlib.Path, duration_seconds: float) -> None:
    """Write a 16 kHz mono PCM16 silent WAV at `path`.

    Used by the long-audio splitter tests; nothing decodes the audio,
    so silence is fine. Frame count is rounded down to keep the math
    deterministic across platforms.
    """

    sample_rate = 16_000
    n_frames = max(1, int(duration_seconds * sample_rate))
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)
        wav.writeframes(b"\x00\x00" * n_frames)


class LoadMimoAsrConfigTest(unittest.TestCase):
    def test_returns_none_when_either_required_path_is_missing(self) -> None:
        # Both keys absent → the loader cannot construct a config and
        # falls back to None so callers treat the provider as
        # unconfigured rather than surfacing an error.
        self.assertIsNone(load_mimo_asr_config({"_test_marker": "1"}))
        # Tokenizer alone is not enough.
        self.assertIsNone(
            load_mimo_asr_config({"VOICELAYER_MIMO_TOKENIZER_PATH": "/tmp/tokenizer"})
        )
        # Model alone is not enough either.
        self.assertIsNone(load_mimo_asr_config({"VOICELAYER_MIMO_MODEL_PATH": "/tmp/model"}))

    def test_reads_full_environment_with_optional_overrides(self) -> None:
        # Every documented knob must round-trip into the dataclass so
        # operator overrides actually take effect on the running
        # worker. Catches a typo'd env key in the loader.
        config = load_mimo_asr_config(
            {
                "VOICELAYER_MIMO_MODEL_PATH": "/abs/model",
                "VOICELAYER_MIMO_TOKENIZER_PATH": "/abs/tokenizer",
                "VOICELAYER_MIMO_REPO_PATH": "/abs/repo",
                "VOICELAYER_MIMO_DEVICE": "cuda:1",
                "VOICELAYER_MIMO_AUDIO_TAG": "<chinese>",
                "VOICELAYER_MIMO_TIMEOUT_SECONDS": "120",
                "VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS": "90",
                "VOICELAYER_MIMO_ARGS": "--foo bar",
            }
        )
        assert config is not None
        self.assertEqual(config.model_path, "/abs/model")
        self.assertEqual(config.tokenizer_path, "/abs/tokenizer")
        self.assertEqual(config.repo_path, "/abs/repo")
        self.assertEqual(config.device, "cuda:1")
        self.assertEqual(config.audio_tag, "<chinese>")
        self.assertEqual(config.timeout_seconds, 120.0)
        self.assertEqual(config.long_audio_split_seconds, 90.0)
        self.assertEqual(config.extra_args, ("--foo", "bar"))

    def test_audio_tag_empty_string_is_treated_as_auto(self) -> None:
        config = load_mimo_asr_config(
            {
                "VOICELAYER_MIMO_MODEL_PATH": "/abs/model",
                "VOICELAYER_MIMO_TOKENIZER_PATH": "/abs/tokenizer",
                "VOICELAYER_MIMO_AUDIO_TAG": "   ",
            }
        )
        assert config is not None
        self.assertIsNone(config.audio_tag)

    def test_repo_path_empty_string_is_treated_as_unset(self) -> None:
        # An operator can blank `VOICELAYER_MIMO_REPO_PATH` to opt out
        # of the sys.path injection once they install the wrapper
        # via pip. The dataclass should record `None`, not an empty
        # string, so `_ensure_repo_on_path` short-circuits cleanly.
        config = load_mimo_asr_config(
            {
                "VOICELAYER_MIMO_MODEL_PATH": "/abs/model",
                "VOICELAYER_MIMO_TOKENIZER_PATH": "/abs/tokenizer",
                "VOICELAYER_MIMO_REPO_PATH": "",
            }
        )
        assert config is not None
        self.assertIsNone(config.repo_path)


class ValidateMimoProviderTest(unittest.TestCase):
    def test_returns_false_when_config_is_none(self) -> None:
        ready, error = mimo_asr.validate_mimo_provider(None)
        self.assertFalse(ready)
        assert error is not None
        self.assertIn("MiMo-V2.5-ASR model paths", error)

    def test_returns_false_when_model_path_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tokenizer = pathlib.Path(tmp)
            config = MimoAsrConfig(
                model_path="/does/not/exist",
                tokenizer_path=str(tokenizer),
                repo_path=None,
                device="cuda:0",
                audio_tag=None,
                timeout_seconds=600.0,
                long_audio_split_seconds=180.0,
                extra_args=(),
            )
            ready, error = mimo_asr.validate_mimo_provider(config)
            self.assertFalse(ready)
            assert error is not None
            self.assertIn("VOICELAYER_MIMO_MODEL_PATH", error)

    def test_returns_false_when_tokenizer_path_does_not_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = pathlib.Path(tmp)
            config = MimoAsrConfig(
                model_path=str(model),
                tokenizer_path="/does/not/exist",
                repo_path=None,
                device="cuda:0",
                audio_tag=None,
                timeout_seconds=600.0,
                long_audio_split_seconds=180.0,
                extra_args=(),
            )
            ready, error = mimo_asr.validate_mimo_provider(config)
            self.assertFalse(ready)
            assert error is not None
            self.assertIn("VOICELAYER_MIMO_TOKENIZER_PATH", error)

    def test_returns_true_when_paths_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model = pathlib.Path(tmp) / "model"
            tokenizer = pathlib.Path(tmp) / "tokenizer"
            model.mkdir()
            tokenizer.mkdir()
            config = MimoAsrConfig(
                model_path=str(model),
                tokenizer_path=str(tokenizer),
                repo_path=None,
                device="cuda:0",
                audio_tag=None,
                timeout_seconds=600.0,
                long_audio_split_seconds=180.0,
                extra_args=(),
            )
            ready, error = mimo_asr.validate_mimo_provider(config)
            self.assertTrue(ready)
            self.assertIsNone(error)


class ResolveAudioTagTest(unittest.TestCase):
    def _config(self, audio_tag: str | None) -> MimoAsrConfig:
        return MimoAsrConfig(
            model_path="/abs/model",
            tokenizer_path="/abs/tokenizer",
            repo_path=None,
            device="cuda:0",
            audio_tag=audio_tag,
            timeout_seconds=600.0,
            long_audio_split_seconds=180.0,
            extra_args=(),
        )

    def test_explicit_zh_routes_to_chinese_tag(self) -> None:
        for code in ("zh", "zh-CN", "zh-cn", "yue", "ZH"):
            with self.subTest(code=code):
                tag = mimo_asr._resolve_audio_tag(code, self._config(audio_tag=None))
                self.assertEqual(tag, "<chinese>")

    def test_explicit_en_routes_to_english_tag(self) -> None:
        for code in ("en", "en-US", "EN-GB"):
            with self.subTest(code=code):
                tag = mimo_asr._resolve_audio_tag(code, self._config(audio_tag=None))
                self.assertEqual(tag, "<english>")

    def test_auto_and_none_fall_back_to_config_default(self) -> None:
        # `auto` must collapse to the operator-configured default so
        # the wrapper can run its own auto-detect when the request
        # does not pin a language.
        config = self._config(audio_tag="<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag(None, config), "<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag("auto", config), "<chinese>")
        self.assertEqual(mimo_asr._resolve_audio_tag("  ", config), "<chinese>")

    def test_unknown_language_falls_back_to_default(self) -> None:
        # `ja` is not in the explicit map; with no configured default
        # we hand `None` to the wrapper so its built-in auto-detect
        # decides instead of receiving a meaningless tag.
        config = self._config(audio_tag=None)
        self.assertIsNone(mimo_asr._resolve_audio_tag("ja", config))


class SplitWavIntoSegmentsTest(unittest.TestCase):
    def test_single_chunk_short_audio_is_returned_unsplit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "short.wav"
            _write_silent_wav(audio, 1.0)
            segments = mimo_asr._split_wav_into_segments(audio, 180.0, tmp_path)
            self.assertEqual(segments, [audio])

    def test_long_audio_is_chunked_to_max_segment_seconds(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.5)
            segments = mimo_asr._split_wav_into_segments(audio, 2.0, tmp_path)
            # 5.5 s ÷ 2.0 s = 3 chunks (two full + one tail). Each
            # chunk is its own WAV file under the runtime dir; the
            # original is preserved on disk so the dictation pipeline
            # can clean it up later.
            self.assertEqual(len(segments), 3)
            for segment in segments:
                self.assertTrue(segment.is_file())
                with wave.open(str(segment), "rb") as wav:
                    self.assertEqual(wav.getnchannels(), 1)
                    self.assertEqual(wav.getframerate(), 16_000)

    def test_zero_max_segment_returns_single_chunk(self) -> None:
        # A misconfigured `0` (or negative) must not divide-by-zero;
        # bail to the original file so the request still goes through.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "any.wav"
            _write_silent_wav(audio, 4.0)
            segments = mimo_asr._split_wav_into_segments(audio, 0.0, tmp_path)
            self.assertEqual(segments, [audio])

    def test_invalid_wav_raises_provider_invocation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            bogus = tmp_path / "bogus.wav"
            bogus.write_bytes(b"not a wav at all")
            with self.assertRaises(ProviderInvocationError):
                mimo_asr._split_wav_into_segments(bogus, 60.0, tmp_path)

    def test_partial_chunks_are_cleaned_up_when_mid_write_fails(self) -> None:
        # A mid-loop failure (disk quota, SIGTERM-driven OSError, ...)
        # would otherwise leak every chunk written before the failure.
        # Patch `wave.open` to fail on the second write so the first
        # chunk file is on disk when the splitter raises; the splitter's
        # finally must roll the partial state back.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.0)
            runtime_dir = tmp_path / "runtime"
            runtime_dir.mkdir()

            real_wave_open = wave.open
            calls = {"count": 0}

            def faulty_wave_open(path: object, mode: str = "rb", *args: object, **kwargs: object):
                calls["count"] += 1
                # Call 1 = read input; call 2 = write chunk 0; call 3 =
                # write chunk 1 — fail there so chunk 0 is on disk and
                # the splitter has to clean it up.
                if mode == "wb" and calls["count"] >= 3:
                    raise OSError("simulated disk failure during chunk write")
                return real_wave_open(path, mode, *args, **kwargs)

            with (
                patch(
                    "voicelayer_orchestrator.providers.mimo_asr.wave.open",
                    side_effect=faulty_wave_open,
                ),
                self.assertRaises(ProviderInvocationError),
            ):
                mimo_asr._split_wav_into_segments(audio, 2.0, runtime_dir)

            leftover = sorted(p.name for p in runtime_dir.glob("mimo-segment-*.wav"))
            self.assertEqual(
                leftover,
                [],
                f"splitter must clean up partial chunks on mid-write failure: {leftover}",
            )


class TranscribeWithMimoTest(unittest.TestCase):
    def _config(self, tmp: pathlib.Path) -> MimoAsrConfig:
        model = tmp / "model"
        tokenizer = tmp / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        return MimoAsrConfig(
            model_path=str(model),
            tokenizer_path=str(tokenizer),
            repo_path=None,
            device="cuda:0",
            audio_tag=None,
            timeout_seconds=600.0,
            long_audio_split_seconds=180.0,
            extra_args=(),
        )

    def test_missing_audio_file_raises_invocation_error(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            config = self._config(pathlib.Path(tmp))
            with self.assertRaises(ProviderInvocationError):
                mimo_asr.transcribe_with_mimo({}, config)

    def test_translate_to_english_is_explicitly_rejected(self) -> None:
        # MiMo's `asr_sft` does not support translation; the worker
        # surfaces the limitation rather than silently dropping the
        # flag, so callers can route translation through the LLM
        # workflow instead of getting a transcript that lies about
        # what it did.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "sample.wav"
            _write_silent_wav(audio, 1.0)
            config = self._config(tmp_path)
            with self.assertRaisesRegex(ProviderInvocationError, "translate_to_english"):
                mimo_asr.transcribe_with_mimo(
                    {"audio_file": str(audio), "translate_to_english": True},
                    config,
                )

    def test_dispatches_each_segment_through_loaded_model(self) -> None:
        # Mock `_run_segment_inference` so we never actually import
        # torch / soundfile / Xiaomi's wrapper at test time. The
        # transcribe function must split the long audio, call the
        # inference fn once per segment with the correct audio_tag,
        # concatenate the per-segment outputs with a single space, and
        # surface the segment count on the notes list.
        #
        # The original draft of this test patched `_load_mimo_model`
        # only and let `_run_segment_inference` execute, which silently
        # required the `mimo` extra (torch + soundfile) to be installed
        # in the test environment. Patching at the inference boundary
        # keeps the default verification chain reliable on a fresh
        # `uv sync --group dev` install.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.0)
            config = MimoAsrConfig(
                model_path=str(tmp_path / "model"),
                tokenizer_path=str(tmp_path / "tokenizer"),
                repo_path=None,
                device="cuda:0",
                audio_tag=None,
                timeout_seconds=600.0,
                long_audio_split_seconds=2.0,
                extra_args=(),
            )
            (tmp_path / "model").mkdir()
            (tmp_path / "tokenizer").mkdir()

            calls: list[tuple[pathlib.Path, str | None]] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                audio_tag: str | None,
            ) -> str:
                calls.append((segment_path, audio_tag))
                return f"chunk-{len(calls)}"

            sentinel_model = object()
            with (
                patch.object(mimo_asr, "_load_mimo_model", return_value=sentinel_model),
                patch.object(mimo_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = mimo_asr.transcribe_with_mimo(
                    {"audio_file": str(audio), "language": "zh"},
                    config,
                )

            self.assertEqual(len(calls), 3)
            for _segment_path, audio_tag in calls:
                # `language=zh` must translate into the wrapper's
                # `<chinese>` audio_tag.
                self.assertEqual(audio_tag, "<chinese>")
            # The concatenated text uses a single space between
            # chunks and is normalized through the shared decorative
            # transcript collapser.
            self.assertEqual(result["text"], "chunk-1 chunk-2 chunk-3")
            # Detected language reflects the explicit `zh` request
            # since MiMo itself does not return a language code.
            self.assertEqual(result["detected_language"], "zh")
            # The notes surface the audio_tag, segment count, and the
            # split mitigation rationale.
            joined_notes = " | ".join(result["notes"])
            self.assertIn("<chinese>", joined_notes)
            self.assertIn("segments processed: 3", joined_notes)
            self.assertIn("issue #6", joined_notes)


def _vad_config_sentinel() -> WhisperVadConfig:
    # `WhisperVadConfig` is a frozen dataclass; populate every required
    # field with the documented defaults so the sentinel survives the
    # frozen-dataclass `__init__` validation.
    return WhisperVadConfig(
        model_path="/abs/silero.onnx",
        threshold=0.5,
        min_speech_ms=250,
        min_silence_ms=100,
        speech_pad_ms=30,
        max_segment_secs=30.0,
        sample_rate=16_000,
    )


class VadPrepassForMimoTest(unittest.TestCase):
    """Pin the silero-vad pre-pass wiring on the MiMo path.

    The whisper chain has applied silero-vad before transcribe since
    `VOICELAYER_WHISPER_VAD_ENABLED=1` was introduced; MiMo now
    honors the same configuration so an operator who turned the
    pre-pass on once gets it on both backends. Without these pins,
    a regression that bypassed `_apply_vad_prepass_for_mimo` would
    surface only as silent hallucinations on a silent recording —
    exactly the failure mode VAD-on-MiMo was added to prevent.
    """

    def _config(self, tmp: pathlib.Path) -> MimoAsrConfig:
        model = tmp / "model"
        tokenizer = tmp / "tokenizer"
        model.mkdir()
        tokenizer.mkdir()
        return MimoAsrConfig(
            model_path=str(model),
            tokenizer_path=str(tokenizer),
            repo_path=None,
            device="cuda:0",
            audio_tag=None,
            timeout_seconds=600.0,
            long_audio_split_seconds=180.0,
            extra_args=(),
        )

    def test_vad_unconfigured_passes_raw_audio_to_mimo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            audio = tmp_path / "raw.wav"
            _write_silent_wav(audio, 1.0)
            config = self._config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _audio_tag: str | None,
            ) -> str:
                inferred.append(segment_path)
                return "raw transcript"

            with (
                patch.object(mimo_asr, "load_whisper_vad_config", return_value=None),
                patch.object(mimo_asr, "_load_mimo_model", return_value=object()),
                patch.object(mimo_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = mimo_asr.transcribe_with_mimo({"audio_file": str(audio)}, config)

            # No VAD config → no trimmed file, no pre-pass note, MiMo
            # sees the original audio.
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
            # Use a real path under the test tmp dir rather than a
            # hard-coded `/tmp/...` string so the cleanup finally block
            # in `transcribe_with_mimo` cannot accidentally delete a
            # file that happens to exist on the host system. The file
            # is pre-created here so the cleanup assertion below can
            # observe a present-then-absent transition rather than a
            # always-absent one.
            trimmed_audio = tmp_path / "silent.vad-empty.wav"
            trimmed_audio.write_bytes(b"")
            config = self._config(tmp_path)

            with (
                patch.object(
                    mimo_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    mimo_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), []),
                ),
                patch.object(mimo_asr, "_load_mimo_model") as load_model,
                patch.object(mimo_asr, "_run_segment_inference") as run_inference,
            ):
                result = mimo_asr.transcribe_with_mimo({"audio_file": str(audio)}, config)

                # The whole point of the short-circuit is to skip the
                # MiMo cold load and the per-segment loop. Pin both.
                load_model.assert_not_called()
                run_inference.assert_not_called()

            self.assertEqual(result["text"], "")
            self.assertIsNone(result["detected_language"])
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD detected no speech", joined_notes)
            self.assertIn("MiMo inference was skipped", joined_notes)
            # The `.vad-empty.wav` sidecar that `apply_vad_prepass`
            # would have written must be cleaned up even though MiMo
            # short-circuited; otherwise every silent capture leaks a
            # file under `runtime_dir/vad/`.
            self.assertFalse(trimmed_audio.exists())

    def test_vad_trimmed_audio_replaces_input_for_mimo(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 5.0)
            trimmed_audio = tmp_path / "raw.vad-trimmed.wav"
            _write_silent_wav(trimmed_audio, 2.0)
            config = self._config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _audio_tag: str | None,
            ) -> str:
                inferred.append(segment_path)
                return "trimmed transcript"

            with (
                patch.object(
                    mimo_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    mimo_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), [(0.5, 1.5), (3.0, 4.5)]),
                ),
                patch.object(mimo_asr, "_load_mimo_model", return_value=object()),
                patch.object(mimo_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = mimo_asr.transcribe_with_mimo({"audio_file": str(raw_audio)}, config)

            # The model is invoked on the trimmed audio, not the
            # original capture; long-audio split runs over the
            # post-VAD WAV so the segments live under runtime_dir.
            self.assertEqual(len(inferred), 1)
            self.assertEqual(inferred[0], trimmed_audio)
            self.assertEqual(result["text"], "trimmed transcript")
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD pre-pass kept 2 speech region(s)", joined_notes)
            # Two regions of 1.0 s + 1.5 s = 2.50 s of detected speech.
            self.assertIn("2.50s total", joined_notes)
            # The trimmed WAV is owned by the worker; the finally block
            # in `transcribe_with_mimo` must delete it once inference
            # completes so `runtime_dir/vad/` does not accumulate one
            # file per VAD-enabled call.
            self.assertFalse(trimmed_audio.exists())
            # The caller-supplied raw WAV is *not* worker-owned; it
            # must still exist after transcribe so the dictation
            # pipeline can clean it up via `keep_audio` later.
            self.assertTrue(raw_audio.exists())

    def test_segment_wavs_are_cleaned_up_after_long_audio_split(self) -> None:
        # Mirrors the trimmed-WAV cleanup but for the per-chunk WAVs the
        # long-audio splitter writes when the input crosses
        # `long_audio_split_seconds`. Both the worker-owned chunks and
        # the trimmed VAD output (when present) live under
        # `provider_runtime_dir(environ)/...`; together they are the
        # full cleanup surface that the dispatcher must clear before
        # returning.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            runtime_root = tmp_path / "runtime"
            runtime_root.mkdir()
            audio = tmp_path / "long.wav"
            _write_silent_wav(audio, 5.0)
            config = MimoAsrConfig(
                model_path=str(tmp_path / "model"),
                tokenizer_path=str(tmp_path / "tokenizer"),
                repo_path=None,
                device="cuda:0",
                audio_tag=None,
                timeout_seconds=600.0,
                long_audio_split_seconds=2.0,
                extra_args=(),
            )
            (tmp_path / "model").mkdir()
            (tmp_path / "tokenizer").mkdir()

            with (
                patch.object(mimo_asr, "load_whisper_vad_config", return_value=None),
                patch.object(mimo_asr, "_load_mimo_model", return_value=object()),
                patch.object(mimo_asr, "_run_segment_inference", return_value="chunk"),
            ):
                mimo_asr.transcribe_with_mimo(
                    {"audio_file": str(audio)},
                    config,
                    environ={"XDG_RUNTIME_DIR": str(runtime_root)},
                )

            mimo_dir = runtime_root / "voicelayer" / "providers" / "mimo"
            # The splitter must have actually written its chunks under
            # `mimo_dir` for the test to be meaningful — otherwise the
            # cleanup assertion below trivially passes whenever the
            # split path was not exercised. With 5 s of audio and
            # `long_audio_split_seconds=2.0` the splitter creates 3
            # chunks before cleanup, so the directory must exist after
            # the call.
            self.assertTrue(
                mimo_dir.exists(),
                "splitter should have created the runtime dir",
            )
            # The directory itself stays (we never rmdir it), but every
            # chunk the splitter wrote must be gone.
            leftover = sorted(p.name for p in mimo_dir.iterdir())
            self.assertEqual(leftover, [])
            # The caller-supplied capture stays alive — only the worker-owned
            # chunk WAVs are cleaned up.
            self.assertTrue(audio.exists())

    def test_vad_trimmed_audio_is_cleaned_up_when_inference_fails(self) -> None:
        # Failure paths are the load-bearing case for try/finally
        # cleanup: a transcribe that raises must still leave
        # `runtime_dir/vad/` empty so a flaky model load cannot turn
        # into a slow disk leak.
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 1.0)
            trimmed_audio = tmp_path / "raw.vad-trimmed.wav"
            _write_silent_wav(trimmed_audio, 1.0)
            config = self._config(tmp_path)

            def failing_inference(*_args: object, **_kwargs: object) -> str:
                raise ProviderInvocationError("simulated MiMo failure")

            with (
                patch.object(
                    mimo_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    mimo_asr,
                    "apply_vad_prepass",
                    return_value=(str(trimmed_audio), [(0.0, 1.0)]),
                ),
                patch.object(mimo_asr, "_load_mimo_model", return_value=object()),
                patch.object(mimo_asr, "_run_segment_inference", side_effect=failing_inference),
                self.assertRaises(ProviderInvocationError),
            ):
                mimo_asr.transcribe_with_mimo({"audio_file": str(raw_audio)}, config)

            self.assertFalse(trimmed_audio.exists())
            # The original capture stays intact across failure too; the
            # caller is responsible for its lifetime.
            self.assertTrue(raw_audio.exists())

    def test_vad_failure_falls_back_to_raw_audio(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = pathlib.Path(tmp)
            raw_audio = tmp_path / "raw.wav"
            _write_silent_wav(raw_audio, 1.0)
            config = self._config(tmp_path)

            inferred: list[pathlib.Path] = []

            def fake_inference(
                _model: object,
                segment_path: pathlib.Path,
                _audio_tag: str | None,
            ) -> str:
                inferred.append(segment_path)
                return "raw transcript after vad failure"

            with (
                patch.object(
                    mimo_asr,
                    "load_whisper_vad_config",
                    return_value=_vad_config_sentinel(),
                ),
                patch.object(
                    mimo_asr,
                    "apply_vad_prepass",
                    side_effect=ProviderInvocationError("silero-vad onnx import failed"),
                ),
                patch.object(mimo_asr, "_load_mimo_model", return_value=object()),
                patch.object(mimo_asr, "_run_segment_inference", side_effect=fake_inference),
            ):
                result = mimo_asr.transcribe_with_mimo({"audio_file": str(raw_audio)}, config)

            # A VAD failure must not make transcribe go dark — the raw
            # WAV still reaches MiMo, and the failure is preserved as
            # a transparent note so the operator sees what happened.
            self.assertEqual(inferred, [raw_audio])
            self.assertEqual(result["text"], "raw transcript after vad failure")
            joined_notes = " | ".join(result["notes"])
            self.assertIn("VAD pre-pass failed", joined_notes)
            self.assertIn("silero-vad onnx import failed", joined_notes)


if __name__ == "__main__":
    unittest.main()
