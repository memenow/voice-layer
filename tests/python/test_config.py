"""Tests for unified worker configuration parsing."""

from __future__ import annotations

import unittest

from voicelayer_worker.config import (
    WorkerConfig,
    configure,
    current_config,
    is_configured,
    reset_configuration_for_tests,
)


class WorkerConfigTest(unittest.TestCase):
    def setUp(self) -> None:
        reset_configuration_for_tests()

    def tearDown(self) -> None:
        reset_configuration_for_tests()

    def test_empty_environment_yields_all_none_sections(self) -> None:
        cfg = WorkerConfig.from_environ({})
        self.assertIsNone(cfg.llm)
        self.assertIsNone(cfg.llama_launch)
        self.assertIsNone(cfg.whisper)
        self.assertIsNone(cfg.whisper_server)
        self.assertIsNone(cfg.vad)
        self.assertIsNone(cfg.mimo_asr)
        self.assertIsNone(cfg.qwen3_asr)

    def test_llm_and_llama_launch_parsing(self) -> None:
        cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_LLM_ENDPOINT": " http://127.0.0.1:8080 ",
                "VOICELAYER_LLM_MODEL": " gemma-4-local ",
                "VOICELAYER_LLM_API_KEY": "secret",
                "VOICELAYER_LLM_TIMEOUT_SECONDS": "42.5",
                "VOICELAYER_LLM_AUTO_START": "true",
                "VOICELAYER_LLAMA_SERVER_BIN": "/opt/llama-server",
                "VOICELAYER_LLAMA_MODEL_PATH": "/models/gemma.gguf",
                "VOICELAYER_LLAMA_SERVER_ARGS": "--ctx-size 4096 --threads 4",
                "VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS": "90",
                "VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS": "0.25",
            }
        )
        assert cfg.llm is not None
        assert cfg.llama_launch is not None
        self.assertEqual(cfg.llm.endpoint, "http://127.0.0.1:8080")
        self.assertEqual(cfg.llm.model, "gemma-4-local")
        self.assertEqual(cfg.llm.api_key, "secret")
        self.assertAlmostEqual(cfg.llm.timeout_seconds, 42.5)
        self.assertEqual(cfg.llama_launch.server_bin, "/opt/llama-server")
        self.assertEqual(cfg.llama_launch.model_path, "/models/gemma.gguf")
        self.assertEqual(cfg.llama_launch.extra_args, ("--ctx-size", "4096", "--threads", "4"))
        self.assertAlmostEqual(cfg.llama_launch.launch_timeout_seconds, 90.0)
        self.assertAlmostEqual(cfg.llama_launch.poll_interval_seconds, 0.25)

    def test_whisper_and_whisper_server_defaults(self) -> None:
        cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_WHISPER_MODEL_PATH": "/models/ggml-base.bin",
                "VOICELAYER_WHISPER_NO_GPU": "yes",
                "VOICELAYER_WHISPER_SERVER_AUTO_START": "1",
                "VOICELAYER_WHISPER_SERVER_BIN": "/usr/local/bin/whisper-server",
            }
        )
        assert cfg.whisper is not None
        assert cfg.whisper_server is not None
        self.assertEqual(cfg.whisper.binary, "whisper-cli")
        self.assertTrue(cfg.whisper.no_gpu)
        self.assertEqual(cfg.whisper_server.host, "127.0.0.1")
        self.assertEqual(cfg.whisper_server.port, 8188)
        self.assertEqual(cfg.whisper_server.model_path, "/models/ggml-base.bin")
        self.assertEqual(cfg.whisper_server.base_url, "http://127.0.0.1:8188")

    def test_vad_requires_both_enabled_and_model_path(self) -> None:
        self.assertIsNone(
            WorkerConfig.from_environ({"VOICELAYER_WHISPER_VAD_MODEL_PATH": "/tmp/vad.onnx"}).vad
        )
        self.assertIsNone(WorkerConfig.from_environ({"VOICELAYER_WHISPER_VAD_ENABLED": "true"}).vad)
        cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_WHISPER_VAD_ENABLED": "1",
                "VOICELAYER_WHISPER_VAD_MODEL_PATH": "/tmp/vad.onnx",
                "VOICELAYER_WHISPER_VAD_THRESHOLD": "0.42",
                "VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS": "200",
                "VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS": "150",
                "VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS": "50",
                "VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS": "15",
                "VOICELAYER_WHISPER_VAD_SAMPLE_RATE": "16000",
            }
        ).vad
        assert cfg is not None
        self.assertEqual(cfg.model_path, "/tmp/vad.onnx")
        self.assertAlmostEqual(cfg.threshold, 0.42)
        self.assertEqual(cfg.min_speech_ms, 200)
        self.assertEqual(cfg.min_silence_ms, 150)
        self.assertEqual(cfg.speech_pad_ms, 50)
        self.assertAlmostEqual(cfg.max_segment_secs, 15.0)
        self.assertEqual(cfg.sample_rate, 16000)

    def test_qwen3_dtype_supports_legacy_and_canonical_env_and_lowercases(self) -> None:
        legacy = WorkerConfig.from_environ(
            {
                "VOICELAYER_QWEN3_ASR_MODEL_PATH": "/models/qwen3",
                "VOICELAYER_QWEN3_ASR_DTYPE": "FLOAT16",
            }
        ).qwen3_asr
        assert legacy is not None
        self.assertEqual(legacy.torch_dtype, "float16")

        canonical = WorkerConfig.from_environ(
            {
                "VOICELAYER_QWEN3_ASR_MODEL_PATH": "/models/qwen3",
                "VOICELAYER_QWEN3_ASR_DTYPE": "float32",
                "VOICELAYER_QWEN3_ASR_TORCH_DTYPE": "BFloat16",
            }
        ).qwen3_asr
        assert canonical is not None
        self.assertEqual(canonical.torch_dtype, "bfloat16")

    def test_mimo_requires_both_model_and_tokenizer_paths(self) -> None:
        self.assertIsNone(WorkerConfig.from_environ({"VOICELAYER_MIMO_MODEL_PATH": "/m"}).mimo_asr)
        cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_MIMO_MODEL_PATH": "/m",
                "VOICELAYER_MIMO_TOKENIZER_PATH": "/t",
                "VOICELAYER_MIMO_REPO_PATH": "   ",
                "VOICELAYER_MIMO_AUDIO_TAG": "",
                "VOICELAYER_MIMO_DEVICE": "cuda:1",
                "VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS": "120",
            }
        ).mimo_asr
        assert cfg is not None
        self.assertIsNone(cfg.repo_path)
        self.assertIsNone(cfg.audio_tag)
        self.assertEqual(cfg.device, "cuda:1")
        self.assertAlmostEqual(cfg.long_audio_split_seconds, 120.0)

    def test_configure_stores_and_resets_process_snapshot(self) -> None:
        self.assertFalse(is_configured())
        with self.assertRaises(RuntimeError):
            current_config()
        configure({"whisper": {"model_path": "/models/base.bin"}})
        self.assertTrue(is_configured())
        assert current_config().whisper is not None
        self.assertEqual(current_config().whisper.model_path, "/models/base.bin")

    def test_malformed_env_numbers_and_zero_payload_values_fall_back_safely(self) -> None:
        env_cfg = WorkerConfig.from_environ(
            {
                "VOICELAYER_WHISPER_MODEL_PATH": "/models/base.bin",
                "VOICELAYER_WHISPER_TIMEOUT_SECONDS": "not-a-float",
                "VOICELAYER_WHISPER_VAD_ENABLED": "true",
                "VOICELAYER_WHISPER_VAD_MODEL_PATH": "/models/vad.onnx",
                "VOICELAYER_WHISPER_VAD_SAMPLE_RATE": "oops",
            }
        )
        assert env_cfg.whisper is not None
        assert env_cfg.vad is not None
        self.assertEqual(env_cfg.whisper.timeout_seconds, 300.0)
        self.assertEqual(env_cfg.vad.sample_rate, 16000)

        zero_cfg = WorkerConfig.from_payload(
            {
                "llm": {
                    "endpoint": "http://127.0.0.1:8080",
                    "model": "gemma",
                    "timeout_seconds": 0.0,
                },
                "whisper": {"model_path": "/models/base.bin", "timeout_seconds": 0.0},
                "vad": {
                    "enabled": True,
                    "model_path": "/models/vad.onnx",
                    "threshold": 0.0,
                    "min_speech_ms": 0,
                    "sample_rate": 0,
                },
            }
        )
        assert zero_cfg.llm is not None
        assert zero_cfg.whisper is not None
        assert zero_cfg.vad is not None
        self.assertEqual(zero_cfg.llm.timeout_seconds, 60.0)
        self.assertEqual(zero_cfg.whisper.timeout_seconds, 300.0)
        self.assertEqual(zero_cfg.vad.threshold, 0.5)
        self.assertEqual(zero_cfg.vad.min_speech_ms, 250)
        self.assertEqual(zero_cfg.vad.sample_rate, 16000)
