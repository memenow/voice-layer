//! Unified VoiceLayer configuration loaded from `config.toml` with
//! `VOICELAYER_*` environment variable overrides.

mod settings;

use std::path::{Path, PathBuf};

use serde::{Deserialize, Serialize};

pub use settings::{
    DaemonSettings, ForegroundPttSettings, LlmSettings, MimoAsrSettings, Qwen3AsrSettings,
    StopAction, VadSettings, WhisperServerSettings, WhisperSettings, WorkerInitPayload,
};

/// Directory for runtime sockets, temporary recordings, and provider locks.
pub fn default_runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("voicelayer")
}

/// Default daemon control socket (`daemon.sock` under [`default_runtime_dir`]).
pub fn default_socket_path() -> PathBuf {
    default_runtime_dir().join("daemon.sock")
}

/// Default repository root used to locate the Python worker.
pub fn default_project_root() -> PathBuf {
    std::env::var_os("VOICELAYER_PROJECT_ROOT")
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

/// Resolve the platform `config.toml` path (for example `~/.config/voicelayer/config.toml` on Linux).
pub fn config_path() -> Result<PathBuf, ConfigError> {
    let project_dirs = directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .ok_or(ConfigError::NoConfigDir)?;
    Ok(project_dirs.config_dir().join("config.toml"))
}

/// Errors encountered while locating, reading, parsing, or writing `config.toml`.
#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("unable to determine the platform config directory for VoiceLayer")]
    NoConfigDir,
    #[error("failed to read config file: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to parse config TOML: {0}")]
    Parse(#[from] toml::de::Error),
    #[error("failed to encode config TOML: {0}")]
    Encode(#[from] toml::ser::Error),
}

/// Root configuration shared by the daemon, Python worker, and CLI.
/// Unknown keys are rejected at parse time via `deny_unknown_fields`.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceLayerConfig {
    pub daemon: DaemonSettings,
    pub llm: LlmSettings,
    pub whisper: WhisperSettings,
    pub whisper_server: WhisperServerSettings,
    pub vad: VadSettings,
    pub mimo_asr: MimoAsrSettings,
    pub qwen3_asr: Qwen3AsrSettings,
    pub foreground_ptt: ForegroundPttSettings,
}

impl VoiceLayerConfig {
    /// Load `config.toml` from the platform config directory (or defaults when absent)
    /// and apply `VOICELAYER_*` environment overrides.
    pub fn load() -> Result<Self, ConfigError> {
        let path = config_path()?;
        Self::load_from(&path)
    }

    /// Build default settings and apply `VOICELAYER_*` environment overrides.
    /// Useful as a fallback when `config.toml` fails to parse so env settings still apply.
    pub fn default_with_env() -> Self {
        let mut config = Self::default();
        config.apply_env_overrides();
        config
    }

    /// Load `config.toml` from `path` without applying environment overrides.
    pub fn load_file_only(path: &Path) -> Result<Self, ConfigError> {
        if path.is_file() {
            Ok(toml::from_str::<Self>(&std::fs::read_to_string(path)?)?)
        } else {
            Ok(Self::default())
        }
    }

    /// Load `config.toml` from `path` (or defaults when absent) and apply environment overrides.
    pub fn load_from(path: &Path) -> Result<Self, ConfigError> {
        let mut config = Self::load_file_only(path)?;
        config.apply_env_overrides();
        Ok(config)
    }

    /// Persist this configuration as formatted TOML to `path`, creating parent directories.
    pub fn save_to(&self, path: &Path) -> Result<(), ConfigError> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let content = toml::to_string_pretty(self)?;
        std::fs::write(path, content)?;
        Ok(())
    }

    /// Serialize provider-facing settings for the worker `initialize` handshake.
    pub fn worker_payload(&self) -> WorkerInitPayload {
        WorkerInitPayload {
            llm: self.llm.clone(),
            whisper: self.whisper.clone(),
            whisper_server: self.whisper_server.clone(),
            vad: self.vad.clone(),
            mimo_asr: self.mimo_asr.clone(),
            qwen3_asr: self.qwen3_asr.clone(),
        }
    }

    /// Apply `VOICELAYER_*` environment overrides in place.
    /// Unset or unparsable variables leave their target field unchanged.
    pub fn apply_env_overrides(&mut self) {
        let env = Env;
        self.apply_daemon_env(&env);
        self.apply_llm_env(&env);
        self.apply_whisper_env(&env);
        self.apply_whisper_server_env(&env);
        self.apply_vad_env(&env);
        self.apply_mimo_env(&env);
        self.apply_qwen3_env(&env);
    }

    fn apply_daemon_env(&mut self, env: &Env) {
        env.string("VOICELAYER_SOCKET_PATH", &mut self.daemon.socket_path_raw);
        env.string("VOICELAYER_PROJECT_ROOT", &mut self.daemon.project_root);
        env.u64(
            "VOICELAYER_WORKER_TIMEOUT_SECONDS",
            &mut self.daemon.worker_timeout_seconds,
        );
    }

    fn apply_llm_env(&mut self, env: &Env) {
        env.string("VOICELAYER_LLM_ENDPOINT", &mut self.llm.endpoint);
        env.string("VOICELAYER_LLM_MODEL", &mut self.llm.model);
        env.string("VOICELAYER_LLM_API_KEY", &mut self.llm.api_key);
        env.f64(
            "VOICELAYER_LLM_TIMEOUT_SECONDS",
            &mut self.llm.timeout_seconds,
        );
        env.flag("VOICELAYER_LLM_AUTO_START", &mut self.llm.auto_start);
        env.string("VOICELAYER_LLAMA_SERVER_BIN", &mut self.llm.server_bin);
        env.string("VOICELAYER_LLAMA_MODEL_PATH", &mut self.llm.model_path);
        env.string("VOICELAYER_LLAMA_HF_REPO", &mut self.llm.hf_repo);
        env.string("VOICELAYER_LLAMA_SERVER_ARGS", &mut self.llm.server_args);
        env.f64(
            "VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS",
            &mut self.llm.launch_timeout_seconds,
        );
        env.f64(
            "VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS",
            &mut self.llm.poll_interval_seconds,
        );
    }

    fn apply_whisper_env(&mut self, env: &Env) {
        env.string(
            "VOICELAYER_WHISPER_MODEL_PATH",
            &mut self.whisper.model_path,
        );
        env.string("VOICELAYER_WHISPER_BIN", &mut self.whisper.binary);
        env.f64(
            "VOICELAYER_WHISPER_TIMEOUT_SECONDS",
            &mut self.whisper.timeout_seconds,
        );
        env.flag("VOICELAYER_WHISPER_NO_GPU", &mut self.whisper.no_gpu);
        env.string("VOICELAYER_WHISPER_ARGS", &mut self.whisper.extra_args);
    }

    fn apply_whisper_server_env(&mut self, env: &Env) {
        env.string(
            "VOICELAYER_WHISPER_SERVER_HOST",
            &mut self.whisper_server.host,
        );
        env.u16(
            "VOICELAYER_WHISPER_SERVER_PORT",
            &mut self.whisper_server.port,
        );
        env.f64(
            "VOICELAYER_WHISPER_SERVER_TIMEOUT_SECONDS",
            &mut self.whisper_server.timeout_seconds,
        );
        env.flag(
            "VOICELAYER_WHISPER_SERVER_AUTO_START",
            &mut self.whisper_server.auto_start,
        );
        env.string(
            "VOICELAYER_WHISPER_SERVER_BIN",
            &mut self.whisper_server.server_bin,
        );
        env.string(
            "VOICELAYER_WHISPER_SERVER_ARGS",
            &mut self.whisper_server.extra_args,
        );
        env.f64(
            "VOICELAYER_WHISPER_SERVER_LAUNCH_TIMEOUT_SECONDS",
            &mut self.whisper_server.launch_timeout_seconds,
        );
        env.f64(
            "VOICELAYER_WHISPER_SERVER_POLL_INTERVAL_SECONDS",
            &mut self.whisper_server.poll_interval_seconds,
        );
    }

    fn apply_vad_env(&mut self, env: &Env) {
        env.flag("VOICELAYER_WHISPER_VAD_ENABLED", &mut self.vad.enabled);
        env.string(
            "VOICELAYER_WHISPER_VAD_MODEL_PATH",
            &mut self.vad.model_path,
        );
        env.f64("VOICELAYER_WHISPER_VAD_THRESHOLD", &mut self.vad.threshold);
        env.u32(
            "VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS",
            &mut self.vad.min_speech_ms,
        );
        env.u32(
            "VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS",
            &mut self.vad.min_silence_ms,
        );
        env.u32(
            "VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS",
            &mut self.vad.speech_pad_ms,
        );
        env.f64(
            "VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS",
            &mut self.vad.max_segment_secs,
        );
        env.u32(
            "VOICELAYER_WHISPER_VAD_SAMPLE_RATE",
            &mut self.vad.sample_rate,
        );
    }

    fn apply_mimo_env(&mut self, env: &Env) {
        env.string("VOICELAYER_MIMO_MODEL_PATH", &mut self.mimo_asr.model_path);
        env.string(
            "VOICELAYER_MIMO_TOKENIZER_PATH",
            &mut self.mimo_asr.tokenizer_path,
        );
        env.string("VOICELAYER_MIMO_REPO_PATH", &mut self.mimo_asr.repo_path);
        env.string("VOICELAYER_MIMO_DEVICE", &mut self.mimo_asr.device);
        env.string("VOICELAYER_MIMO_AUDIO_TAG", &mut self.mimo_asr.audio_tag);
        env.f64(
            "VOICELAYER_MIMO_TIMEOUT_SECONDS",
            &mut self.mimo_asr.timeout_seconds,
        );
        env.f64(
            "VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS",
            &mut self.mimo_asr.long_audio_split_seconds,
        );
        env.string("VOICELAYER_MIMO_ARGS", &mut self.mimo_asr.extra_args);
    }

    fn apply_qwen3_env(&mut self, env: &Env) {
        env.string(
            "VOICELAYER_QWEN3_ASR_MODEL_PATH",
            &mut self.qwen3_asr.model_path,
        );
        env.string("VOICELAYER_QWEN3_ASR_DEVICE", &mut self.qwen3_asr.device);
        // Apply legacy alias first so the canonical spelling wins when both are set.
        env.string(
            "VOICELAYER_QWEN3_ASR_DTYPE",
            &mut self.qwen3_asr.torch_dtype,
        );
        env.string(
            "VOICELAYER_QWEN3_ASR_TORCH_DTYPE",
            &mut self.qwen3_asr.torch_dtype,
        );
        env.f64(
            "VOICELAYER_QWEN3_ASR_TIMEOUT_SECONDS",
            &mut self.qwen3_asr.timeout_seconds,
        );
        env.f64(
            "VOICELAYER_QWEN3_ASR_LONG_AUDIO_SPLIT_SECONDS",
            &mut self.qwen3_asr.long_audio_split_seconds,
        );
        env.string("VOICELAYER_QWEN3_ASR_ARGS", &mut self.qwen3_asr.extra_args);
    }
}

struct Env;

impl Env {
    fn get(&self, key: &str) -> Option<String> {
        std::env::var(key).ok().filter(|value| !value.is_empty())
    }

    fn string(&self, key: &str, target: &mut Option<String>) {
        if let Some(value) = self.get(key) {
            *target = Some(value);
        }
    }

    fn flag(&self, key: &str, target: &mut bool) {
        if let Some(value) = self.get(key) {
            *target = matches!(
                value.trim().to_ascii_lowercase().as_str(),
                "1" | "true" | "yes" | "on"
            );
        }
    }

    fn u64(&self, key: &str, target: &mut u64) {
        if let Some(value) = self.get(key).and_then(|v| v.parse().ok()) {
            *target = value;
        }
    }

    fn u16(&self, key: &str, target: &mut Option<u16>) {
        if let Some(value) = self.get(key).and_then(|v| v.parse().ok()) {
            *target = Some(value);
        }
    }

    fn u32(&self, key: &str, target: &mut u32) {
        if let Some(value) = self.get(key).and_then(|v| v.parse().ok()) {
            *target = value;
        }
    }

    fn f64(&self, key: &str, target: &mut f64) {
        if let Some(value) = self.get(key).and_then(|v| v.parse().ok()) {
            *target = value;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use voicelayer_testkit::EnvGuard;

    use super::VoiceLayerConfig;

    #[test]
    fn default_config_uses_sane_nonzero_provider_defaults() {
        let payload = VoiceLayerConfig::default().worker_payload();
        assert!(!payload.llm.is_configured());
        assert!(!payload.whisper_server.is_in_play());
        assert!(!payload.vad.enabled);
        assert_eq!(payload.llm.timeout_seconds, 60.0);
        assert_eq!(payload.llm.launch_timeout_seconds, 45.0);
        assert_eq!(payload.llm.poll_interval_seconds, 0.5);
        assert_eq!(payload.whisper.timeout_seconds, 300.0);
        assert_eq!(payload.whisper_server.timeout_seconds, 60.0);
        assert_eq!(payload.whisper_server.launch_timeout_seconds, 30.0);
        assert_eq!(payload.whisper_server.poll_interval_seconds, 0.5);
        assert_eq!(payload.vad.threshold, 0.5);
        assert_eq!(payload.vad.min_speech_ms, 250);
        assert_eq!(payload.vad.min_silence_ms, 100);
        assert_eq!(payload.vad.speech_pad_ms, 30);
        assert_eq!(payload.vad.max_segment_secs, 30.0);
        assert_eq!(payload.vad.sample_rate, 16000);
        assert_eq!(payload.mimo_asr.timeout_seconds, 600.0);
        assert_eq!(payload.mimo_asr.long_audio_split_seconds, 180.0);
        assert_eq!(payload.qwen3_asr.timeout_seconds, 600.0);
    }

    #[test]
    fn config_round_trips_through_toml_and_accepts_example_keys() {
        let mut config = VoiceLayerConfig::default();
        config.daemon.socket_path_raw = Some("/tmp/vl-test.sock".to_owned());
        config.llm.endpoint = Some("http://127.0.0.1:8080".to_owned());
        config.whisper.model_path = Some("/models/ggml-base.bin".to_owned());
        let encoded = toml::to_string_pretty(&config).unwrap();
        assert!(encoded.contains("socket_path = \"/tmp/vl-test.sock\""));
        let decoded: VoiceLayerConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.daemon.socket_path(),
            PathBuf::from("/tmp/vl-test.sock")
        );

        let example_path = PathBuf::from(format!(
            "{}/../../config/config.toml.example",
            env!("CARGO_MANIFEST_DIR")
        ));
        let raw = std::fs::read_to_string(&example_path).expect("read config.toml.example");
        let uncommented: String = raw
            .lines()
            .filter_map(|line| {
                let trimmed = line.trim();
                if trimmed.starts_with('[') {
                    return Some(trimmed.to_owned());
                }
                let rest = trimmed.strip_prefix("# ")?;
                rest.contains('=').then(|| rest.to_owned())
            })
            .collect::<Vec<_>>()
            .join("\n");
        let parsed: VoiceLayerConfig =
            toml::from_str(&uncommented).expect("example config.toml must parse cleanly");
        assert_eq!(
            parsed.daemon.socket_path(),
            PathBuf::from("/run/user/1000/voicelayer/daemon.sock")
        );
    }

    #[test]
    fn env_overrides_populate_mimo_and_qwen3_in_worker_payload() {
        let mut env = EnvGuard::lock();
        env.set("VOICELAYER_MIMO_MODEL_PATH", "/models/mimo");
        env.set("VOICELAYER_MIMO_TOKENIZER_PATH", "/models/mimo-tok");
        env.set("VOICELAYER_QWEN3_ASR_MODEL_PATH", "/models/qwen3");
        env.set("VOICELAYER_QWEN3_ASR_DTYPE", "float16");
        env.set("VOICELAYER_QWEN3_ASR_TORCH_DTYPE", "bfloat16");

        let payload = VoiceLayerConfig::default_with_env().worker_payload();
        assert_eq!(payload.mimo_asr.model_path.as_deref(), Some("/models/mimo"));
        assert_eq!(
            payload.mimo_asr.tokenizer_path.as_deref(),
            Some("/models/mimo-tok")
        );
        assert_eq!(
            payload.qwen3_asr.model_path.as_deref(),
            Some("/models/qwen3")
        );
        assert_eq!(payload.qwen3_asr.torch_dtype.as_deref(), Some("bfloat16"));
    }
}
