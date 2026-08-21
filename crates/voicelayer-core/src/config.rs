//! Unified VoiceLayer configuration.
//!
//! A single TOML file (default: platform config dir, e.g.
//! `~/.config/voicelayer/config.toml` on Linux) is the source of truth for
//! the daemon, the worker, and the CLI. `VOICELAYER_*` environment variables
//! act as an explicit override layer applied after the file is loaded.
//!
//! The daemon serializes the provider-facing sections (`llm`, `whisper`,
//! `whisper_server`, `vad`) into the worker `initialize` handshake payload;
//! the Python worker never reads the environment directly.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

/// Directory for sockets, recordings, and provider state.
pub fn default_runtime_dir() -> PathBuf {
    std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir)
        .join("voicelayer")
}

pub fn default_socket_path() -> PathBuf {
    default_runtime_dir().join("daemon.sock")
}

pub fn default_project_root() -> PathBuf {
    std::env::var_os("VOICELAYER_PROJECT_ROOT")
        .map(PathBuf::from)
        .or_else(|| std::env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn config_path() -> Result<PathBuf, ConfigError> {
    let project_dirs = directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .ok_or(ConfigError::NoConfigDir)?;
    Ok(project_dirs.config_dir().join("config.toml"))
}

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

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VoiceLayerConfig {
    pub daemon: DaemonSettings,
    pub llm: LlmSettings,
    pub whisper: WhisperSettings,
    pub whisper_server: WhisperServerSettings,
    pub vad: VadSettings,
    pub foreground_ptt: ForegroundPttSettings,
}

impl VoiceLayerConfig {
    /// Load the config file (or defaults when absent) and apply the
    /// `VOICELAYER_*` environment override layer.
    pub fn load() -> Result<Self, ConfigError> {
        let path = config_path()?;
        Self::load_from(&path)
    }

    pub fn load_from(path: &std::path::Path) -> Result<Self, ConfigError> {
        let mut config = if path.is_file() {
            toml::from_str::<Self>(&std::fs::read_to_string(path)?)?
        } else {
            Self::default()
        };
        config.apply_env_overrides();
        Ok(config)
    }

    /// The subset handed to the Python worker in the `initialize` handshake.
    pub fn worker_payload(&self) -> WorkerInitPayload {
        WorkerInitPayload {
            llm: self.llm.clone(),
            whisper: self.whisper.clone(),
            whisper_server: self.whisper_server.clone(),
            vad: self.vad.clone(),
        }
    }

    pub fn apply_env_overrides(&mut self) {
        let env = Env;

        env.string("VOICELAYER_SOCKET_PATH", &mut self.daemon.socket_path_raw);
        env.string("VOICELAYER_PROJECT_ROOT", &mut self.daemon.project_root);
        env.u64(
            "VOICELAYER_WORKER_TIMEOUT_SECONDS",
            &mut self.daemon.worker_timeout_seconds,
        );

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
}

/// Provider-facing configuration delivered to the worker at `initialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInitPayload {
    pub llm: LlmSettings,
    pub whisper: WhisperSettings,
    pub whisper_server: WhisperServerSettings,
    pub vad: VadSettings,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonSettings {
    /// Raw string form so `VOICELAYER_SOCKET_PATH` and TOML share a field;
    /// resolved via [`DaemonSettings::socket_path`].
    pub socket_path_raw: Option<String>,
    /// Repository root used to locate the Python worker. Essential for
    /// service-launched daemons (systemd/launchd start with cwd=`/`).
    pub project_root: Option<String>,
    pub worker_timeout_seconds: u64,
}

impl Default for DaemonSettings {
    fn default() -> Self {
        Self {
            socket_path_raw: None,
            project_root: None,
            worker_timeout_seconds: 600,
        }
    }
}

impl DaemonSettings {
    pub fn socket_path(&self) -> PathBuf {
        self.socket_path_raw
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(default_socket_path)
    }

    pub fn project_root(&self) -> PathBuf {
        self.project_root
            .as_deref()
            .map(PathBuf::from)
            .unwrap_or_else(default_project_root)
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct LlmSettings {
    pub endpoint: Option<String>,
    pub model: Option<String>,
    pub api_key: Option<String>,
    pub timeout_seconds: f64,
    pub auto_start: bool,
    pub server_bin: Option<String>,
    pub model_path: Option<String>,
    pub hf_repo: Option<String>,
    pub server_args: Option<String>,
    pub launch_timeout_seconds: f64,
    pub poll_interval_seconds: f64,
}

impl LlmSettings {
    pub fn is_configured(&self) -> bool {
        self.endpoint.is_some() && self.model.is_some()
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WhisperSettings {
    pub model_path: Option<String>,
    pub binary: Option<String>,
    pub timeout_seconds: f64,
    pub no_gpu: bool,
    pub extra_args: Option<String>,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WhisperServerSettings {
    pub host: Option<String>,
    pub port: Option<u16>,
    pub timeout_seconds: f64,
    pub auto_start: bool,
    pub server_bin: Option<String>,
    pub extra_args: Option<String>,
    pub launch_timeout_seconds: f64,
    pub poll_interval_seconds: f64,
}

impl WhisperServerSettings {
    /// The server path is in play when any server knob is set, matching the
    /// worker's previous environment-based semantics.
    pub fn is_in_play(&self) -> bool {
        self.host.is_some() || self.port.is_some() || self.server_bin.is_some() || self.auto_start
    }
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct VadSettings {
    pub enabled: bool,
    pub model_path: Option<String>,
    pub threshold: f64,
    pub min_speech_ms: u32,
    pub min_silence_ms: u32,
    pub speech_pad_ms: u32,
    pub max_segment_secs: f64,
    pub sample_rate: u32,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopAction {
    #[default]
    None,
    Copy,
    Inject,
    Save,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForegroundPttSettings {
    pub language: Option<String>,
    pub translate_to_english: bool,
    pub keep_audio: bool,
    /// Key name: one of space, enter, tab, f8, f9, f10.
    pub key: String,
    pub tmux_target_pane: Option<String>,
    pub wezterm_target_pane_id: Option<String>,
    pub kitty_match: Option<String>,
    pub copy_on_stop: bool,
    pub default_stop_action: StopAction,
    pub restore_clipboard_on_exit: bool,
    pub save_dir: Option<String>,
}

impl Default for ForegroundPttSettings {
    fn default() -> Self {
        Self {
            language: None,
            translate_to_english: false,
            keep_audio: false,
            key: "space".to_owned(),
            tmux_target_pane: None,
            wezterm_target_pane_id: None,
            kitty_match: None,
            copy_on_stop: false,
            default_stop_action: StopAction::None,
            restore_clipboard_on_exit: false,
            save_dir: None,
        }
    }
}

/// Reader for the `VOICELAYER_*` override layer. Each setter leaves the
/// target untouched when the variable is unset or unparsable.
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
    use super::VoiceLayerConfig;

    #[test]
    fn default_config_produces_unconfigured_worker_payload() {
        let payload = VoiceLayerConfig::default().worker_payload();
        assert!(!payload.llm.is_configured());
        assert!(!payload.whisper_server.is_in_play());
        assert!(!payload.vad.enabled);
    }

    #[test]
    fn config_round_trips_through_toml() {
        let mut config = VoiceLayerConfig::default();
        config.llm.endpoint = Some("http://127.0.0.1:8080".to_owned());
        config.whisper.model_path = Some("/models/ggml-base.bin".to_owned());
        let encoded = toml::to_string_pretty(&config).unwrap();
        let decoded: VoiceLayerConfig = toml::from_str(&encoded).unwrap();
        assert_eq!(
            decoded.llm.endpoint.as_deref(),
            config.llm.endpoint.as_deref()
        );
        assert_eq!(
            decoded.whisper.model_path.as_deref(),
            config.whisper.model_path.as_deref()
        );
    }
}
