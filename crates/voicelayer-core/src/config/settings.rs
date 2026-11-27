//! Configuration section structs and defaults shared across `voicelayerd`, `vl`, and the worker.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use super::{default_project_root, default_socket_path};

/// Provider settings delivered to the Python worker during `initialize`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WorkerInitPayload {
    pub llm: LlmSettings,
    pub whisper: WhisperSettings,
    pub whisper_server: WhisperServerSettings,
    pub vad: VadSettings,
    pub mimo_asr: MimoAsrSettings,
    pub qwen3_asr: Qwen3AsrSettings,
}

/// Daemon process settings: control socket path, worker project root, and RPC timeout.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct DaemonSettings {
    /// Raw socket path override (`socket_path` in TOML); resolved via [`DaemonSettings::socket_path`].
    #[serde(rename = "socket_path")]
    pub socket_path_raw: Option<String>,
    /// Repository root containing the Python worker; required under systemd/launchd.
    pub project_root: Option<String>,
    /// Timeout budget in seconds for worker inference JSON-RPC calls (default 600).
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
    /// Resolved socket path, falling back to [`default_socket_path`].
    pub fn socket_path(&self) -> PathBuf {
        self.socket_path_raw
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_socket_path)
    }

    /// Resolved project root, falling back to [`default_project_root`].
    pub fn project_root(&self) -> PathBuf {
        self.project_root
            .as_deref()
            .filter(|value| !value.trim().is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(default_project_root)
    }
}

/// OpenAI-compatible LLM endpoint and optional `llama-server` autostart settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for LlmSettings {
    fn default() -> Self {
        Self {
            endpoint: None,
            model: None,
            api_key: None,
            timeout_seconds: 60.0,
            auto_start: false,
            server_bin: None,
            model_path: None,
            hf_repo: None,
            server_args: None,
            launch_timeout_seconds: 45.0,
            poll_interval_seconds: 0.5,
        }
    }
}

impl LlmSettings {
    /// Returns `true` when both `endpoint` and `model` are configured.
    pub fn is_configured(&self) -> bool {
        self.endpoint.is_some() && self.model.is_some()
    }
}

/// One-shot `whisper-cli` subprocess settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct WhisperSettings {
    pub model_path: Option<String>,
    pub binary: Option<String>,
    pub timeout_seconds: f64,
    pub no_gpu: bool,
    pub extra_args: Option<String>,
}

impl Default for WhisperSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            binary: None,
            timeout_seconds: 300.0,
            no_gpu: false,
            extra_args: None,
        }
    }
}

/// Persistent `whisper-server` HTTP endpoint and autostart settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for WhisperServerSettings {
    fn default() -> Self {
        Self {
            host: None,
            port: None,
            timeout_seconds: 60.0,
            auto_start: false,
            server_bin: None,
            extra_args: None,
            launch_timeout_seconds: 30.0,
            poll_interval_seconds: 0.5,
        }
    }
}

impl WhisperServerSettings {
    /// Returns `true` when any server setting is configured.
    pub fn is_in_play(&self) -> bool {
        self.host.is_some() || self.port.is_some() || self.server_bin.is_some() || self.auto_start
    }
}

/// Silero VAD pre-pass settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
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

impl Default for VadSettings {
    fn default() -> Self {
        Self {
            enabled: false,
            model_path: None,
            threshold: 0.5,
            min_speech_ms: 250,
            min_silence_ms: 100,
            speech_pad_ms: 30,
            max_segment_secs: 30.0,
            sample_rate: 16000,
        }
    }
}

/// Optional Xiaomi MiMo-V2.5-ASR CUDA provider settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct MimoAsrSettings {
    pub model_path: Option<String>,
    pub tokenizer_path: Option<String>,
    pub repo_path: Option<String>,
    pub device: Option<String>,
    pub audio_tag: Option<String>,
    pub timeout_seconds: f64,
    pub long_audio_split_seconds: f64,
    pub extra_args: Option<String>,
}

impl Default for MimoAsrSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            tokenizer_path: None,
            repo_path: None,
            device: None,
            audio_tag: None,
            timeout_seconds: 600.0,
            long_audio_split_seconds: 180.0,
            extra_args: None,
        }
    }
}

/// Optional Alibaba Qwen3-ASR-1.7B provider settings.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Qwen3AsrSettings {
    pub model_path: Option<String>,
    pub device: Option<String>,
    pub torch_dtype: Option<String>,
    pub timeout_seconds: f64,
    pub long_audio_split_seconds: f64,
    pub extra_args: Option<String>,
}

impl Default for Qwen3AsrSettings {
    fn default() -> Self {
        Self {
            model_path: None,
            device: None,
            torch_dtype: None,
            timeout_seconds: 600.0,
            long_audio_split_seconds: 0.0,
            extra_args: None,
        }
    }
}

/// Action taken when `vl dictation foreground-ptt` completes a capture.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StopAction {
    #[default]
    None,
    Copy,
    Inject,
    Save,
}

/// Defaults for `vl dictation foreground-ptt`.
#[derive(Debug, Clone, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct ForegroundPttSettings {
    pub language: Option<String>,
    pub translate_to_english: bool,
    pub keep_audio: bool,
    /// Hold-to-talk key name: `space`, `enter`, `tab`, `f8`, `f9`, or `f10`.
    pub key: String,
    pub tmux_target_pane: Option<String>,
    pub wezterm_target_pane_id: Option<String>,
    pub kitty_match: Option<String>,
    pub copy_on_stop: bool,
    pub default_stop_action: StopAction,
    pub restore_clipboard_on_exit: bool,
    pub save_dir: Option<String>,
    pub provider_id: Option<String>,
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
            provider_id: None,
        }
    }
}
