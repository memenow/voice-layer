//! Shared domain types, configuration, injection planning, and optional
//! Unix-socket client for VoiceLayer.
//!
//! Public serde representations in [`domain`] and [`provider`] define the `/v1`
//! control API contract mirrored in `openapi/voicelayerd.v1.yaml`.

#[cfg(feature = "client")]
pub mod client;
pub mod config;
pub mod domain;
pub mod injection;
pub mod provider;

pub use config::{
    ConfigError, DaemonSettings, ForegroundPttSettings, LlmSettings, MimoAsrSettings,
    Qwen3AsrSettings, StopAction, VadSettings, VoiceLayerConfig, WhisperServerSettings,
    WhisperSettings, WorkerInitPayload, config_path, default_project_root, default_runtime_dir,
    default_socket_path,
};
pub use domain::*;
pub use provider::*;
