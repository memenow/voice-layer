//! Shared domain contracts for the VoiceLayer daemon, CLI, desktop shell,
//! and Python worker. The types declared here are the wire vocabulary of
//! `/v1` and the JSON-RPC stdio bridge: their serde representation is the
//! source of truth for `openapi/voicelayerd.v1.yaml`.
//!
//! Any change to a public field, enum variant, or serde rename in this
//! crate is a contract change. Update `openapi/voicelayerd.v1.yaml` in
//! the same commit so downstream clients and the in-repo drift guards
//! stay aligned.

pub mod config;
pub mod domain;
pub mod injection;
pub mod provider;

pub use config::{
    DaemonSettings, ForegroundPttSettings, LlmSettings, StopAction, VadSettings, VoiceLayerConfig,
    WhisperServerSettings, WhisperSettings, WorkerInitPayload, config_path, default_project_root,
    default_runtime_dir, default_socket_path,
};
pub use domain::*;
pub use provider::*;
