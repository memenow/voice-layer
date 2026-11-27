//! Health probe response payloads for `GET /v1/health` and `POST /v1/health/refresh`.

use serde::{Deserialize, Serialize};

/// Top-level daemon health response.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthResponse {
    /// `"ok"` when the worker probe succeeded, else `"degraded"`.
    pub status: String,
    pub socket_path: String,
    pub version: String,
    pub worker: WorkerHealthSummary,
}

/// Readiness report for the Python worker, configured backends, and host hotkey backend.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerHealthSummary {
    pub status: String,
    pub command: String,
    pub asr_configured: bool,
    pub asr_model_path: Option<String>,
    pub asr_binary: Option<String>,
    pub asr_error: Option<String>,
    /// Active whisper dispatch mode: `"server"`, `"cli"`, or `"unconfigured"`.
    #[serde(default)]
    pub whisper_mode: Option<String>,
    /// Base URL of the configured `whisper-server` endpoint, if enabled.
    #[serde(default)]
    pub whisper_server_url: Option<String>,
    /// `true` when MiMo-V2.5-ASR paths exist on disk (weights load lazily on first transcribe).
    #[serde(default)]
    pub mimo_configured: bool,
    #[serde(default)]
    pub mimo_model_path: Option<String>,
    #[serde(default)]
    pub mimo_error: Option<String>,
    /// `true` when the Qwen3-ASR-1.7B model directory exists on disk.
    #[serde(default)]
    pub qwen3_asr_configured: bool,
    #[serde(default)]
    pub qwen3_asr_model_path: Option<String>,
    #[serde(default)]
    pub qwen3_asr_error: Option<String>,
    pub llm_configured: bool,
    pub llm_model: Option<String>,
    pub llm_endpoint: Option<String>,
    pub llm_reachable: bool,
    pub llm_error: Option<String>,
    pub global_hotkeys_available: bool,
    pub global_hotkeys_backend: Option<String>,
    pub global_hotkeys_detail: Option<String>,
    pub message: Option<String>,
}
