//! State machine for the VoiceLayer desktop shell.
//!
//! The desktop shell is intentionally small: a single window that probes
//! daemon health, exposes start/stop buttons for a dictation session, and
//! displays the resulting transcript. Portal-backed global shortcuts are
//! best-effort; when the portal is unavailable we fall back to F9 inside
//! the focused window.

use std::sync::Arc;

use serde::{Deserialize, Serialize};
use uuid::Uuid;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DaemonStatus {
    #[default]
    Unknown,
    Probing,
    Healthy,
    Unreachable,
}

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SessionStage {
    #[default]
    Idle,
    Starting,
    Listening,
    Stopping,
    Completed,
    Failed,
}

#[derive(Debug, Clone)]
pub struct HotkeyStatus {
    pub portal_available: bool,
    pub portal_error: Option<String>,
}

impl Default for HotkeyStatus {
    fn default() -> Self {
        Self {
            portal_available: false,
            portal_error: Some("not probed".into()),
        }
    }
}

#[derive(Debug, Default, Clone)]
pub struct State {
    pub daemon: DaemonStatus,
    pub session_stage: SessionStage,
    pub session_id: Option<Uuid>,
    pub transcript: Option<String>,
    pub detected_language: Option<String>,
    pub last_notes: Vec<String>,
    pub error: Option<String>,
    pub hotkey: HotkeyStatus,
}

#[derive(Debug, Serialize)]
pub struct StartDictationRequest {
    pub trigger: &'static str,
    pub segmentation: Segmentation,
}

#[derive(Debug, Serialize)]
pub struct Segmentation {
    pub mode: &'static str,
}

#[derive(Debug, Serialize)]
pub struct StopDictationRequest {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Deserialize)]
pub struct CaptureSession {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Deserialize)]
pub struct TranscriptionResult {
    pub text: String,
    #[serde(default)]
    pub detected_language: Option<String>,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct DictationCaptureResult {
    // `session` is part of the daemon response but the shell only consumes
    // the transcription and failure classification.
    #[allow(dead_code)]
    pub session: CaptureSession,
    pub transcription: TranscriptionResult,
    #[serde(default)]
    pub failure_kind: Option<String>,
}

#[derive(Debug, Clone, Deserialize)]
pub struct HealthResponse {
    // Kept so the JSON payload shape is validated during deserialization,
    // even though the shell only treats the probe as a boolean right now.
    #[allow(dead_code)]
    #[serde(default)]
    pub status: String,
}

/// A shareable error message. `Arc<String>` is cheap to clone so the app can
/// bubble the same error through multiple Messages without copies.
pub type SharedError = Arc<String>;

pub fn render_daemon_status(status: DaemonStatus) -> &'static str {
    match status {
        DaemonStatus::Unknown => "unknown",
        DaemonStatus::Probing => "probing daemon...",
        DaemonStatus::Healthy => "daemon healthy",
        DaemonStatus::Unreachable => "daemon unreachable",
    }
}

pub fn render_session_stage(stage: SessionStage) -> &'static str {
    match stage {
        SessionStage::Idle => "idle",
        SessionStage::Starting => "starting dictation...",
        SessionStage::Listening => "listening — speak now",
        SessionStage::Stopping => "stopping...",
        SessionStage::Completed => "completed",
        SessionStage::Failed => "failed",
    }
}
