//! State for the VoiceLayer desktop shell.
//!
//! All wire types come from `voicelayer-core`; this module only holds the
//! shell's own UI state.

use std::sync::Arc;

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
    pub available: bool,
    pub backend: Option<String>,
    pub detail: Option<String>,
}

impl Default for HotkeyStatus {
    fn default() -> Self {
        Self {
            available: false,
            backend: None,
            detail: Some("not probed".into()),
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
