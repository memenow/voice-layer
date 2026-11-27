//! Core domain types for sessions, dictation, text workflows, health reports,
//! SSE events, and host injection.

mod api;
mod dictation;
mod events;
mod health;
mod session;

use std::time::{SystemTime, UNIX_EPOCH};

use thiserror::Error;

pub use api::{
    ComposeRequest, CompositionArchetype, CompositionReceipt, InjectRequest, InjectTarget,
    InjectionPlan, PreviewArtifact, PreviewStatus, ProblemDetails, RewriteRequest, RewriteStyle,
    TranslateRequest,
};
pub use dictation::{
    DictationCaptureRequest, DictationCaptureResult, SegmentProbeRequest, SegmentProbeResult,
    SegmentationMode, SpeechRegion, StartDictationRequest, StitchWavSegmentsRequest,
    StitchWavSegmentsResult, StopDictationRequest, TranscribeRequest, TranscriptionResult,
};
pub use events::{DaemonEvent, EventEnvelope};
pub use health::{HealthResponse, WorkerHealthSummary};
pub use session::{
    CaptureSession, LanguageProfile, LanguageStrategy, SessionMode, SessionState, TranscriptChunk,
    TriggerKind,
};

/// Opening escape sequence for terminal bracketed-paste mode.
pub const BRACKETED_PASTE_START: &str = "\u{1b}[200~";
/// Closing escape sequence for terminal bracketed-paste mode.
pub const BRACKETED_PASTE_END: &str = "\u{1b}[201~";

/// Domain-level errors returned by core helpers.
#[derive(Debug, Error)]
pub enum VoiceLayerError {
    #[error("injection target is not supported")]
    UnsupportedInjectionTarget,
}

/// Current Unix timestamp in milliseconds, returning `0` if the system clock precedes the epoch.
pub fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}
