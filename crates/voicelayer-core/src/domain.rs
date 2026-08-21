use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use uuid::Uuid;

pub const BRACKETED_PASTE_START: &str = "\u{1b}[200~";
pub const BRACKETED_PASTE_END: &str = "\u{1b}[201~";

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionMode {
    Dictation,
    Compose,
    Rewrite,
    Translate,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum SessionState {
    Idle,
    Listening,
    Transcribing,
    Previewing,
    AwaitingConfirmation,
    Completed,
    Failed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum TriggerKind {
    PushToTalk,
    Toggle,
    Cli,
    Tui,
    TrayButton,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum LanguageStrategy {
    AutoDetect,
    Locked,
    FollowPrevious,
    Bilingual,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct LanguageProfile {
    pub strategy: LanguageStrategy,
    pub input_languages: Vec<String>,
    pub output_language: Option<String>,
}

impl Default for LanguageProfile {
    fn default() -> Self {
        Self {
            strategy: LanguageStrategy::AutoDetect,
            input_languages: vec![
                "en".to_owned(),
                "zh".to_owned(),
                "ja".to_owned(),
                "ko".to_owned(),
                "es".to_owned(),
            ],
            output_language: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CaptureSession {
    pub session_id: Uuid,
    pub mode: SessionMode,
    pub state: SessionState,
    pub trigger: TriggerKind,
    pub language_profile: LanguageProfile,
    pub created_at_millis: u64,
}

impl CaptureSession {
    pub fn new(mode: SessionMode, trigger: TriggerKind, language_profile: LanguageProfile) -> Self {
        Self {
            session_id: Uuid::new_v4(),
            mode,
            state: SessionState::Listening,
            trigger,
            language_profile,
            created_at_millis: now_epoch_millis(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptChunk {
    pub session_id: Uuid,
    pub text: String,
    pub is_final: bool,
    pub language: Option<String>,
    pub confidence_basis_points: Option<u16>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompositionArchetype {
    Email,
    CoverLetter,
    DailyReport,
    Issue,
    PullRequestDescription,
    Prompt,
    TechnicalSummary,
    Custom,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StartDictationRequest {
    pub trigger: TriggerKind,
    #[serde(default)]
    pub language_profile: Option<LanguageProfile>,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
    #[serde(default)]
    pub segmentation: SegmentationMode,
}

/// Describes how audio capture should be segmented for transcription.
///
/// `OneShot` (the default) captures continuously from start to stop and the
/// entire audio is transcribed once at the end.
///
/// `Fixed` cuts `segment_secs`-sized chunks out of the continuous capture
/// buffer and streams each chunk to the worker while capture continues, so
/// stop-to-text latency stays bounded by the chunk length plus the
/// configured whisper warm-state cost rather than the full session length.
/// Chunks are cut from a single uninterrupted stream, so no audio is lost
/// at chunk boundaries. Word-level stitching at boundaries (via whisper
/// timestamps) is deferred; boundary-clipped words are accepted for now.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SegmentationMode {
    #[default]
    OneShot,
    Fixed {
        segment_secs: u32,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureRequest {
    pub trigger: TriggerKind,
    #[serde(default)]
    pub language_profile: Option<LanguageProfile>,
    pub duration_seconds: u32,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureResult {
    pub session: CaptureSession,
    pub transcription: TranscriptionResult,
    pub audio_file: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StopDictationRequest {
    pub session_id: Uuid,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComposeRequest {
    pub spoken_prompt: String,
    #[serde(default)]
    pub archetype: Option<CompositionArchetype>,
    #[serde(default)]
    pub output_language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteStyle {
    MoreFormal,
    Shorter,
    Politer,
    MoreTechnical,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteRequest {
    pub source_text: String,
    pub style: RewriteStyle,
    #[serde(default)]
    pub output_language: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslateRequest {
    pub source_text: String,
    pub target_language: String,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscribeRequest {
    pub audio_file: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub translate_to_english: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionResult {
    pub text: String,
    pub detected_language: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreviewStatus {
    Ready,
    Rejected,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreviewArtifact {
    pub artifact_id: Uuid,
    pub status: PreviewStatus,
    pub title: String,
    pub generated_text: Option<String>,
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositionReceipt {
    pub job_id: Uuid,
    pub preview: PreviewArtifact,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InjectTarget {
    GuiAccessible,
    GuiClipboard,
    TerminalBracketedPaste,
    TerminalKittyRemote,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InjectRequest {
    pub target: InjectTarget,
    pub text: String,
    #[serde(default)]
    pub auto_submit: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InjectionPlan {
    pub target: InjectTarget,
    pub payload: String,
    pub auto_submit: bool,
}

/// RFC 9457 problem details returned by the daemon on non-2xx responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProblemDetails {
    #[serde(rename = "type")]
    pub problem_type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
}

impl ProblemDetails {
    pub fn new(
        problem_type: impl Into<String>,
        title: impl Into<String>,
        status: u16,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            problem_type: problem_type.into(),
            title: title.into(),
            status,
            detail: detail.into(),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct HealthResponse {
    pub status: String,
    pub socket_path: String,
    pub version: String,
    pub worker: WorkerHealthSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct WorkerHealthSummary {
    pub status: String,
    pub command: String,
    pub asr_configured: bool,
    pub asr_model_path: Option<String>,
    pub asr_binary: Option<String>,
    pub asr_error: Option<String>,
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

/// Typed daemon event streamed over `GET /v1/events/stream` (SSE).
///
/// The serde tag (`event_type`) doubles as the SSE `event:` field via
/// [`DaemonEvent::name`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum DaemonEvent {
    DictationSessionCreated {
        session_id: Uuid,
    },
    DictationListening {
        session_id: Uuid,
    },
    DictationTranscribing {
        session_id: Uuid,
    },
    DictationCompleted {
        session_id: Uuid,
        transcript_chars: usize,
    },
    DictationFailed {
        session_id: Uuid,
        detail: String,
    },
    DictationSegmentedStarted {
        session_id: Uuid,
        segment_secs: u32,
    },
    SegmentRecorded {
        session_id: Uuid,
        segment_id: u32,
    },
    SegmentTranscribed {
        session_id: Uuid,
        segment_id: u32,
        transcript_chars: usize,
    },
    SegmentTranscribeFailed {
        session_id: Uuid,
        segment_id: u32,
        detail: String,
    },
    ComposeJobCreated {
        title: String,
    },
    RewriteJobCreated {
        title: String,
    },
    TranslateJobCreated {
        title: String,
    },
    TranscriptionCompleted {
        transcript_chars: usize,
    },
    WorkerProvidersUnavailable {
        detail: String,
    },
    /// Synthesized by the SSE handler when a subscriber lagged behind the
    /// broadcast channel; carries the number of dropped events.
    EventsLost {
        count: u64,
    },
}

impl DaemonEvent {
    pub fn name(&self) -> &'static str {
        match self {
            Self::DictationSessionCreated { .. } => "dictation_session_created",
            Self::DictationListening { .. } => "dictation_listening",
            Self::DictationTranscribing { .. } => "dictation_transcribing",
            Self::DictationCompleted { .. } => "dictation_completed",
            Self::DictationFailed { .. } => "dictation_failed",
            Self::DictationSegmentedStarted { .. } => "dictation_segmented_started",
            Self::SegmentRecorded { .. } => "segment_recorded",
            Self::SegmentTranscribed { .. } => "segment_transcribed",
            Self::SegmentTranscribeFailed { .. } => "segment_transcribe_failed",
            Self::ComposeJobCreated { .. } => "compose_job_created",
            Self::RewriteJobCreated { .. } => "rewrite_job_created",
            Self::TranslateJobCreated { .. } => "translate_job_created",
            Self::TranscriptionCompleted { .. } => "transcription_completed",
            Self::WorkerProvidersUnavailable { .. } => "worker_providers_unavailable",
            Self::EventsLost { .. } => "events_lost",
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventEnvelope {
    pub created_at_millis: u64,
    #[serde(flatten)]
    pub event: DaemonEvent,
}

impl EventEnvelope {
    pub fn new(event: DaemonEvent) -> Self {
        Self {
            created_at_millis: now_epoch_millis(),
            event,
        }
    }
}

pub fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}

#[cfg(test)]
mod tests {
    use super::{DaemonEvent, EventEnvelope};
    use uuid::Uuid;

    #[test]
    fn event_envelope_flattens_typed_event() {
        let envelope = EventEnvelope::new(DaemonEvent::SegmentRecorded {
            session_id: Uuid::nil(),
            segment_id: 3,
        });
        let encoded = serde_json::to_value(&envelope).unwrap();
        assert_eq!(encoded["event_type"], "segment_recorded");
        assert_eq!(encoded["segment_id"], 3);
        assert!(encoded["created_at_millis"].is_number());

        let decoded: EventEnvelope = serde_json::from_value(encoded).unwrap();
        assert_eq!(decoded.event.name(), "segment_recorded");
        assert_eq!(decoded, envelope);
    }
}
