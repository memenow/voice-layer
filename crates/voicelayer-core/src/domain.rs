use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};
use thiserror::Error;
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

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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
    pub recorder_backend: Option<RecorderBackend>,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecorderBackend {
    Auto,
    Pipewire,
    Alsa,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureRequest {
    pub trigger: TriggerKind,
    #[serde(default)]
    pub language_profile: Option<LanguageProfile>,
    pub duration_seconds: u32,
    #[serde(default)]
    pub recorder_backend: Option<RecorderBackend>,
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
    Translate,
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
    NeedsProvider,
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

impl PreviewArtifact {
    pub fn needs_provider(title: &str, mode: SessionMode) -> Self {
        let mode_label = match mode {
            SessionMode::Dictation => "dictation",
            SessionMode::Compose => "composition",
            SessionMode::Rewrite => "rewrite",
            SessionMode::Translate => "translation",
        };

        Self {
            artifact_id: Uuid::new_v4(),
            status: PreviewStatus::NeedsProvider,
            title: title.to_owned(),
            generated_text: None,
            notes: vec![
                format!("No configured provider is available for the {mode_label} workflow."),
                "Configure a local or cloud provider before requesting generated text.".to_owned(),
            ],
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositionReceipt {
    pub job_id: Uuid,
    pub preview: PreviewArtifact,
}

impl CompositionReceipt {
    pub fn needs_provider(title: &str, mode: SessionMode) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            preview: PreviewArtifact::needs_provider(title, mode),
        }
    }
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
    pub global_shortcuts_portal_available: bool,
    pub global_shortcuts_portal_version: Option<u32>,
    pub global_shortcuts_portal_error: Option<String>,
    pub message: Option<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct EventEnvelope {
    pub event_type: String,
    pub session_id: Option<Uuid>,
    pub created_at_millis: u64,
    pub message: String,
}

impl EventEnvelope {
    pub fn new(
        event_type: impl Into<String>,
        session_id: Option<Uuid>,
        message: impl Into<String>,
    ) -> Self {
        Self {
            event_type: event_type.into(),
            session_id,
            created_at_millis: now_epoch_millis(),
            message: message.into(),
        }
    }
}

#[derive(Debug, Error)]
pub enum VoiceLayerError {
    #[error("injection target is not supported")]
    UnsupportedInjectionTarget,
}

pub fn now_epoch_millis() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_millis() as u64)
        .unwrap_or_default()
}
