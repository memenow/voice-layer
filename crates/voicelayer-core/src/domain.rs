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
    pub recorder_backend: Option<RecorderBackend>,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
    #[serde(default)]
    pub segmentation: SegmentationMode,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RecorderBackend {
    Auto,
    Pipewire,
    Alsa,
}

/// Describes how audio capture should be segmented for transcription.
///
/// `OneShot` (the default) keeps the pre-Phase-3 behavior: a single recorder
/// subprocess runs from start to stop and the entire audio file is
/// transcribed once at the end.
///
/// `Fixed` rolls the recorder every `segment_secs` seconds and streams each
/// finalized chunk to the worker while the next chunk captures, so stop-to-text
/// latency stays bounded by the chunk length plus the configured whisper
/// warm-state cost rather than the full session length.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SegmentationMode {
    #[default]
    OneShot,
    Fixed {
        segment_secs: u32,
        #[serde(default)]
        overlap_secs: u32,
    },
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

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum DictationFailureKind {
    /// The recorder subprocess failed to capture audio or stop cleanly.
    RecordingFailed,
    /// The ASR worker reached the audio but failed to produce a transcript.
    AsrFailed,
    /// The transcript was produced but injecting it into the foreground target failed.
    /// Only the client (CLI, desktop shell) can classify this; the daemon never sets it.
    InjectionFailed,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureResult {
    pub session: CaptureSession,
    pub transcription: TranscriptionResult,
    pub audio_file: Option<String>,
    #[serde(default)]
    pub failure_kind: Option<DictationFailureKind>,
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
    /// Which whisper path the Python worker would dispatch to on the next
    /// `transcribe` call: `"server"` (persistent whisper-server over HTTP),
    /// `"cli"` (one-shot whisper-cli subprocess), or `"unconfigured"`.
    /// `None` when an older daemon / worker is serving and did not emit
    /// this field.
    #[serde(default)]
    pub whisper_mode: Option<String>,
    /// Base URL of the configured whisper-server endpoint (e.g.
    /// `http://127.0.0.1:8188`). `None` when whisper-server is not
    /// configured or when the worker predates this field.
    #[serde(default)]
    pub whisper_server_url: Option<String>,
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

#[cfg(test)]
mod tests {
    use super::{DictationFailureKind, WorkerHealthSummary};

    #[test]
    fn dictation_failure_kind_serializes_to_snake_case_variants() {
        assert_eq!(
            serde_json::to_string(&DictationFailureKind::RecordingFailed).unwrap(),
            "\"recording_failed\""
        );
        assert_eq!(
            serde_json::to_string(&DictationFailureKind::AsrFailed).unwrap(),
            "\"asr_failed\""
        );
        assert_eq!(
            serde_json::to_string(&DictationFailureKind::InjectionFailed).unwrap(),
            "\"injection_failed\""
        );
    }

    #[test]
    fn dictation_failure_kind_round_trips_through_json() {
        let original = DictationFailureKind::AsrFailed;
        let encoded = serde_json::to_string(&original).unwrap();
        let decoded: DictationFailureKind = serde_json::from_str(&encoded).unwrap();
        assert_eq!(decoded, original);
    }

    // Drift guard for the /v1/health response shape. Adding a pub field to
    // WorkerHealthSummary breaks the sentinel construction below at compile
    // time, so the next contributor must either document the field in
    // openapi/voicelayerd.v1.yaml or explicitly opt out by zeroing the
    // sentinel field; the serialized keys drive the openapi assertion.
    //
    // Known blind spot: this pattern assumes no WorkerHealthSummary field
    // uses `#[serde(skip_serializing_if = "Option::is_none")]`. If that
    // attribute is added, `to_value(sentinel_with_None).as_object().keys()`
    // will drop the key and the guard silently stops covering it. Either
    // seed the sentinel field with `Some(...)` or remove the attribute to
    // keep coverage intact.
    #[test]
    fn openapi_worker_health_summary_documents_every_field() {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");

        // Restrict the substring search to the WorkerHealthSummary component
        // slice. Generic property names such as `status`, `command`, and
        // `message` also appear under other schemas at the same indent, so an
        // unscoped match would silently pass even after a genuine deletion
        // inside WorkerHealthSummary. Walking line-by-line lets us terminate
        // at the next top-level schema heading (exactly 4-space indent)
        // instead of the first 6/8-space sub-property.
        let mut section = String::new();
        let mut collecting = false;
        for line in contents.lines() {
            if line == "    WorkerHealthSummary:" {
                collecting = true;
            } else if collecting
                && line.starts_with("    ")
                && !line.starts_with("     ")
                && !line.trim().is_empty()
            {
                break;
            }
            if collecting {
                section.push_str(line);
                section.push('\n');
            }
        }
        assert!(
            !section.is_empty(),
            "WorkerHealthSummary component not found in openapi file",
        );

        // Derive field names from the Rust struct so the guard self-updates
        // with renames. Adding a pub field forces a compile error here until
        // the sentinel is extended, and serde's default snake_case naming
        // matches the openapi property names verbatim.
        let sentinel = WorkerHealthSummary {
            status: String::new(),
            command: String::new(),
            asr_configured: false,
            asr_binary: None,
            asr_model_path: None,
            asr_error: None,
            whisper_mode: None,
            whisper_server_url: None,
            llm_configured: false,
            llm_model: None,
            llm_endpoint: None,
            llm_reachable: false,
            llm_error: None,
            global_shortcuts_portal_available: false,
            global_shortcuts_portal_version: None,
            global_shortcuts_portal_error: None,
            message: None,
        };
        let value = serde_json::to_value(&sentinel)
            .expect("WorkerHealthSummary serializes to JSON without error");
        let object = value
            .as_object()
            .expect("WorkerHealthSummary serializes to a JSON object");

        for field in object.keys() {
            let needle = format!("        {field}:");
            assert!(
                section.contains(&needle),
                "openapi WorkerHealthSummary schema is missing `{field}` \
                 (looked for `{needle}` inside the component). \
                 Update openapi/voicelayerd.v1.yaml to keep the contract in sync.",
            );
        }

        // Cross-check the `required` list: every non-Option Rust field must
        // appear under required, and every Option field must not. `None`
        // serialises as JSON `null`, so the serialized value's nullness
        // identifies Rust-side optionality without any type reflection. If
        // required shifts to inline form (`required: [a, b]`) the scan below
        // breaks — the header assertion calls that out loudly.
        let required_header = section
            .lines()
            .find(|line| line.trim_start().starts_with("required:"))
            .expect("WorkerHealthSummary declares a required list");
        assert_eq!(
            required_header, "      required:",
            "required list must stay in YAML block form so the scan can parse it \
             (found {required_header:?})",
        );
        let required_set: std::collections::BTreeSet<&str> = section
            .lines()
            .skip_while(|line| *line != "      required:")
            .skip(1)
            .take_while(|line| line.starts_with("        - "))
            .map(|line| line.trim_start_matches("        - ").trim())
            .collect();

        for (field, field_value) in object {
            let is_option = field_value.is_null();
            let listed = required_set.contains(field.as_str());
            match (is_option, listed) {
                (false, false) => panic!(
                    "openapi WorkerHealthSummary required list is missing `{field}`; \
                     the Rust field is non-Option so it always serialises — the schema \
                     must list it under required."
                ),
                (true, true) => panic!(
                    "openapi WorkerHealthSummary required list contains `{field}`, but \
                     the Rust field is Option<_> and may legitimately be absent. Drop it \
                     from required (or change the Rust side to a non-Option type)."
                ),
                _ => {}
            }
        }
    }
}
