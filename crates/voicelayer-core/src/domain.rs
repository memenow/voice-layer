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
    /// VAD-gated segmentation. The recorder rolls every `probe_secs`
    /// seconds; each probe is classified by the Python worker's
    /// `segment_probe` RPC. The orchestrator accumulates speech probes
    /// into a pending buffer and flushes them as one logical speech unit
    /// on the first run of `silence_gap_probes` silent probes or when the
    /// buffered duration reaches `max_segment_secs` (whichever comes
    /// first). Flushed units are transcribed in the background, exactly
    /// as in `Fixed` mode.
    VadGated {
        probe_secs: u32,
        max_segment_secs: u32,
        #[serde(default = "default_silence_gap_probes")]
        silence_gap_probes: u32,
    },
}

fn default_silence_gap_probes() -> u32 {
    1
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

/// Closed speech region inside a single audio file, in seconds from the
/// file start. Emitted by the Python worker's `segment_probe` RPC after a
/// silero-vad pass.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechRegion {
    pub start_secs: f32,
    pub end_secs: f32,
}

/// Request body for the Python worker's `segment_probe` JSON-RPC method.
///
/// `audio_file` is an absolute path to a WAV file the worker will read
/// synchronously. Callers supply probe-sized captures (typically 1-2 s) —
/// the worker does not itself cap the input length.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentProbeRequest {
    pub audio_file: String,
}

/// Result of a `segment_probe` RPC. `speech_ratio` is the fraction of the
/// input's duration that silero-vad classified as speech, in `[0.0, 1.0]`.
/// `has_speech` is `true` iff the regions list is non-empty. `notes`
/// carries provider-side diagnostics (e.g. a fallback explanation); it is
/// empty on the happy path.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentProbeResult {
    pub has_speech: bool,
    pub speech_ratio: f32,
    pub regions: Vec<SpeechRegion>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Request body for the Python worker's `stitch_wav_segments` RPC.
/// Concatenates N probe WAV files written by the same recorder (same
/// sample rate, bit depth, channel count) into a single output WAV. The
/// daemon orchestrator uses it to assemble a VAD-gated speech unit from
/// pending probes before handing the merged path to `transcribe`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StitchWavSegmentsRequest {
    pub audio_files: Vec<String>,
    pub out_file: String,
}

/// Result of a `stitch_wav_segments` RPC. Echoes the output path (so
/// callers need not remember it) along with the count of inputs merged
/// and the resulting duration for logging/diagnostics.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchWavSegmentsResult {
    pub audio_file: String,
    pub segment_count: usize,
    pub duration_secs: f32,
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
    use super::{
        CaptureSession, ComposeRequest, CompositionArchetype, CompositionReceipt,
        DictationCaptureRequest, DictationCaptureResult, DictationFailureKind, HealthResponse,
        InjectRequest, InjectTarget, InjectionPlan, LanguageProfile, LanguageStrategy,
        PreviewArtifact, PreviewStatus, RecorderBackend, RewriteRequest, RewriteStyle,
        SegmentationMode, SessionMode, SessionState, StartDictationRequest, StopDictationRequest,
        TranscribeRequest, TranscriptionResult, TranslateRequest, TriggerKind, WorkerHealthSummary,
    };
    // `ProviderDescriptor` and `ProviderKind` live in `provider.rs`;
    // pulled in here because the openapi drift guard machinery is
    // centralised in this module's tests rather than duplicated.
    use crate::{ProviderDescriptor, ProviderKind};
    use uuid::Uuid;

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

    // Drift guard template for `components.schemas.<Name>` blocks in
    // openapi/voicelayerd.v1.yaml. Adding a `pub` field to a guarded
    // struct breaks the matching sentinel at compile time, so a future
    // contributor must either document the field in the openapi schema
    // or explicitly opt out by extending the sentinel.
    //
    // Sentinel convention: every Option field is `None` (so it
    // serialises to JSON `null`, which the required-list check uses to
    // identify Rust-side optionality without type reflection); every
    // non-Option field is the type's natural zero/default value.
    //
    // Known blind spot: a future field annotated with
    // `#[serde(skip_serializing_if = "Option::is_none")]` would drop
    // out of `to_value(sentinel).as_object().keys()` and the guard
    // would silently stop covering it. Seed such a field with
    // `Some(...)` in the sentinel, or remove the attribute, to keep
    // coverage intact.

    /// Compare every field in `sentinel` (as serialised by serde_json)
    /// against the openapi component schema named `schema_name`. Asserts
    /// two contracts:
    ///
    /// 1. **Property presence** — every serialised field appears as a
    ///    property under the schema component (via the literal
    ///    `        <field>:` line — eight-space indent matches every
    ///    schema component's top-level property indent).
    /// 2. **Required-list cross-check** — every non-Option field
    ///    (serialised as a non-null value) appears under `required:`,
    ///    and every Option field (serialised as `null`) does not.
    ///
    /// `serde_default_fields` lists Rust fields that are non-Option but
    /// annotated with `#[serde(default)]`, so they serialise as a real
    /// value yet are wire-optional. Such fields are excluded from the
    /// required-list cross-check (the openapi schema may legitimately
    /// list them as required or omit them — the contract is the absence
    /// of a hard requirement, not its presence). Property-presence is
    /// still enforced. Pass `&[]` for schemas with no such fields.
    ///
    /// Supports both YAML required-list forms in the source file:
    /// block (`      required:\n        - foo`) and inline
    /// (`      required: [foo, bar]`).
    fn assert_openapi_documents_every_field<T: serde::Serialize>(
        schema_name: &str,
        sentinel: &T,
        serde_default_fields: &[&str],
    ) {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");

        let section = isolate_schema_section(&contents, schema_name);
        assert!(
            !section.is_empty(),
            "schema component {schema_name} not found in openapi file",
        );

        let value = serde_json::to_value(sentinel)
            .unwrap_or_else(|err| panic!("{schema_name} sentinel must serialise: {err}"));
        let object = value
            .as_object()
            .unwrap_or_else(|| panic!("{schema_name} sentinel must serialise to a JSON object"));

        // 1. Property presence.
        for field in object.keys() {
            let needle = format!("        {field}:");
            assert!(
                section.contains(&needle),
                "openapi {schema_name} schema is missing `{field}` \
                 (looked for `{needle}` inside the component). \
                 Update openapi/voicelayerd.v1.yaml to keep the contract in sync.",
            );
        }

        // 2. Required-list cross-check.
        let required_set = parse_required_set(&section);
        if let Err(detail) =
            check_required_consistency(schema_name, object, &required_set, serde_default_fields)
        {
            panic!("{detail}");
        }
    }

    /// Cross-check a serialised sentinel against an openapi schema's
    /// `required` set. Returns `Ok(())` when every non-Option field is
    /// listed (and every Option field is not), or `Err(detail)` with a
    /// human-readable explanation of the first mismatch encountered.
    ///
    /// Fields named in `serde_default_fields` are skipped in either
    /// direction — their wire contract is "may be absent on input,
    /// daemon defaults", so the openapi schema may legitimately list or
    /// omit them.
    ///
    /// Extracted from `assert_openapi_documents_every_field` so the
    /// override-loop's behaviour can be unit-tested without a real
    /// openapi file on disk.
    fn check_required_consistency(
        schema_name: &str,
        object: &serde_json::Map<String, serde_json::Value>,
        required_set: &std::collections::BTreeSet<String>,
        serde_default_fields: &[&str],
    ) -> Result<(), String> {
        for (field, field_value) in object {
            if serde_default_fields.contains(&field.as_str()) {
                continue;
            }
            let is_option = field_value.is_null();
            let listed = required_set.contains(field.as_str());
            match (is_option, listed) {
                (false, false) => {
                    return Err(format!(
                        "openapi {schema_name} required list is missing `{field}`; \
                         the Rust field is non-Option so it always serialises — the schema \
                         must list it under required.",
                    ));
                }
                (true, true) => {
                    return Err(format!(
                        "openapi {schema_name} required list contains `{field}`, but \
                         the Rust field is Option<_> and may legitimately be absent. Drop it \
                         from required (or change the Rust side to a non-Option type).",
                    ));
                }
                _ => {}
            }
        }
        Ok(())
    }

    /// Walk the openapi YAML line-by-line and return the slice that
    /// belongs to `components.schemas.<schema_name>`. Generic property
    /// names (`status`, `command`, `message`) also appear under other
    /// schemas, so an unscoped substring search would silently pass
    /// after a genuine deletion. The walker terminates at the next
    /// exactly-4-space heading line (the indent that distinguishes a
    /// new component from a sub-property at 6/8/10 spaces).
    fn isolate_schema_section(contents: &str, schema_name: &str) -> String {
        let heading = format!("    {schema_name}:");
        let mut section = String::new();
        let mut collecting = false;
        for line in contents.lines() {
            if line == heading {
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
        section
    }

    /// Parse the schema's `required:` list out of the isolated section,
    /// supporting both block and inline forms. Block form wins when
    /// both forms appear (which should never happen in a well-formed
    /// schema). An empty set is returned when no required list is
    /// declared — callers treat that as "no fields are required".
    fn parse_required_set(section: &str) -> std::collections::BTreeSet<String> {
        let mut block: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        let mut in_block = false;
        for line in section.lines() {
            if line == "      required:" {
                in_block = true;
                continue;
            }
            if in_block {
                if let Some(name) = line.strip_prefix("        - ") {
                    block.insert(name.trim().to_owned());
                } else {
                    in_block = false;
                }
            }
        }
        if !block.is_empty() {
            return block;
        }

        // Inline form: scan for `required: [a, b, c]` at the schema's
        // own indent level (six spaces) so nested oneOf branches don't
        // bleed into the outer schema's required list. Slice the inner
        // list at the first `]` so a future trailing comment
        // (`required: [a, b]  # ...`) cannot leak into the last entry.
        let mut inline: std::collections::BTreeSet<String> = std::collections::BTreeSet::new();
        for line in section.lines() {
            if let Some(rest) = line.strip_prefix("      required: [") {
                let inner = match rest.find(']') {
                    Some(end) => &rest[..end],
                    None => continue,
                };
                for entry in inner.split(',') {
                    let name = entry.trim();
                    if !name.is_empty() {
                        inline.insert(name.to_owned());
                    }
                }
                break;
            }
        }
        inline
    }

    /// Parse a top-level string-enum schema's `enum: [...]` line.
    ///
    /// Targets schemas like `RecorderBackend` whose entire body is a
    /// `type: string` plus an inline `enum: [...]` at the schema's own
    /// property indent (six spaces). Returns the enum values in source
    /// order, or an empty vec when no top-level enum is declared.
    /// The `]` slice mirrors `parse_required_set`'s inline form so a
    /// future trailing comment cannot leak into the last entry.
    fn parse_top_level_string_enum(section: &str) -> Vec<String> {
        for line in section.lines() {
            if let Some(rest) = line.strip_prefix("      enum: [") {
                let inner = match rest.find(']') {
                    Some(end) => &rest[..end],
                    None => continue,
                };
                return inner
                    .split(',')
                    .map(|entry| entry.trim().to_owned())
                    .filter(|entry| !entry.is_empty())
                    .collect();
            }
        }
        Vec::new()
    }

    /// Collect every discriminator value declared inside a `oneOf`
    /// schema's branches.
    ///
    /// Targets schemas like `SegmentationMode` whose body is
    /// `oneOf: [{properties: {mode: {enum: [<value>]}}}, ...]`. Each
    /// branch's discriminator enum sits at fourteen-space indent
    /// (`              enum: [<value>]`); collecting these in source
    /// order yields the full discriminator set without needing a real
    /// YAML parser. Trailing-comment defense matches the other inline
    /// scanners.
    fn parse_oneof_discriminator_values(section: &str) -> Vec<String> {
        let mut values = Vec::new();
        for line in section.lines() {
            if let Some(rest) = line.strip_prefix("              enum: [") {
                let inner = match rest.find(']') {
                    Some(end) => &rest[..end],
                    None => continue,
                };
                for entry in inner.split(',') {
                    let name = entry.trim();
                    if !name.is_empty() {
                        values.push(name.to_owned());
                    }
                }
            }
        }
        values
    }

    /// Parse a property-scoped string enum out of an isolated schema
    /// section. Targets a single property at eight-space indent
    /// (`        <property_name>:`) and recovers either the inline form
    /// (`          enum: [a, b]`) or the block form (`          enum:`
    /// followed by `            - value` lines), in source order.
    ///
    /// Block form may include `- null` to model a nullable field; the
    /// parser returns `null` verbatim and leaves filtering to the
    /// caller — the rust side models nullability via `Option<_>`, not a
    /// separate variant, so the caller must drop `null` before
    /// cross-checking against rust variants.
    ///
    /// Returns an empty vec when the property has no enum (or the
    /// property itself is not present).
    fn parse_property_string_enum(section: &str, property_name: &str) -> Vec<String> {
        let property_heading = format!("        {property_name}:");
        let mut in_property = false;
        let mut in_block = false;
        let mut block: Vec<String> = Vec::new();

        for line in section.lines() {
            if !in_property {
                if line == property_heading {
                    in_property = true;
                }
                continue;
            }
            // Sibling property at the same eight-space indent ends the
            // property scope. Lines with deeper indent (10/12 spaces)
            // and blank lines stay within the body.
            if line.starts_with("        ")
                && !line.starts_with("         ")
                && !line.trim().is_empty()
            {
                break;
            }
            // Inline form wins as soon as we see it; matches the
            // resolution order the existing top-level scanner uses.
            if let Some(rest) = line.strip_prefix("          enum: [")
                && let Some(end) = rest.find(']')
            {
                return rest[..end]
                    .split(',')
                    .map(|entry| entry.trim().to_owned())
                    .filter(|entry| !entry.is_empty())
                    .collect();
            }
            if line == "          enum:" {
                in_block = true;
                continue;
            }
            if in_block {
                if let Some(name) = line.strip_prefix("            - ") {
                    block.push(name.trim().to_owned());
                } else {
                    in_block = false;
                }
            }
        }
        block
    }

    /// Cross-check every variant of a Rust string enum against an
    /// openapi top-level enum schema. Each variant must serialize to a
    /// JSON string that appears in the openapi list, and every openapi
    /// value must correspond to a Rust variant — drift in either
    /// direction trips the test.
    ///
    /// Pair this with an `enumerate_*_variants` helper whose body
    /// matches every variant exhaustively, so adding a new Rust
    /// variant fails compile until the helper is updated.
    fn assert_openapi_documents_every_string_enum_variant<T: serde::Serialize>(
        schema_name: &str,
        rust_variants: &[T],
    ) {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");
        let section = isolate_schema_section(&contents, schema_name);
        assert!(
            !section.is_empty(),
            "schema component {schema_name} not found in openapi file",
        );

        let openapi_variants = parse_top_level_string_enum(&section);
        assert!(
            !openapi_variants.is_empty(),
            "openapi {schema_name} schema has no top-level `enum:` list — \
             use the struct drift guard instead, this helper is for pure enums",
        );

        let rust_serialized: Vec<String> = rust_variants
            .iter()
            .map(|variant| match serde_json::to_value(variant) {
                Ok(serde_json::Value::String(s)) => s,
                Ok(other) => {
                    panic!("{schema_name} variant must serialize to a JSON string; got {other:?}",)
                }
                Err(err) => panic!("{schema_name} variant must serialize: {err}"),
            })
            .collect();

        for serialized in &rust_serialized {
            assert!(
                openapi_variants.iter().any(|value| value == serialized),
                "openapi {schema_name} enum is missing `{serialized}` (rust enum has it). \
                 Update openapi/voicelayerd.v1.yaml to keep the contract in sync.",
            );
        }
        for openapi_variant in &openapi_variants {
            assert!(
                rust_serialized.iter().any(|value| value == openapi_variant),
                "openapi {schema_name} enum lists `{openapi_variant}` but the rust enum does not. \
                 Either add the variant to the rust enum or drop it from the openapi list.",
            );
        }
    }

    /// Cross-check every variant of a Rust `#[serde(tag = "...")]`
    /// enum against an openapi `oneOf` schema's discriminator values.
    /// Each Rust variant must serialize to a JSON object that contains
    /// `discriminator` as a string; every such string must appear in
    /// the openapi `oneOf` branches, and every openapi discriminator
    /// must correspond to a Rust variant.
    fn assert_openapi_documents_every_oneof_discriminator<T: serde::Serialize>(
        schema_name: &str,
        discriminator: &str,
        rust_variants: &[T],
    ) {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");
        let section = isolate_schema_section(&contents, schema_name);
        assert!(
            !section.is_empty(),
            "schema component {schema_name} not found in openapi file",
        );

        let openapi_values = parse_oneof_discriminator_values(&section);
        assert!(
            !openapi_values.is_empty(),
            "openapi {schema_name} schema has no oneOf discriminator enums — \
             this helper expects each branch to declare `enum: [<value>]`",
        );

        let rust_values: Vec<String> = rust_variants
            .iter()
            .map(|variant| {
                let value = serde_json::to_value(variant)
                    .unwrap_or_else(|err| panic!("{schema_name} variant must serialize: {err}"));
                let object = value.as_object().unwrap_or_else(|| {
                    panic!("{schema_name} variant must serialize to a JSON object; got {value:?}")
                });
                let raw = object.get(discriminator).unwrap_or_else(|| {
                    panic!(
                        "{schema_name} variant must contain `{discriminator}` field; got {object:?}",
                    )
                });
                match raw {
                    serde_json::Value::String(s) => s.clone(),
                    other => panic!(
                        "{schema_name}.{discriminator} must serialize as string; got {other:?}",
                    ),
                }
            })
            .collect();

        for serialized in &rust_values {
            assert!(
                openapi_values.iter().any(|value| value == serialized),
                "openapi {schema_name} oneOf is missing the `{discriminator}: {serialized}` \
                 branch (rust variant exists). Update openapi/voicelayerd.v1.yaml.",
            );
        }
        for openapi_value in &openapi_values {
            assert!(
                rust_values.iter().any(|value| value == openapi_value),
                "openapi {schema_name} oneOf has `{discriminator}: {openapi_value}` but the \
                 rust enum has no matching variant. Either add the variant or drop the branch.",
            );
        }
    }

    /// Cross-check every variant of a Rust string enum against the
    /// openapi enum list embedded inside a single property of a
    /// composite schema (e.g. `RewriteRequest.style`,
    /// `DictationCaptureResult.failure_kind`). The rust side is
    /// modeled as a separate enum but the openapi side inlines the
    /// enum under one property — drift would be invisible to the
    /// struct-shape and top-level enum guards.
    ///
    /// `null` entries in the openapi list are filtered before the
    /// cross-check: rust models nullability with `Option<_>`, not a
    /// separate variant, so the caller's enumerator never produces
    /// `null` and the openapi list legitimately includes it for
    /// nullable fields.
    fn assert_openapi_documents_every_property_enum_variant<T: serde::Serialize>(
        schema_name: &str,
        property_name: &str,
        rust_variants: &[T],
    ) {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");
        let section = isolate_schema_section(&contents, schema_name);
        assert!(
            !section.is_empty(),
            "schema component {schema_name} not found in openapi file",
        );

        let openapi_values: Vec<String> = parse_property_string_enum(&section, property_name)
            .into_iter()
            .filter(|value| value != "null")
            .collect();
        assert!(
            !openapi_values.is_empty(),
            "openapi {schema_name}.{property_name} has no string enum values \
             (after filtering `null`); use a different drift guard if the field is not enum-typed",
        );

        let rust_serialized: Vec<String> = rust_variants
            .iter()
            .map(|variant| match serde_json::to_value(variant) {
                Ok(serde_json::Value::String(s)) => s,
                Ok(other) => panic!(
                    "{schema_name}.{property_name} variant must serialize to a JSON string; \
                     got {other:?}",
                ),
                Err(err) => panic!("{schema_name}.{property_name} variant must serialize: {err}"),
            })
            .collect();

        for serialized in &rust_serialized {
            assert!(
                openapi_values.iter().any(|value| value == serialized),
                "openapi {schema_name}.{property_name} enum is missing `{serialized}` \
                 (rust enum has it). Update openapi/voicelayerd.v1.yaml.",
            );
        }
        for openapi_value in &openapi_values {
            assert!(
                rust_serialized.iter().any(|value| value == openapi_value),
                "openapi {schema_name}.{property_name} enum lists `{openapi_value}` but the \
                 rust enum has no matching variant. Either add the variant or drop it from \
                 the openapi list.",
            );
        }
    }

    /// Collect every `#/components/schemas/<Name>` reference that
    /// appears as the value of a `$ref:` line. Tolerates both quoted
    /// (`"..."`) and unquoted forms; the only requirement is the
    /// `$ref:` token at the start of the line (after trimming) and a
    /// matching component-prefixed value.
    fn collect_schema_refs(contents: &str) -> std::collections::BTreeSet<String> {
        let prefix = "#/components/schemas/";
        let mut refs = std::collections::BTreeSet::new();
        for line in contents.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("$ref:") else {
                continue;
            };
            let unquoted = rest.trim().trim_matches(['"', '\'']);
            if let Some(name) = unquoted.strip_prefix(prefix) {
                refs.insert(name.to_owned());
            }
        }
        refs
    }

    /// Collect every PascalCase schema heading at the canonical
    /// four-space `components.schemas.<Name>:` indent. Path keys
    /// (e.g. `/v1/health:`) live at the same indent under `paths:`
    /// but always start with `/`, so the leading-uppercase filter
    /// excludes them. Future lower-case schema names would slip
    /// through, but openapi convention (and every existing schema
    /// in this contract) uses PascalCase.
    fn collect_defined_schema_names(contents: &str) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        for line in contents.lines() {
            let Some(stripped) = line.strip_prefix("    ") else {
                continue;
            };
            // 4-space-only indent: deeper indents must be excluded.
            if stripped.starts_with(' ') {
                continue;
            }
            let Some(name) = stripped.strip_suffix(':') else {
                continue;
            };
            // PascalCase guard: paths start with `/`, descriptions
            // would have spaces, etc.
            if !name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) {
                continue;
            }
            if name.contains(' ') {
                continue;
            }
            names.insert(name.to_owned());
        }
        names
    }

    /// Pull the first quoted-string value bound to `key` in a TOML
    /// document. Matches lines of the form `key = "..."` ignoring
    /// leading whitespace; returns `None` when the key is missing.
    /// Section headers (`[workspace.package]` etc.) are not tracked
    /// — every `key = "..."` line in the file is fair game, and the
    /// first match wins. The pinned values this helper covers
    /// (`channel`, `rust-version`) appear once each in their target
    /// files, so first-match is the right semantics.
    fn extract_toml_string_value(contents: &str, key: &str) -> Option<String> {
        let prefix = format!("{key} = \"");
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix(&prefix)
                && let Some(end) = rest.find('"')
            {
                return Some(rest[..end].to_owned());
            }
        }
        None
    }

    /// Pull the first quoted-string value bound to `key` in a YAML
    /// document. Matches lines of the form `key: "..."` ignoring
    /// leading whitespace. As with `extract_toml_string_value`, no
    /// section / nesting tracking — the helper is intended for
    /// exactly-once keys like the workflow's `toolchain:` step input.
    fn extract_yaml_string_value(contents: &str, key: &str) -> Option<String> {
        let prefix = format!("{key}: \"");
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if let Some(rest) = trimmed.strip_prefix(&prefix)
                && let Some(end) = rest.find('"')
            {
                return Some(rest[..end].to_owned());
            }
        }
        None
    }

    /// Pull the python version floor out of `pyproject.toml`. The
    /// field is shaped `requires-python = ">=X.Y"`; this helper
    /// strips the `>=` (or bare `>`) prefix so callers compare the
    /// raw `X.Y` token rather than the comparator-bearing string.
    /// Returns `None` when the field is absent or carries no
    /// recognised version literal.
    fn extract_python_version_floor_from_pyproject(contents: &str) -> Option<String> {
        let value = extract_toml_string_value(contents, "requires-python")?;
        let trimmed = value.trim();
        let stripped = trimmed
            .strip_prefix(">=")
            .or_else(|| trimmed.strip_prefix('>'))
            .unwrap_or(trimmed);
        let candidate = stripped.trim();
        if candidate.is_empty() {
            None
        } else {
            Some(candidate.to_owned())
        }
    }

    /// Pull the python version CI installs out of a `uv python
    /// install <version>` shell line in a workflow yaml. The version
    /// token is the first whitespace-delimited word after the install
    /// command. Returns `None` when the file does not invoke
    /// `uv python install` at all.
    fn extract_uv_python_install_version(contents: &str) -> Option<String> {
        let needle = "uv python install ";
        for line in contents.lines() {
            if let Some(idx) = line.find(needle) {
                let after = &line[idx + needle.len()..];
                let token: String = after.chars().take_while(|c| !c.is_whitespace()).collect();
                if !token.is_empty() {
                    return Some(token);
                }
            }
        }
        None
    }

    #[test]
    fn isolate_schema_section_returns_empty_for_unknown_schema() {
        let yaml = "    components:\n    Foo:\n      type: object\n";
        assert!(isolate_schema_section(yaml, "Bar").is_empty());
    }

    #[test]
    fn isolate_schema_section_terminates_at_next_component_heading() {
        // Two adjacent schema components at the canonical 4-space
        // heading indent. The walker must capture every line under
        // `Foo` and stop the moment the next 4-space heading (`Bar:`)
        // appears, so generic fields like `name` under Bar never leak
        // into Foo's slice.
        let yaml = "
    Foo:
      type: object
      required: [a]
      properties:
        a:
          type: string
    Bar:
      type: object
      properties:
        name:
          type: string
";
        let section = isolate_schema_section(yaml, "Foo");
        assert!(section.contains("        a:"));
        assert!(
            !section.contains("name:"),
            "Foo's section must not bleed into Bar's properties; got:\n{section}",
        );
    }

    #[test]
    fn parse_required_set_handles_block_form() {
        let section = "\
    Foo:
      required:
        - a
        - b
      properties:
        a:
          type: string
        b:
          type: string
";
        let set = parse_required_set(section);
        assert!(set.contains("a"));
        assert!(set.contains("b"));
        assert_eq!(set.len(), 2);
    }

    #[test]
    fn parse_required_set_handles_inline_form() {
        let section = "\
    Foo:
      required: [a, b, c]
      properties:
        a:
          type: string
";
        let set = parse_required_set(section);
        assert!(set.contains("a"));
        assert!(set.contains("b"));
        assert!(set.contains("c"));
        assert_eq!(set.len(), 3);
    }

    #[test]
    fn parse_required_set_inline_form_drops_trailing_yaml_comment() {
        // Future-proofs the inline branch: if a maintainer adds a
        // trailing comment (perfectly valid YAML), the parser must
        // slice at `]` rather than trimming `]` off the end. Without
        // the `find(']')` slice this would yield an entry like
        // `b  # comment`.
        let section = "\
    Foo:
      required: [a, b]  # documents Foo's contract
      properties:
        a:
          type: string
";
        let set = parse_required_set(section);
        assert!(set.contains("a"));
        assert!(set.contains("b"));
        assert_eq!(set.len(), 2, "got: {set:?}");
    }

    #[test]
    fn parse_required_set_returns_empty_when_no_required_declared() {
        let section = "\
    Foo:
      type: object
      properties:
        a:
          type: string
";
        assert!(parse_required_set(section).is_empty());
    }

    #[test]
    fn parse_required_set_block_form_stops_at_next_key() {
        // The block-form scanner must release `in_block` the moment a
        // line at the schema-key indent (or any non-`        - `
        // prefix) appears, so neighboring keys like `properties` never
        // get mis-parsed as required entries.
        let section = "\
    Foo:
      required:
        - a
      properties:
        a:
          type: string
";
        let set = parse_required_set(section);
        assert_eq!(set.len(), 1);
        assert!(set.contains("a"));
        assert!(
            !set.iter().any(|name| name.contains("properties")),
            "block scan must not absorb sibling keys; got: {set:?}",
        );
    }

    #[test]
    fn parse_top_level_string_enum_returns_values_in_source_order() {
        let section = "\
    Foo:
      type: string
      enum: [first, second, third]
";
        let values = parse_top_level_string_enum(section);
        assert_eq!(values, vec!["first", "second", "third"]);
    }

    #[test]
    fn parse_top_level_string_enum_returns_empty_when_no_enum_declared() {
        let section = "\
    Foo:
      type: object
      properties:
        a:
          type: string
";
        assert!(parse_top_level_string_enum(section).is_empty());
    }

    #[test]
    fn parse_top_level_string_enum_drops_trailing_yaml_comment() {
        // Same defense as the inline `parse_required_set` scanner: slice
        // at the first `]` so a `# ...` comment after the closing
        // bracket cannot leak into the last entry.
        let section = "\
    Foo:
      type: string
      enum: [a, b]  # trailing note
";
        let values = parse_top_level_string_enum(section);
        assert_eq!(values, vec!["a", "b"]);
    }

    #[test]
    fn parse_oneof_discriminator_values_collects_each_branch_in_source_order() {
        let section = "\
    Foo:
      oneOf:
        - type: object
          properties:
            mode:
              type: string
              enum: [alpha]
        - type: object
          properties:
            mode:
              type: string
              enum: [beta]
";
        let values = parse_oneof_discriminator_values(section);
        assert_eq!(values, vec!["alpha", "beta"]);
    }

    #[test]
    fn parse_oneof_discriminator_values_returns_empty_when_no_branches_declared() {
        let section = "\
    Foo:
      type: string
      enum: [single]
";
        assert!(parse_oneof_discriminator_values(section).is_empty());
    }

    #[test]
    fn parse_property_string_enum_handles_inline_form() {
        let section = "\
    Foo:
      properties:
        style:
          type: string
          enum: [a, b, c]
        other:
          type: string
";
        let values = parse_property_string_enum(section, "style");
        assert_eq!(values, vec!["a", "b", "c"]);
    }

    #[test]
    fn parse_property_string_enum_handles_block_form_with_null() {
        // Block form including the openapi-style nullable marker. The
        // parser keeps `null` verbatim; the assert helper filters it
        // before cross-checking against the rust variants. A trailing
        // sibling property at eight-space indent must end the scan
        // without falling into its body.
        let section = "\
    Foo:
      properties:
        kind:
          type:
            - string
            - \"null\"
          enum:
            - first
            - second
            - null
        other:
          type: string
";
        let values = parse_property_string_enum(section, "kind");
        assert_eq!(values, vec!["first", "second", "null"]);
    }

    #[test]
    fn parse_property_string_enum_returns_empty_for_unknown_property() {
        let section = "\
    Foo:
      properties:
        style:
          type: string
          enum: [a]
";
        assert!(parse_property_string_enum(section, "kind").is_empty());
    }

    #[test]
    fn parse_property_string_enum_returns_empty_when_property_has_no_enum() {
        // Catches the regression where a property exists but only
        // declares `type:` — the helper must report an empty enum
        // (not silently pick up enum lines from an adjacent property).
        let section = "\
    Foo:
      properties:
        style:
          type: string
        kind:
          type: string
          enum: [a, b]
";
        assert!(parse_property_string_enum(section, "style").is_empty());
    }

    #[test]
    fn collect_schema_refs_handles_double_and_single_quoted_forms() {
        // Both quoting styles are valid YAML and the parser must
        // accept either. An unquoted `$ref: #/...` form would be
        // interpreted by YAML as `$ref:` (empty) with the rest as a
        // comment, so it is intentionally not included here.
        let yaml = "\
paths:
  /a:
    get:
      responses:
        \"200\":
          content:
            application/json:
              schema:
                $ref: \"#/components/schemas/Alpha\"
        \"201\":
          content:
            application/json:
              schema:
                $ref: '#/components/schemas/Beta'
";
        let refs = collect_schema_refs(yaml);
        assert!(refs.contains("Alpha"));
        assert!(refs.contains("Beta"));
    }

    #[test]
    fn collect_schema_refs_ignores_non_components_refs() {
        let yaml = "\
paths:
  /a:
    parameters:
      - $ref: \"#/components/parameters/SomeParam\"
    get:
      responses:
        \"200\":
          content:
            application/json:
              schema:
                $ref: \"#/components/schemas/Alpha\"
";
        let refs = collect_schema_refs(yaml);
        assert!(refs.contains("Alpha"));
        assert!(
            !refs.contains("SomeParam"),
            "non-schema refs must be ignored; got: {refs:?}",
        );
    }

    #[test]
    fn collect_defined_schema_names_skips_paths_and_deeper_indents() {
        let yaml = "\
paths:
  /v1/health:
    get:
      responses: {}
components:
  schemas:
    Foo:
      type: object
    Bar:
      type: object
      properties:
        nested_field:
          type: string
";
        let names = collect_defined_schema_names(yaml);
        // PascalCase 4-space-indent headings are picked up.
        assert!(names.contains("Foo"));
        assert!(names.contains("Bar"));
        // Path keys start with `/` and fail the uppercase guard.
        assert!(!names.iter().any(|n| n.starts_with('/')));
        // Deeper-indent fields like `nested_field:` (8 spaces) must
        // be excluded too.
        assert!(!names.contains("nested_field"));
    }

    #[test]
    fn extract_toml_string_value_matches_indent_and_quoted_value() {
        let toml = concat!(
            "[workspace.package]\n",
            "version = \"0.1.0\"\n",
            "rust-version = \"1.88\"\n",
            "\n",
            "[toolchain]\n",
            "  channel = \"1.90\"\n", // intentionally indented to prove trim
        );
        assert_eq!(
            extract_toml_string_value(toml, "rust-version").as_deref(),
            Some("1.88"),
        );
        assert_eq!(
            extract_toml_string_value(toml, "channel").as_deref(),
            Some("1.90"),
        );
    }

    #[test]
    fn extract_toml_string_value_returns_none_when_key_absent() {
        let toml = "version = \"0.1.0\"\n";
        assert!(extract_toml_string_value(toml, "rust-version").is_none());
    }

    #[test]
    fn extract_yaml_string_value_handles_indented_step_inputs() {
        let yaml = concat!(
            "jobs:\n",
            "  verify:\n",
            "    steps:\n",
            "      - uses: dtolnay/rust-toolchain@master\n",
            "        with:\n",
            "          toolchain: \"1.88\"\n",
            "          components: rustfmt, clippy\n",
        );
        assert_eq!(
            extract_yaml_string_value(yaml, "toolchain").as_deref(),
            Some("1.88"),
        );
    }

    #[test]
    fn extract_yaml_string_value_returns_none_when_key_absent() {
        let yaml = "jobs:\n  verify:\n    runs-on: ubuntu-latest\n";
        assert!(extract_yaml_string_value(yaml, "toolchain").is_none());
    }

    #[test]
    fn extract_python_version_floor_strips_comparator_prefix() {
        let toml = concat!(
            "[project]\n",
            "name = \"voicelayer-orchestrator\"\n",
            "requires-python = \">=3.12\"\n",
        );
        assert_eq!(
            extract_python_version_floor_from_pyproject(toml).as_deref(),
            Some("3.12"),
        );
    }

    #[test]
    fn extract_python_version_floor_handles_bare_greater_than_and_no_prefix() {
        let with_gt = "requires-python = \">3.11\"\n";
        assert_eq!(
            extract_python_version_floor_from_pyproject(with_gt).as_deref(),
            Some("3.11"),
        );
        let bare = "requires-python = \"3.10\"\n";
        assert_eq!(
            extract_python_version_floor_from_pyproject(bare).as_deref(),
            Some("3.10"),
        );
    }

    #[test]
    fn extract_python_version_floor_returns_none_when_field_absent() {
        let toml = "[project]\nname = \"foo\"\n";
        assert!(extract_python_version_floor_from_pyproject(toml).is_none());
    }

    #[test]
    fn extract_uv_python_install_version_reads_first_token_after_command() {
        let yaml = concat!(
            "      - name: Install Python\n",
            "        run: uv python install 3.12\n",
            "      - name: Sync deps\n",
            "        run: uv sync --group dev\n",
        );
        assert_eq!(
            extract_uv_python_install_version(yaml).as_deref(),
            Some("3.12"),
        );
    }

    #[test]
    fn extract_uv_python_install_version_returns_none_when_command_absent() {
        let yaml = "jobs:\n  verify:\n    steps:\n      - run: cargo build\n";
        assert!(extract_uv_python_install_version(yaml).is_none());
    }

    /// Pins both directions of the soft-list override:
    /// 1. With `flag` listed in `serde_default_fields`, the cross-check
    ///    returns `Ok(())` even though `flag` is non-null and missing
    ///    from `required`.
    /// 2. Without that override, the *same input* produces an `Err`
    ///    naming the mismatched field — proving the override is doing
    ///    real work, not just masking a no-op path.
    ///
    /// The check operates on an in-memory section + sentinel, so it
    /// pins `check_required_consistency` directly instead of needing
    /// a fake openapi file on disk.
    #[test]
    fn check_required_consistency_skips_serde_default_fields_only_when_listed() {
        #[derive(serde::Serialize)]
        struct Sample {
            id: String,
            flag: bool,
        }

        let section = "\
    Sample:
      required: [id]
      properties:
        id:
          type: string
        flag:
          type: boolean
";
        let value = serde_json::to_value(Sample {
            id: String::new(),
            flag: false,
        })
        .expect("serialise Sample");
        let object = value.as_object().expect("Sample is a JSON object");
        let required_set = parse_required_set(section);

        check_required_consistency("Sample", object, &required_set, &["flag"])
            .expect("soft-list override must silence the missing-required complaint for `flag`");

        let detail = check_required_consistency("Sample", object, &required_set, &[])
            .expect_err("without the override, `flag` must trip the required-list check");
        assert!(
            detail.contains("`flag`"),
            "the error must name the offending field; got: {detail}",
        );
        assert!(
            detail.contains("non-Option"),
            "the error must explain the rule that fired; got: {detail}",
        );
    }

    fn worker_health_summary_sentinel() -> WorkerHealthSummary {
        WorkerHealthSummary {
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
        }
    }

    fn capture_session_sentinel() -> CaptureSession {
        CaptureSession {
            session_id: Uuid::nil(),
            mode: SessionMode::Dictation,
            state: SessionState::Idle,
            trigger: TriggerKind::Cli,
            language_profile: LanguageProfile::default(),
            created_at_millis: 0,
        }
    }

    fn preview_artifact_sentinel() -> PreviewArtifact {
        PreviewArtifact {
            artifact_id: Uuid::nil(),
            status: PreviewStatus::Ready,
            title: String::new(),
            generated_text: None,
            notes: Vec::new(),
        }
    }

    fn transcription_result_sentinel() -> TranscriptionResult {
        TranscriptionResult {
            text: String::new(),
            detected_language: None,
            notes: Vec::new(),
        }
    }

    /// Materialise every `RecorderBackend` variant with a compile-time
    /// exhaustiveness guard. The unused closure forces a missing-arm
    /// error if a future variant is added without updating the list,
    /// so the openapi drift guard cannot silently fall behind the rust
    /// enum.
    fn enumerate_recorder_backend_variants() -> Vec<RecorderBackend> {
        let _exhaustive: fn(RecorderBackend) -> &'static str = |variant| match variant {
            RecorderBackend::Auto => "auto",
            RecorderBackend::Pipewire => "pipewire",
            RecorderBackend::Alsa => "alsa",
        };
        vec![
            RecorderBackend::Auto,
            RecorderBackend::Pipewire,
            RecorderBackend::Alsa,
        ]
    }

    /// Materialise every `SegmentationMode` variant. The exhaustiveness
    /// closure pins the discriminator labels every new variant must
    /// add to the openapi `oneOf` schema. The numeric payload values
    /// are placeholders — only the discriminator field feeds into the
    /// drift guard.
    fn enumerate_segmentation_mode_variants() -> Vec<SegmentationMode> {
        let _exhaustive: fn(SegmentationMode) -> &'static str = |variant| match variant {
            SegmentationMode::OneShot => "one_shot",
            SegmentationMode::Fixed { .. } => "fixed",
            SegmentationMode::VadGated { .. } => "vad_gated",
        };
        vec![
            SegmentationMode::OneShot,
            SegmentationMode::Fixed {
                segment_secs: 0,
                overlap_secs: 0,
            },
            SegmentationMode::VadGated {
                probe_secs: 0,
                max_segment_secs: 0,
                silence_gap_probes: 0,
            },
        ]
    }

    /// Materialise every `DictationFailureKind` variant. The
    /// exhaustiveness closure pins the wire labels alongside the
    /// openapi `failure_kind` enum entries; `null` is the absence of a
    /// variant and is contributed by `Option<_>`, not the enum itself.
    fn enumerate_dictation_failure_kind_variants() -> Vec<DictationFailureKind> {
        let _exhaustive: fn(DictationFailureKind) -> &'static str = |variant| match variant {
            DictationFailureKind::RecordingFailed => "recording_failed",
            DictationFailureKind::AsrFailed => "asr_failed",
            DictationFailureKind::InjectionFailed => "injection_failed",
        };
        vec![
            DictationFailureKind::RecordingFailed,
            DictationFailureKind::AsrFailed,
            DictationFailureKind::InjectionFailed,
        ]
    }

    /// Materialise every `RewriteStyle` variant. Same exhaustiveness
    /// pattern as the other enumerators — adding a future style fails
    /// compile until both this list and the openapi schema are updated.
    fn enumerate_rewrite_style_variants() -> Vec<RewriteStyle> {
        let _exhaustive: fn(RewriteStyle) -> &'static str = |variant| match variant {
            RewriteStyle::MoreFormal => "more_formal",
            RewriteStyle::Shorter => "shorter",
            RewriteStyle::Politer => "politer",
            RewriteStyle::MoreTechnical => "more_technical",
            RewriteStyle::Translate => "translate",
        };
        vec![
            RewriteStyle::MoreFormal,
            RewriteStyle::Shorter,
            RewriteStyle::Politer,
            RewriteStyle::MoreTechnical,
            RewriteStyle::Translate,
        ]
    }

    /// Materialise every `TriggerKind` variant. The exhaustiveness
    /// closure pins the wire labels every schema embedding this enum
    /// must list. The same set is duplicated inline under three
    /// schemas (`StartDictationRequest`, `DictationCaptureRequest`,
    /// `CaptureSession`) so each gets its own drift-guard test below.
    fn enumerate_trigger_kind_variants() -> Vec<TriggerKind> {
        let _exhaustive: fn(TriggerKind) -> &'static str = |variant| match variant {
            TriggerKind::PushToTalk => "push_to_talk",
            TriggerKind::Toggle => "toggle",
            TriggerKind::Cli => "cli",
            TriggerKind::Tui => "tui",
            TriggerKind::TrayButton => "tray_button",
        };
        vec![
            TriggerKind::PushToTalk,
            TriggerKind::Toggle,
            TriggerKind::Cli,
            TriggerKind::Tui,
            TriggerKind::TrayButton,
        ]
    }

    /// Materialise every `InjectTarget` variant. The same set is
    /// duplicated inline under `InjectRequest` and `InjectionPlan` so
    /// each gets its own drift-guard test below.
    fn enumerate_inject_target_variants() -> Vec<InjectTarget> {
        let _exhaustive: fn(InjectTarget) -> &'static str = |variant| match variant {
            InjectTarget::GuiAccessible => "gui_accessible",
            InjectTarget::GuiClipboard => "gui_clipboard",
            InjectTarget::TerminalBracketedPaste => "terminal_bracketed_paste",
            InjectTarget::TerminalKittyRemote => "terminal_kitty_remote",
        };
        vec![
            InjectTarget::GuiAccessible,
            InjectTarget::GuiClipboard,
            InjectTarget::TerminalBracketedPaste,
            InjectTarget::TerminalKittyRemote,
        ]
    }

    /// Materialise every `SessionMode` variant. Inlined under
    /// `CaptureSession.mode`; the exhaustiveness guard keeps the wire
    /// labels in lockstep with the openapi schema.
    fn enumerate_session_mode_variants() -> Vec<SessionMode> {
        let _exhaustive: fn(SessionMode) -> &'static str = |variant| match variant {
            SessionMode::Dictation => "dictation",
            SessionMode::Compose => "compose",
            SessionMode::Rewrite => "rewrite",
            SessionMode::Translate => "translate",
        };
        vec![
            SessionMode::Dictation,
            SessionMode::Compose,
            SessionMode::Rewrite,
            SessionMode::Translate,
        ]
    }

    /// Materialise every `SessionState` variant. Inlined under
    /// `CaptureSession.state`; the longest of the inline enums and
    /// the most exposed to drift since each new session-machine
    /// state must be added here as well as on the openapi side.
    fn enumerate_session_state_variants() -> Vec<SessionState> {
        let _exhaustive: fn(SessionState) -> &'static str = |variant| match variant {
            SessionState::Idle => "idle",
            SessionState::Listening => "listening",
            SessionState::Transcribing => "transcribing",
            SessionState::Previewing => "previewing",
            SessionState::AwaitingConfirmation => "awaiting_confirmation",
            SessionState::Completed => "completed",
            SessionState::Failed => "failed",
        };
        vec![
            SessionState::Idle,
            SessionState::Listening,
            SessionState::Transcribing,
            SessionState::Previewing,
            SessionState::AwaitingConfirmation,
            SessionState::Completed,
            SessionState::Failed,
        ]
    }

    /// Materialise every `LanguageStrategy` variant. Inlined under
    /// `LanguageProfile.strategy`. Drives `whisper-cli`'s `--language`
    /// flag selection on the daemon side.
    fn enumerate_language_strategy_variants() -> Vec<LanguageStrategy> {
        let _exhaustive: fn(LanguageStrategy) -> &'static str = |variant| match variant {
            LanguageStrategy::AutoDetect => "auto_detect",
            LanguageStrategy::Locked => "locked",
            LanguageStrategy::FollowPrevious => "follow_previous",
            LanguageStrategy::Bilingual => "bilingual",
        };
        vec![
            LanguageStrategy::AutoDetect,
            LanguageStrategy::Locked,
            LanguageStrategy::FollowPrevious,
            LanguageStrategy::Bilingual,
        ]
    }

    /// Materialise every `PreviewStatus` variant. Inlined under
    /// `PreviewArtifact.status`; gates whether the UI shows the
    /// `needs_provider` warning.
    fn enumerate_preview_status_variants() -> Vec<PreviewStatus> {
        let _exhaustive: fn(PreviewStatus) -> &'static str = |variant| match variant {
            PreviewStatus::Ready => "ready",
            PreviewStatus::NeedsProvider => "needs_provider",
            PreviewStatus::Rejected => "rejected",
        };
        vec![
            PreviewStatus::Ready,
            PreviewStatus::NeedsProvider,
            PreviewStatus::Rejected,
        ]
    }

    /// Materialise every `CompositionArchetype` variant. Inlined under
    /// `ComposeRequest.archetype`; the prompt-template registry on the
    /// worker side keys off these labels.
    fn enumerate_composition_archetype_variants() -> Vec<CompositionArchetype> {
        let _exhaustive: fn(CompositionArchetype) -> &'static str = |variant| match variant {
            CompositionArchetype::Email => "email",
            CompositionArchetype::CoverLetter => "cover_letter",
            CompositionArchetype::DailyReport => "daily_report",
            CompositionArchetype::Issue => "issue",
            CompositionArchetype::PullRequestDescription => "pull_request_description",
            CompositionArchetype::Prompt => "prompt",
            CompositionArchetype::TechnicalSummary => "technical_summary",
            CompositionArchetype::Custom => "custom",
        };
        vec![
            CompositionArchetype::Email,
            CompositionArchetype::CoverLetter,
            CompositionArchetype::DailyReport,
            CompositionArchetype::Issue,
            CompositionArchetype::PullRequestDescription,
            CompositionArchetype::Prompt,
            CompositionArchetype::TechnicalSummary,
            CompositionArchetype::Custom,
        ]
    }

    /// Materialise every `ProviderKind` variant. Inlined under
    /// `ProviderDescriptor.kind`. Adding a new provider category
    /// (e.g. a `Vad` kind) must update both this list and the openapi
    /// schema.
    fn enumerate_provider_kind_variants() -> Vec<ProviderKind> {
        let _exhaustive: fn(ProviderKind) -> &'static str = |variant| match variant {
            ProviderKind::Asr => "asr",
            ProviderKind::Llm => "llm",
            ProviderKind::Tts => "tts",
            ProviderKind::HostAdapter => "host_adapter",
        };
        vec![
            ProviderKind::Asr,
            ProviderKind::Llm,
            ProviderKind::Tts,
            ProviderKind::HostAdapter,
        ]
    }

    /// Sentinel for `ProviderDescriptor`. Strings empty / booleans
    /// false so the cross-check exercises the property-presence and
    /// required-list contract without depending on real catalog data.
    /// The `kind` arm is materialised explicitly to keep the sentinel
    /// readable; it's unrelated to the variant set the property-enum
    /// guard pins.
    fn provider_descriptor_sentinel() -> ProviderDescriptor {
        ProviderDescriptor {
            id: String::new(),
            kind: ProviderKind::Asr,
            transport: String::new(),
            local: false,
            default_enabled: false,
            experimental: false,
            license: String::new(),
        }
    }

    #[test]
    fn openapi_worker_health_summary_documents_every_field() {
        assert_openapi_documents_every_field(
            "WorkerHealthSummary",
            &worker_health_summary_sentinel(),
            &[],
        );
    }

    #[test]
    fn openapi_health_response_documents_every_field() {
        let sentinel = HealthResponse {
            status: String::new(),
            socket_path: String::new(),
            version: String::new(),
            worker: worker_health_summary_sentinel(),
        };
        assert_openapi_documents_every_field("HealthResponse", &sentinel, &[]);
    }

    #[test]
    fn openapi_capture_session_documents_every_field() {
        assert_openapi_documents_every_field("CaptureSession", &capture_session_sentinel(), &[]);
    }

    #[test]
    fn openapi_preview_artifact_documents_every_field() {
        assert_openapi_documents_every_field("PreviewArtifact", &preview_artifact_sentinel(), &[]);
    }

    #[test]
    fn openapi_composition_receipt_documents_every_field() {
        let sentinel = CompositionReceipt {
            job_id: Uuid::nil(),
            preview: preview_artifact_sentinel(),
        };
        assert_openapi_documents_every_field("CompositionReceipt", &sentinel, &[]);
    }

    #[test]
    fn openapi_injection_plan_documents_every_field() {
        let sentinel = InjectionPlan {
            target: InjectTarget::GuiAccessible,
            payload: String::new(),
            auto_submit: false,
        };
        assert_openapi_documents_every_field("InjectionPlan", &sentinel, &[]);
    }

    #[test]
    fn openapi_transcription_result_documents_every_field() {
        assert_openapi_documents_every_field(
            "TranscriptionResult",
            &transcription_result_sentinel(),
            &[],
        );
    }

    #[test]
    fn openapi_dictation_capture_result_documents_every_field() {
        let sentinel = DictationCaptureResult {
            session: capture_session_sentinel(),
            transcription: transcription_result_sentinel(),
            audio_file: None,
            failure_kind: None,
        };
        assert_openapi_documents_every_field("DictationCaptureResult", &sentinel, &[]);
    }

    // Request-side drift guards. Pattern matches the response-side
    // ones above, with the addition that fields annotated with
    // `#[serde(default)]` on the Rust struct are passed in
    // `serde_default_fields` so the required-list cross-check tolerates
    // either listing them under `required:` or omitting them — the
    // wire contract is "may be absent on input, daemon defaults".

    #[test]
    fn openapi_start_dictation_request_documents_every_field() {
        let sentinel = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: None,
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::OneShot,
        };
        assert_openapi_documents_every_field(
            "StartDictationRequest",
            &sentinel,
            &["translate_to_english", "keep_audio", "segmentation"],
        );
    }

    #[test]
    fn openapi_stop_dictation_request_documents_every_field() {
        let sentinel = StopDictationRequest {
            session_id: Uuid::nil(),
        };
        assert_openapi_documents_every_field("StopDictationRequest", &sentinel, &[]);
    }

    #[test]
    fn openapi_dictation_capture_request_documents_every_field() {
        let sentinel = DictationCaptureRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            duration_seconds: 0,
            recorder_backend: None,
            translate_to_english: false,
            keep_audio: false,
        };
        assert_openapi_documents_every_field(
            "DictationCaptureRequest",
            &sentinel,
            &["translate_to_english", "keep_audio"],
        );
    }

    #[test]
    fn openapi_compose_request_documents_every_field() {
        let sentinel = ComposeRequest {
            spoken_prompt: String::new(),
            archetype: None,
            output_language: None,
        };
        assert_openapi_documents_every_field("ComposeRequest", &sentinel, &[]);
    }

    #[test]
    fn openapi_rewrite_request_documents_every_field() {
        let sentinel = RewriteRequest {
            source_text: String::new(),
            style: RewriteStyle::MoreFormal,
            output_language: None,
        };
        assert_openapi_documents_every_field("RewriteRequest", &sentinel, &[]);
    }

    #[test]
    fn openapi_translate_request_documents_every_field() {
        let sentinel = TranslateRequest {
            source_text: String::new(),
            target_language: String::new(),
        };
        assert_openapi_documents_every_field("TranslateRequest", &sentinel, &[]);
    }

    #[test]
    fn openapi_transcribe_request_documents_every_field() {
        let sentinel = TranscribeRequest {
            audio_file: String::new(),
            language: None,
            translate_to_english: false,
        };
        assert_openapi_documents_every_field(
            "TranscribeRequest",
            &sentinel,
            &["translate_to_english"],
        );
    }

    #[test]
    fn openapi_inject_request_documents_every_field() {
        let sentinel = InjectRequest {
            target: InjectTarget::GuiAccessible,
            text: String::new(),
            auto_submit: false,
        };
        assert_openapi_documents_every_field("InjectRequest", &sentinel, &["auto_submit"]);
    }

    // Enum-shape drift guards. Pair every top-level enum schema with
    // its rust counterpart so adding a variant to one and forgetting
    // the other (in either direction) trips the test instead of
    // shipping silently. The struct-shape guards above only enforce
    // property presence and the required-list contract; they do not
    // pin the variant set of an embedded enum.

    #[test]
    fn openapi_recorder_backend_documents_every_variant() {
        assert_openapi_documents_every_string_enum_variant(
            "RecorderBackend",
            &enumerate_recorder_backend_variants(),
        );
    }

    #[test]
    fn openapi_segmentation_mode_documents_every_discriminator() {
        assert_openapi_documents_every_oneof_discriminator(
            "SegmentationMode",
            "mode",
            &enumerate_segmentation_mode_variants(),
        );
    }

    // Property-scoped enum drift guards. Pin enums that live inside
    // a parent schema's `properties` block instead of as a top-level
    // component. The struct-shape and top-level-enum guards above do
    // not cover this case, so a forgotten `failure_kind` variant
    // (rust adds, openapi doesn't, or vice versa) would slip through.

    #[test]
    fn openapi_dictation_capture_result_failure_kind_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "DictationCaptureResult",
            "failure_kind",
            &enumerate_dictation_failure_kind_variants(),
        );
    }

    #[test]
    fn openapi_rewrite_request_style_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "RewriteRequest",
            "style",
            &enumerate_rewrite_style_variants(),
        );
    }

    // `TriggerKind` is duplicated inline under three schemas. Pin
    // each occurrence so a drift in any one of them — e.g. someone
    // forgets `tray_button` while rewriting one schema's enum — is
    // caught instead of silently desynchronising the contract.

    #[test]
    fn openapi_start_dictation_request_trigger_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "StartDictationRequest",
            "trigger",
            &enumerate_trigger_kind_variants(),
        );
    }

    #[test]
    fn openapi_dictation_capture_request_trigger_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "DictationCaptureRequest",
            "trigger",
            &enumerate_trigger_kind_variants(),
        );
    }

    #[test]
    fn openapi_capture_session_trigger_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "CaptureSession",
            "trigger",
            &enumerate_trigger_kind_variants(),
        );
    }

    // `InjectTarget` is duplicated inline under two schemas; same
    // rationale.

    #[test]
    fn openapi_inject_request_target_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "InjectRequest",
            "target",
            &enumerate_inject_target_variants(),
        );
    }

    #[test]
    fn openapi_injection_plan_target_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "InjectionPlan",
            "target",
            &enumerate_inject_target_variants(),
        );
    }

    // Single-occurrence inline-enum drift guards. Each rust enum is
    // embedded under exactly one schema's properties block; pinning
    // them caps the property-enum drift surface for the contract.

    #[test]
    fn openapi_capture_session_mode_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "CaptureSession",
            "mode",
            &enumerate_session_mode_variants(),
        );
    }

    #[test]
    fn openapi_capture_session_state_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "CaptureSession",
            "state",
            &enumerate_session_state_variants(),
        );
    }

    #[test]
    fn openapi_language_profile_strategy_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "LanguageProfile",
            "strategy",
            &enumerate_language_strategy_variants(),
        );
    }

    #[test]
    fn openapi_preview_artifact_status_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "PreviewArtifact",
            "status",
            &enumerate_preview_status_variants(),
        );
    }

    #[test]
    fn openapi_compose_request_archetype_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "ComposeRequest",
            "archetype",
            &enumerate_composition_archetype_variants(),
        );
    }

    // `ProviderDescriptor` lives in `provider.rs` but its openapi
    // contract sits next to the rest of the schema components.
    // Closing the drift surface for the last public component:

    #[test]
    fn openapi_provider_descriptor_documents_every_field() {
        assert_openapi_documents_every_field(
            "ProviderDescriptor",
            &provider_descriptor_sentinel(),
            &[],
        );
    }

    #[test]
    fn openapi_provider_descriptor_kind_documents_every_variant() {
        assert_openapi_documents_every_property_enum_variant(
            "ProviderDescriptor",
            "kind",
            &enumerate_provider_kind_variants(),
        );
    }

    /// Structural integrity check on the openapi YAML: every
    /// `$ref: "#/components/schemas/<Name>"` must point to a schema
    /// component that the file actually declares. Unlike the rust↔
    /// openapi drift guards above, this test is purely a YAML lint
    /// — it catches typos and rename-without-update regressions
    /// inside the contract file itself, regardless of whether the
    /// rust side has matching types. A broken `$ref` would still
    /// load via `serde_json::from_str` (the daemon never resolves
    /// refs at runtime), so the only place to catch it is at
    /// build/test time.
    #[test]
    fn openapi_every_schema_ref_resolves_to_a_defined_component() {
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );
        let contents =
            std::fs::read_to_string(&openapi_path).expect("openapi contract file should exist");

        let defined = collect_defined_schema_names(&contents);
        let refs = collect_schema_refs(&contents);
        assert!(
            !refs.is_empty(),
            "expected at least one $ref in the openapi file; \
             collect_schema_refs may be misparsing",
        );

        for reference in &refs {
            assert!(
                defined.contains(reference),
                "openapi `$ref: #/components/schemas/{reference}` does not resolve — \
                 schema is not declared. Either fix the typo or add the missing schema. \
                 Defined components: {defined:?}",
            );
        }
    }

    /// The rust toolchain pin lives in three places that must agree:
    /// `rust-toolchain.toml` (rustup), the `[workspace.package]
    /// rust-version` in the root `Cargo.toml` (used by `cargo` to
    /// reject older toolchains), and the `toolchain:` step input in
    /// `.github/workflows/ci.yml` (which downloads the version CI
    /// runs). The header comment in `rust-toolchain.toml` calls out
    /// the cross-file requirement, but no test enforced it — this
    /// pin closes that drift so a future bump must update all three
    /// or fail at `cargo test` instead of silently letting CI run
    /// against a different toolchain than developers do.
    #[test]
    fn rust_toolchain_pin_is_consistent_across_repo_files() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));

        let toolchain_toml = std::fs::read_to_string(format!("{repo_root}/rust-toolchain.toml"))
            .expect("read rust-toolchain.toml");
        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let ci_yml = std::fs::read_to_string(format!("{repo_root}/.github/workflows/ci.yml"))
            .expect("read .github/workflows/ci.yml");

        let rustup_channel = extract_toml_string_value(&toolchain_toml, "channel").expect(
            "rust-toolchain.toml must declare `channel = \"...\"`; \
             extract_toml_string_value may be misparsing",
        );
        let cargo_rust_version = extract_toml_string_value(&cargo_toml, "rust-version").expect(
            "Cargo.toml must declare `rust-version = \"...\"`; \
             extract_toml_string_value may be misparsing",
        );
        let ci_toolchain = extract_yaml_string_value(&ci_yml, "toolchain").expect(
            ".github/workflows/ci.yml must declare `toolchain: \"...\"`; \
             extract_yaml_string_value may be misparsing",
        );

        assert_eq!(
            rustup_channel, cargo_rust_version,
            "rust-toolchain.toml `channel = \"{rustup_channel}\"` does not match \
             Cargo.toml `rust-version = \"{cargo_rust_version}\"`. Bump both together \
             so `cargo` and `rustup` agree on the floor.",
        );
        assert_eq!(
            rustup_channel, ci_toolchain,
            "rust-toolchain.toml `channel = \"{rustup_channel}\"` does not match \
             ci.yml `toolchain: \"{ci_toolchain}\"`. CI must run on the same toolchain \
             developers use locally.",
        );
    }

    /// Mirror of `rust_toolchain_pin_is_consistent_across_repo_files`
    /// for python: `pyproject.toml`'s `requires-python` floor must
    /// match the version `ci.yml` installs via `uv python install`.
    /// A bump that updates one without the other would let CI run on
    /// a newer interpreter than the manifest claims to support — or
    /// the inverse, where `pyproject.toml` widens the floor but CI
    /// still runs on the older version.
    #[test]
    fn python_version_pin_is_consistent_across_pyproject_and_ci() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));

        let pyproject = std::fs::read_to_string(format!("{repo_root}/pyproject.toml"))
            .expect("read pyproject.toml");
        let ci_yml = std::fs::read_to_string(format!("{repo_root}/.github/workflows/ci.yml"))
            .expect("read .github/workflows/ci.yml");

        let pyproject_floor = extract_python_version_floor_from_pyproject(&pyproject).expect(
            "pyproject.toml must declare `requires-python = \">=...\"`; \
                 extract_python_version_floor_from_pyproject may be misparsing",
        );
        let ci_install = extract_uv_python_install_version(&ci_yml).expect(
            ".github/workflows/ci.yml must invoke `uv python install <version>`; \
             extract_uv_python_install_version may be misparsing",
        );

        assert_eq!(
            pyproject_floor, ci_install,
            "python version drift: pyproject.toml `requires-python = \">={pyproject_floor}\"` \
             but ci.yml runs `uv python install {ci_install}`. Update both together so the \
             test matrix matches the manifest's declared floor.",
        );
    }
}
