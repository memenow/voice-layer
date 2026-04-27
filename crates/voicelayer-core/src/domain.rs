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

    /// Pull a YAML scalar value bound to `key`, accepting either the
    /// bare form (`key: value`) or a quoted form (`key: "value"` /
    /// `key: 'value'`). Stops at the first whitespace or quote
    /// inside the value, so a trailing `# comment` cannot leak into
    /// the token. Companion to `extract_yaml_string_value`, which
    /// requires the quoted form — this helper is for fields like
    /// openapi's `info.version: 0.1.0` written as a bare scalar.
    fn extract_yaml_bare_string_value(contents: &str, key: &str) -> Option<String> {
        let prefix = format!("{key}: ");
        for line in contents.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix(&prefix) else {
                continue;
            };
            let stripped = rest.trim_start_matches(['"', '\'']);
            let token: String = stripped
                .chars()
                .take_while(|c| !c.is_whitespace() && *c != '"' && *c != '\'')
                .collect();
            if !token.is_empty() {
                return Some(token);
            }
        }
        None
    }

    /// Walk a Python source string and collect every JSON-RPC error
    /// code defined as a `_CODE = -3XXXX` constant. Targets the
    /// constants in `python/voicelayer_orchestrator/worker.py`
    /// (`PROVIDER_UNAVAILABLE_CODE`, `METHOD_NOT_FOUND_CODE`, etc.).
    /// Filters to the JSON-RPC reserved range (`-32099` to `-32700`)
    /// so unrelated negative numerics elsewhere in the file would
    /// not pollute the set.
    fn extract_python_jsonrpc_error_codes(contents: &str) -> std::collections::BTreeSet<i32> {
        let mut codes = std::collections::BTreeSet::new();
        for line in contents.lines() {
            let Some(idx) = line.find("_CODE = ") else {
                continue;
            };
            let after = &line[idx + "_CODE = ".len()..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '-')
                .collect();
            if let Ok(value) = token.parse::<i32>()
                && (-32700..=-32000).contains(&value)
            {
                codes.insert(value);
            }
        }
        codes
    }

    /// Walk a Markdown doc string and collect every backtick-quoted
    /// JSON-RPC error code (`` `-3XXXX` ``). Returns the dedup'd set
    /// of values; the doc may mention the same code in several
    /// passages (e.g. an Error Policy enumeration plus inline usage
    /// notes), but for the drift guard only presence matters. Same
    /// JSON-RPC range filter as `extract_python_jsonrpc_error_codes`.
    fn extract_doc_jsonrpc_error_codes(contents: &str) -> std::collections::BTreeSet<i32> {
        let mut codes = std::collections::BTreeSet::new();
        for line in contents.lines() {
            let mut search = line;
            while let Some(idx) = search.find("`-") {
                let after = &search[idx + 1..]; // start at "-..."
                let token: String = after
                    .chars()
                    .take_while(|c| c.is_ascii_digit() || *c == '-')
                    .collect();
                if let Ok(value) = token.parse::<i32>()
                    && (-32700..=-32000).contains(&value)
                {
                    codes.insert(value);
                }
                search = &after[token.len()..];
                if search.is_empty() {
                    break;
                }
            }
        }
        codes
    }

    /// Walk a Cargo.toml string and pull out every quoted path inside
    /// the `[workspace] members = [...]` array. The parser opens at
    /// the `members = [` line and closes at the next `]` so trailing
    /// top-level keys (`resolver`, `[workspace.package]`, etc.) do
    /// not leak in. Each entry is the path verbatim
    /// (e.g. `"crates/voicelayer-core"`).
    fn extract_cargo_workspace_members(contents: &str) -> std::collections::BTreeSet<String> {
        let mut members = std::collections::BTreeSet::new();
        let mut in_block = false;
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("members = [") {
                in_block = true;
                continue;
            }
            if in_block {
                if trimmed == "]" {
                    break;
                }
                let unquoted = trimmed
                    .trim_start_matches('"')
                    .trim_end_matches(',')
                    .trim_end_matches('"');
                if !unquoted.is_empty() {
                    members.insert(unquoted.to_owned());
                }
            }
        }
        members
    }

    /// Pull the SPDX literal from pyproject's inline-table license
    /// form: `license = { text = "Apache-2.0" }`. Returns `None`
    /// when the field is absent or written in any other shape (e.g.
    /// `license = "..."` for a bare-string form, which pyproject
    /// also accepts but voicelayer doesn't use today).
    fn extract_pyproject_license_text(contents: &str) -> Option<String> {
        for line in contents.lines() {
            let trimmed = line.trim_start();
            if !trimmed.starts_with("license = ") {
                continue;
            }
            let needle = "text = \"";
            if let Some(idx) = line.find(needle) {
                let after = &line[idx + needle.len()..];
                if let Some(end) = after.find('"') {
                    return Some(after[..end].to_owned());
                }
            }
        }
        None
    }

    /// Walk a Markdown body and pull out every backtick-quoted file
    /// path that contains at least one `/` and ends in a recognised
    /// source extension. Filters out runtime paths
    /// (`~/.config/...`, `/usr/local/...`, `${HOME}/...`) by
    /// requiring the first character to be an ASCII lowercase
    /// letter — every repo-relative path the docs reference today
    /// starts with one of `crates/`, `docs/`, `openapi/`, `python/`,
    /// `providers/`, `scripts/`, `systemd/`, or `tests/`. Returns
    /// the raw path text — callers are responsible for resolving
    /// shorthand conventions (e.g. `providers/foo.py` →
    /// `python/voicelayer_orchestrator/providers/foo.py`).
    fn extract_doc_file_path_refs(contents: &str) -> std::collections::BTreeSet<String> {
        let exts = [".rs", ".py", ".toml", ".sh", ".yaml", ".md", ".service"];
        let mut paths = std::collections::BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find('`') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let inner = &after[..close];
            search = &after[close + 1..];
            if !inner.contains('/') {
                continue;
            }
            if !exts.iter().any(|ext| inner.ends_with(ext)) {
                continue;
            }
            if inner.chars().any(char::is_whitespace) {
                continue;
            }
            if !inner.chars().next().is_some_and(|c| c.is_ascii_lowercase()) {
                continue;
            }
            paths.insert(inner.to_owned());
        }
        paths
    }

    /// Walk a Markdown body and pull out every `[text](path.md)` or
    /// `[text](path.md#fragment)` link target whose path component
    /// ends in `.md`. Skips absolute URLs (`http://`, `https://`,
    /// `mailto:`) — those are not local cross-references and live
    /// outside the repo's drift surface. The fragment is stripped
    /// before matching the `.md` suffix so anchored links resolve
    /// to the same file.
    fn extract_markdown_md_links(contents: &str) -> Vec<String> {
        let mut links = Vec::new();
        let mut search = contents;
        while let Some(idx) = search.find("](") {
            let after = &search[idx + "](".len()..];
            let Some(close) = after.find(')') else {
                break;
            };
            let link = &after[..close];
            search = &after[close + 1..];
            if link.starts_with("http://")
                || link.starts_with("https://")
                || link.starts_with("mailto:")
            {
                continue;
            }
            let path_part = link.split('#').next().unwrap_or(link);
            if path_part.ends_with(".md") {
                links.push(path_part.to_owned());
            }
        }
        links
    }

    /// Walk an `install.sh` shell script and pull out every basename
    /// it copies into `${BIN_DIR}/`. Targets the canonical line
    /// shape `install -m 0755 "..." "${BIN_DIR}/<name>"` (the form
    /// `scripts/install.sh` uses today). The basename runs from
    /// `${BIN_DIR}/` to the next `"`.
    fn extract_install_sh_bin_targets(source: &str) -> std::collections::BTreeSet<String> {
        let needle = "\"${BIN_DIR}/";
        let mut targets = std::collections::BTreeSet::new();
        for line in source.lines() {
            if let Some(idx) = line.find(needle) {
                let after = &line[idx + needle.len()..];
                if let Some(end) = after.find('"')
                    && !after[..end].is_empty()
                {
                    targets.insert(after[..end].to_owned());
                }
            }
        }
        targets
    }

    /// Walk an `install.sh` shell script and pull out every binary
    /// name passed to `cargo build` via `--bin <name>`. The match is
    /// purely textual — a line like
    /// `cargo build --release --bin vl --bin vl-desktop` returns
    /// `{"vl", "vl-desktop"}`. Stops at whitespace or a `\`
    /// continuation marker, so a name with internal hyphens
    /// (`vl-desktop`) is captured intact.
    fn extract_install_sh_cargo_build_bins(source: &str) -> std::collections::BTreeSet<String> {
        let mut bins = std::collections::BTreeSet::new();
        let mut search = source;
        while let Some(idx) = search.find("--bin ") {
            let after = &search[idx + "--bin ".len()..];
            let end = after
                .find(|c: char| c.is_whitespace() || matches!(c, '\\' | ')' | '"' | '\''))
                .unwrap_or(after.len());
            let name = after[..end].trim();
            if !name.is_empty() {
                bins.insert(name.to_owned());
            }
            search = &after[end..];
        }
        bins
    }

    /// Walk an `install.sh` shell script and pull out every
    /// `(source, dest)` pair from `install -m <mode> "<source>" "<dest>"`
    /// invocations. Multi-line install commands are collapsed via
    /// `\\\n` → ` ` so the source and dest end up on the same logical
    /// line before scanning.
    ///
    /// Returns shell literals as-written, so `${REPO_ROOT}/...` and
    /// `${BIN_DIR}/...` are preserved. Callers do their own
    /// substitution (via `strip_prefix`) when resolving against the
    /// real filesystem.
    fn extract_install_sh_install_pairs(source: &str) -> Vec<(String, String)> {
        let mut pairs = Vec::new();
        let collapsed = source.replace("\\\n", " ");
        for line in collapsed.lines() {
            let trimmed = line.trim();
            if !trimmed.starts_with("install -m") {
                continue;
            }
            let mut search = trimmed;
            let mut quoted: Vec<String> = Vec::new();
            while let Some(idx) = search.find('"') {
                let after = &search[idx + 1..];
                let Some(close) = after.find('"') else {
                    break;
                };
                quoted.push(after[..close].to_owned());
                search = &after[close + 1..];
            }
            if quoted.len() >= 2 {
                pairs.push((quoted[0].clone(), quoted[1].clone()));
            }
        }
        pairs
    }

    /// Walk a systemd `.service` unit file and pull out the path
    /// from every `EnvironmentFile=` line, dropping the optional
    /// leading `-` (which marks the file as optional rather than
    /// required). Returns paths exactly as written, so a `%h/...`
    /// runtime placeholder is preserved for the caller to expand.
    fn extract_systemd_environment_file_paths(source: &str) -> Vec<String> {
        let mut paths = Vec::new();
        for line in source.lines() {
            let trimmed = line.trim();
            if let Some(rest) = trimmed.strip_prefix("EnvironmentFile=") {
                let path = rest.strip_prefix('-').unwrap_or(rest).trim().to_owned();
                if !path.is_empty() {
                    paths.push(path);
                }
            }
        }
        paths
    }

    /// Pull the default value of the `ENV_DIR` shell variable from
    /// `scripts/install.sh`. Targets the canonical bash idiom
    /// `ENV_DIR="${VOICELAYER_INSTALL_ENV_DIR:-${HOME}/.config/voicelayer}"`,
    /// returning the `:-`-default substring (here
    /// `${HOME}/.config/voicelayer`). Returns `None` if the variable
    /// is missing or written in a different shape.
    fn extract_install_sh_default_env_dir(source: &str) -> Option<String> {
        for line in source.lines() {
            let trimmed = line.trim();
            let Some(rest) = trimmed.strip_prefix("ENV_DIR=\"") else {
                continue;
            };
            let idx = rest.find(":-")?;
            let after = &rest[idx + ":-".len()..];
            let end = after.find("}\"")?;
            return Some(after[..end].to_owned());
        }
        None
    }

    /// Walk a systemd `.service` unit file and pull out the basename
    /// of every `ExecStart=%h/.local/bin/<name>` line. systemd
    /// expands `%h` to the user's home directory at runtime; the
    /// basename is what install.sh must put on disk.
    fn extract_systemd_execstart_basenames(source: &str) -> std::collections::BTreeSet<String> {
        let prefix = "ExecStart=%h/.local/bin/";
        let mut basenames = std::collections::BTreeSet::new();
        for line in source.lines() {
            if let Some(rest) = line.strip_prefix(prefix) {
                let basename: String = rest.chars().take_while(|c| !c.is_whitespace()).collect();
                if !basename.is_empty() {
                    basenames.insert(basename);
                }
            }
        }
        basenames
    }

    /// Walk a Rust source string and pull out every shortcut id
    /// declared as `pub const SHORTCUT_<NAME>: &str = "<value>";`.
    /// Targets `vl-desktop::portal::SHORTCUT_TOGGLE` and any future
    /// siblings. The fixed shape leaves no room for fuzzy matches:
    /// a literal that is not a shortcut const cannot satisfy the
    /// prefix.
    fn collect_rust_shortcut_ids(source: &str) -> std::collections::BTreeSet<String> {
        let mut ids = std::collections::BTreeSet::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("pub const SHORTCUT_") else {
                continue;
            };
            let Some(equals_pos) = rest.find(" = \"") else {
                continue;
            };
            let after_equals = &rest[equals_pos + " = \"".len()..];
            if let Some(end) = after_equals.find('"') {
                ids.insert(after_equals[..end].to_owned());
            }
        }
        ids
    }

    /// Walk a `Cargo.toml` body and pull out every dependency name
    /// declared inside the `[workspace.dependencies]` section. The
    /// scanner uses brace-depth tracking so a multi-line inline
    /// table value (e.g. a long `features = [...]` array continued
    /// across several lines) does not leak its sub-keys as fake
    /// dep names. Comment lines, section headings, and continuation
    /// rows are all rejected by the same `classify_cargo_dep_header`
    /// helper that classifies member-crate dep entries.
    fn extract_workspace_dependency_names(contents: &str) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        let mut in_block = false;
        let mut depth: i32 = 0;
        for line in contents.lines() {
            let trimmed = line.trim();
            if depth == 0 && trimmed.starts_with('[') {
                in_block = trimmed.starts_with("[workspace.dependencies]");
                continue;
            }
            if in_block
                && depth == 0
                && let Some((name, _kind)) = classify_cargo_dep_header(trimmed)
            {
                names.insert(name);
            }
            let opens = line.matches('{').count() as i32;
            let closes = line.matches('}').count() as i32;
            depth = (depth + opens - closes).max(0);
        }
        names
    }

    /// Classify every dependency entry in a member crate's
    /// `Cargo.toml` `[dependencies]`, `[dev-dependencies]`, and
    /// `[build-dependencies]` blocks. Returns two disjoint sets:
    ///
    /// - `workspace_inherited`: declared as `<name>.workspace = true`
    ///   (or any `.workspace = ...` form, in case future entries
    ///   override individual fields).
    /// - `inline_or_literal`: any other shape — bare `<name> = "1.0"`
    ///   literal-version pin, single- or multi-line inline table
    ///   `<name> = { version = "1.0", features = [...] }`, or path
    ///   dep `<name> = { path = "..." }`.
    ///
    /// Path deps are intentionally lumped with literal pins: they
    /// never appear in `[workspace.dependencies]`, so the test that
    /// intersects this set with the workspace dep names will always
    /// drop them. Bundling them keeps the helper focused on the one
    /// distinction that matters: workspace-inherited vs. not.
    fn classify_crate_dependency_forms(
        contents: &str,
    ) -> (
        std::collections::BTreeSet<String>,
        std::collections::BTreeSet<String>,
    ) {
        let mut workspace_inherited = std::collections::BTreeSet::new();
        let mut inline_or_literal = std::collections::BTreeSet::new();
        let mut in_dep_block = false;
        let mut depth: i32 = 0;
        for line in contents.lines() {
            let trimmed = line.trim();
            if depth == 0 && trimmed.starts_with('[') {
                in_dep_block = trimmed.starts_with("[dependencies]")
                    || trimmed.starts_with("[dev-dependencies]")
                    || trimmed.starts_with("[build-dependencies]");
                continue;
            }
            if in_dep_block
                && depth == 0
                && let Some((name, kind)) = classify_cargo_dep_header(trimmed)
            {
                match kind {
                    CargoDepKind::WorkspaceInherited => {
                        workspace_inherited.insert(name);
                    }
                    CargoDepKind::InlineOrLiteral => {
                        inline_or_literal.insert(name);
                    }
                }
            }
            let opens = line.matches('{').count() as i32;
            let closes = line.matches('}').count() as i32;
            depth = (depth + opens - closes).max(0);
        }
        (workspace_inherited, inline_or_literal)
    }

    enum CargoDepKind {
        WorkspaceInherited,
        InlineOrLiteral,
    }

    /// Parse a single dep-header candidate line and return the dep
    /// name plus whether it uses workspace inheritance. A header line
    /// has the shape `<name> = ...` or `<name>.workspace = ...`,
    /// where `<name>` is `[a-zA-Z][a-zA-Z0-9_-]*`. Continuation lines,
    /// comments, and blank lines all return `None`.
    fn classify_cargo_dep_header(trimmed: &str) -> Option<(String, CargoDepKind)> {
        let no_comment = trimmed.split('#').next()?.trim();
        if no_comment.is_empty() {
            return None;
        }
        let (lhs, rhs) = no_comment.split_once('=')?;
        if rhs.trim().is_empty() {
            return None;
        }
        let lhs = lhs.trim();
        let (name, suffix) = lhs
            .split_once('.')
            .map_or((lhs, ""), |(n, s)| (n, s.trim()));
        if name.is_empty()
            || !name.starts_with(|c: char| c.is_ascii_alphabetic())
            || !name
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '_' || c == '-')
        {
            return None;
        }
        let kind = if suffix == "workspace" {
            CargoDepKind::WorkspaceInherited
        } else {
            CargoDepKind::InlineOrLiteral
        };
        Some((name.to_owned(), kind))
    }

    /// Walk a Markdown body and pull out every backtick-quoted
    /// `VOICELAYER_<NAME>` literal where `<NAME>` is uppercase
    /// letters, digits, and underscores. Rejects two prose
    /// patterns the docs use today:
    /// - `` `VOICELAYER_FOO_*` `` — wildcard mentions where the
    ///   trailing `*` is dropped during scan and the captured text
    ///   ends in `_`.
    /// - Any literal containing whitespace, which is prose, not a
    ///   real env var name.
    fn extract_doc_voicelayer_env_var_mentions(
        contents: &str,
    ) -> std::collections::BTreeSet<String> {
        let mut mentions = std::collections::BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find('`') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let inner = &after[..close];
            search = &after[close + 1..];
            if inner.chars().any(char::is_whitespace) {
                continue;
            }
            let Some(rest) = inner.strip_prefix("VOICELAYER_") else {
                continue;
            };
            if rest.is_empty() || rest.ends_with('_') {
                continue;
            }
            if !rest
                .chars()
                .all(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || c == '_')
            {
                continue;
            }
            mentions.insert(format!("VOICELAYER_{rest}"));
        }
        mentions
    }

    /// Walk an env example file and pull every `VOICELAYER_<NAME>`
    /// key on an assignment-shaped line: optional leading `#` (for
    /// commented overrides), the key, an `=` (with the value
    /// optional). Returns the bare key. Mirrors the parsing in
    /// `tests/python/test_env_example.py`'s
    /// `ENV_EXAMPLE_ASSIGNMENT_PATTERN` so callers compare against
    /// the same key set the python-side guard sees.
    fn extract_env_example_voicelayer_keys(contents: &str) -> std::collections::BTreeSet<String> {
        let mut keys = std::collections::BTreeSet::new();
        for line in contents.lines() {
            let trimmed = line.trim_start();
            let after_hash = trimmed.trim_start_matches('#').trim_start();
            let Some(rest) = after_hash.strip_prefix("VOICELAYER_") else {
                continue;
            };
            let name: String = rest
                .chars()
                .take_while(|c| c.is_ascii_uppercase() || c.is_ascii_digit() || *c == '_')
                .collect();
            if name.is_empty() {
                continue;
            }
            let after_name = &rest[name.len()..];
            if !after_name.starts_with('=') {
                continue;
            }
            keys.insert(format!("VOICELAYER_{name}"));
        }
        keys
    }

    /// Walk `CLAUDE.md` and pull out every backtick-quoted
    /// PascalCase identifier inside the `## Domain Vocabulary`
    /// section. Stops at the next `## ` heading so prose mentions
    /// of types elsewhere in the file (e.g. examples in
    /// `## Architecture Defaults`) do not leak into the captured
    /// set.
    fn extract_claude_md_domain_vocabulary_types(
        contents: &str,
    ) -> std::collections::BTreeSet<String> {
        let mut types = std::collections::BTreeSet::new();
        let mut in_section = false;
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("## ") {
                in_section = trimmed == "## Domain Vocabulary";
                continue;
            }
            if !in_section {
                continue;
            }
            let mut search = line;
            while let Some(idx) = search.find('`') {
                let after = &search[idx + 1..];
                let Some(close) = after.find('`') else {
                    break;
                };
                let inner = &after[..close];
                search = &after[close + 1..];
                if inner.is_empty() {
                    continue;
                }
                if !inner.starts_with(|c: char| c.is_ascii_uppercase()) {
                    continue;
                }
                if !inner.chars().all(|c| c.is_ascii_alphanumeric()) {
                    continue;
                }
                types.insert(inner.to_owned());
            }
        }
        types
    }

    /// Walk a Rust source string and pull every `pub struct <Name>`
    /// and `pub enum <Name>` declaration. The token after the
    /// keyword is the type name; trailing generic parameters
    /// (`<T>`), where-clauses, and brace bodies are stripped by
    /// taking only alphanumeric/underscore characters until the
    /// first delimiter.
    fn extract_voicelayer_core_pub_type_names(source: &str) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        for line in source.lines() {
            let trimmed = line.trim_start();
            for prefix in ["pub struct ", "pub enum "] {
                if let Some(rest) = trimmed.strip_prefix(prefix) {
                    let name: String = rest
                        .chars()
                        .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                        .collect();
                    if !name.is_empty() && name.starts_with(|c: char| c.is_ascii_uppercase()) {
                        names.insert(name);
                    }
                }
            }
        }
        names
    }

    /// Walk a Markdown body and pull out every backtick-quoted
    /// `crates/<name>` literal where `<name>` is a single segment
    /// (no further `/`), composed of alphanumerics, hyphens, or
    /// underscores. Designed for the README's Architecture section
    /// and any prose mention of a workspace member crate. Rejects
    /// deeper paths like `crates/vl/src/main.rs` (those are
    /// already covered by `extract_doc_file_path_refs`) and
    /// non-crate mentions like `crates/` (the bare directory).
    fn extract_readme_crate_path_refs(contents: &str) -> std::collections::BTreeSet<String> {
        let mut refs = std::collections::BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find('`') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let inner = &after[..close];
            search = &after[close + 1..];
            let Some(rest) = inner.strip_prefix("crates/") else {
                continue;
            };
            if rest.is_empty() || rest.contains('/') {
                continue;
            }
            if !rest
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
            {
                continue;
            }
            refs.insert(rest.to_owned());
        }
        refs
    }

    /// Pull the version floor a README list-item declares for a
    /// language requirement. Targets the canonical
    /// `- <Lang> <X.Y>+` shape (e.g. `- Rust 1.88+` or
    /// `- Python 3.12+`) in the Requirements section. Stops at the
    /// first character that is not a digit or `.`, so the trailing
    /// `+` (and any prose after it) does not leak into the captured
    /// token.
    ///
    /// Returns `None` when the language line is absent or written in
    /// a different shape — callers are expected to surface that as
    /// "the README has been refactored; this drift guard needs to
    /// follow", not as a silent pass.
    fn extract_readme_requirement_floor(contents: &str, lang: &str) -> Option<String> {
        let prefix = format!("- {lang} ");
        for line in contents.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix(&prefix) else {
                continue;
            };
            let token: String = rest
                .chars()
                .take_while(|c| c.is_ascii_digit() || *c == '.')
                .collect();
            if !token.is_empty() {
                return Some(token);
            }
        }
        None
    }

    /// Walk `CLAUDE.md` and pull out the verification chain — the
    /// `&&`-joined sequence of commands inside the first fenced
    /// ```bash``` block. Strips the leading `&& ` continuation
    /// marker and the trailing `\` line-join, so callers compare bare
    /// command strings. The block is stable enough today that a
    /// stronger regex isn't justified; if the chain ever moves to a
    /// non-bash fence, this helper will simply return empty and the
    /// dependent test will surface it as a missing-fence panic.
    fn extract_claude_md_verification_chain(contents: &str) -> Vec<String> {
        let mut commands = Vec::new();
        let mut in_block = false;
        for line in contents.lines() {
            let trimmed = line.trim();
            if !in_block {
                if trimmed.starts_with("```bash") {
                    in_block = true;
                }
                continue;
            }
            if trimmed.starts_with("```") {
                break;
            }
            let cleaned = trimmed
                .trim_end_matches('\\')
                .trim()
                .trim_start_matches("&& ")
                .trim()
                .to_owned();
            if !cleaned.is_empty() {
                commands.push(cleaned);
            }
        }
        commands
    }

    /// Walk `.github/workflows/ci.yml` and pull the `run:` line of
    /// every step in the `verify` job that invokes `cargo` or
    /// `uv run`. The match is intentionally narrow — it ignores
    /// `name:`, `uses:`, the `apt-get` install step, and any other
    /// scaffolding — so the resulting list is exactly the
    /// verification chain CI gates merges on. The order is preserved
    /// from file order, mirroring the verification chain's order in
    /// `CLAUDE.md`.
    fn extract_ci_verify_run_commands(contents: &str) -> Vec<String> {
        let mut commands = Vec::new();
        for line in contents.lines() {
            let trimmed = line.trim_start();
            let Some(rest) = trimmed.strip_prefix("run: ") else {
                continue;
            };
            let cmd = rest.trim();
            if cmd.starts_with("cargo ") || cmd.starts_with("uv run ") {
                commands.push(cmd.to_owned());
            }
        }
        commands
    }

    /// Normalise a `cargo fmt` invocation to its canonical
    /// write-mode form. CI runs `cargo fmt --all -- --check` (read-
    /// only, fails on diff); `CLAUDE.md` documents the local
    /// write-mode `cargo fmt --all`. They are the same step in
    /// effect, just with the `-- --check` flag stripped on the CI
    /// side. Other commands pass through unchanged.
    fn normalise_fmt_command(cmd: &str) -> String {
        if let Some(stripped) = cmd.strip_suffix(" -- --check")
            && stripped.starts_with("cargo fmt")
        {
            return stripped.to_owned();
        }
        cmd.to_owned()
    }

    /// Pull every package name declared under
    /// `[dependency-groups].<group>` in pyproject. Same array shape
    /// as `[project.optional-dependencies]` (multi-line or single-
    /// line, with PEP 508 specifiers) — only the section heading
    /// differs. Reuses `extract_pep508_names_from_inline` for the
    /// per-line extraction so the version-specifier strip behaviour
    /// stays in lockstep.
    fn extract_pyproject_dependency_group(
        contents: &str,
        group: &str,
    ) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        let mut in_section = false;
        let mut in_array = false;
        let array_open = format!("{group} = [");
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_section = trimmed.starts_with("[dependency-groups]");
                in_array = false;
                continue;
            }
            if !in_section {
                continue;
            }
            if !in_array {
                if let Some(rest) = trimmed.strip_prefix(&array_open) {
                    in_array = true;
                    extract_pep508_names_from_inline(rest, &mut names);
                    if rest.contains(']') {
                        in_array = false;
                    }
                }
                continue;
            }
            if trimmed.starts_with(']') {
                in_array = false;
                continue;
            }
            extract_pep508_names_from_inline(trimmed, &mut names);
        }
        names
    }

    /// Pull every package name declared under
    /// `[project.optional-dependencies].<group>` in pyproject. Handles
    /// the canonical multi-line array form (`<group> = [` opens, each
    /// `"<spec>"` row contributes one entry, `]` on its own line
    /// closes) and single-line arrays. The version specifier (`>=`,
    /// `<=`, `==`, `~=`, `!=`, etc.) is stripped from each entry so
    /// callers compare bare PyPI package names.
    ///
    /// Returns an empty set if the group is absent — that is a valid
    /// pyproject (the group has not been declared) and is the same
    /// shape as a group declared with no entries.
    fn extract_pyproject_optional_deps(
        contents: &str,
        group: &str,
    ) -> std::collections::BTreeSet<String> {
        let mut names = std::collections::BTreeSet::new();
        let mut in_section = false;
        let mut in_array = false;
        let array_open = format!("{group} = [");
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with('[') {
                in_section = trimmed.starts_with("[project.optional-dependencies]");
                in_array = false;
                continue;
            }
            if !in_section {
                continue;
            }
            if !in_array {
                if let Some(rest) = trimmed.strip_prefix(&array_open) {
                    in_array = true;
                    extract_pep508_names_from_inline(rest, &mut names);
                    if rest.contains(']') {
                        in_array = false;
                    }
                }
                continue;
            }
            if trimmed.starts_with(']') {
                in_array = false;
                continue;
            }
            extract_pep508_names_from_inline(trimmed, &mut names);
        }
        names
    }

    /// Pull every quoted PEP 508 entry on a single line, strip the
    /// version specifier, and insert the bare name into `out`. Used
    /// by `extract_pyproject_optional_deps` for both single-line
    /// arrays and one-entry-per-line continuation rows.
    fn extract_pep508_names_from_inline(line: &str, out: &mut std::collections::BTreeSet<String>) {
        let mut search = line;
        while let Some(idx) = search.find('"') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('"') else {
                break;
            };
            let spec = &after[..close];
            search = &after[close + 1..];
            let stop = ['<', '>', '=', '!', '~', '[', '(', ' ', ';'];
            let end = spec.find(stop).unwrap_or(spec.len());
            let name = spec[..end].trim();
            if !name.is_empty() {
                out.insert(name.to_owned());
            }
        }
    }

    /// Walk every `.py` file under `start` recursively and collect the
    /// root package name of each `import X` and `from X import Y`
    /// statement. The root is the leftmost dotted segment, so
    /// `from foo.bar import baz` contributes `foo`. `import a, b`
    /// (comma-separated form) contributes both `a` and `b`. Aliases
    /// (`import numpy as np`) are stripped down to the package name.
    ///
    /// Comment-only lines are skipped; inline trailing comments are
    /// trimmed before parsing (so `import numpy  # noqa` parses
    /// cleanly). Indented imports inside function bodies, try/except
    /// blocks, or class definitions are still captured — what matters
    /// is that the package is referenced anywhere the interpreter
    /// could execute.
    fn collect_python_imports_under(
        start: &std::path::Path,
        names: &mut std::collections::BTreeSet<String>,
    ) -> std::io::Result<()> {
        for entry in std::fs::read_dir(start)? {
            let entry = entry?;
            let path = entry.path();
            if entry.file_type()?.is_dir() {
                collect_python_imports_under(&path, names)?;
            } else if path.extension().and_then(|s| s.to_str()) == Some("py") {
                let contents = std::fs::read_to_string(&path)?;
                gather_python_imports_from_source(&contents, names);
            }
        }
        Ok(())
    }

    fn gather_python_imports_from_source(
        source: &str,
        out: &mut std::collections::BTreeSet<String>,
    ) {
        for line in source.lines() {
            let trimmed = line.trim_start();
            if trimmed.starts_with('#') {
                continue;
            }
            let no_comment = trimmed.split('#').next().unwrap_or(trimmed).trim_end();
            if let Some(rest) = no_comment.strip_prefix("import ") {
                for part in rest.split(',') {
                    let pkg = part.split_whitespace().next().unwrap_or("");
                    let root = pkg.split('.').next().unwrap_or(pkg);
                    if !root.is_empty() {
                        out.insert(root.to_owned());
                    }
                }
            } else if let Some(rest) = no_comment.strip_prefix("from ") {
                let pkg = rest.split_whitespace().next().unwrap_or("");
                let root = pkg.split('.').next().unwrap_or(pkg);
                if !root.is_empty() {
                    out.insert(root.to_owned());
                }
            }
        }
    }

    /// Walk a Markdown doc string and pull out every backtick-quoted
    /// `voicelayer.<word>` literal. Excludes `dictation.*` SSE event
    /// names (different prefix) and any other non-shortcut content.
    /// The contents-rejection on whitespace and emptiness keeps a
    /// trailing prose match (`\`voicelayer prefix\``) from polluting
    /// the set.
    fn collect_doc_shortcut_ids(contents: &str) -> std::collections::BTreeSet<String> {
        let needle = "`voicelayer.";
        let mut ids = std::collections::BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find(needle) {
            let after = &search[idx + 1..]; // skip the opening backtick
            let inner: String = after.chars().take_while(|c| *c != '`').collect();
            if !inner.is_empty() && !inner.contains(' ') {
                ids.insert(inner.clone());
            }
            let consumed = inner.len() + 1; // backtick + content
            if consumed >= search.len() - idx {
                break;
            }
            search = &search[idx + consumed..];
        }
        ids
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

    #[test]
    fn extract_yaml_bare_string_value_handles_both_bare_and_quoted_forms() {
        let yaml = concat!(
            "info:\n",
            "  title: Local Control API\n",
            "  version: 0.1.0\n",
            "  description: |\n",
            "    multi-line block\n",
        );
        // Bare-scalar version (no quotes) is the default openapi style.
        assert_eq!(
            extract_yaml_bare_string_value(yaml, "version").as_deref(),
            Some("0.1.0"),
        );
        // Quoted form must work too — both styles are valid YAML.
        let quoted = "info:\n  version: \"0.2.0\"\n";
        assert_eq!(
            extract_yaml_bare_string_value(quoted, "version").as_deref(),
            Some("0.2.0"),
        );
    }

    #[test]
    fn extract_yaml_bare_string_value_stops_at_trailing_inline_comment() {
        // Pin the comment-tolerance so a trailing `# note` cannot leak
        // into the token. The take-while exits at whitespace, so the
        // helper trims at the first space — comment text never bleeds
        // into the version string.
        let yaml = "info:\n  version: 1.2.3  # bumped 2026-04-25\n";
        assert_eq!(
            extract_yaml_bare_string_value(yaml, "version").as_deref(),
            Some("1.2.3"),
        );
    }

    #[test]
    fn extract_yaml_bare_string_value_returns_none_when_key_absent() {
        let yaml = "info:\n  title: nope\n";
        assert!(extract_yaml_bare_string_value(yaml, "version").is_none());
    }

    #[test]
    fn extract_python_jsonrpc_error_codes_collects_named_constants() {
        let py = concat!(
            "PROVIDER_UNAVAILABLE_CODE = -32004\n",
            "PROVIDER_REQUEST_FAILED_CODE = -32005\n",
            "INVALID_REQUEST_CODE = -32600\n",
            "METHOD_NOT_FOUND_CODE = -32601\n",
            "PARSE_ERROR_CODE = -32700\n",
        );
        let codes = extract_python_jsonrpc_error_codes(py);
        assert_eq!(codes.len(), 5);
        assert!(codes.contains(&-32004));
        assert!(codes.contains(&-32700));
    }

    #[test]
    fn extract_python_jsonrpc_error_codes_filters_out_unrelated_negatives() {
        // A bare `-1 = ...` style assignment must not be picked up;
        // only values in the JSON-RPC reserved range (-32700..=-32000)
        // are real error codes per the spec.
        let py = concat!(
            "RETRY_BACKOFF_CODE = -1\n",
            "PROVIDER_UNAVAILABLE_CODE = -32004\n",
            "ARBITRARY_OFFSET = -100000\n",
        );
        let codes = extract_python_jsonrpc_error_codes(py);
        assert_eq!(codes.len(), 1);
        assert!(codes.contains(&-32004));
    }

    #[test]
    fn extract_doc_jsonrpc_error_codes_dedupes_across_mentions() {
        // The same code appears multiple times — once in an Error
        // Policy enumeration, once in an inline usage description.
        // The set semantics dedup them.
        let md = concat!(
            "Returns `-32004` when the provider is not configured.\n",
            "## Error Policy\n",
            "- Provider unavailable: `-32004`\n",
            "- Method not found: `-32601`\n",
        );
        let codes = extract_doc_jsonrpc_error_codes(md);
        assert_eq!(codes.len(), 2);
        assert!(codes.contains(&-32004));
        assert!(codes.contains(&-32601));
    }

    #[test]
    fn extract_doc_jsonrpc_error_codes_ignores_out_of_range_negatives() {
        let md = "See ticket `-1234` and ASCII codepoint `-65`.\n";
        let codes = extract_doc_jsonrpc_error_codes(md);
        assert!(codes.is_empty());
    }

    #[test]
    fn extract_cargo_workspace_members_collects_quoted_paths() {
        let toml = concat!(
            "[workspace]\n",
            "members = [\n",
            "  \"crates/voicelayer-core\",\n",
            "  \"crates/voicelayerd\",\n",
            "  \"crates/vl\",\n",
            "  \"crates/vl-desktop\",\n",
            "]\n",
            "resolver = \"2\"\n",
        );
        let members = extract_cargo_workspace_members(toml);
        assert_eq!(members.len(), 4);
        assert!(members.contains("crates/voicelayer-core"));
        assert!(members.contains("crates/vl-desktop"));
    }

    #[test]
    fn extract_pyproject_license_text_handles_inline_table_form() {
        let toml = concat!(
            "[project]\n",
            "name = \"voicelayer-orchestrator\"\n",
            "license = { text = \"Apache-2.0\" }\n",
        );
        assert_eq!(
            extract_pyproject_license_text(toml).as_deref(),
            Some("Apache-2.0"),
        );
    }

    #[test]
    fn extract_pyproject_license_text_returns_none_when_field_absent() {
        let toml = "[project]\nname = \"foo\"\n";
        assert!(extract_pyproject_license_text(toml).is_none());
    }

    #[test]
    fn extract_doc_file_path_refs_filters_to_path_shaped_tokens() {
        let md = concat!(
            "See `crates/voicelayerd/src/lib.rs` and `providers/foo.py`.\n",
            "Bare identifier `compose_payload` is not a path.\n",
            "Single-segment basename `whisper-cli` has no slash.\n",
            "Spaces inside backticks `not a path.py` should be skipped.\n",
            "Wrong extension `path/to/data.txt` should be skipped.\n",
            "Runtime paths `~/.config/voicelayer/config.toml`, `/usr/bin/foo.sh`, \n",
            "`${HOME}/foo.py` must not be classified as repo refs.\n",
        );
        let paths = extract_doc_file_path_refs(md);
        assert_eq!(paths.len(), 2);
        assert!(paths.contains("crates/voicelayerd/src/lib.rs"));
        assert!(paths.contains("providers/foo.py"));
    }

    #[test]
    fn extract_markdown_md_links_strips_fragments_and_skips_urls() {
        let md = concat!(
            "See [Desktop](./desktop.md) and [Setup](./setup.md#install).\n",
            "External: [crates.io](https://crates.io/crates/voicelayerd).\n",
            "Mailto: [team](mailto:team@example.com).\n",
            "Repo path: [overview](docs/architecture/overview.md).\n",
            "Non-md link: [readme](./README.txt).\n",
        );
        let links = extract_markdown_md_links(md);
        assert_eq!(links.len(), 3);
        assert!(links.contains(&"./desktop.md".to_owned()));
        assert!(links.contains(&"./setup.md".to_owned()));
        assert!(links.contains(&"docs/architecture/overview.md".to_owned()));
    }

    #[test]
    fn extract_install_sh_bin_targets_picks_up_quoted_basenames() {
        let sh = concat!(
            "BIN_DIR=\"${VOICELAYER_INSTALL_BIN_DIR:-${HOME}/.local/bin}\"\n",
            "install -m 0755 \"${REPO_ROOT}/target/release/vl\" \"${BIN_DIR}/vl\"\n",
            "install -m 0755 \"${REPO_ROOT}/scripts/wrapper.sh\" \\\n",
            "  \"${BIN_DIR}/voicelayer-whisper-server-run.sh\"\n",
        );
        let targets = extract_install_sh_bin_targets(sh);
        assert_eq!(targets.len(), 2);
        assert!(targets.contains("vl"));
        assert!(targets.contains("voicelayer-whisper-server-run.sh"));
    }

    #[test]
    fn extract_systemd_execstart_basenames_drops_arguments() {
        // The full line is `ExecStart=%h/.local/bin/vl daemon run ...`
        // — only `vl` is the binary basename. Pin that the parser
        // stops at whitespace so trailing args do not leak.
        let unit = concat!(
            "[Service]\n",
            "ExecStart=%h/.local/bin/vl daemon run --socket-path %t/x.sock\n",
            "ExecStart=%h/.local/bin/voicelayer-whisper-server-run.sh\n",
        );
        let basenames = extract_systemd_execstart_basenames(unit);
        assert_eq!(basenames.len(), 2);
        assert!(basenames.contains("vl"));
        assert!(basenames.contains("voicelayer-whisper-server-run.sh"));
    }

    #[test]
    fn collect_rust_shortcut_ids_extracts_quoted_const_value() {
        let source = concat!(
            "pub const SHORTCUT_TOGGLE: &str = \"voicelayer.dictation_toggle\";\n",
            "pub const SHORTCUT_PUSH_TO_TALK: &str = \"voicelayer.ptt\";\n",
            "// `pub const NOT_A_SHORTCUT: &str = \"prose\"` should be ignored\n",
        );
        let ids = collect_rust_shortcut_ids(source);
        assert_eq!(ids.len(), 2);
        assert!(ids.contains("voicelayer.dictation_toggle"));
        assert!(ids.contains("voicelayer.ptt"));
    }

    #[test]
    fn collect_doc_shortcut_ids_dedupes_repeated_mentions() {
        let md = concat!(
            "the GUI registers `voicelayer.dictation_toggle` via portal.\n",
            "later: `vl-desktop` registers `voicelayer.dictation_toggle` again.\n",
            "an SSE event like `dictation.completed` does not match the prefix.\n",
        );
        let ids = collect_doc_shortcut_ids(md);
        assert_eq!(ids.len(), 1);
        assert!(ids.contains("voicelayer.dictation_toggle"));
    }

    #[test]
    fn extract_cargo_workspace_members_stops_at_closing_bracket() {
        // Pin scope: a key after the array close (`resolver = "2"`)
        // must not be consumed. A future workspace key with a quoted
        // value would otherwise drift into the members set.
        let toml = concat!(
            "[workspace]\n",
            "members = [\n",
            "  \"crates/foo\",\n",
            "]\n",
            "resolver = \"2\"\n",
            "[workspace.package]\n",
            "version = \"0.1.0\"\n",
        );
        let members = extract_cargo_workspace_members(toml);
        assert_eq!(members.len(), 1);
        assert!(members.contains("crates/foo"));
        assert!(!members.iter().any(|m| m.starts_with('0')));
        assert!(!members.iter().any(|m| m.starts_with('2')));
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

    /// Pin `openapi.info.version` against the workspace
    /// `[workspace.package] version` in the root `Cargo.toml`. Both
    /// claim to describe the same release artefact, so a bump that
    /// updated only one side would publish an openapi document
    /// labeled with the wrong release. Operators reading the yaml in
    /// isolation (e.g. someone consuming the contract through a code
    /// generator) would see a stale version and silently target an
    /// older daemon.
    #[test]
    fn openapi_info_version_matches_cargo_workspace_version() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));

        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let openapi = std::fs::read_to_string(format!("{repo_root}/openapi/voicelayerd.v1.yaml"))
            .expect("read openapi contract");

        let cargo_version = extract_toml_string_value(&cargo_toml, "version").expect(
            "Cargo.toml must declare `version = \"...\"` under [workspace.package]; \
             extract_toml_string_value may be misparsing",
        );
        let openapi_version = extract_yaml_bare_string_value(&openapi, "version").expect(
            "openapi/voicelayerd.v1.yaml must declare `info.version: ...`; \
             extract_yaml_bare_string_value may be misparsing",
        );

        assert_eq!(
            cargo_version, openapi_version,
            "version drift: Cargo.toml `version = \"{cargo_version}\"` does not match \
             openapi/voicelayerd.v1.yaml `info.version: {openapi_version}`. Bump both \
             together so the contract document and the published crate share a label.",
        );
    }

    /// Pin `pyproject.toml`'s `version` against the Cargo workspace
    /// and openapi labels already cross-checked in
    /// `openapi_info_version_matches_cargo_workspace_version`.
    /// `vl --version` reports the Cargo value (#58); a divergent
    /// pyproject version would publish the Python orchestrator
    /// against a different release tag than the Rust binaries that
    /// drive it. Operators bisecting a bug across both stacks would
    /// see two version stamps and have no idea which package
    /// shipped the regression.
    #[test]
    fn pyproject_version_matches_cargo_workspace_version() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml =
            std::fs::read_to_string(format!("{repo_root}/Cargo.toml")).expect("read Cargo.toml");
        let pyproject = std::fs::read_to_string(format!("{repo_root}/pyproject.toml"))
            .expect("read pyproject.toml");

        let cargo_version = extract_toml_string_value(&cargo_toml, "version")
            .expect("Cargo.toml must declare `version = \"...\"` under [workspace.package]");
        let pyproject_version = extract_toml_string_value(&pyproject, "version")
            .expect("pyproject.toml must declare `version = \"...\"` under [project]");

        assert_eq!(
            cargo_version, pyproject_version,
            "version drift: Cargo.toml `version = \"{cargo_version}\"` does not match \
             pyproject.toml `version = \"{pyproject_version}\"`. Bump both together so \
             `vl --version` and the Python orchestrator's release tag share a label.",
        );
    }

    /// Pin ruff's `target-version` against pyproject's
    /// `requires-python` floor. Ruff drives lint rules by language
    /// version (e.g. PEP 695 generic syntax requires py312); a stale
    /// target would silently disable rules the codebase already
    /// relies on. The comparison normalises both to `py3XX`/`>=3.XX`
    /// shape: the floor's `.` is collapsed and prefixed with `py`,
    /// then matched against the literal target value.
    #[test]
    fn ruff_target_version_matches_pyproject_requires_python() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let pyproject = std::fs::read_to_string(format!("{repo_root}/pyproject.toml"))
            .expect("read pyproject.toml");

        let floor = extract_python_version_floor_from_pyproject(&pyproject)
            .expect("pyproject.toml must declare `requires-python = \">=...\"`");
        let ruff_target = extract_toml_string_value(&pyproject, "target-version")
            .expect("pyproject.toml must declare `[tool.ruff] target-version = \"...\"`");

        let expected_target = format!("py{}", floor.replace('.', ""));
        assert_eq!(
            ruff_target, expected_target,
            "lint version drift: pyproject `requires-python = \">={floor}\"` but \
             `[tool.ruff] target-version = \"{ruff_target}\"` (expected \
             \"{expected_target}\"). Ruff would silently disable lint rules the \
             codebase already requires; bump both together.",
        );
    }

    /// Pin pyproject's `license = { text = "..." }` against Cargo's
    /// `license = "..."`. The LICENSE file on disk is already
    /// cross-checked against Cargo (#63); this closes the missing
    /// edge so all three (Cargo, pyproject, LICENSE) stay coherent.
    /// A pyproject that drifts to a different SPDX while Cargo and
    /// LICENSE keep Apache-2.0 would publish the Python orchestrator
    /// against the wrong license metadata on PyPI without breaking
    /// the Cargo side.
    #[test]
    fn pyproject_license_matches_cargo_license() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml =
            std::fs::read_to_string(format!("{repo_root}/Cargo.toml")).expect("read Cargo.toml");
        let pyproject = std::fs::read_to_string(format!("{repo_root}/pyproject.toml"))
            .expect("read pyproject.toml");

        let cargo_license = extract_toml_string_value(&cargo_toml, "license").expect(
            "Cargo.toml must declare `license = \"...\"`; \
             extract_toml_string_value may be misparsing",
        );
        let pyproject_license = extract_pyproject_license_text(&pyproject).expect(
            "pyproject.toml must declare `license = { text = \"...\" }`; \
             extract_pyproject_license_text may be misparsing",
        );

        assert_eq!(
            cargo_license, pyproject_license,
            "license drift: Cargo.toml `license = \"{cargo_license}\"` does not match \
             pyproject.toml `license = {{ text = \"{pyproject_license}\" }}`. The \
             Rust workspace and Python orchestrator are released as a single artefact \
             — both metadata files must declare the same SPDX, and the LICENSE file \
             on disk (already pinned in #63) must follow.",
        );
    }

    /// Cross-check JSON-RPC error codes between the Python worker
    /// (where they are emitted as `_CODE = -3XXXX` constants) and
    /// the worker-protocol doc (which enumerates them in an Error
    /// Policy block plus inline usage notes). A drift on either
    /// side would mislead operators reading the doc to debug a
    /// `-32004` they receive — they would either find no entry or
    /// find an entry for a code the worker no longer emits.
    #[test]
    fn jsonrpc_error_codes_in_python_match_the_protocol_doc() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));

        let python_source = std::fs::read_to_string(format!(
            "{repo_root}/python/voicelayer_orchestrator/worker.py"
        ))
        .expect("read python worker.py");
        let doc_source = std::fs::read_to_string(format!(
            "{repo_root}/docs/architecture/python-worker-protocol.md",
        ))
        .expect("read python-worker-protocol.md");

        let python_codes = extract_python_jsonrpc_error_codes(&python_source);
        let doc_codes = extract_doc_jsonrpc_error_codes(&doc_source);

        assert!(
            !python_codes.is_empty(),
            "python worker.py must declare at least one `_CODE = -3XXXX` constant; \
             extract_python_jsonrpc_error_codes may be misparsing",
        );
        assert!(
            !doc_codes.is_empty(),
            "python-worker-protocol.md must mention at least one `\\`-3XXXX\\`` code; \
             extract_doc_jsonrpc_error_codes may be misparsing",
        );

        for code in &python_codes {
            assert!(
                doc_codes.contains(code),
                "python worker emits `{code}` but no matching `\\`{code}\\`` mention \
                 in docs/architecture/python-worker-protocol.md. Add a line under the \
                 Error Policy enumeration so operators reading the doc can identify \
                 the code they receive.",
            );
        }
        for code in &doc_codes {
            assert!(
                python_codes.contains(code),
                "docs/architecture/python-worker-protocol.md mentions `\\`{code}\\`` \
                 but no matching `_CODE = {code}` constant exists in worker.py. \
                 Either drop the mention from the doc or add the constant on the \
                 python side.",
            );
        }
    }

    /// Pin every `crates/*` directory against the
    /// `[workspace] members = [...]` list in the root `Cargo.toml`.
    /// `cargo build --workspace` already errors if a listed member
    /// is missing on disk, so the failure mode this guard catches is
    /// the inverse: a new crate directory committed without a
    /// matching members entry. Such a crate compiles standalone via
    /// `cargo build -p ...` but is silently excluded from
    /// `cargo test --all` and the rest of the verification chain,
    /// drifting out of CI coverage with no signal.
    #[test]
    fn cargo_workspace_members_match_crates_directory_listing() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let members = extract_cargo_workspace_members(&cargo_toml);
        assert!(
            !members.is_empty(),
            "Cargo.toml must declare `members = [...]`; \
             extract_cargo_workspace_members may be misparsing",
        );

        let crates_dir = std::path::PathBuf::from(format!("{repo_root}/crates"));
        let on_disk: std::collections::BTreeSet<String> = std::fs::read_dir(&crates_dir)
            .expect("read crates/ directory")
            .filter_map(|entry| {
                let entry = entry.ok()?;
                if entry.file_type().ok()?.is_dir() {
                    let name = entry.file_name().into_string().ok()?;
                    // Skip hidden / build artefacts (e.g. `target`)
                    // even though `crates/` should not contain
                    // any. The filter keeps the test robust against
                    // future tooling that drops dotfiles.
                    if name.starts_with('.') || name == "target" {
                        return None;
                    }
                    Some(format!("crates/{name}"))
                } else {
                    None
                }
            })
            .collect();
        assert!(
            !on_disk.is_empty(),
            "expected at least one crate directory under `crates/`; \
             the test may be running from the wrong CWD",
        );

        for path in &on_disk {
            assert!(
                members.contains(path),
                "crate directory `{path}` exists on disk but is not declared in \
                 Cargo.toml `[workspace] members = [...]`. Add the entry so \
                 `cargo test --all` and the verification chain pick it up.",
            );
        }
        for path in &members {
            assert!(
                on_disk.contains(path),
                "Cargo.toml lists workspace member `{path}` but the directory \
                 does not exist on disk. Either restore the crate or drop the \
                 entry — `cargo build --workspace` will refuse to load otherwise.",
            );
        }
    }

    /// Pin the SPDX `license` identifier in `Cargo.toml` against the
    /// actual content of `LICENSE`. The most plausible drift is a
    /// SPDX bump (e.g. `Apache-2.0` → `MIT`) that forgets to swap
    /// the file — `cargo publish` would reject the package against
    /// crates.io's policy, but only at release time. The repo is
    /// internal-first, so no `cargo publish` runs in CI; without
    /// this guard the discrepancy would only surface during the
    /// first publish attempt long after the change shipped.
    ///
    /// Today we ship Apache-2.0; the check looks for the canonical
    /// `Apache License` + `Version 2.0` header lines. Adding a
    /// future SPDX-to-signature row here is the right place to
    /// extend coverage when the licence changes.
    #[test]
    fn cargo_license_spdx_matches_license_file_signature() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let license_file =
            std::fs::read_to_string(format!("{repo_root}/LICENSE")).expect("read LICENSE file");

        let spdx = extract_toml_string_value(&cargo_toml, "license").expect(
            "Cargo.toml must declare `license = \"...\"`; \
             extract_toml_string_value may be misparsing",
        );

        match spdx.as_str() {
            "Apache-2.0" => {
                assert!(
                    license_file.contains("Apache License"),
                    "Cargo.toml claims `Apache-2.0` but LICENSE file is missing \
                     the canonical `Apache License` header. Either fix the SPDX \
                     identifier or restore the correct LICENSE.",
                );
                assert!(
                    license_file.contains("Version 2.0"),
                    "Cargo.toml claims `Apache-2.0` but LICENSE file is missing \
                     the `Version 2.0` line. Confirm the file is the right \
                     Apache release.",
                );
            }
            other => panic!(
                "Cargo.toml `license = \"{other}\"` is not covered by this drift \
                 guard. Add a match arm here pinning the SPDX identifier to a \
                 signature line in LICENSE — silently letting the SPDX float free \
                 of the file would defeat the point of the check.",
            ),
        }
    }

    /// Pin every per-crate `Cargo.toml` `name = "..."` against its
    /// directory basename. Workspace `path = ...` deps are resolved
    /// by manifest name, not directory; a typo such as
    /// `name = "voicelayer_core"` (underscore) for a crate at
    /// `crates/voicelayer-core/` (hyphen) would silently break
    /// every dependent's path-based dep — `cargo build --workspace`
    /// loads each member's manifest in isolation and only complains
    /// about the unresolved name when a dependent tries to import.
    #[test]
    fn each_crate_manifest_name_matches_its_directory_basename() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let members = extract_cargo_workspace_members(&cargo_toml);
        assert!(
            !members.is_empty(),
            "Cargo.toml must declare at least one member; cannot pin per-crate \
             name without a member list to walk",
        );

        for member_path in &members {
            let manifest_path = format!("{repo_root}/{member_path}/Cargo.toml");
            let manifest = std::fs::read_to_string(&manifest_path)
                .unwrap_or_else(|err| panic!("read {manifest_path}: {err}"));
            let declared_name = extract_toml_string_value(&manifest, "name").unwrap_or_else(|| {
                panic!(
                    "crate manifest at `{manifest_path}` must declare \
                         `name = \"...\"` under [package]; \
                         extract_toml_string_value may be misparsing"
                )
            });
            let dir_basename = std::path::Path::new(member_path)
                .file_name()
                .and_then(|name| name.to_str())
                .expect("member path has a directory component");
            assert_eq!(
                declared_name, dir_basename,
                "crate at `{member_path}` declares \
                 `name = \"{declared_name}\"` but the directory basename is \
                 `{dir_basename}`. Workspace `path = ...` deps resolve by \
                 manifest name; a mismatch breaks `cargo build`'s resolution \
                 silently across dependents.",
            );
        }
    }

    /// Pin every per-crate `Cargo.toml` to inherit `version` from the
    /// workspace via `version.workspace = true`. A literal
    /// `version = "X.Y.Z"` line on a member would diverge from
    /// `[workspace.package] version` and silently defeat the
    /// openapi/cargo version pin (#55) — operators reading the
    /// openapi document would see one version while a particular
    /// crate published a different one.
    #[test]
    fn each_crate_manifest_inherits_workspace_version() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let cargo_toml = std::fs::read_to_string(format!("{repo_root}/Cargo.toml"))
            .expect("read root Cargo.toml");
        let members = extract_cargo_workspace_members(&cargo_toml);
        assert!(
            !members.is_empty(),
            "Cargo.toml must declare at least one member; cannot pin per-crate \
             version inheritance without a member list to walk",
        );

        for member_path in &members {
            let manifest_path = format!("{repo_root}/{member_path}/Cargo.toml");
            let manifest = std::fs::read_to_string(&manifest_path)
                .unwrap_or_else(|err| panic!("read {manifest_path}: {err}"));
            assert!(
                manifest.contains("version.workspace = true"),
                "crate at `{member_path}` does not contain \
                 `version.workspace = true`. Inherit the workspace version \
                 instead of declaring a literal — a divergent version would \
                 defeat the openapi/cargo version pin and confuse anyone \
                 matching `vl --version` against the published crate.",
            );
        }
    }

    /// Pin every `pub const SHORTCUT_*: &str = "..."` value in
    /// `vl-desktop::portal` against the corresponding mention in
    /// `docs/architecture/overview.md`. The shortcut id flows in
    /// both directions: the Rust constant feeds
    /// `org.freedesktop.portal.GlobalShortcuts.bind_shortcuts`, and
    /// the doc explains the resulting hotkey to operators. A rename
    /// on either side without the other would silently mislead a
    /// reader trying to match what they see in their portal UI
    /// against the doc.
    #[test]
    fn vl_desktop_shortcut_ids_match_overview_doc() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let portal_src =
            std::fs::read_to_string(format!("{repo_root}/crates/vl-desktop/src/portal.rs"))
                .expect("read vl-desktop portal.rs");
        let overview =
            std::fs::read_to_string(format!("{repo_root}/docs/architecture/overview.md"))
                .expect("read overview.md");

        let rust_ids = collect_rust_shortcut_ids(&portal_src);
        let doc_ids = collect_doc_shortcut_ids(&overview);

        assert!(
            !rust_ids.is_empty(),
            "vl-desktop portal.rs must declare at least one `pub const SHORTCUT_*` \
             constant; collect_rust_shortcut_ids may be misparsing",
        );
        assert!(
            !doc_ids.is_empty(),
            "overview.md must mention at least one `\\`voicelayer.<id>\\`` shortcut; \
             collect_doc_shortcut_ids may be misparsing",
        );

        for id in &rust_ids {
            assert!(
                doc_ids.contains(id),
                "Rust constant SHORTCUT_*=\"{id}\" is never mentioned in \
                 docs/architecture/overview.md. Add a backtick-quoted reference \
                 so operators can match the portal UI against the doc.",
            );
        }
        for id in &doc_ids {
            assert!(
                rust_ids.contains(id),
                "overview.md mentions `\\`{id}\\`` but no matching \
                 `pub const SHORTCUT_*: &str = \"{id}\";` exists in \
                 crates/vl-desktop/src/portal.rs. Either drop the doc \
                 mention or add the Rust constant.",
            );
        }
    }

    /// Every `ExecStart=%h/.local/bin/<basename>` referenced by a
    /// systemd unit must correspond to an `install -m 0755 "..."
    /// "${BIN_DIR}/<basename>"` line in `scripts/install.sh`. A unit
    /// that points at a binary the installer doesn't lay down would
    /// fail at first `systemctl --user enable --now <unit>` with
    /// "Executable file not found" — *after* the operator already
    /// committed to the install. The reverse direction is
    /// intentionally one-way: install.sh is allowed to lay down
    /// extras (e.g. `vl-desktop`, the GUI shell) that no unit
    /// supervises.
    #[test]
    fn systemd_execstart_binaries_are_all_installed_by_install_sh() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let install_sh = std::fs::read_to_string(format!("{repo_root}/scripts/install.sh"))
            .expect("read scripts/install.sh");
        let bin_targets = extract_install_sh_bin_targets(&install_sh);
        assert!(
            !bin_targets.is_empty(),
            "scripts/install.sh must install at least one file to ${{BIN_DIR}}; \
             extract_install_sh_bin_targets may be misparsing",
        );

        let unit_dir = std::path::PathBuf::from(format!("{repo_root}/systemd"));
        let mut execstart_basenames = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(&unit_dir).expect("read systemd directory") {
            let entry = entry.expect("read entry");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("service") {
                continue;
            }
            let unit = std::fs::read_to_string(&path).expect("read service file");
            execstart_basenames.extend(extract_systemd_execstart_basenames(&unit));
        }
        assert!(
            !execstart_basenames.is_empty(),
            "expected at least one systemd unit with `ExecStart=%h/.local/bin/...`; \
             extract_systemd_execstart_basenames may be misparsing",
        );

        for basename in &execstart_basenames {
            assert!(
                bin_targets.contains(basename),
                "systemd unit ExecStart references `%h/.local/bin/{basename}` but \
                 scripts/install.sh does not install a file named `{basename}` to \
                 BIN_DIR. Add the install line — `systemctl --user enable --now` \
                 would otherwise fail post-install with `Executable file not found`.",
            );
        }
    }

    /// Walk every `.md` file under `docs/` plus the repo-root
    /// `README.md`, pull out every relative `[text](path.md)` link
    /// (fragments stripped), and assert each resolves to an existing
    /// file. Markdown convention: link targets are relative to the
    /// containing file's parent directory, so a link from
    /// `docs/guides/systemd.md` to `./desktop.md` resolves to
    /// `docs/guides/desktop.md`.
    ///
    /// Catches: a doc renamed without updating inbound links, or a
    /// new link typo that points at a nonexistent file. Either would
    /// produce a 404 in any rendered view (GitHub, mdBook, etc.).
    #[test]
    fn every_markdown_doc_cross_reference_resolves_to_an_existing_file() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let mut md_files = vec![repo_root.join("README.md")];
        voicelayer_doc_test_utils::collect_markdown_files(&repo_root.join("docs"), &mut md_files)
            .expect("walk docs/ directory");
        assert!(
            md_files.len() > 1,
            "expected README.md plus at least one doc under docs/; \
             collect_markdown_files may be misparsing",
        );

        let mut broken: Vec<String> = Vec::new();
        for md_file in &md_files {
            let contents = match std::fs::read_to_string(md_file) {
                Ok(s) => s,
                Err(_) => continue,
            };
            let parent = md_file.parent().unwrap_or_else(|| std::path::Path::new(""));
            for link in extract_markdown_md_links(&contents) {
                let resolved = parent.join(&link);
                if !resolved.exists() {
                    broken.push(format!(
                        "{} → `{link}` (resolved to {})",
                        md_file
                            .strip_prefix(&repo_root)
                            .unwrap_or(md_file)
                            .display(),
                        resolved.display(),
                    ));
                }
            }
        }
        assert!(
            broken.is_empty(),
            "broken markdown cross-references:\n  - {}\n\n\
             Either rename the link target back, drop the reference, \
             or add a stub doc at the target path.",
            broken.join("\n  - "),
        );
    }

    /// Walk every `.md` under `docs/` plus the repo-root `README.md`,
    /// collect every backtick-quoted file path with a path
    /// separator and a recognised source extension, and assert each
    /// resolves to an existing file. The check tolerates the
    /// shorthand convention used in
    /// `docs/architecture/python-worker-protocol.md`: a path
    /// starting with `providers/` is implicitly anchored under
    /// `python/voicelayer_orchestrator/`. Anything else is resolved
    /// against the repo root.
    ///
    /// Catches: a renamed source file (`scripts/foo.sh`) or doc
    /// (`docs/guides/desktop.md`) whose inbound prose references
    /// were missed during the rename — Markdown's `()` link form is
    /// already covered by #69, but bare backtick references are
    /// not, and contributors use those just as often.
    #[test]
    fn every_doc_file_path_reference_resolves_to_an_existing_file() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let mut md_files = vec![repo_root.join("README.md")];
        voicelayer_doc_test_utils::collect_markdown_files(&repo_root.join("docs"), &mut md_files)
            .expect("walk docs/ directory");

        let mut broken: Vec<String> = Vec::new();
        for md_file in &md_files {
            let contents = match std::fs::read_to_string(md_file) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for path in extract_doc_file_path_refs(&contents) {
                let primary = repo_root.join(&path);
                let shorthand = repo_root
                    .join("python")
                    .join("voicelayer_orchestrator")
                    .join(&path);
                if primary.exists() || shorthand.exists() {
                    continue;
                }
                broken.push(format!(
                    "{} → `{path}`",
                    md_file
                        .strip_prefix(&repo_root)
                        .unwrap_or(md_file)
                        .display(),
                ));
            }
        }
        assert!(
            broken.is_empty(),
            "broken doc file-path references:\n  - {}\n\n\
             Either fix the path, drop the mention, or restore the \
             missing file. Paths under `providers/` are also tried \
             with the `python/voicelayer_orchestrator/` prefix to \
             match the shorthand convention used in the python-worker \
             protocol doc.",
            broken.join("\n  - "),
        );
    }

    #[test]
    fn extract_workspace_dependency_names_handles_single_and_inline_table_forms() {
        let toml = "\
[workspace]
members = [\"crates/foo\"]

[workspace.dependencies]
arboard = \"3.6.1\"
axum = { version = \"0.8.6\", features = [\"http1\"] }
# a comment line should not be captured
serde = { version = \"1.0\", features = [\"derive\"] }

[workspace.package]
version = \"0.1.0\"
";
        let names = extract_workspace_dependency_names(toml);
        assert_eq!(
            names,
            ["arboard", "axum", "serde"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_workspace_dependency_names_skips_continuation_rows_inside_inline_tables() {
        // A multi-line inline table whose `features = [` array
        // continues across rows. The sub-key name (`features`) and
        // the indented array literals must NOT be captured as fake
        // workspace deps.
        let toml = "\
[workspace.dependencies]
ashpd = { version = \"0.13\", default-features = false, features = [
    \"tokio\",
    \"global_shortcuts\",
] }
bytes = \"1.10.1\"
";
        let names = extract_workspace_dependency_names(toml);
        assert_eq!(
            names,
            ["ashpd", "bytes"].iter().map(|s| (*s).to_owned()).collect(),
        );
    }

    #[test]
    fn classify_crate_dependency_forms_separates_workspace_from_literal_entries() {
        let toml = "\
[package]
name = \"foo\"

[dependencies]
arboard.workspace = true
axum = \"0.8.6\"
iced = { version = \"0.14\", features = [\"tokio\"] }
voicelayer-core = { path = \"../voicelayer-core\" }

[dev-dependencies]
tempfile.workspace = true
";
        let (inherited, literal) = classify_crate_dependency_forms(toml);
        assert_eq!(
            inherited,
            ["arboard", "tempfile"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
        assert_eq!(
            literal,
            ["axum", "iced", "voicelayer-core"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn classify_crate_dependency_forms_handles_multiline_inline_table_continuations() {
        // The exact shape used by `crates/vl-desktop/Cargo.toml`
        // for `ashpd`. The continuation rows must not be captured
        // as separate deps, and `ashpd` itself must land in the
        // literal-or-inline bucket because it has no
        // `.workspace = true` form.
        let toml = "\
[dependencies]
ashpd = { version = \"0.13\", default-features = false, features = [
    \"tokio\",
    \"global_shortcuts\",
] }
bytes.workspace = true
";
        let (inherited, literal) = classify_crate_dependency_forms(toml);
        assert_eq!(
            inherited,
            ["bytes"].iter().map(|s| (*s).to_owned()).collect(),
        );
        assert_eq!(literal, ["ashpd"].iter().map(|s| (*s).to_owned()).collect(),);
    }

    #[test]
    fn classify_crate_dependency_forms_ignores_non_dependency_sections() {
        // Keys defined under `[package]` or `[features]` are not
        // dependency entries and must never appear in either set.
        let toml = "\
[package]
name = \"foo\"
version = \"0.1.0\"

[features]
default = [\"std\"]
std = []

[dependencies]
serde.workspace = true
";
        let (inherited, literal) = classify_crate_dependency_forms(toml);
        assert_eq!(
            inherited,
            ["serde"].iter().map(|s| (*s).to_owned()).collect(),
        );
        assert!(
            literal.is_empty(),
            "no literal deps expected, got {literal:?}",
        );
    }

    /// For every member crate listed in the workspace `Cargo.toml`,
    /// any dependency whose name is centralised in
    /// `[workspace.dependencies]` MUST be declared in the member's
    /// `[dependencies]` / `[dev-dependencies]` / `[build-dependencies]`
    /// as `<name>.workspace = true`. A literal version pin or inline
    /// table on a centralised dep silently diverges from the
    /// workspace's single source of truth — and Cargo will happily
    /// build either form, so the bug surfaces only when the workspace
    /// pin is bumped and one crate is left behind.
    ///
    /// Crate-local literal pins for deps that are NOT in
    /// `[workspace.dependencies]` (e.g. `vl-desktop`'s `ashpd` and
    /// `iced` blocks) are unaffected: the intersection with the
    /// workspace dep set drops them automatically.
    #[test]
    fn every_workspace_centralised_dep_uses_workspace_inheritance_in_member_crates() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let workspace_toml = std::fs::read_to_string(repo_root.join("Cargo.toml"))
            .expect("read workspace Cargo.toml");
        let workspace_deps = extract_workspace_dependency_names(&workspace_toml);
        assert!(
            !workspace_deps.is_empty(),
            "extract_workspace_dependency_names returned empty — \
             [workspace.dependencies] block may have moved or the \
             scanner is mis-parsing the new shape",
        );
        let members = extract_cargo_workspace_members(&workspace_toml);
        assert!(
            !members.is_empty(),
            "extract_cargo_workspace_members returned empty — \
             [workspace] members list may have moved",
        );

        let mut violations: Vec<String> = Vec::new();
        for member in &members {
            let manifest_path = repo_root.join(member).join("Cargo.toml");
            let contents = std::fs::read_to_string(&manifest_path).unwrap_or_else(|err| {
                panic!("read {}: {err}", manifest_path.display());
            });
            let (inherited, literal_or_inline) = classify_crate_dependency_forms(&contents);
            for name in literal_or_inline.intersection(&workspace_deps) {
                if inherited.contains(name) {
                    continue;
                }
                violations.push(format!("{member}: `{name}` is centralised in workspace.dependencies but the crate uses a literal/inline form"));
            }
        }
        assert!(
            violations.is_empty(),
            "workspace dep inheritance drift:\n  - {}\n\n\
             Replace each offender with `<name>.workspace = true` so \
             the version stays pinned in `Cargo.toml` only. If you \
             genuinely need a per-crate override (different feature \
             set, etc.), drop the dep from `[workspace.dependencies]` \
             so the centralised pin no longer claims to govern it.",
            violations.join("\n  - "),
        );
    }

    #[test]
    fn extract_pyproject_optional_deps_handles_multiline_array_form() {
        let pyproject = "\
[project]
name = \"foo\"

[project.optional-dependencies]
vad = [
    \"numpy>=2.0\",
    \"onnxruntime>=1.18\",
]
";
        let names = extract_pyproject_optional_deps(pyproject, "vad");
        assert_eq!(
            names,
            ["numpy", "onnxruntime"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_pyproject_optional_deps_strips_pep508_specifiers_and_extras() {
        // The stripper has to drop `>=`, `<`, `==`, `~=`, `!=`, plus
        // extras (`[ssl]`), env markers (`; python_version < \"3.12\"`),
        // and trailing whitespace. All survive as a bare name.
        let pyproject = "\
[project.optional-dependencies]
mixed = [
    \"requests[ssl]>=2.31\",
    \"urllib3==1.26.0\",
    \"pytest>=8 ; python_version < '3.12'\",
    \"trio~=0.23\",
]
";
        let names = extract_pyproject_optional_deps(pyproject, "mixed");
        assert_eq!(
            names,
            ["pytest", "requests", "trio", "urllib3"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_pyproject_optional_deps_returns_empty_when_group_absent() {
        let pyproject = "\
[project.optional-dependencies]
other = [\"foo\"]
";
        let names = extract_pyproject_optional_deps(pyproject, "vad");
        assert!(
            names.is_empty(),
            "expected empty set for missing group, got {names:?}",
        );
    }

    #[test]
    fn gather_python_imports_from_source_handles_indented_lazy_imports() {
        // The exact shape used in
        // `python/voicelayer_orchestrator/providers/vad_segmenter.py`:
        // `try: <indent> import X as Y  # noqa: PLC0415`. The trim and
        // alias-strip both have to fire, plus the inline comment must
        // not pollute the captured name.
        let source = "\
from __future__ import annotations

def lazy():
    try:
        import numpy as np  # noqa: PLC0415
    except ImportError:
        raise
    try:
        import onnxruntime as ort
    except ImportError:
        raise
    return np, ort
";
        let mut names = std::collections::BTreeSet::new();
        gather_python_imports_from_source(source, &mut names);
        assert!(names.contains("numpy"));
        assert!(names.contains("onnxruntime"));
        assert!(names.contains("__future__"));
    }

    #[test]
    fn gather_python_imports_from_source_handles_comma_separated_and_dotted_forms() {
        let source = "\
import os, sys
from pathlib import Path
from typing import Any
from voicelayer_orchestrator.providers import whisper_cli
";
        let mut names = std::collections::BTreeSet::new();
        gather_python_imports_from_source(source, &mut names);
        // `from voicelayer_orchestrator.providers` collapses to its
        // root segment.
        for expected in ["os", "sys", "pathlib", "typing", "voicelayer_orchestrator"] {
            assert!(
                names.contains(expected),
                "expected `{expected}` in {names:?}",
            );
        }
    }

    #[test]
    fn extract_claude_md_verification_chain_strips_continuation_and_join_markers() {
        let md = "\
# Heading

Some prose.

```bash
cargo fmt --all \\
  && cargo clippy -- -D warnings \\
  && cargo test --all
```

More prose.
";
        let commands = extract_claude_md_verification_chain(md);
        assert_eq!(
            commands,
            vec![
                "cargo fmt --all".to_owned(),
                "cargo clippy -- -D warnings".to_owned(),
                "cargo test --all".to_owned(),
            ],
        );
    }

    #[test]
    fn extract_ci_verify_run_commands_filters_to_cargo_and_uv_invocations() {
        let yaml = "\
jobs:
  verify:
    steps:
      - name: Install deps
        run: apt-get install -y foo
      - name: Install Rust
        uses: dtolnay/rust-toolchain@master
      - name: cargo test
        run: cargo test --all
      - name: ruff
        run: uv run ruff check python
";
        let commands = extract_ci_verify_run_commands(yaml);
        assert_eq!(
            commands,
            vec![
                "cargo test --all".to_owned(),
                "uv run ruff check python".to_owned(),
            ],
        );
    }

    #[test]
    fn normalise_fmt_command_drops_check_suffix_only_for_cargo_fmt() {
        assert_eq!(
            normalise_fmt_command("cargo fmt --all -- --check"),
            "cargo fmt --all",
        );
        assert_eq!(normalise_fmt_command("cargo fmt -- --check"), "cargo fmt",);
        // Not a fmt invocation — pass through unchanged even if a
        // similarly-shaped suffix appears.
        assert_eq!(
            normalise_fmt_command("cargo clippy --all -- -D warnings"),
            "cargo clippy --all -- -D warnings",
        );
    }

    /// `README.md` and `CLAUDE.md` both document the verification
    /// chain. The README copy is the contributor-facing list (plus
    /// the one-time `uv sync --group dev` setup step); the
    /// `CLAUDE.md` copy is the per-task chain Claude / agents are
    /// expected to run after the env is already synced. The two
    /// must agree on the *task-level* steps so a contributor's
    /// expectation matches the agent's behaviour: filtering
    /// `uv sync` out of the README list, the rest must be a
    /// byte-for-byte match.
    ///
    /// The drift mode is the README quietly adding a check the
    /// agent never runs (or vice versa): a human reading the README
    /// sees green, the agent reading `CLAUDE.md` sees green, but
    /// the two were checking different things.
    #[test]
    fn readme_verification_chain_matches_claude_md_chain_modulo_uv_sync() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let readme = std::fs::read_to_string(repo_root.join("README.md")).expect("read README.md");
        let claude_md =
            std::fs::read_to_string(repo_root.join("CLAUDE.md")).expect("read CLAUDE.md");

        let readme_raw = extract_claude_md_verification_chain(&readme);
        let claude_raw = extract_claude_md_verification_chain(&claude_md);
        assert!(
            !readme_raw.is_empty(),
            "README's first ```bash``` block (under \"Verification Chain\") not found",
        );
        assert!(
            !claude_raw.is_empty(),
            "CLAUDE.md's first ```bash``` block not found",
        );

        let readme_filtered: Vec<String> = readme_raw
            .into_iter()
            .filter(|cmd| !cmd.starts_with("uv sync"))
            .collect();

        assert_eq!(
            readme_filtered, claude_raw,
            "verification chain drift between README.md and CLAUDE.md.\n\
             README (filtered, no `uv sync`): {readme_filtered:#?}\n\
             CLAUDE.md: {claude_raw:#?}\n\n\
             The README's contributor-facing list and CLAUDE.md's agent-facing \
             list must agree on every task-level step. The only sanctioned \
             addition in README is `uv sync --group dev` (a one-time setup \
             that runs before the chain, not as part of it).",
        );
    }

    /// `CLAUDE.md` documents the verification chain that contributors
    /// (and Claude) must run before closing a task; `.github/workflows/ci.yml`
    /// must run the same chain so CI mirrors local-pass discipline. A
    /// step added to one without the other is the failure mode: a
    /// contributor follows `CLAUDE.md`, sees green locally, but CI
    /// either skips a check (and lets the regression in) or runs an
    /// extra check that the doc never mentions (and blocks an honest
    /// PR with a surprise red).
    ///
    /// The single allowance: `cargo fmt --all` in `CLAUDE.md`
    /// (write-mode) maps to `cargo fmt --all -- --check` in CI
    /// (read-only). Both run the same formatter; only CI needs to
    /// fail rather than rewrite. `normalise_fmt_command` collapses
    /// the two onto a single canonical form before comparison.
    #[test]
    fn ci_verify_chain_matches_claude_md_verification_chain() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let claude_md =
            std::fs::read_to_string(repo_root.join("CLAUDE.md")).expect("read CLAUDE.md");
        let ci_yaml = std::fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))
            .expect("read .github/workflows/ci.yml");

        let chain_raw = extract_claude_md_verification_chain(&claude_md);
        assert!(
            !chain_raw.is_empty(),
            "CLAUDE.md verification chain not found — the first ```bash``` block \
             may have moved or been renamed",
        );
        let ci_raw = extract_ci_verify_run_commands(&ci_yaml);
        assert!(
            !ci_raw.is_empty(),
            "ci.yml verify steps not found — the workflow may have been refactored \
             to call the chain through a script",
        );

        let chain: Vec<String> = chain_raw.iter().map(|s| normalise_fmt_command(s)).collect();
        let ci: Vec<String> = ci_raw.iter().map(|s| normalise_fmt_command(s)).collect();

        assert_eq!(
            chain, ci,
            "verification chain drift between CLAUDE.md and .github/workflows/ci.yml.\n\
             CLAUDE.md (normalised): {chain:#?}\n\
             ci.yml (normalised): {ci:#?}\n\n\
             Bring the two in sync: a step in `CLAUDE.md` that CI does not \
             gate on lets contributors close work CI would have rejected, \
             and a CI step missing from `CLAUDE.md` blocks PRs with a check \
             nobody is told to run. The only sanctioned divergence is \
             `cargo fmt --all` (CLAUDE.md, write-mode) vs \
             `cargo fmt --all -- --check` (ci.yml, read-only); \
             `normalise_fmt_command` already collapses that pair.",
        );
    }

    /// Every package declared in `[project.optional-dependencies].vad`
    /// must actually be imported somewhere under
    /// `python/voicelayer_orchestrator/`. A dep listed in pyproject
    /// but never imported is dead documentation: an operator who runs
    /// `uv sync --extra vad` installs it for nothing, and the next
    /// dependabot bump touches a package the codebase no longer cares
    /// about.
    ///
    /// Reverse direction (every imported third-party package is in
    /// the optional-deps group) requires a stdlib reference list that
    /// would drift with each Python release; not enforced here.
    #[test]
    fn vad_optional_deps_are_all_imported_under_python_orchestrator() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let pyproject =
            std::fs::read_to_string(repo_root.join("pyproject.toml")).expect("read pyproject.toml");
        let vad_deps = extract_pyproject_optional_deps(&pyproject, "vad");
        assert!(
            !vad_deps.is_empty(),
            "expected `[project.optional-dependencies].vad` to declare at least one \
             package — pyproject may have been refactored away from optional-deps",
        );

        let mut imports = std::collections::BTreeSet::new();
        collect_python_imports_under(
            &repo_root.join("python").join("voicelayer_orchestrator"),
            &mut imports,
        )
        .expect("walk python/voicelayer_orchestrator/");

        let missing: Vec<&String> = vad_deps.difference(&imports).collect();
        assert!(
            missing.is_empty(),
            "vad optional-deps are not imported anywhere under \
             python/voicelayer_orchestrator/: {missing:?}\n\n\
             Either drop the dead entries from `[project.optional-dependencies].vad` \
             in pyproject.toml or wire the import in the provider that needs them. \
             A dead extra installs cost without effect when an operator runs \
             `uv sync --extra vad`.",
        );
    }

    #[test]
    fn extract_install_sh_cargo_build_bins_collects_repeated_bin_flags() {
        let sh = "\
#!/usr/bin/env bash
cargo build --release --bin vl --bin vl-desktop
";
        let bins = extract_install_sh_cargo_build_bins(sh);
        assert_eq!(
            bins,
            ["vl", "vl-desktop"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_install_sh_cargo_build_bins_handles_line_continuations() {
        let sh = "\
cargo build --release \\
  --bin vl \\
  --bin vl-desktop
";
        let bins = extract_install_sh_cargo_build_bins(sh);
        assert_eq!(
            bins,
            ["vl", "vl-desktop"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_install_sh_cargo_build_bins_strips_trailing_subshell_paren() {
        // The shape used in `scripts/install.sh` today wraps the build
        // in a `(cd ... && cargo build ...)` subshell, so the final
        // `--bin <name>` runs straight into a `)`. Without dropping
        // the paren the extractor returns `"vl-desktop)"` and any
        // membership check against `"vl-desktop"` fails silently.
        let sh = "(cd \"${REPO_ROOT}\" && cargo build --release --bin vl --bin vl-desktop)";
        let bins = extract_install_sh_cargo_build_bins(sh);
        assert_eq!(
            bins,
            ["vl", "vl-desktop"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_install_sh_install_pairs_collapses_continuation_lines() {
        // The shape used in `scripts/install.sh` for the
        // whisper-server-run wrapper: source on line N, dest on
        // line N+1, separated by `\` line-join.
        let sh = "\
install -m 0755 \"${REPO_ROOT}/scripts/voicelayer-whisper-server-run.sh\" \\
  \"${BIN_DIR}/voicelayer-whisper-server-run.sh\"
install -m 0755 \"${REPO_ROOT}/target/release/vl\" \"${BIN_DIR}/vl\"
";
        let pairs = extract_install_sh_install_pairs(sh);
        assert_eq!(
            pairs,
            vec![
                (
                    "${REPO_ROOT}/scripts/voicelayer-whisper-server-run.sh".to_owned(),
                    "${BIN_DIR}/voicelayer-whisper-server-run.sh".to_owned(),
                ),
                (
                    "${REPO_ROOT}/target/release/vl".to_owned(),
                    "${BIN_DIR}/vl".to_owned(),
                ),
            ],
        );
    }

    /// Every `install -m 0755 "<source>" "${BIN_DIR}/<dest>"` line in
    /// `scripts/install.sh` must point at a source that this script
    /// actually produces or ships:
    ///
    /// - `${REPO_ROOT}/target/release/<name>` requires a matching
    ///   `--bin <name>` flag in the `cargo build` invocation; without
    ///   it cargo never emits the binary and the install fails at
    ///   runtime.
    /// - `${REPO_ROOT}/<rel>` (any other repo-relative path, e.g.
    ///   `scripts/foo.sh`) requires the file to exist on disk.
    ///
    /// Catches a future contributor adding `--bin voicelayerd` to an
    /// install line without remembering to add it to the
    /// `cargo build` flags, or adding a wrapper script install
    /// without committing the wrapper.
    #[test]
    fn install_sh_install_targets_resolve_to_either_built_or_shipped_sources() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let install_sh = std::fs::read_to_string(repo_root.join("scripts/install.sh"))
            .expect("read scripts/install.sh");

        let built_bins = extract_install_sh_cargo_build_bins(&install_sh);
        assert!(
            !built_bins.is_empty(),
            "expected at least one `--bin` flag in install.sh's `cargo build`",
        );
        let pairs = extract_install_sh_install_pairs(&install_sh);
        assert!(
            !pairs.is_empty(),
            "expected at least one `install -m` line in install.sh",
        );

        let mut violations: Vec<String> = Vec::new();
        for (source, dest) in pairs {
            if !dest.contains("${BIN_DIR}") {
                continue;
            }
            if let Some(name) = source.strip_prefix("${REPO_ROOT}/target/release/") {
                if !built_bins.contains(name) {
                    violations.push(format!(
                        "{source} -> {dest}: install line copies a cargo binary that \
                         the script never builds (no `--bin {name}` flag)",
                    ));
                }
            } else if let Some(rel) = source.strip_prefix("${REPO_ROOT}/") {
                let abs = repo_root.join(rel);
                if !abs.exists() {
                    violations.push(format!(
                        "{source} -> {dest}: source path `{}` does not exist on disk",
                        abs.display(),
                    ));
                }
            }
        }
        assert!(
            violations.is_empty(),
            "scripts/install.sh references missing or unbuilt sources:\n  - {}\n\n\
             For a `target/release/<name>` source, add `--bin <name>` to the \
             `cargo build` line. For a repo-relative source (e.g. `scripts/foo.sh`), \
             commit the file at the referenced path.",
            violations.join("\n  - "),
        );
    }

    #[test]
    fn extract_readme_requirement_floor_strips_trailing_plus_and_prose() {
        let md = "\
### Requirements

- Rust 1.88+
- Python 3.12+ (managed through `uv`)
- Ubuntu with PipeWire
";
        assert_eq!(
            extract_readme_requirement_floor(md, "Rust"),
            Some("1.88".to_owned()),
        );
        assert_eq!(
            extract_readme_requirement_floor(md, "Python"),
            Some("3.12".to_owned()),
        );
    }

    #[test]
    fn extract_readme_requirement_floor_returns_none_when_language_absent() {
        let md = "\
### Requirements

- Rust 1.88+
";
        assert!(extract_readme_requirement_floor(md, "Python").is_none());
    }

    /// The README's "Requirements" list documents the toolchain
    /// floors a contributor needs locally. Those floors must agree
    /// with the canonical sources:
    /// - `Rust X.Y+` ↔ `Cargo.toml` `[workspace.package].rust-version`
    /// - `Python X.Y+` ↔ `pyproject.toml` `requires-python = ">=X.Y"`
    ///
    /// The drift mode is silent and especially confusing for new
    /// contributors: the README invites them to install Rust 1.87,
    /// the workspace MSRV is now 1.88, and `cargo build` fails with
    /// a one-line "package requires rustc 1.88 or newer" error that
    /// reads like a build-environment problem rather than a
    /// documentation lag.
    ///
    /// `cargo` and `pyproject` are already pinned to each other
    /// (PR #67 for the cargo↔pyproject leg, the MSRV-triangle test
    /// for cargo↔rust-toolchain↔ci); this test closes the README
    /// leg.
    #[test]
    fn readme_requirements_match_cargo_and_pyproject_toolchain_floors() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let readme = std::fs::read_to_string(repo_root.join("README.md")).expect("read README.md");
        let cargo_toml = std::fs::read_to_string(repo_root.join("Cargo.toml"))
            .expect("read workspace Cargo.toml");
        let pyproject =
            std::fs::read_to_string(repo_root.join("pyproject.toml")).expect("read pyproject.toml");

        let readme_rust = extract_readme_requirement_floor(&readme, "Rust")
            .expect("README must declare `- Rust <X.Y>+` in Requirements — line may have moved");
        let readme_python = extract_readme_requirement_floor(&readme, "Python")
            .expect("README must declare `- Python <X.Y>+` in Requirements — line may have moved");

        let cargo_rust = extract_toml_string_value(&cargo_toml, "rust-version")
            .expect("Cargo.toml [workspace.package].rust-version not found");
        let pyproject_python = extract_python_version_floor_from_pyproject(&pyproject)
            .expect("pyproject.toml requires-python not found or unparseable");

        assert_eq!(
            readme_rust, cargo_rust,
            "README declares `Rust {readme_rust}+` but `Cargo.toml` rust-version is \
             `{cargo_rust}`. Bump the README list when the workspace MSRV moves.",
        );
        assert_eq!(
            readme_python, pyproject_python,
            "README declares `Python {readme_python}+` but `pyproject.toml` \
             requires-python floor is `{pyproject_python}`. Bump the README list \
             when the python floor moves.",
        );
    }

    #[test]
    fn extract_systemd_environment_file_paths_drops_optional_dash_prefix() {
        let unit = "\
[Service]
Type=simple
EnvironmentFile=-%h/.config/voicelayer/voicelayerd.env
EnvironmentFile=/etc/voicelayerd.env
ExecStart=%h/.local/bin/vl daemon run
";
        let paths = extract_systemd_environment_file_paths(unit);
        assert_eq!(
            paths,
            vec![
                "%h/.config/voicelayer/voicelayerd.env".to_owned(),
                "/etc/voicelayerd.env".to_owned(),
            ],
        );
    }

    #[test]
    fn extract_install_sh_default_env_dir_picks_up_canonical_braced_default() {
        let sh = "\
#!/usr/bin/env bash
BIN_DIR=\"${VOICELAYER_INSTALL_BIN_DIR:-${HOME}/.local/bin}\"
ENV_DIR=\"${VOICELAYER_INSTALL_ENV_DIR:-${HOME}/.config/voicelayer}\"
";
        assert_eq!(
            extract_install_sh_default_env_dir(sh),
            Some("${HOME}/.config/voicelayer".to_owned()),
        );
    }

    #[test]
    fn extract_install_sh_default_env_dir_returns_none_when_var_absent() {
        let sh = "BIN_DIR=\"${VOICELAYER_INSTALL_BIN_DIR:-${HOME}/.local/bin}\"\n";
        assert!(extract_install_sh_default_env_dir(sh).is_none());
    }

    /// Every systemd unit's `EnvironmentFile=` directive must point
    /// at the path `scripts/install.sh` seeds. The two surfaces use
    /// different placeholder syntaxes — systemd writes `%h/...` (the
    /// home-dir runtime variable), install.sh writes `${HOME}/...`
    /// (the bash variable) — but they expand to the same disk path,
    /// so the suffix after the placeholder must agree.
    ///
    /// The drift mode is silent and operator-facing: install.sh
    /// happily writes `~/.config/voicelayer-orchestrator/voicelayerd.env`
    /// while systemd keeps reading `~/.config/voicelayer/voicelayerd.env`,
    /// the unit starts but sees no env vars, and the daemon falls
    /// back to in-source defaults that look like a misconfigured
    /// installation. The fix is one-line in either file but the
    /// failure has no obvious signal.
    #[test]
    fn systemd_environment_file_paths_resolve_to_install_sh_env_dir_default() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let install_sh = std::fs::read_to_string(repo_root.join("scripts/install.sh"))
            .expect("read scripts/install.sh");
        let env_dir_default = extract_install_sh_default_env_dir(&install_sh).expect(
            "scripts/install.sh `ENV_DIR=\"${...:-${HOME}/...}\"` default not found — \
             the bash idiom may have moved",
        );
        let install_suffix = env_dir_default
            .strip_prefix("${HOME}/")
            .unwrap_or(&env_dir_default);
        let install_full = format!("{install_suffix}/voicelayerd.env");

        let units: &[&std::path::Path] = &[
            std::path::Path::new("systemd/voicelayerd.service"),
            std::path::Path::new("systemd/voicelayer-whisper-server.service"),
        ];
        for unit_rel in units {
            let unit_abs = repo_root.join(unit_rel);
            let unit_source = std::fs::read_to_string(&unit_abs)
                .unwrap_or_else(|err| panic!("read {}: {err}", unit_abs.display()));
            let paths = extract_systemd_environment_file_paths(&unit_source);
            assert!(
                !paths.is_empty(),
                "{} declares no `EnvironmentFile=` line — the unit may be loading env via \
                 a different mechanism (drop-in, [Service] inline `Environment=`), in \
                 which case this drift guard needs to follow that move",
                unit_rel.display(),
            );
            for path in paths {
                let suffix = path.strip_prefix("%h/").unwrap_or(&path);
                assert_eq!(
                    suffix,
                    install_full,
                    "{}: `EnvironmentFile=` resolves to `{path}` (suffix `{suffix}`) but \
                     `scripts/install.sh` seeds `{install_full}`. Bring the two in sync \
                     so the unit reads the file install.sh actually writes.",
                    unit_rel.display(),
                );
            }
        }
    }

    #[test]
    fn extract_readme_crate_path_refs_collects_single_segment_paths() {
        let md = "\
- `crates/voicelayer-core`: shared types
- `crates/voicelayerd`: daemon
- `crates/vl`: CLI
- See also `crates/vl/src/main.rs` for the entry point.
- The bare directory `crates/` is not a crate.
";
        let refs = extract_readme_crate_path_refs(md);
        assert_eq!(
            refs,
            ["vl", "voicelayer-core", "voicelayerd"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "deeper paths and bare `crates/` must be rejected",
        );
    }

    #[test]
    fn extract_readme_crate_path_refs_rejects_non_alphanumeric_segments() {
        // A backtick-quoted `crates/foo bar` is prose, not a crate
        // reference, and should be rejected.
        let md = "Open `crates/foo bar` to see the demo.";
        assert!(extract_readme_crate_path_refs(md).is_empty());
    }

    /// Every workspace member listed in `Cargo.toml` must be
    /// mentioned in the README — typically in the Architecture
    /// section, but anywhere in the file qualifies. And every
    /// backtick-quoted `crates/<name>` reference in the README
    /// must point at a real workspace member; a typo like
    /// `crates/voicelayer-cor` becomes a broken cross-reference
    /// the moment a contributor clicks through it.
    ///
    /// The drift mode in both directions is silent. Forward: a new
    /// crate gets added to the workspace, the architecture section
    /// stays a list of three, and a reader walks away thinking the
    /// new crate is unimportant or doesn't exist. Reverse: a typo
    /// or a renamed crate leaves the README pointing at nothing,
    /// and `cargo` happily builds because the README is not part
    /// of any compile path.
    #[test]
    fn readme_crate_mentions_match_workspace_members_bidirectionally() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let cargo_toml = std::fs::read_to_string(repo_root.join("Cargo.toml"))
            .expect("read workspace Cargo.toml");
        let readme = std::fs::read_to_string(repo_root.join("README.md")).expect("read README.md");

        let members = extract_cargo_workspace_members(&cargo_toml);
        let member_names: std::collections::BTreeSet<String> = members
            .iter()
            .filter_map(|m| m.strip_prefix("crates/").map(str::to_owned))
            .collect();
        assert!(
            !member_names.is_empty(),
            "expected at least one `crates/<name>` member in Cargo.toml — \
             extract_cargo_workspace_members may be misparsing",
        );

        let readme_refs = extract_readme_crate_path_refs(&readme);
        assert!(
            !readme_refs.is_empty(),
            "expected at least one `crates/<name>` reference in README — \
             the Architecture section may have moved",
        );

        let missing_in_readme: Vec<&String> = member_names.difference(&readme_refs).collect();
        assert!(
            missing_in_readme.is_empty(),
            "workspace members not mentioned in README: {missing_in_readme:?}\n\n\
             Add a backtick-quoted `crates/<name>` reference in the README \
             (typically under \"Architecture\") so a reader can locate the \
             crate from the project front page.",
        );

        let typo_in_readme: Vec<&String> = readme_refs.difference(&member_names).collect();
        assert!(
            typo_in_readme.is_empty(),
            "README references crates that the workspace does not declare: {typo_in_readme:?}\n\n\
             Fix the typo, drop the mention, or add the member to \
             [workspace] members in `Cargo.toml`.",
        );
    }

    #[test]
    fn extract_pyproject_dependency_group_handles_multiline_array_form() {
        let pyproject = "\
[dependency-groups]
dev = [
    \"pytest>=9.0.3,<10\",
    \"ruff>=0.15.11\",
]

[tool.setuptools]
package-dir = { \"\" = \"python\" }
";
        let names = extract_pyproject_dependency_group(pyproject, "dev");
        assert_eq!(
            names,
            ["pytest", "ruff"].iter().map(|s| (*s).to_owned()).collect(),
        );
    }

    #[test]
    fn extract_pyproject_dependency_group_returns_empty_when_section_absent() {
        let pyproject = "[project]\nname = \"foo\"\n";
        let names = extract_pyproject_dependency_group(pyproject, "dev");
        assert!(names.is_empty());
    }

    #[test]
    fn extract_pyproject_dependency_group_does_not_leak_optional_deps_section() {
        // A `vad` group declared under `[project.optional-dependencies]`
        // (the surface PR #74 covers) must NOT appear when the caller
        // asks for `[dependency-groups].vad`. The two sections live
        // under different headings even though they share the array
        // shape; mixing them up would wire the dev-deps drift guard
        // against the wrong source.
        let pyproject = "\
[project.optional-dependencies]
vad = [\"numpy>=2.0\", \"onnxruntime>=1.18\"]

[dependency-groups]
dev = [\"pytest>=9.0.3,<10\"]
";
        assert_eq!(
            extract_pyproject_dependency_group(pyproject, "dev"),
            ["pytest"].iter().map(|s| (*s).to_owned()).collect(),
        );
        assert!(
            extract_pyproject_dependency_group(pyproject, "vad").is_empty(),
            "vad lives under [project.optional-dependencies], not [dependency-groups]",
        );
    }

    /// Every package declared under `[dependency-groups].dev` in
    /// `pyproject.toml` must be invoked at least once in
    /// `.github/workflows/ci.yml`'s verify job, in the canonical
    /// `uv run <package>` shape.
    ///
    /// The drift mode is silent and gating-side: a contributor adds
    /// `mypy` to `dev` thinking it now runs in CI, the CI workflow
    /// keeps invoking only `pytest` and `ruff`, and a regression
    /// `mypy` would have caught lands cleanly.
    ///
    /// The reverse direction (every `uv run` invocation matches a
    /// dev-dep) is intentionally not enforced — `uv run pytest` and
    /// `uv run ruff` are the only `uv run` invocations today, and
    /// both ARE in dev-deps; if a future workflow runs e.g.
    /// `uv run python` for a one-off probe, the test should not
    /// fight that.
    #[test]
    fn every_pyproject_dev_dep_is_invoked_in_ci_verify_job() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let pyproject =
            std::fs::read_to_string(repo_root.join("pyproject.toml")).expect("read pyproject.toml");
        let ci_yaml = std::fs::read_to_string(repo_root.join(".github/workflows/ci.yml"))
            .expect("read .github/workflows/ci.yml");

        let dev_deps = extract_pyproject_dependency_group(&pyproject, "dev");
        assert!(
            !dev_deps.is_empty(),
            "expected at least one package in `[dependency-groups].dev` — \
             pyproject may have been refactored away from dependency-groups",
        );

        let ci_commands = extract_ci_verify_run_commands(&ci_yaml);
        let invoked: std::collections::BTreeSet<String> = ci_commands
            .iter()
            .filter_map(|cmd| cmd.strip_prefix("uv run "))
            .filter_map(|tail| tail.split_whitespace().next().map(str::to_owned))
            .collect();
        assert!(
            !invoked.is_empty(),
            "expected at least one `uv run <package>` step in CI — \
             extract_ci_verify_run_commands may be misparsing",
        );

        let unused: Vec<&String> = dev_deps.difference(&invoked).collect();
        assert!(
            unused.is_empty(),
            "pyproject [dependency-groups].dev declares packages CI never \
             invokes via `uv run`: {unused:?}\n\nEither drop the dep from \
             pyproject.toml or add a `uv run <package> ...` step to the \
             verify job in .github/workflows/ci.yml.",
        );
    }

    #[test]
    fn extract_doc_voicelayer_env_var_mentions_drops_wildcard_and_prose_forms() {
        let md = "\
- `VOICELAYER_LLM_ENDPOINT` is the chat completion URL.
- `VOICELAYER_WHISPER_SERVER_*` is shorthand for several knobs.
- `VOICELAYER_FOO BAR` looks like prose, not a real var.
- The `VOICELAYER_LLM_TIMEOUT_SECONDS` knob bounds requests.
";
        let mentions = extract_doc_voicelayer_env_var_mentions(md);
        assert_eq!(
            mentions,
            ["VOICELAYER_LLM_ENDPOINT", "VOICELAYER_LLM_TIMEOUT_SECONDS"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
        );
    }

    #[test]
    fn extract_env_example_voicelayer_keys_handles_live_and_commented_assignments() {
        let env = "\
# header comment

VOICELAYER_LIVE_KNOB=hello
#VOICELAYER_COMMENTED_KNOB=world
#  VOICELAYER_PADDED_KNOB=42

# VOICELAYER_PROSE_ONLY is documented but not declared as KEY=
";
        let keys = extract_env_example_voicelayer_keys(env);
        assert_eq!(
            keys,
            [
                "VOICELAYER_COMMENTED_KNOB",
                "VOICELAYER_LIVE_KNOB",
                "VOICELAYER_PADDED_KNOB",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            "prose mentions without `=` must not be captured",
        );
    }

    /// Every backtick-quoted `VOICELAYER_<NAME>` mention in any
    /// `docs/**/*.md` file must either:
    ///
    /// - Appear as a `KEY=` (or `#KEY=`) line in
    ///   `systemd/voicelayerd.env.example`, OR
    /// - Sit on the explicit allow-list for vars that legitimately
    ///   live outside the daemon's env file (vl-desktop client-side
    ///   knobs and install-script-only knobs).
    ///
    /// The drift mode is silent and operator-facing: a doc claims
    /// `VOICELAYER_FANCY_NEW_KNOB` is configurable, but the env
    /// example never declares it, so an operator copying the env
    /// file as a starting template never sees the knob and has no
    /// way to know they should add it themselves.
    ///
    /// Companion to the python-side
    /// `tests/python/test_env_example.py::test_every_env_example_key_is_referenced_by_python_or_rust_daemon`,
    /// which closes the env-example ↔ code loop. Together the two
    /// tests pin docs ↔ env example ↔ code in a single chain.
    #[test]
    fn every_doc_voicelayer_env_var_mention_is_declared_in_env_example_or_allowlisted() {
        const ALLOWLIST: &[&str] = &[
            // vl-desktop client-side knobs — read by the GUI
            // process, not the daemon. Documented in
            // docs/guides/desktop.md.
            "VOICELAYER_LOG",
            "VOICELAYER_VL_BIN",
            // scripts/install.sh internal overrides — read by the
            // installer, not the daemon. Documented in
            // docs/guides/systemd.md.
            "VOICELAYER_INSTALL_BIN_DIR",
            "VOICELAYER_INSTALL_ENV_DIR",
            "VOICELAYER_INSTALL_UNIT_DIR",
        ];

        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let env_example =
            std::fs::read_to_string(repo_root.join("systemd/voicelayerd.env.example"))
                .expect("read systemd/voicelayerd.env.example");
        let env_keys = extract_env_example_voicelayer_keys(&env_example);
        let allowed: std::collections::BTreeSet<String> = env_keys
            .iter()
            .cloned()
            .chain(ALLOWLIST.iter().map(|s| (*s).to_owned()))
            .collect();
        assert!(
            !env_keys.is_empty(),
            "expected at least one VOICELAYER_KEY= entry in env example",
        );

        let mut docs = Vec::new();
        voicelayer_doc_test_utils::collect_markdown_files(&repo_root.join("docs"), &mut docs)
            .expect("walk docs/ directory");

        let mut violations: Vec<String> = Vec::new();
        for doc_path in &docs {
            let contents = match std::fs::read_to_string(doc_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for mention in extract_doc_voicelayer_env_var_mentions(&contents) {
                if allowed.contains(&mention) {
                    continue;
                }
                violations.push(format!(
                    "{}: `{mention}`",
                    doc_path
                        .strip_prefix(&repo_root)
                        .unwrap_or(doc_path)
                        .display(),
                ));
            }
        }
        assert!(
            violations.is_empty(),
            "docs reference VOICELAYER env vars not declared in \
             systemd/voicelayerd.env.example and not on the allowlist:\n  - {}\n\n\
             Either add the var to the env example as a `KEY=` or `#KEY=` \
             line so operators can override it, drop the doc mention, or \
             extend the in-test ALLOWLIST if the var legitimately lives \
             outside the daemon's env file (vl-desktop client-side or \
             install-script-only).",
            violations.join("\n  - "),
        );
    }

    #[test]
    fn extract_claude_md_domain_vocabulary_types_terminates_at_next_section() {
        let md = "\
# VoiceLayer

## Domain Vocabulary

Public types use the domain terms `CaptureSession` and `InjectionPlan`.

## Architecture Defaults

The `axum::Router` here is not a domain type and must not leak in.
";
        let types = extract_claude_md_domain_vocabulary_types(md);
        assert_eq!(
            types,
            ["CaptureSession", "InjectionPlan"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "the post-section `axum::Router` must not be captured",
        );
    }

    #[test]
    fn extract_voicelayer_core_pub_type_names_collects_struct_and_enum_declarations() {
        let source = "\
pub struct CaptureSession {}
pub enum DictationFailureKind { Foo, Bar }
struct PrivateThing;
pub(crate) struct CrateLocal;
pub struct GenericThing<T> { _phantom: std::marker::PhantomData<T> }
";
        let names = extract_voicelayer_core_pub_type_names(source);
        assert_eq!(
            names,
            ["CaptureSession", "DictationFailureKind", "GenericThing"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "private and pub(crate) types must be excluded; generic params \
             must be stripped from `GenericThing<T>`",
        );
    }

    /// Every PascalCase type the `## Domain Vocabulary` section of
    /// `CLAUDE.md` claims is a public type must actually appear as
    /// a `pub struct` or `pub enum` declaration in some file under
    /// `crates/voicelayer-core/src/`. The drift mode is silent doc
    /// rot: a refactor renames `CaptureSession` to
    /// `RecordingSession`, `CLAUDE.md` lags behind, and a future
    /// agent reading the contributor guide reaches for the wrong
    /// name when generating new code.
    ///
    /// Reverse direction (every `pub struct` is named in the doc)
    /// is intentionally not enforced — the doc lists the *core*
    /// vocabulary, not every internal helper type.
    #[test]
    fn every_claude_md_domain_vocabulary_type_exists_as_a_voicelayer_core_pub_type() {
        let repo_root = std::path::PathBuf::from(format!("{}/../..", env!("CARGO_MANIFEST_DIR")));

        let claude_md =
            std::fs::read_to_string(repo_root.join("CLAUDE.md")).expect("read CLAUDE.md");
        let vocab = extract_claude_md_domain_vocabulary_types(&claude_md);
        assert!(
            !vocab.is_empty(),
            "expected at least one PascalCase type in CLAUDE.md `## Domain Vocabulary` — \
             the section may have moved or the parser may be misparsing",
        );

        let core_src = repo_root.join("crates/voicelayer-core/src");
        let mut pub_types = std::collections::BTreeSet::new();
        for entry in std::fs::read_dir(&core_src).expect("read voicelayer-core/src") {
            let entry = entry.expect("read_dir entry");
            let path = entry.path();
            if path.extension().and_then(|s| s.to_str()) != Some("rs") {
                continue;
            }
            let contents = std::fs::read_to_string(&path)
                .unwrap_or_else(|err| panic!("read {}: {err}", path.display()));
            for name in extract_voicelayer_core_pub_type_names(&contents) {
                pub_types.insert(name);
            }
        }
        assert!(
            !pub_types.is_empty(),
            "expected at least one `pub struct`/`pub enum` declaration in \
             voicelayer-core — extract_voicelayer_core_pub_type_names may be \
             misparsing",
        );

        let missing: Vec<&String> = vocab.difference(&pub_types).collect();
        assert!(
            missing.is_empty(),
            "CLAUDE.md `## Domain Vocabulary` claims these types are public \
             but voicelayer-core does not declare them: {missing:?}\n\n\
             Either rename the doc reference to match the current code, \
             promote the type to `pub`, or drop the mention.",
        );
    }
}
