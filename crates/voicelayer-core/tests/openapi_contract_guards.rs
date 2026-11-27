//! Contract guards pinning `voicelayer-core` public types against `openapi/voicelayerd.v1.yaml`.

use std::collections::BTreeSet;

use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionArchetype, CompositionReceipt,
    DictationCaptureRequest, DictationCaptureResult, HealthResponse, InjectRequest, InjectTarget,
    InjectionPlan, LanguageProfile, LanguageStrategy, PreviewArtifact, PreviewStatus,
    ProviderDescriptor, ProviderKind, RewriteRequest, RewriteStyle, SegmentationMode, SessionMode,
    SessionState, StartDictationRequest, StopDictationRequest, TranscribeRequest,
    TranscriptionResult, TranslateRequest, TriggerKind, WorkerHealthSummary,
};

fn openapi_contents() -> String {
    let path = format!(
        "{}/../../openapi/voicelayerd.v1.yaml",
        env!("CARGO_MANIFEST_DIR")
    );
    std::fs::read_to_string(&path).expect("read openapi/voicelayerd.v1.yaml")
}

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

fn parse_required_set(section: &str) -> BTreeSet<String> {
    let mut block = BTreeSet::new();
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

    let mut inline = BTreeSet::new();
    for line in section.lines() {
        if let Some(rest) = line.strip_prefix("      required: [")
            && let Some(end) = rest.find(']')
        {
            for entry in rest[..end].split(',') {
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

fn parse_oneof_discriminator_values(section: &str) -> Vec<String> {
    let mut values = Vec::new();
    for line in section.lines() {
        if let Some(rest) = line.strip_prefix("              enum: [")
            && let Some(end) = rest.find(']')
        {
            for entry in rest[..end].split(',') {
                let name = entry.trim();
                if !name.is_empty() {
                    values.push(name.to_owned());
                }
            }
        }
    }
    values
}

fn parse_property_string_enum(section: &str, property_name: &str) -> Vec<String> {
    let heading = format!("        {property_name}:");
    let mut in_property = false;
    let mut in_block = false;
    let mut block = Vec::new();

    for line in section.lines() {
        if !in_property {
            if line == heading {
                in_property = true;
            }
            continue;
        }
        if line.starts_with("        ") && !line.starts_with("         ") && !line.trim().is_empty()
        {
            break;
        }
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

fn check_required_consistency(
    schema_name: &str,
    object: &serde_json::Map<String, serde_json::Value>,
    required_set: &BTreeSet<String>,
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
                    "openapi {schema_name} required list is missing non-Option field `{field}`"
                ));
            }
            (true, true) => {
                return Err(format!(
                    "openapi {schema_name} required list contains Option field `{field}`"
                ));
            }
            _ => {}
        }
    }
    Ok(())
}

fn assert_schema_fields<T: serde::Serialize>(
    schema_name: &str,
    sentinel: &T,
    serde_default_fields: &[&str],
) {
    let contents = openapi_contents();
    let section = isolate_schema_section(&contents, schema_name);
    assert!(
        !section.is_empty(),
        "schema component `{schema_name}` not found in openapi/voicelayerd.v1.yaml"
    );

    let value = serde_json::to_value(sentinel)
        .unwrap_or_else(|err| panic!("{schema_name} sentinel must serialize: {err}"));
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{schema_name} sentinel must serialize to a JSON object"));

    for field in object.keys() {
        let needle = format!("        {field}:");
        assert!(
            section.contains(&needle),
            "openapi {schema_name} schema is missing property `{field}`"
        );
    }

    let required_set = parse_required_set(&section);
    if let Err(detail) =
        check_required_consistency(schema_name, object, &required_set, serde_default_fields)
    {
        panic!("{detail}");
    }
}

fn assert_oneof_discriminators<T: serde::Serialize>(
    schema_name: &str,
    discriminator: &str,
    rust_variants: &[T],
) {
    let contents = openapi_contents();
    let section = isolate_schema_section(&contents, schema_name);
    assert!(!section.is_empty(), "schema `{schema_name}` not found");

    let openapi_values = parse_oneof_discriminator_values(&section);
    assert!(
        !openapi_values.is_empty(),
        "openapi {schema_name} has no oneOf discriminator values"
    );

    let rust_values: Vec<String> = rust_variants
        .iter()
        .map(|variant| {
            let value = serde_json::to_value(variant).expect("serialize variant");
            value
                .as_object()
                .and_then(|obj| obj.get(discriminator))
                .and_then(serde_json::Value::as_str)
                .unwrap_or_else(|| panic!("{schema_name} missing string `{discriminator}`"))
                .to_owned()
        })
        .collect();

    for serialized in &rust_values {
        assert!(
            openapi_values.iter().any(|v| v == serialized),
            "openapi {schema_name} is missing `{discriminator}: {serialized}`"
        );
    }
    for openapi_value in &openapi_values {
        assert!(
            rust_values.iter().any(|v| v == openapi_value),
            "openapi {schema_name} declares unknown `{discriminator}: {openapi_value}`"
        );
    }
}

fn assert_property_enum<T: serde::Serialize>(
    schema_name: &str,
    property_name: &str,
    rust_variants: &[T],
) {
    let contents = openapi_contents();
    let section = isolate_schema_section(&contents, schema_name);
    assert!(!section.is_empty(), "schema `{schema_name}` not found");

    let openapi_values: Vec<String> = parse_property_string_enum(&section, property_name)
        .into_iter()
        .filter(|value| value != "null")
        .collect();
    assert!(
        !openapi_values.is_empty(),
        "openapi {schema_name}.{property_name} has no enum values"
    );

    let rust_values: Vec<String> = rust_variants
        .iter()
        .map(|variant| {
            serde_json::to_value(variant)
                .expect("serialize variant")
                .as_str()
                .expect("variant must serialize to string")
                .to_owned()
        })
        .collect();

    for serialized in &rust_values {
        assert!(
            openapi_values.iter().any(|v| v == serialized),
            "openapi {schema_name}.{property_name} missing `{serialized}`"
        );
    }
    for openapi_value in &openapi_values {
        assert!(
            rust_values.iter().any(|v| v == openapi_value),
            "openapi {schema_name}.{property_name} has unknown variant `{openapi_value}`"
        );
    }
}

fn collect_schema_refs(contents: &str) -> BTreeSet<String> {
    let prefix = "#/components/schemas/";
    let mut refs = BTreeSet::new();
    for line in contents.lines() {
        let Some(rest) = line.trim_start().strip_prefix("$ref:") else {
            continue;
        };
        let unquoted = rest.trim().trim_matches(['"', '\'']);
        if let Some(name) = unquoted.strip_prefix(prefix) {
            refs.insert(name.to_owned());
        }
    }
    refs
}

fn collect_defined_schema_names(contents: &str) -> BTreeSet<String> {
    let mut names = BTreeSet::new();
    for line in contents.lines() {
        let Some(stripped) = line.strip_prefix("    ") else {
            continue;
        };
        if stripped.starts_with(' ') {
            continue;
        }
        let Some(name) = stripped.strip_suffix(':') else {
            continue;
        };
        if name.chars().next().is_some_and(|c| c.is_ascii_uppercase()) && !name.contains(' ') {
            names.insert(name.to_owned());
        }
    }
    names
}

fn worker_health_sentinel() -> WorkerHealthSummary {
    WorkerHealthSummary {
        status: String::new(),
        command: String::new(),
        asr_configured: false,
        asr_binary: None,
        asr_model_path: None,
        asr_error: None,
        whisper_mode: None,
        whisper_server_url: None,
        mimo_configured: false,
        mimo_model_path: None,
        mimo_error: None,
        qwen3_asr_configured: false,
        qwen3_asr_model_path: None,
        qwen3_asr_error: None,
        llm_configured: false,
        llm_model: None,
        llm_endpoint: None,
        llm_reachable: false,
        llm_error: None,
        global_hotkeys_available: false,
        global_hotkeys_backend: None,
        global_hotkeys_detail: None,
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

fn transcription_sentinel() -> TranscriptionResult {
    TranscriptionResult {
        text: String::new(),
        detected_language: None,
        notes: Vec::new(),
    }
}

fn all_triggers() -> Vec<TriggerKind> {
    let _exhaustive: fn(TriggerKind) -> &'static str = |v| match v {
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

fn all_inject_targets() -> Vec<InjectTarget> {
    let _exhaustive: fn(InjectTarget) -> &'static str = |v| match v {
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

#[test]
fn openapi_response_schemas_match_rust_structs() {
    assert_schema_fields("WorkerHealthSummary", &worker_health_sentinel(), &[]);
    assert_schema_fields(
        "HealthResponse",
        &HealthResponse {
            status: String::new(),
            socket_path: String::new(),
            version: String::new(),
            worker: worker_health_sentinel(),
        },
        &[],
    );
    assert_schema_fields("CaptureSession", &capture_session_sentinel(), &[]);
    assert_schema_fields("PreviewArtifact", &preview_artifact_sentinel(), &[]);
    assert_schema_fields(
        "CompositionReceipt",
        &CompositionReceipt {
            job_id: Uuid::nil(),
            preview: preview_artifact_sentinel(),
        },
        &[],
    );
    assert_schema_fields(
        "InjectionPlan",
        &InjectionPlan {
            target: InjectTarget::GuiAccessible,
            payload: String::new(),
            auto_submit: false,
        },
        &[],
    );
    assert_schema_fields("TranscriptionResult", &transcription_sentinel(), &[]);
    assert_schema_fields(
        "DictationCaptureResult",
        &DictationCaptureResult {
            session: capture_session_sentinel(),
            transcription: transcription_sentinel(),
            audio_file: None,
        },
        &[],
    );
    assert_schema_fields(
        "ProviderDescriptor",
        &ProviderDescriptor {
            id: String::new(),
            kind: ProviderKind::Asr,
            transport: String::new(),
            local: false,
            default_enabled: false,
            experimental: false,
            license: String::new(),
        },
        &[],
    );
}

#[test]
fn openapi_request_schemas_match_rust_structs() {
    assert_schema_fields(
        "StartDictationRequest",
        &StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::OneShot,
            provider_id: None,
        },
        &[
            "translate_to_english",
            "keep_audio",
            "segmentation",
            "provider_id",
        ],
    );
    assert_schema_fields(
        "StopDictationRequest",
        &StopDictationRequest {
            session_id: Uuid::nil(),
        },
        &[],
    );
    assert_schema_fields(
        "DictationCaptureRequest",
        &DictationCaptureRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            duration_seconds: 0,
            translate_to_english: false,
            keep_audio: false,
            provider_id: None,
        },
        &["translate_to_english", "keep_audio", "provider_id"],
    );
    assert_schema_fields(
        "ComposeRequest",
        &ComposeRequest {
            spoken_prompt: String::new(),
            archetype: None,
            output_language: None,
        },
        &[],
    );
    assert_schema_fields(
        "RewriteRequest",
        &RewriteRequest {
            source_text: String::new(),
            style: RewriteStyle::MoreFormal,
            output_language: None,
        },
        &[],
    );
    assert_schema_fields(
        "TranslateRequest",
        &TranslateRequest {
            source_text: String::new(),
            target_language: String::new(),
        },
        &[],
    );
    assert_schema_fields(
        "TranscribeRequest",
        &TranscribeRequest {
            audio_file: String::new(),
            language: None,
            translate_to_english: false,
            provider_id: None,
        },
        &["translate_to_english", "provider_id"],
    );
    assert_schema_fields(
        "InjectRequest",
        &InjectRequest {
            target: InjectTarget::GuiAccessible,
            text: String::new(),
            auto_submit: false,
        },
        &["auto_submit"],
    );
}

#[test]
fn openapi_enums_and_discriminators_match_rust_variants() {
    let seg_exhaustive: fn(SegmentationMode) -> &'static str = |v| match v {
        SegmentationMode::OneShot => "one_shot",
        SegmentationMode::Fixed { .. } => "fixed",
        SegmentationMode::VadGated { .. } => "vad_gated",
    };
    let _ = seg_exhaustive;
    assert_oneof_discriminators(
        "SegmentationMode",
        "mode",
        &[
            SegmentationMode::OneShot,
            SegmentationMode::Fixed { segment_secs: 0 },
            SegmentationMode::VadGated {
                probe_secs: 0,
                max_segment_secs: 0,
                silence_gap_probes: 0,
            },
        ],
    );

    let triggers = all_triggers();
    assert_property_enum("StartDictationRequest", "trigger", &triggers);
    assert_property_enum("DictationCaptureRequest", "trigger", &triggers);
    assert_property_enum("CaptureSession", "trigger", &triggers);

    let targets = all_inject_targets();
    assert_property_enum("InjectRequest", "target", &targets);
    assert_property_enum("InjectionPlan", "target", &targets);

    assert_property_enum(
        "RewriteRequest",
        "style",
        &[
            RewriteStyle::MoreFormal,
            RewriteStyle::Shorter,
            RewriteStyle::Politer,
            RewriteStyle::MoreTechnical,
            RewriteStyle::Translate,
        ],
    );
    assert_property_enum(
        "CaptureSession",
        "mode",
        &[
            SessionMode::Dictation,
            SessionMode::Compose,
            SessionMode::Rewrite,
            SessionMode::Translate,
        ],
    );
    assert_property_enum(
        "CaptureSession",
        "state",
        &[
            SessionState::Idle,
            SessionState::Listening,
            SessionState::Transcribing,
            SessionState::Previewing,
            SessionState::AwaitingConfirmation,
            SessionState::Completed,
            SessionState::Failed,
        ],
    );
    assert_property_enum(
        "LanguageProfile",
        "strategy",
        &[
            LanguageStrategy::AutoDetect,
            LanguageStrategy::Locked,
            LanguageStrategy::FollowPrevious,
            LanguageStrategy::Bilingual,
        ],
    );
    assert_property_enum(
        "PreviewArtifact",
        "status",
        &[
            PreviewStatus::Ready,
            PreviewStatus::NeedsProvider,
            PreviewStatus::Rejected,
        ],
    );
    assert_property_enum(
        "ComposeRequest",
        "archetype",
        &[
            CompositionArchetype::Email,
            CompositionArchetype::CoverLetter,
            CompositionArchetype::DailyReport,
            CompositionArchetype::Issue,
            CompositionArchetype::PullRequestDescription,
            CompositionArchetype::Prompt,
            CompositionArchetype::TechnicalSummary,
            CompositionArchetype::Custom,
        ],
    );
    assert_property_enum(
        "ProviderDescriptor",
        "kind",
        &[
            ProviderKind::Asr,
            ProviderKind::Llm,
            ProviderKind::Tts,
            ProviderKind::HostAdapter,
        ],
    );
}

#[test]
fn openapi_every_schema_ref_resolves_to_a_defined_component() {
    let contents = openapi_contents();
    let defined = collect_defined_schema_names(&contents);
    let refs = collect_schema_refs(&contents);
    assert!(!refs.is_empty(), "expected at least one schema $ref");

    for reference in &refs {
        assert!(
            defined.contains(reference),
            "unresolved `$ref: #/components/schemas/{reference}`; defined: {defined:?}"
        );
    }
}

#[test]
fn capture_session_new_initializes_listening_uuid_v4_and_monotonic_clock() {
    let first = CaptureSession::new(
        SessionMode::Dictation,
        TriggerKind::Cli,
        LanguageProfile::default(),
    );
    let second = CaptureSession::new(
        SessionMode::Dictation,
        TriggerKind::Cli,
        LanguageProfile::default(),
    );
    assert_eq!(first.state, SessionState::Listening);
    assert_ne!(first.session_id, Uuid::nil());
    assert_eq!(first.session_id.get_version_num(), 4);
    assert!(first.created_at_millis > 0);
    assert!(second.created_at_millis >= first.created_at_millis);

    let encoded = serde_json::to_string(&first).expect("serialize");
    let decoded: CaptureSession = serde_json::from_str(&encoded).expect("deserialize");
    assert_eq!(decoded, first);
}
