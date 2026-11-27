//! Unit tests for desktop application state transitions and policies.

use std::path::PathBuf;
use std::sync::Arc;

use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, DaemonEvent, EventEnvelope, LanguageProfile, ProviderDescriptor, ProviderKind,
    SessionMode, TriggerKind,
};

use super::{
    App, Message, QuitPlan, capture_is_active, dictation_translate_supported, quit_plan,
    reconcile_selected_asr_provider, validated_dictation_translate,
};
use crate::daemon_client::Client;
use crate::forms::{ComposeForm, RewriteForm, TranslateForm};
use crate::state::{
    DaemonStatus, HotkeyStatus, Preferences, SegChoice, Session, SessionStage, SystemA11y,
    WorkflowTab,
};

fn test_app() -> App {
    let preferences = Preferences::default();
    App {
        main_window: None,
        hud_window: None,
        client: Client::new(PathBuf::from("/tmp/old.sock")),
        socket_input: "/tmp/old.sock".to_owned(),
        daemon_epoch: 0,
        daemon: DaemonStatus::Healthy,
        tab: WorkflowTab::default(),
        session: Session::default(),
        providers: None,
        providers_error: None,
        health: None,
        hotkey: HotkeyStatus::default(),
        last_event: None,
        error: None,
        elapsed: 0.0,
        last_frame: None,
        dictation_provider: None,
        dictation_language: String::new(),
        dictation_segmentation: SegChoice::default(),
        dictation_translate: false,
        capture_in_flight: false,
        compose: ComposeForm::new(&preferences),
        rewrite: RewriteForm::new(&preferences),
        translate: TranslateForm::new(&preferences),
        sessions: None,
        sessions_error: None,
        quitting: false,
        preferences,
        system_a11y: SystemA11y::default(),
    }
}

fn provider(id: &str, kind: ProviderKind) -> ProviderDescriptor {
    ProviderDescriptor {
        id: id.to_owned(),
        kind,
        transport: "test".to_owned(),
        local: true,
        default_enabled: true,
        experimental: false,
        license: "Apache-2.0".to_owned(),
    }
}

#[test]
fn provider_refresh_keeps_only_dispatchable_selected_asr() {
    let providers = vec![
        provider("whisper_cpp", ProviderKind::Asr),
        provider("voxtral_realtime", ProviderKind::Asr),
        provider("writer", ProviderKind::Llm),
    ];

    let mut selected = Some("whisper_cpp".to_owned());
    reconcile_selected_asr_provider(&mut selected, Some(&providers));
    assert_eq!(selected.as_deref(), Some("whisper_cpp"));

    for bad in ["voxtral_realtime", "missing", "writer"] {
        let mut candidate = Some(bad.to_owned());
        reconcile_selected_asr_provider(&mut candidate, Some(&providers));
        assert_eq!(candidate, None);
    }
}

#[test]
fn provider_failures_are_scoped_and_cleared_on_refresh() {
    let mut app = test_app();
    app.providers = Some(vec![provider("whisper_cpp", ProviderKind::Asr)]);
    app.dictation_provider = Some("whisper_cpp".to_owned());
    app.error = Some("unrelated daemon error".to_owned());

    drop(app.update(Message::ProvidersListed(
        app.daemon_epoch,
        Err(Arc::new("provider refresh failed".to_owned())),
    )));
    assert!(app.providers.is_none());
    assert!(app.dictation_provider.is_none());
    assert_eq!(
        app.providers_error.as_deref(),
        Some("provider refresh failed")
    );
    assert_eq!(app.error.as_deref(), Some("unrelated daemon error"));

    drop(app.update(Message::RefreshProvidersPressed));
    assert!(app.providers_error.is_none());
}

#[test]
fn history_failures_preserve_cached_sessions_and_are_scoped() {
    let mut app = test_app();
    app.sessions = Some(vec![CaptureSession::new(
        SessionMode::Dictation,
        TriggerKind::TrayButton,
        LanguageProfile::default(),
    )]);

    drop(app.update(Message::SessionsListed(
        app.daemon_epoch,
        Err(Arc::new("history refresh failed".to_owned())),
    )));
    assert_eq!(app.sessions.as_ref().map(Vec::len), Some(1));
    assert_eq!(
        app.sessions_error.as_deref(),
        Some("history refresh failed")
    );

    drop(app.update(Message::SessionsListed(app.daemon_epoch, Ok(Vec::new()))));
    assert!(app.sessions_error.is_none());
    assert_eq!(app.sessions.as_ref().map(Vec::len), Some(0));
}

#[test]
fn translation_toggle_is_restricted_to_whisper_cpp() {
    assert!(dictation_translate_supported(None));
    assert!(dictation_translate_supported(Some("whisper_cpp")));
    assert!(!dictation_translate_supported(Some("mimo_v2_5_asr")));
    assert!(!dictation_translate_supported(Some("qwen3_asr_1_7b")));
    assert!(validated_dictation_translate(None, true));
    assert!(!validated_dictation_translate(Some("mimo_v2_5_asr"), true));

    let mut app = test_app();
    app.dictation_translate = true;
    drop(app.update(Message::DictationProviderSelected(Some(
        "mimo_v2_5_asr".to_owned(),
    ))));
    assert!(!app.dictation_translate);
    drop(app.update(Message::DictationTranslateToggled));
    assert!(!app.dictation_translate);
}

#[test]
fn socket_retarget_clears_daemon_state_and_drops_stale_replies() {
    let mut app = test_app();
    app.providers = Some(vec![provider("whisper_cpp", ProviderKind::Asr)]);
    drop(app.update(Message::SocketPathEdited("/tmp/new.sock".to_owned())));
    assert_eq!(app.daemon_epoch, 1);
    assert_eq!(app.daemon, DaemonStatus::Unknown);
    assert!(app.providers.is_none());

    drop(app.update(Message::EventReceived(
        0,
        EventEnvelope::new(DaemonEvent::DictationFailed {
            session_id: Uuid::nil(),
            detail: "stale".to_owned(),
        }),
    )));
    assert!(app.last_event.is_none());
}

#[test]
fn socket_retarget_is_rejected_while_capture_is_active() {
    let mut app = test_app();
    app.capture_in_flight = true;
    drop(app.update(Message::SocketPathEdited("/tmp/new.sock".to_owned())));
    assert_eq!(app.daemon_epoch, 0);
    assert!(
        app.error
            .as_deref()
            .is_some_and(|e| e.contains("socket path"))
    );
}

#[test]
fn quit_plan_and_capture_active_cover_all_stages() {
    assert!(capture_is_active(true, SessionStage::Idle));
    assert!(capture_is_active(false, SessionStage::Listening));
    assert!(!capture_is_active(false, SessionStage::Completed));

    let id = Uuid::new_v4();
    assert_eq!(
        quit_plan(false, SessionStage::Listening, Some(id)),
        QuitPlan::StopThenExit(id)
    );
    assert_eq!(
        quit_plan(false, SessionStage::Starting, None),
        QuitPlan::AwaitPendingReply
    );
    assert_eq!(
        quit_plan(true, SessionStage::Listening, Some(id)),
        QuitPlan::Exit
    );

    let mut app = test_app();
    app.session.mark_listening(id);
    drop(app.request_quit());
    assert!(app.quitting);
    assert_eq!(app.session.stage, SessionStage::Stopping);
}
