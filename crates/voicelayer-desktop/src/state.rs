//! Pure UI state types and dictation session transitions for the desktop shell.

use std::sync::Arc;

use uuid::Uuid;
use voicelayer_core::{
    DictationCaptureResult, InjectTarget, LanguageProfile, LanguageStrategy, SegmentationMode,
};
use voicelayer_ui::accessibility::Accessibility;

/// Cheaply cloneable error string carried inside iced messages.
pub type SharedError = Arc<String>;

/// Last observed liveness of the local daemon.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DaemonStatus {
    #[default]
    Unknown,
    Probing,
    Healthy,
    Unreachable,
}

/// Dictation capture lifecycle state.
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

/// Top-level navigation destinations shown in the sidebar.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum WorkflowTab {
    #[default]
    Dictation,
    Compose,
    Rewrite,
    Translate,
    Providers,
    Doctor,
    History,
    Settings,
}

impl WorkflowTab {
    pub const ALL: [WorkflowTab; 8] = [
        WorkflowTab::Dictation,
        WorkflowTab::Compose,
        WorkflowTab::Rewrite,
        WorkflowTab::Translate,
        WorkflowTab::Providers,
        WorkflowTab::Doctor,
        WorkflowTab::History,
        WorkflowTab::Settings,
    ];

    pub fn label(self) -> &'static str {
        match self {
            WorkflowTab::Dictation => "Dictation",
            WorkflowTab::Compose => "Compose",
            WorkflowTab::Rewrite => "Rewrite",
            WorkflowTab::Translate => "Translate",
            WorkflowTab::Providers => "Providers",
            WorkflowTab::Doctor => "Doctor",
            WorkflowTab::History => "History",
            WorkflowTab::Settings => "Settings",
        }
    }
}

/// Global hotkey availability probe result.
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

/// Active and most recent dictation session state.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub stage: SessionStage,
    pub id: Option<Uuid>,
    pub transcript: Option<String>,
    pub detected_language: Option<String>,
    pub notes: Vec<String>,
}

impl Session {
    pub fn begin_starting(&mut self) {
        self.stage = SessionStage::Starting;
    }

    pub fn begin_stopping(&mut self) {
        self.stage = SessionStage::Stopping;
    }

    pub fn mark_listening(&mut self, id: Uuid) {
        self.stage = SessionStage::Listening;
        self.id = Some(id);
    }

    pub fn mark_failed(&mut self) {
        self.stage = SessionStage::Failed;
        self.id = None;
    }

    pub fn apply_capture(&mut self, result: DictationCaptureResult) -> Option<String> {
        self.id = None;
        self.transcript = Some(result.transcription.text);
        self.detected_language = result.transcription.detected_language;
        self.notes = result.transcription.notes;
        self.stage = SessionStage::Completed;
        None
    }
}

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
        SessionStage::Listening => "listening, speak now",
        SessionStage::Stopping => "stopping...",
        SessionStage::Completed => "completed",
        SessionStage::Failed => "failed",
    }
}

/// Segmentation strategy selection for streaming dictation.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SegChoice {
    #[default]
    OneShot,
    Fixed,
    VadGated,
}

impl SegChoice {
    pub const ALL: [SegChoice; 3] = [SegChoice::OneShot, SegChoice::Fixed, SegChoice::VadGated];

    pub fn label(self) -> &'static str {
        match self {
            SegChoice::OneShot => "One-shot",
            SegChoice::Fixed => "Fixed window",
            SegChoice::VadGated => "VAD-gated",
        }
    }

    pub fn to_mode(self) -> SegmentationMode {
        match self {
            SegChoice::OneShot => SegmentationMode::OneShot,
            SegChoice::Fixed => SegmentationMode::Fixed { segment_secs: 15 },
            SegChoice::VadGated => SegmentationMode::VadGated {
                probe_secs: 1,
                max_segment_secs: 30,
                silence_gap_probes: 3,
            },
        }
    }
}

impl std::fmt::Display for SegChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.label())
    }
}

/// Parse a free-text language box into a [`LanguageProfile`] (`None` = auto-detect).
pub fn language_profile_from_input(input: &str) -> Option<LanguageProfile> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return None;
    }
    Some(LanguageProfile {
        strategy: LanguageStrategy::Locked,
        input_languages: vec![trimmed.to_owned()],
        output_language: None,
    })
}

/// Local desktop preferences edited in the Settings panel.
///
/// `glass_opacity` and `reduce_transparency` persist to `desktop.toml`
/// (see [`crate::config`]); workflow defaults are session-scoped.
#[derive(Debug, Clone, PartialEq)]
pub struct Preferences {
    pub default_output_language: String,
    pub default_inject_target: InjectTarget,
    pub capture_seconds: u32,
    pub glass_opacity: f32,
    pub reduce_transparency: bool,
}

pub const DEFAULT_GLASS_OPACITY: f32 = 0.5;

impl Default for Preferences {
    fn default() -> Self {
        Self {
            default_output_language: String::new(),
            default_inject_target: InjectTarget::GuiAccessible,
            capture_seconds: 8,
            glass_opacity: DEFAULT_GLASS_OPACITY,
            reduce_transparency: false,
        }
    }
}

/// Live OS accessibility preferences mirrored from the desktop environment.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SystemA11y {
    pub increase_contrast: bool,
    pub reduce_motion: bool,
}

/// Combine persisted user preferences and live OS flags into [`Accessibility`].
pub fn resolve_accessibility(prefs: &Preferences, system: SystemA11y) -> Accessibility {
    Accessibility {
        reduce_transparency: prefs.reduce_transparency,
        increase_contrast: system.increase_contrast,
        reduce_motion: system.reduce_motion,
        opacity: prefs.glass_opacity,
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DaemonStatus, HotkeyStatus, Preferences, SegChoice, Session, SessionStage, SystemA11y,
        WorkflowTab, language_profile_from_input, render_daemon_status, render_session_stage,
        resolve_accessibility,
    };
    use uuid::Uuid;
    use voicelayer_core::{
        CaptureSession, DictationCaptureResult, InjectTarget, LanguageProfile, LanguageStrategy,
        SegmentationMode, SessionMode, SessionState, TranscriptionResult, TriggerKind,
    };

    #[test]
    fn render_daemon_status_pins_user_visible_labels() {
        assert_eq!(render_daemon_status(DaemonStatus::Unknown), "unknown");
        assert_eq!(
            render_daemon_status(DaemonStatus::Probing),
            "probing daemon..."
        );
        assert_eq!(
            render_daemon_status(DaemonStatus::Healthy),
            "daemon healthy"
        );
        assert_eq!(
            render_daemon_status(DaemonStatus::Unreachable),
            "daemon unreachable",
        );
    }

    #[test]
    fn render_session_stage_pins_user_visible_labels() {
        assert_eq!(render_session_stage(SessionStage::Idle), "idle");
        assert_eq!(
            render_session_stage(SessionStage::Starting),
            "starting dictation...",
        );
        assert_eq!(
            render_session_stage(SessionStage::Listening),
            "listening, speak now",
        );
        assert_eq!(render_session_stage(SessionStage::Stopping), "stopping...");
        assert_eq!(render_session_stage(SessionStage::Completed), "completed");
        assert_eq!(render_session_stage(SessionStage::Failed), "failed");
    }

    #[test]
    fn workflow_tabs_pin_order_and_labels() {
        assert_eq!(WorkflowTab::ALL.len(), 8);
        assert_eq!(WorkflowTab::default(), WorkflowTab::Dictation);
        let labels: Vec<_> = WorkflowTab::ALL.iter().map(|t| t.label()).collect();
        assert_eq!(
            labels,
            vec![
                "Dictation",
                "Compose",
                "Rewrite",
                "Translate",
                "Providers",
                "Doctor",
                "History",
                "Settings",
            ],
        );
    }

    #[test]
    fn preferences_default_is_local_and_conservative() {
        let prefs = Preferences::default();
        assert!(prefs.default_output_language.is_empty());
        assert_eq!(prefs.default_inject_target, InjectTarget::GuiAccessible);
        assert_eq!(prefs.capture_seconds, 8);
        assert_eq!(prefs.glass_opacity, 0.5);
        assert!(!prefs.reduce_transparency);
    }

    #[test]
    fn resolve_accessibility_splits_user_and_system_inputs() {
        let prefs = Preferences {
            glass_opacity: 0.8,
            reduce_transparency: true,
            ..Default::default()
        };
        let system = SystemA11y {
            increase_contrast: true,
            reduce_motion: true,
        };
        let a11y = resolve_accessibility(&prefs, system);
        assert!(a11y.reduce_transparency);
        assert!(a11y.increase_contrast);
        assert!(a11y.reduce_motion);
        assert_eq!(a11y.opacity, 0.8);
    }

    #[test]
    fn resolve_accessibility_defaults() {
        let a11y = resolve_accessibility(&Preferences::default(), SystemA11y::default());
        assert!(!a11y.reduce_transparency);
        assert!(!a11y.increase_contrast);
        assert!(!a11y.reduce_motion);
        assert_eq!(a11y.opacity, 0.5);
    }

    #[test]
    fn seg_choice_maps_to_wire_segmentation_mode() {
        assert_eq!(SegChoice::ALL.len(), 3);
        assert_eq!(SegChoice::default(), SegChoice::OneShot);
        assert_eq!(SegChoice::OneShot.to_mode(), SegmentationMode::OneShot);
        assert!(matches!(
            SegChoice::Fixed.to_mode(),
            SegmentationMode::Fixed { .. }
        ));
        assert!(matches!(
            SegChoice::VadGated.to_mode(),
            SegmentationMode::VadGated { .. }
        ));
    }

    #[test]
    fn language_profile_from_input_blank_is_auto_detect() {
        assert!(language_profile_from_input("   ").is_none());
        let locked = language_profile_from_input(" en ").expect("non-blank locks language");
        assert_eq!(locked.strategy, LanguageStrategy::Locked);
        assert_eq!(locked.input_languages, vec!["en".to_owned()]);
        assert!(locked.output_language.is_none());
    }

    #[test]
    fn hotkey_status_default_marks_portal_unprobed() {
        let status = HotkeyStatus::default();
        assert!(!status.portal_available);
        assert_eq!(status.portal_error.as_deref(), Some("not probed"));
    }

    #[test]
    fn session_default_starts_idle_with_no_capture() {
        let session = Session::default();
        assert_eq!(session.stage, SessionStage::Idle);
        assert!(session.id.is_none());
        assert!(session.transcript.is_none());
        assert!(session.detected_language.is_none());
        assert!(session.notes.is_empty());
    }

    #[test]
    fn session_mark_listening_records_id_and_stage() {
        let mut session = Session::default();
        let id = Uuid::nil();
        session.mark_listening(id);
        assert_eq!(session.stage, SessionStage::Listening);
        assert_eq!(session.id, Some(id));
    }

    fn capture_result(text: &str) -> DictationCaptureResult {
        DictationCaptureResult {
            session: CaptureSession {
                session_id: Uuid::nil(),
                mode: SessionMode::Dictation,
                state: SessionState::Completed,
                trigger: TriggerKind::TrayButton,
                language_profile: LanguageProfile::default(),
                created_at_millis: 0,
            },
            transcription: TranscriptionResult {
                text: text.to_owned(),
                detected_language: Some("en".to_owned()),
                notes: vec!["note".to_owned()],
            },
            audio_file: None,
        }
    }

    #[test]
    fn session_apply_capture_success_completes_with_transcript() {
        let mut session = Session::default();
        let message = session.apply_capture(capture_result("hello world"));
        assert!(message.is_none());
        assert_eq!(session.stage, SessionStage::Completed);
        assert_eq!(session.transcript.as_deref(), Some("hello world"));
        assert_eq!(session.detected_language.as_deref(), Some("en"));
        assert_eq!(session.notes, vec!["note".to_owned()]);
        assert!(session.id.is_none());
    }
}
