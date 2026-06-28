//! UI state vocabulary and the dictation session state machine for the
//! VoiceLayer desktop shell.
//!
//! This module owns the *pure*, unit-testable parts of the shell's state: the
//! daemon/session status enums, the workflow navigation tabs, and the
//! per-session transition logic. The live, window-aware program state lives in
//! [`crate::app`]; the wire types exchanged with the daemon come from
//! [`voicelayer_core`] via [`crate::api`].

use std::sync::Arc;

use uuid::Uuid;
use voicelayer_core::{
    DictationCaptureResult, DictationFailureKind, InjectTarget, LanguageProfile, LanguageStrategy,
    RecorderBackend, SegmentationMode,
};
use voicelayer_ui::a11y::Accessibility;

/// A shareable error message. `Arc<String>` is cheap to clone so the app can
/// bubble the same error through multiple messages without copies.
pub type SharedError = Arc<String>;

/// Liveness of the daemon as last observed by a health probe.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum DaemonStatus {
    #[default]
    Unknown,
    Probing,
    Healthy,
    Unreachable,
}

/// Where a dictation capture is in its lifecycle.
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

/// The top-level workflow destinations shown in the navigation sidebar; each
/// maps to one content panel. Preview-before-inject is intentionally *not* a
/// tab — it is an inline stage of the generative flows (compose / rewrite /
/// translate), landing with those views in P3.
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
    /// Sidebar order, top to bottom.
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

    /// The user-visible sidebar label.
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

/// Whether the portal-backed global shortcut registered, and why not if it did
/// not. Drives the hotkey hint in the dictation panel.
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

/// The dictation session sub-state. Dictation is the one workflow wired in the
/// P2 skeleton; compose / rewrite / translate grow their own state in P3.
#[derive(Debug, Default, Clone)]
pub struct Session {
    pub stage: SessionStage,
    pub id: Option<Uuid>,
    pub transcript: Option<String>,
    pub detected_language: Option<String>,
    pub notes: Vec<String>,
}

impl Session {
    /// Move to `Starting` while the start request is in flight.
    pub fn begin_starting(&mut self) {
        self.stage = SessionStage::Starting;
    }

    /// Move to `Stopping` while the stop request is in flight.
    pub fn begin_stopping(&mut self) {
        self.stage = SessionStage::Stopping;
    }

    /// The daemon accepted the session and is now listening.
    pub fn mark_listening(&mut self, id: Uuid) {
        self.stage = SessionStage::Listening;
        self.id = Some(id);
    }

    /// A start or stop request failed; clear any in-flight session id.
    pub fn mark_failed(&mut self) {
        self.stage = SessionStage::Failed;
        self.id = None;
    }

    /// Apply a completed capture: store the transcript, detected language, and
    /// notes, then settle into `Completed` — or `Failed` when the daemon flagged
    /// a failure kind, in which case the returned string is a user-facing
    /// explanation to surface.
    pub fn apply_capture(&mut self, result: DictationCaptureResult) -> Option<String> {
        self.id = None;
        self.transcript = Some(result.transcription.text);
        self.detected_language = result.transcription.detected_language;
        self.notes = result.transcription.notes;
        match result.failure_kind {
            Some(kind) => {
                self.stage = SessionStage::Failed;
                Some(format!(
                    "[{}] capture did not complete cleanly",
                    render_failure_kind(kind)
                ))
            }
            None => {
                self.stage = SessionStage::Completed;
                None
            }
        }
    }
}

/// User-visible label for a daemon status. A new `DaemonStatus` variant forces
/// this match to fail compile until wired, pinning the wording operators read.
pub fn render_daemon_status(status: DaemonStatus) -> &'static str {
    match status {
        DaemonStatus::Unknown => "unknown",
        DaemonStatus::Probing => "probing daemon...",
        DaemonStatus::Healthy => "daemon healthy",
        DaemonStatus::Unreachable => "daemon unreachable",
    }
}

/// User-visible label for a session stage; pinned for the same reason.
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

/// The wire-style label for a dictation failure kind, matching the daemon's
/// OpenAPI vocabulary so an operator sees the same token in the UI and the logs.
pub fn render_failure_kind(kind: DictationFailureKind) -> &'static str {
    match kind {
        DictationFailureKind::RecordingFailed => "recording_failed",
        DictationFailureKind::AsrFailed => "asr_failed",
        DictationFailureKind::InjectionFailed => "injection_failed",
    }
}

/// How the dictation panel labels and constructs a [`SegmentationMode`]. The
/// richer per-mode parameters (segment / overlap / probe seconds) carry
/// conservative defaults here; surfacing them for tuning is a later milestone,
/// and the daemon validates the values it receives.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub enum SegChoice {
    #[default]
    OneShot,
    Fixed,
    VadGated,
}

impl SegChoice {
    /// Picker order, top to bottom.
    pub const ALL: [SegChoice; 3] = [SegChoice::OneShot, SegChoice::Fixed, SegChoice::VadGated];

    /// The user-visible picker label.
    pub fn label(self) -> &'static str {
        match self {
            SegChoice::OneShot => "One-shot",
            SegChoice::Fixed => "Fixed window",
            SegChoice::VadGated => "VAD-gated",
        }
    }

    /// Build the wire [`SegmentationMode`]. The non-one-shot variants use
    /// conservative default windows until per-mode tuning lands.
    pub fn to_mode(self) -> SegmentationMode {
        match self {
            SegChoice::OneShot => SegmentationMode::OneShot,
            SegChoice::Fixed => SegmentationMode::Fixed {
                segment_secs: 15,
                overlap_secs: 0,
            },
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

/// Map the dictation panel's free-text language box to a [`LanguageProfile`].
/// Blank input yields `None` so the daemon keeps its auto-detect default; any
/// value locks the input language to it.
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

/// Local-only desktop preferences edited in the Settings panel. There is no
/// `/v1` configuration route, so these stay in the shell and seed each
/// workflow's defaults; daemon-side configuration remains the CLI's domain. The
/// glass-opacity slider and the manual Reduce Transparency toggle are persisted
/// to the desktop shell's own `desktop.toml` (see [`crate::config`]); the other
/// fields are session defaults.
#[derive(Debug, Clone, PartialEq)]
pub struct Preferences {
    pub default_output_language: String,
    pub default_inject_target: InjectTarget,
    pub recorder_backend: RecorderBackend,
    pub capture_seconds: u32,
    /// User glass opacity, `0.0` (clearest) .. `1.0` (frosted). Persisted.
    pub glass_opacity: f32,
    /// Manual Reduce Transparency. GNOME exposes no system signal for it (unlike
    /// macOS), so it is a user toggle rather than read from the OS. Persisted.
    pub reduce_transparency: bool,
}

/// The default glass opacity, mirroring [`voicelayer_ui::a11y::Accessibility`]'s
/// 2026 baseline so the desktop shell and the shared contract agree.
pub const DEFAULT_GLASS_OPACITY: f32 = 0.5;

impl Default for Preferences {
    fn default() -> Self {
        Self {
            default_output_language: String::new(),
            default_inject_target: InjectTarget::GuiAccessible,
            recorder_backend: RecorderBackend::Auto,
            capture_seconds: 8,
            glass_opacity: DEFAULT_GLASS_OPACITY,
            reduce_transparency: false,
        }
    }
}

/// Live OS accessibility state mirrored from the desktop environment — the part
/// VoiceLayer reads rather than persists. GNOME publishes it through the XDG
/// Settings portal (`org.freedesktop.appearance`); see [`crate::a11y`]. Reduce
/// Transparency is absent here because GNOME has no such signal.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct SystemA11y {
    pub increase_contrast: bool,
    pub reduce_motion: bool,
}

/// Combine the persisted [`Preferences`] with the live [`SystemA11y`] into the
/// final Liquid Glass [`Accessibility`] contract the renderer applies. Opacity
/// and Reduce Transparency are user-controlled; Increase Contrast and Reduce
/// Motion follow the OS.
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
        DaemonStatus, DictationFailureKind, HotkeyStatus, Preferences, SegChoice, Session,
        SessionStage, SystemA11y, WorkflowTab, language_profile_from_input, render_daemon_status,
        render_failure_kind, render_session_stage, resolve_accessibility,
    };
    use uuid::Uuid;
    use voicelayer_core::{
        CaptureSession, DictationCaptureResult, InjectTarget, LanguageProfile, LanguageStrategy,
        RecorderBackend, SegmentationMode, SessionMode, SessionState, TranscriptionResult,
        TriggerKind,
    };

    /// Adding a `DaemonStatus` variant forces the `render_*` match to fail
    /// compile until the new arm is wired; this pins the *labels* operators see.
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
            "listening — speak now",
        );
        assert_eq!(render_session_stage(SessionStage::Stopping), "stopping...");
        assert_eq!(render_session_stage(SessionStage::Completed), "completed");
        assert_eq!(render_session_stage(SessionStage::Failed), "failed");
    }

    /// The failure-kind labels are shared with the daemon's OpenAPI wire
    /// vocabulary; pin them so a rename can't silently desync UI from logs.
    #[test]
    fn render_failure_kind_matches_wire_vocabulary() {
        assert_eq!(
            render_failure_kind(DictationFailureKind::RecordingFailed),
            "recording_failed"
        );
        assert_eq!(
            render_failure_kind(DictationFailureKind::AsrFailed),
            "asr_failed"
        );
        assert_eq!(
            render_failure_kind(DictationFailureKind::InjectionFailed),
            "injection_failed"
        );
    }

    /// The sidebar renders `WorkflowTab::ALL` in order; pin the order and labels
    /// so the shell's information architecture is explicit and a reordering is a
    /// deliberate, reviewed change.
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

    /// Preferences default to local, conservative values: no forced output
    /// language, the safest GUI inject target, automatic recorder selection, and
    /// a short one-shot capture window.
    #[test]
    fn preferences_default_is_local_and_conservative() {
        let prefs = Preferences::default();
        assert!(prefs.default_output_language.is_empty());
        assert_eq!(prefs.default_inject_target, InjectTarget::GuiAccessible);
        assert_eq!(prefs.recorder_backend, RecorderBackend::Auto);
        assert_eq!(prefs.capture_seconds, 8);
        assert_eq!(prefs.glass_opacity, 0.5);
        assert!(!prefs.reduce_transparency);
    }

    /// The final accessibility contract splits its inputs: opacity and Reduce
    /// Transparency are user preferences; Increase Contrast and Reduce Motion are
    /// read from the OS. Pin that wiring so a future refactor can't cross them.
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
        assert!(a11y.reduce_transparency, "transparency is the user toggle");
        assert!(a11y.increase_contrast, "contrast follows the OS");
        assert!(a11y.reduce_motion, "motion follows the OS");
        assert_eq!(a11y.opacity, 0.8, "opacity is the user slider");
    }

    #[test]
    fn resolve_accessibility_defaults_are_calm() {
        let a11y = resolve_accessibility(&Preferences::default(), SystemA11y::default());
        assert!(!a11y.reduce_transparency);
        assert!(!a11y.increase_contrast);
        assert!(!a11y.reduce_motion);
        assert_eq!(a11y.opacity, 0.5);
    }

    /// The dictation panel's segmentation picker maps onto the wire enum; pin
    /// the variant mapping and the picker arity.
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

    /// A blank language box means auto-detect (`None`); any value locks the
    /// input language, trimming surrounding whitespace.
    #[test]
    fn language_profile_from_input_blank_is_auto_detect() {
        assert!(language_profile_from_input("   ").is_none());
        let locked = language_profile_from_input(" en ").expect("non-blank locks the language");
        assert_eq!(locked.strategy, LanguageStrategy::Locked);
        assert_eq!(locked.input_languages, vec!["en".to_owned()]);
        assert!(locked.output_language.is_none());
    }

    /// `HotkeyStatus::default()` starts unavailable with an explanatory reason
    /// so the UI never claims the hotkey works before the first probe.
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

    fn capture_result(
        failure_kind: Option<DictationFailureKind>,
        text: &str,
    ) -> DictationCaptureResult {
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
            failure_kind,
        }
    }

    #[test]
    fn session_apply_capture_success_completes_with_transcript() {
        let mut session = Session::default();
        let message = session.apply_capture(capture_result(None, "hello world"));
        assert!(message.is_none());
        assert_eq!(session.stage, SessionStage::Completed);
        assert_eq!(session.transcript.as_deref(), Some("hello world"));
        assert_eq!(session.detected_language.as_deref(), Some("en"));
        assert_eq!(session.notes, vec!["note".to_owned()]);
        assert!(session.id.is_none());
    }

    #[test]
    fn session_apply_capture_failure_marks_failed_with_message() {
        let mut session = Session::default();
        let message = session.apply_capture(capture_result(
            Some(DictationFailureKind::AsrFailed),
            "partial",
        ));
        assert_eq!(session.stage, SessionStage::Failed);
        assert_eq!(
            message.as_deref(),
            Some("[asr_failed] capture did not complete cleanly"),
        );
        // The transcript is still captured even on failure, for inspection.
        assert_eq!(session.transcript.as_deref(), Some("partial"));
    }
}
