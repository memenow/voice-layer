//! The VoiceLayer desktop program controller.
//!
//! An [`iced::daemon`] (not a single-window `application`) so the main window,
//! the capture HUD (P5), and a settings window (P3) can be sibling windows with
//! their own content, title, and chrome — daemon `view`/`title`/`theme` receive
//! the [`window::Id`] being drawn, which `application` does not expose. The
//! daemon also outlives its windows, matching VoiceLayer's "background voice
//! layer" model; closing the main window quits via [`iced::exit`].
//!
//! This module owns state, message handling, async tasks, and the event/portal
//! subscriptions. Rendering lives in [`crate::view`]; the typed `/v1` client and
//! the SSE reader live in [`crate::api`].

use std::path::PathBuf;
use std::time::{Duration, Instant};

use iced::futures::SinkExt;
use iced::futures::channel::mpsc as futures_mpsc;
use iced::widget::text_editor;
use iced::{Element, Subscription, Task, keyboard, stream, window};

use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionArchetype, CompositionReceipt,
    DictationCaptureRequest, DictationCaptureResult, EventEnvelope, HealthResponse, InjectRequest,
    InjectTarget, InjectionPlan, ProviderDescriptor, RecorderBackend, RewriteRequest, RewriteStyle,
    StartDictationRequest, TranslateRequest, TriggerKind,
};
use voicelayer_ui::a11y::Accessibility;

use crate::a11y;
use crate::api::{self, Client};
use crate::config;
use crate::forms::{
    ComposeForm, JobStage, RewriteForm, TranslateForm, apply_job_result, optional_text,
};
use crate::launcher::spawn_daemon;
use crate::portal::{self, PortalProbe, SHORTCUT_TOGGLE};
use crate::state::{
    DaemonStatus, HotkeyStatus, Preferences, SegChoice, Session, SessionStage, SharedError,
    SystemA11y, WorkflowTab, language_profile_from_input, resolve_accessibility,
};
use crate::view;

/// Window chrome for the main navigation window.
fn main_window_settings() -> window::Settings {
    window::Settings {
        size: iced::Size::new(960.0, 680.0),
        min_size: Some(iced::Size::new(720.0, 520.0)),
        position: window::Position::Centered,
        // Request a transparent surface so later edge/rounded-corner polish can
        // land. iced_wgpu picks the composite alpha mode internally
        // (PostMultiplied → PreMultiplied → Auto) — there is no app-level knob.
        // The view always paints an opaque backdrop beneath the glass shader, so
        // a compositor without real transparency simply shows that opaque tier.
        transparent: true,
        ..window::Settings::default()
    }
}

/// Window chrome for the capture HUD overlay: a small borderless, transparent,
/// best-effort always-on-top child window. On GNOME Wayland the compositor owns
/// placement (`window::move_to` is unsupported) and silently ignores the level,
/// so `Position::Centered` and `Level::AlwaysOnTop` are hints that wlroots/KWin
/// honor; the layer-shell path that would force placement is left unimplemented
/// (see [`crate::hud`]). `exit_on_close_request: false` keeps the daemon alive if
/// the HUD is closed, and it is non-resizable since the shader owns its size.
fn hud_window_settings() -> window::Settings {
    window::Settings {
        size: iced::Size::new(360.0, 132.0),
        position: window::Position::Centered,
        resizable: false,
        decorations: false,
        transparent: true,
        level: window::Level::AlwaysOnTop,
        exit_on_close_request: false,
        ..window::Settings::default()
    }
}

/// Top-level daemon state. Windows are tracked by id so the HUD and settings
/// windows can join later without restructuring the routing. Fields the view
/// reads are `pub(crate)`; the wire client stays private to this controller.
pub struct App {
    pub(crate) main_window: Option<window::Id>,
    /// The capture HUD overlay window, present only while a capture is active.
    /// [`App::sync_hud`] opens and closes it from the capture state.
    pub(crate) hud_window: Option<window::Id>,
    client: Client,
    pub(crate) socket_input: String,
    pub(crate) daemon: DaemonStatus,
    pub(crate) tab: WorkflowTab,
    pub(crate) session: Session,
    pub(crate) providers: Option<Vec<ProviderDescriptor>>,
    pub(crate) health: Option<HealthResponse>,
    pub(crate) hotkey: HotkeyStatus,
    pub(crate) last_event: Option<EventEnvelope>,
    pub(crate) error: Option<String>,
    /// Monotonic seconds clock driving the glass shader, advanced by
    /// `window::frames`. `last_frame` holds the previous frame instant so the
    /// delta accumulates regardless of frame pacing.
    pub(crate) elapsed: f32,
    last_frame: Option<Instant>,
    /// Local-only preferences edited in Settings; seed each workflow's defaults.
    pub(crate) preferences: Preferences,
    /// Live OS accessibility state mirrored from the XDG settings portal; feeds
    /// the final [`Accessibility`] contract alongside [`Self::preferences`].
    pub(crate) system_a11y: SystemA11y,
    // Dictation panel selections (the P3 enhancement over the hardcoded P2 start).
    pub(crate) dictation_provider: Option<String>,
    pub(crate) dictation_language: String,
    pub(crate) dictation_segmentation: SegChoice,
    pub(crate) dictation_translate: bool,
    pub(crate) capture_in_flight: bool,
    // Generative workflow forms.
    pub(crate) compose: ComposeForm,
    pub(crate) rewrite: RewriteForm,
    pub(crate) translate: TranslateForm,
    // History: capture sessions known to the daemon, lazily loaded.
    pub(crate) sessions: Option<Vec<CaptureSession>>,
}

#[derive(Debug, Clone)]
pub enum Message {
    MainWindowOpened(window::Id),
    HudWindowOpened(window::Id),
    WindowClosed(window::Id),
    /// Tray menu: raise the main window. Best-effort — GNOME Wayland ignores
    /// programmatic focus while wlroots/KWin honor it.
    #[cfg(feature = "tray")]
    TrayShowRequested,
    /// Tray menu: quit the desktop shell.
    #[cfg(feature = "tray")]
    TrayQuitRequested,
    TabSelected(WorkflowTab),
    SocketPathEdited(String),
    ProbeDaemonPressed,
    // The big payloads (`HealthResponse`, `DictationCaptureResult`) are boxed:
    // iced clones every `Message` on dispatch, so keeping the enum small keeps
    // those clones cheap and satisfies `clippy::large_enum_variant`.
    DaemonProbed(Result<Box<HealthResponse>, SharedError>),
    StartDaemonPressed,
    DaemonSpawnResult(Result<(), SharedError>),
    RefreshProvidersPressed,
    ProvidersListed(Result<Vec<ProviderDescriptor>, SharedError>),
    PortalProbed(PortalProbe),
    HotkeyReceived,
    StartPressed,
    StopPressed,
    SessionStarted(Result<CaptureSession, SharedError>),
    SessionStopped(Result<Box<DictationCaptureResult>, SharedError>),
    EventReceived(EventEnvelope),
    /// Frame tick from `window::frames`, carrying the frame instant; advances the
    /// glass shader's animation clock.
    Tick(Instant),

    // Dictation panel controls (P3 enhancement).
    DictationProviderSelected(Option<String>),
    DictationLanguageEdited(String),
    DictationSegmentationSelected(SegChoice),
    DictationTranslateToggled,
    CapturePressed,
    Captured(Result<Box<DictationCaptureResult>, SharedError>),

    // Compose workflow.
    ComposePromptEdited(text_editor::Action),
    ComposeArchetypeSelected(CompositionArchetype),
    ComposeLanguageEdited(String),
    ComposeSubmitPressed,
    Composed(Result<Box<CompositionReceipt>, SharedError>),

    // Rewrite workflow.
    RewriteSourceEdited(text_editor::Action),
    RewriteStyleSelected(RewriteStyle),
    RewriteLanguageEdited(String),
    RewriteSubmitPressed,
    Rewritten(Result<Box<CompositionReceipt>, SharedError>),

    // Translate workflow.
    TranslateSourceEdited(text_editor::Action),
    TranslateTargetEdited(String),
    TranslateSubmitPressed,
    Translated(Result<Box<CompositionReceipt>, SharedError>),

    // Preview → inject, shared by the three generative workflows above.
    InjectTargetSelected(WorkflowTab, InjectTarget),
    InjectAutoSubmitToggled(WorkflowTab),
    InjectPressed(WorkflowTab),
    Injected(WorkflowTab, Result<Box<InjectionPlan>, SharedError>),

    // Settings (local preferences).
    PrefOutputLanguageEdited(String),
    PrefInjectTargetSelected(InjectTarget),
    PrefRecorderSelected(RecorderBackend),
    PrefCaptureSecondsSelected(u32),
    PrefGlassOpacityChanged(f32),
    PrefReduceTransparencyToggled,
    /// Live OS accessibility state from the XDG settings portal — the initial
    /// probe and subsequent `SettingChanged` updates.
    SystemA11yChanged(SystemA11y),

    // History.
    HistoryRefreshPressed,
    SessionsListed(Result<Vec<CaptureSession>, SharedError>),
}

impl App {
    pub fn boot() -> (Self, Task<Message>) {
        let client = Client::new(api::desktop_socket_path());
        let socket_input = client.socket_path().display().to_string();
        // A daemon opens no window on its own; show the main window from boot.
        let (main_id, open_main) = window::open(main_window_settings());
        // Seed the persisted glass preferences from `desktop.toml` (best-effort;
        // defaults on a missing or unreadable file).
        let mut preferences = Preferences::default();
        config::load().apply_to(&mut preferences);
        let app = Self {
            main_window: Some(main_id),
            hud_window: None,
            client: client.clone(),
            socket_input,
            daemon: DaemonStatus::Probing,
            tab: WorkflowTab::default(),
            session: Session::default(),
            providers: None,
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
            preferences,
            system_a11y: SystemA11y::default(),
        };
        let startup = Task::batch(vec![
            open_main.map(Message::MainWindowOpened),
            probe_health(client.clone()),
            list_providers(client),
            Task::perform(portal::probe(), Message::PortalProbed),
            // Read the initial OS accessibility state; the subscription keeps it live.
            Task::perform(a11y::probe(), Message::SystemA11yChanged),
        ]);
        (app, startup)
    }

    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::MainWindowOpened(id) => {
                self.main_window = Some(id);
                Task::none()
            }
            Message::HudWindowOpened(_id) => {
                // The id was reserved synchronously in `sync_hud`, which is the
                // source of truth; the open `Task` completing carries no new state
                // (iced just requires it be mapped to a message).
                Task::none()
            }
            Message::WindowClosed(id) => {
                if self.main_window == Some(id) {
                    // The daemon does not self-terminate on last-window-close; quit
                    // deliberately when the main window goes away.
                    iced::exit()
                } else {
                    // The HUD closing — by us via `sync_hud` or by the compositor —
                    // only clears its slot; the daemon keeps running.
                    if self.hud_window == Some(id) {
                        self.hud_window = None;
                    }
                    Task::none()
                }
            }
            #[cfg(feature = "tray")]
            Message::TrayShowRequested => match self.main_window {
                // `gain_focus` is a no-op on GNOME Wayland (the compositor owns
                // focus) but is honored by wlroots/KWin; harmless either way.
                Some(id) => window::gain_focus(id),
                None => Task::none(),
            },
            #[cfg(feature = "tray")]
            Message::TrayQuitRequested => iced::exit(),
            Message::TabSelected(tab) => {
                self.tab = tab;
                // Refresh the read-only panels on entry so they show live data;
                // the dictation provider picker also needs the provider list.
                match tab {
                    WorkflowTab::Providers => list_providers(self.client.clone()),
                    WorkflowTab::Doctor => probe_health(self.client.clone()),
                    WorkflowTab::History => fetch_sessions(self.client.clone()),
                    WorkflowTab::Dictation if self.providers.is_none() => {
                        list_providers(self.client.clone())
                    }
                    _ => Task::none(),
                }
            }
            Message::SocketPathEdited(next) => {
                self.client = Client::new(PathBuf::from(next.trim()));
                self.socket_input = next;
                Task::none()
            }
            Message::ProbeDaemonPressed => {
                self.daemon = DaemonStatus::Probing;
                self.error = None;
                probe_health(self.client.clone())
            }
            Message::DaemonProbed(Ok(health)) => {
                self.daemon = DaemonStatus::Healthy;
                self.health = Some(*health);
                self.error = None;
                Task::none()
            }
            Message::DaemonProbed(Err(error)) => {
                self.daemon = DaemonStatus::Unreachable;
                self.error = Some((*error).clone());
                Task::none()
            }
            Message::StartDaemonPressed => {
                self.error = None;
                Task::perform(spawn_daemon(), Message::DaemonSpawnResult)
            }
            Message::DaemonSpawnResult(Ok(())) => {
                self.daemon = DaemonStatus::Probing;
                probe_health(self.client.clone())
            }
            Message::DaemonSpawnResult(Err(error)) => {
                self.error = Some(format!("Failed to start daemon: {error}"));
                Task::none()
            }
            Message::RefreshProvidersPressed => list_providers(self.client.clone()),
            Message::ProvidersListed(Ok(list)) => {
                self.providers = Some(list);
                Task::none()
            }
            Message::ProvidersListed(Err(error)) => {
                self.error = Some((*error).clone());
                Task::none()
            }
            Message::PortalProbed(PortalProbe::Available) => {
                self.hotkey = HotkeyStatus {
                    portal_available: true,
                    portal_error: None,
                };
                Task::none()
            }
            Message::PortalProbed(PortalProbe::Unavailable(reason)) => {
                self.hotkey = HotkeyStatus {
                    portal_available: false,
                    portal_error: Some(reason),
                };
                Task::none()
            }
            // The capture-affecting arms reconcile the HUD overlay after mutating
            // session state: `sync_hud` opens it when capture begins and closes it
            // when capture ends (see [`Self::capture_active`]).
            Message::HotkeyReceived => {
                let task = self.toggle_session();
                Task::batch([task, self.sync_hud()])
            }
            Message::StartPressed => {
                let task = self.start_session();
                Task::batch([task, self.sync_hud()])
            }
            Message::StopPressed => {
                let task = self.stop_session();
                Task::batch([task, self.sync_hud()])
            }
            Message::SessionStarted(Ok(session)) => {
                self.session.mark_listening(session.session_id);
                self.error = None;
                self.sync_hud()
            }
            Message::SessionStarted(Err(error)) => {
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }
            Message::SessionStopped(Ok(result)) => {
                self.error = self.session.apply_capture(*result);
                self.sync_hud()
            }
            Message::SessionStopped(Err(error)) => {
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }
            Message::EventReceived(event) => {
                self.last_event = Some(event);
                Task::none()
            }
            Message::Tick(now) => {
                if let Some(prev) = self.last_frame {
                    self.elapsed += now.saturating_duration_since(prev).as_secs_f32();
                }
                self.last_frame = Some(now);
                Task::none()
            }

            // --- Dictation panel controls ---
            Message::DictationProviderSelected(provider) => {
                self.dictation_provider = provider;
                Task::none()
            }
            Message::DictationLanguageEdited(value) => {
                self.dictation_language = value;
                Task::none()
            }
            Message::DictationSegmentationSelected(choice) => {
                self.dictation_segmentation = choice;
                Task::none()
            }
            Message::DictationTranslateToggled => {
                self.dictation_translate = !self.dictation_translate;
                Task::none()
            }
            Message::CapturePressed => {
                let task = self.capture_session();
                Task::batch([task, self.sync_hud()])
            }
            Message::Captured(Ok(result)) => {
                self.capture_in_flight = false;
                self.error = self.session.apply_capture(*result);
                self.sync_hud()
            }
            Message::Captured(Err(error)) => {
                self.capture_in_flight = false;
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }

            // --- Compose ---
            Message::ComposePromptEdited(action) => {
                self.compose.prompt.perform(action);
                Task::none()
            }
            Message::ComposeArchetypeSelected(archetype) => {
                self.compose.archetype = Some(archetype);
                Task::none()
            }
            Message::ComposeLanguageEdited(value) => {
                self.compose.language = value;
                Task::none()
            }
            Message::ComposeSubmitPressed => self.submit_compose(),
            Message::Composed(result) => {
                apply_job_result(&mut self.compose.job, result);
                Task::none()
            }

            // --- Rewrite ---
            Message::RewriteSourceEdited(action) => {
                self.rewrite.source.perform(action);
                Task::none()
            }
            Message::RewriteStyleSelected(style) => {
                self.rewrite.style = style;
                Task::none()
            }
            Message::RewriteLanguageEdited(value) => {
                self.rewrite.language = value;
                Task::none()
            }
            Message::RewriteSubmitPressed => self.submit_rewrite(),
            Message::Rewritten(result) => {
                apply_job_result(&mut self.rewrite.job, result);
                Task::none()
            }

            // --- Translate ---
            Message::TranslateSourceEdited(action) => {
                self.translate.source.perform(action);
                Task::none()
            }
            Message::TranslateTargetEdited(value) => {
                self.translate.target = value;
                Task::none()
            }
            Message::TranslateSubmitPressed => self.submit_translate(),
            Message::Translated(result) => {
                apply_job_result(&mut self.translate.job, result);
                Task::none()
            }

            // --- Preview → inject (shared by the three generative workflows) ---
            Message::InjectTargetSelected(tab, target) => {
                if let Some(job) = self.job_mut(tab) {
                    job.inject_target = target;
                }
                Task::none()
            }
            Message::InjectAutoSubmitToggled(tab) => {
                if let Some(job) = self.job_mut(tab) {
                    job.auto_submit = !job.auto_submit;
                }
                Task::none()
            }
            Message::InjectPressed(tab) => self.submit_inject(tab),
            Message::Injected(tab, result) => {
                if let Some(job) = self.job_mut(tab) {
                    job.injecting = false;
                    match result {
                        Ok(plan) => {
                            job.plan = Some(*plan);
                            job.error = None;
                        }
                        Err(error) => job.error = Some((*error).clone()),
                    }
                }
                Task::none()
            }

            // --- Settings (local preferences) ---
            Message::PrefOutputLanguageEdited(value) => {
                self.preferences.default_output_language = value;
                Task::none()
            }
            Message::PrefInjectTargetSelected(target) => {
                self.preferences.default_inject_target = target;
                Task::none()
            }
            Message::PrefRecorderSelected(backend) => {
                self.preferences.recorder_backend = backend;
                Task::none()
            }
            Message::PrefCaptureSecondsSelected(seconds) => {
                self.preferences.capture_seconds = seconds;
                Task::none()
            }
            Message::PrefGlassOpacityChanged(value) => {
                self.preferences.glass_opacity = value.clamp(0.0, 1.0);
                self.persist_preferences();
                Task::none()
            }
            Message::PrefReduceTransparencyToggled => {
                self.preferences.reduce_transparency = !self.preferences.reduce_transparency;
                self.persist_preferences();
                Task::none()
            }
            Message::SystemA11yChanged(system) => {
                self.system_a11y = system;
                Task::none()
            }

            // --- History ---
            Message::HistoryRefreshPressed => fetch_sessions(self.client.clone()),
            Message::SessionsListed(Ok(list)) => {
                self.sessions = Some(list);
                Task::none()
            }
            Message::SessionsListed(Err(error)) => {
                self.error = Some((*error).clone());
                Task::none()
            }
        }
    }

    fn toggle_session(&mut self) -> Task<Message> {
        match self.session.stage {
            SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => {
                self.start_session()
            }
            SessionStage::Listening => self.stop_session(),
            SessionStage::Starting | SessionStage::Stopping => Task::none(),
        }
    }

    /// Whether a capture is active and the HUD overlay should be on screen: any
    /// streaming stage that holds the microphone, or a one-shot capture in
    /// flight. The terminal stages (idle / completed / failed) are not active.
    fn capture_active(&self) -> bool {
        self.capture_in_flight
            || matches!(
                self.session.stage,
                SessionStage::Starting | SessionStage::Listening | SessionStage::Stopping
            )
    }

    /// Reconcile the capture HUD window with [`Self::capture_active`]: open it
    /// when capture begins, close it when it ends. `window::open` allocates the id
    /// synchronously, so it is stored before the open `Task` resolves; the close
    /// path drops our handle and asks iced to destroy the window. Idempotent —
    /// safe to call after every capture-affecting message.
    fn sync_hud(&mut self) -> Task<Message> {
        match (self.capture_active(), self.hud_window) {
            (true, None) => {
                let (id, open) = window::open(hud_window_settings());
                self.hud_window = Some(id);
                open.map(Message::HudWindowOpened)
            }
            (false, Some(id)) => {
                self.hud_window = None;
                window::close(id)
            }
            _ => Task::none(),
        }
    }

    fn start_session(&mut self) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            self.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        self.session.begin_starting();
        self.error = None;
        let client = self.client.clone();
        let request = StartDictationRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            recorder_backend: Some(self.preferences.recorder_backend),
            translate_to_english: self.dictation_translate,
            keep_audio: false,
            segmentation: self.dictation_segmentation.to_mode(),
            provider_id: self.dictation_provider.clone(),
        };
        Task::perform(
            async move { client.start_dictation(request).await },
            Message::SessionStarted,
        )
    }

    /// One-shot, fixed-duration capture (`POST /v1/dictation/capture`). Unlike
    /// streaming dictation this does not enter the `Listening` stage; it records
    /// for the configured window and returns a single result. The in-flight flag
    /// drives the capture button while the streaming stage machine is untouched.
    fn capture_session(&mut self) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            self.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        self.capture_in_flight = true;
        self.error = None;
        let request = DictationCaptureRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            duration_seconds: self.preferences.capture_seconds,
            recorder_backend: Some(self.preferences.recorder_backend),
            translate_to_english: self.dictation_translate,
            keep_audio: false,
            provider_id: self.dictation_provider.clone(),
        };
        let client = self.client.clone();
        Task::perform(
            async move { client.capture(request).await.map(Box::new) },
            Message::Captured,
        )
    }

    /// The mutable [`JobStage`] for a generative workflow tab, if it has one.
    fn job_mut(&mut self, tab: WorkflowTab) -> Option<&mut JobStage> {
        match tab {
            WorkflowTab::Compose => Some(&mut self.compose.job),
            WorkflowTab::Rewrite => Some(&mut self.rewrite.job),
            WorkflowTab::Translate => Some(&mut self.translate.job),
            _ => None,
        }
    }

    fn submit_compose(&mut self) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            self.compose.job.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        let prompt = self.compose.prompt.text();
        let prompt = prompt.trim();
        if prompt.is_empty() {
            self.compose.job.error = Some("Enter a prompt to compose from.".into());
            return Task::none();
        }
        let request = ComposeRequest {
            spoken_prompt: prompt.to_owned(),
            archetype: self.compose.archetype.clone(),
            output_language: optional_text(&self.compose.language),
        };
        self.compose.job.begin_submit();
        let client = self.client.clone();
        Task::perform(
            async move { client.compose(request).await.map(Box::new) },
            Message::Composed,
        )
    }

    fn submit_rewrite(&mut self) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            self.rewrite.job.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        let source = self.rewrite.source.text();
        let source = source.trim();
        if source.is_empty() {
            self.rewrite.job.error = Some("Enter text to rewrite.".into());
            return Task::none();
        }
        let request = RewriteRequest {
            source_text: source.to_owned(),
            style: self.rewrite.style.clone(),
            output_language: optional_text(&self.rewrite.language),
        };
        self.rewrite.job.begin_submit();
        let client = self.client.clone();
        Task::perform(
            async move { client.rewrite(request).await.map(Box::new) },
            Message::Rewritten,
        )
    }

    fn submit_translate(&mut self) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            self.translate.job.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        let source = self.translate.source.text();
        let source = source.trim();
        let target = self.translate.target.trim().to_owned();
        if source.is_empty() {
            self.translate.job.error = Some("Enter text to translate.".into());
            return Task::none();
        }
        if target.is_empty() {
            self.translate.job.error = Some("Enter a target language.".into());
            return Task::none();
        }
        let request = TranslateRequest {
            source_text: source.to_owned(),
            target_language: target,
        };
        self.translate.job.begin_submit();
        let client = self.client.clone();
        Task::perform(
            async move { client.translate(request).await.map(Box::new) },
            Message::Translated,
        )
    }

    /// Ask the daemon to plan an injection of the active preview's text into the
    /// chosen target (`POST /v1/inject`).
    fn submit_inject(&mut self, tab: WorkflowTab) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            if let Some(job) = self.job_mut(tab) {
                job.error = Some("Daemon is not healthy; start it first.".into());
            }
            return Task::none();
        }
        let Some(job) = self.job_mut(tab) else {
            return Task::none();
        };
        let text = job
            .preview
            .as_ref()
            .and_then(|preview| preview.generated_text.clone())
            .filter(|text| !text.is_empty());
        let Some(text) = text else {
            job.error = Some("This preview has no generated text to inject.".into());
            return Task::none();
        };
        let request = InjectRequest {
            target: job.inject_target.clone(),
            text,
            auto_submit: job.auto_submit,
        };
        job.injecting = true;
        job.error = None;
        job.plan = None;
        let client = self.client.clone();
        Task::perform(
            async move { client.inject(request).await.map(Box::new) },
            move |result| Message::Injected(tab, result),
        )
    }

    fn stop_session(&mut self) -> Task<Message> {
        let Some(session_id) = self.session.id else {
            self.error = Some("No active session to stop.".into());
            return Task::none();
        };
        self.session.begin_stopping();
        self.error = None;
        let client = self.client.clone();
        Task::perform(
            async move { client.stop_dictation(session_id).await.map(Box::new) },
            Message::SessionStopped,
        )
    }

    /// The live Liquid Glass accessibility contract: the persisted preferences
    /// (opacity, Reduce Transparency) combined with the live OS state (Increase
    /// Contrast, Reduce Motion). Views thread this into the glass styling.
    pub(crate) fn accessibility(&self) -> Accessibility {
        resolve_accessibility(&self.preferences, self.system_a11y)
    }

    /// Persist the user-controlled glass preferences to `desktop.toml`,
    /// best-effort (a failed write is logged inside [`config::save`]).
    fn persist_preferences(&self) {
        config::save(&config::DesktopConfig::from_preferences(&self.preferences));
    }

    pub fn view(&self, window: window::Id) -> Element<'_, Message> {
        view::window_view(self, window)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let keys = keyboard::listen().filter_map(|event| match event {
            keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(keyboard::key::Named::F9),
                ..
            } => Some(Message::HotkeyReceived),
            _ => None,
        });
        let mut subscriptions = vec![
            keys,
            Subscription::run(portal_stream),
            // Mirror the OS accessibility state (contrast / reduced motion) as it
            // changes, via the XDG settings portal.
            Subscription::run(a11y_stream),
            // Keyed by the socket path so editing it retargets the live stream.
            // A non-capturing closure bridges `&PathBuf` to the owned-path
            // `sse_stream`; it coerces to `run_with`'s `fn(&PathBuf)` builder.
            Subscription::run_with(self.client.socket_path().to_path_buf(), |path: &PathBuf| {
                sse_stream(path.clone())
            }),
            window::close_events().map(Message::WindowClosed),
        ];
        // The glass shader's animation clock. Reduce Motion gates it off: when the
        // OS asks for reduced motion we stop subscribing to per-frame ticks, so the
        // shader freezes on its current frame (still drawn, just not animated).
        if self.accessibility().animations_enabled() {
            subscriptions.push(window::frames().map(Message::Tick));
        }
        // The optional system tray bridges its menu activations into the runtime
        // through the same channel-backed subscription shape as the portal.
        #[cfg(feature = "tray")]
        subscriptions.push(Subscription::run(tray_stream));
        Subscription::batch(subscriptions)
    }
}

fn probe_health(client: Client) -> Task<Message> {
    Task::perform(
        async move { client.health().await.map(Box::new) },
        Message::DaemonProbed,
    )
}

fn list_providers(client: Client) -> Task<Message> {
    Task::perform(
        async move { client.providers().await },
        Message::ProvidersListed,
    )
}

fn fetch_sessions(client: Client) -> Task<Message> {
    Task::perform(
        async move { client.sessions().await },
        Message::SessionsListed,
    )
}

/// Best-effort stream for portal-activated hotkeys; mirrors the listener task
/// into the iced runtime. Silent when the portal is unavailable.
fn portal_stream() -> impl iced::futures::Stream<Item = Message> {
    stream::channel(8, async |mut output: futures_mpsc::Sender<Message>| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<String>();
        let registration = tokio::spawn(async move {
            if let Err(error) = portal::run_listener(tx).await {
                tracing::warn!(error = %error, "portal listener returned error");
            }
        });
        while let Some(shortcut_id) = rx.recv().await {
            if shortcut_id != SHORTCUT_TOGGLE {
                continue;
            }
            if output.send(Message::HotkeyReceived).await.is_err() {
                break;
            }
        }
        registration.abort();
    })
}

/// Best-effort stream mirroring OS accessibility changes into the runtime via the
/// XDG settings portal. Mirrors [`portal_stream`]: a spawned task forwards fresh
/// [`SystemA11y`] snapshots through a channel this subscription relays as
/// `Message::SystemA11yChanged`. Silent when the portal is unavailable.
fn a11y_stream() -> impl iced::futures::Stream<Item = Message> {
    stream::channel(8, async |mut output: futures_mpsc::Sender<Message>| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<SystemA11y>();
        let watcher = tokio::spawn(async move {
            if let Err(error) = a11y::watch(tx).await {
                tracing::debug!(error = %error, "accessibility portal watch ended");
            }
        });
        while let Some(system) = rx.recv().await {
            if output
                .send(Message::SystemA11yChanged(system))
                .await
                .is_err()
            {
                break;
            }
        }
        watcher.abort();
    })
}

/// Best-effort bridge from the optional system tray to the iced runtime. Mirrors
/// [`portal_stream`]: the `ksni` service runs on a spawned task and forwards menu
/// activations through a channel that this subscription relays as `Message`s. It
/// logs once and goes quiet when no StatusNotifierHost is available.
#[cfg(feature = "tray")]
fn tray_stream() -> impl iced::futures::Stream<Item = Message> {
    use crate::tray::{self, TrayCommand};

    stream::channel(8, async |mut output: futures_mpsc::Sender<Message>| {
        let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<TrayCommand>();
        let service = tokio::spawn(async move {
            if let Err(error) = tray::run(tx).await {
                tracing::info!(error = %error, "system tray unavailable; continuing without it");
            }
        });
        while let Some(command) = rx.recv().await {
            let message = match command {
                TrayCommand::ToggleDictation => Message::HotkeyReceived,
                TrayCommand::ShowMain => Message::TrayShowRequested,
                TrayCommand::Quit => Message::TrayQuitRequested,
            };
            if output.send(message).await.is_err() {
                break;
            }
        }
        service.abort();
    })
}

/// Long-lived Server-Sent Events stream from the daemon, reconnecting on every
/// drop. Takes an owned `PathBuf` (not `&PathBuf`) so the returned stream
/// captures no input lifetime and stays `'static` — the bridging closure in
/// [`App::subscription`] hands it a clone per keyed socket path.
fn sse_stream(socket_path: PathBuf) -> impl iced::futures::Stream<Item = Message> {
    stream::channel(
        64,
        async move |mut output: futures_mpsc::Sender<Message>| {
            loop {
                let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel::<EventEnvelope>();
                let reader_path = socket_path.clone();
                let reader = tokio::spawn(async move {
                    if let Err(error) = api::stream_events(&reader_path, tx).await {
                        tracing::debug!(error = %error, "event stream ended");
                    }
                });
                while let Some(event) = rx.recv().await {
                    if output.send(Message::EventReceived(event)).await.is_err() {
                        reader.abort();
                        return;
                    }
                }
                reader.abort();
                // The daemon closed the stream or the socket was unreachable; pause
                // before reconnecting so a down daemon does not spin.
                tokio::time::sleep(Duration::from_millis(1500)).await;
            }
        },
    )
}
