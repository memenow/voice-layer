//! The VoiceLayer desktop program controller.
//!
//! An [`iced::daemon`] (not a single-window `application`) so the main window,
//! the capture HUD (P5), and a settings window (P3) can be sibling windows with
//! their own content, title, and chrome — daemon `view`/`title`/`theme` receive
//! the [`window::Id`] being drawn, which `application` does not expose. The
//! daemon also outlives its windows, matching VoiceLayer's "background voice
//! layer" model; closing the main window quits via [`iced::exit`], after
//! [`App::request_quit`] has stopped any active dictation so the daemon-owned
//! recorder is never orphaned.
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

use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionArchetype, CompositionReceipt,
    DictationCaptureRequest, DictationCaptureResult, EventEnvelope, HealthResponse, InjectRequest,
    InjectTarget, InjectionPlan, ProviderDescriptor, ProviderKind, RecorderBackend, RewriteRequest,
    RewriteStyle, StartDictationRequest, TranslateRequest, TriggerKind,
    is_supported_transcribe_provider_id,
};
use voicelayer_ui::a11y::Accessibility;

use crate::a11y;
use crate::api::{self, Client};
use crate::config;
use crate::forms::{ComposeForm, JobStage, RewriteForm, TranslateForm, apply_job_result};
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
    /// Generation of the active daemon target. Async replies carry this value so
    /// a late response from a previously selected socket cannot refill its cache.
    daemon_epoch: u64,
    pub(crate) daemon: DaemonStatus,
    pub(crate) tab: WorkflowTab,
    pub(crate) session: Session,
    pub(crate) providers: Option<Vec<ProviderDescriptor>>,
    pub(crate) providers_error: Option<String>,
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
    pub(crate) sessions_error: Option<String>,
    /// Set once a quit has been requested while dictation cleanup is still in
    /// flight; a second quit request forces the exit immediately.
    quitting: bool,
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
    DaemonProbed(u64, Result<Box<HealthResponse>, SharedError>),
    StartDaemonPressed,
    DaemonSpawnResult(u64, Result<(), SharedError>),
    RefreshProvidersPressed,
    ProvidersListed(u64, Result<Vec<ProviderDescriptor>, SharedError>),
    PortalProbed(PortalProbe),
    HotkeyReceived,
    StartPressed,
    StopPressed,
    SessionStarted(Result<CaptureSession, SharedError>),
    SessionStopped(Result<Box<DictationCaptureResult>, SharedError>),
    EventReceived(u64, EventEnvelope),
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
    ComposeArchetypeSelected(Option<CompositionArchetype>),
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
    // The `u64` is the inject generation this reply belongs to; a stale reply
    // (superseded by a newer submission or inject) is dropped on arrival.
    Injected(WorkflowTab, u64, Result<Box<InjectionPlan>, SharedError>),

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
    SessionsListed(u64, Result<Vec<CaptureSession>, SharedError>),
    /// Internal: pre-quit dictation cleanup settled (successfully or not), so
    /// the deferred exit requested by [`App::request_quit`] can proceed.
    QuitNow,
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
            daemon_epoch: 0,
            daemon: DaemonStatus::Probing,
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
        };
        let startup = Task::batch(vec![
            open_main.map(Message::MainWindowOpened),
            probe_health(client.clone(), 0),
            list_providers(client, 0),
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
                    // deliberately when the main window goes away — but stop an
                    // active dictation first so the microphone is not orphaned.
                    self.request_quit()
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
            Message::TrayQuitRequested => self.request_quit(),
            Message::TabSelected(tab) => {
                self.tab = tab;
                // Refresh the read-only panels on entry so they show live data;
                // the dictation provider picker also needs the provider list.
                match tab {
                    WorkflowTab::Providers => self.refresh_providers(),
                    WorkflowTab::Doctor => probe_health(self.client.clone(), self.daemon_epoch),
                    WorkflowTab::History => self.refresh_sessions(),
                    WorkflowTab::Dictation if self.providers.is_none() => self.refresh_providers(),
                    _ => Task::none(),
                }
            }
            Message::SocketPathEdited(next) => {
                let socket_path = PathBuf::from(next.trim());
                if socket_path == self.client.socket_path() {
                    self.socket_input = next;
                    return Task::none();
                }
                if self.daemon_request_active() {
                    self.error = Some(
                        "Finish the active daemon request before changing the socket path."
                            .to_owned(),
                    );
                    return Task::none();
                }

                self.client = Client::new(socket_path);
                self.socket_input = next;
                self.daemon_epoch = self.daemon_epoch.wrapping_add(1);
                self.daemon = DaemonStatus::Unknown;
                self.session = Session::default();
                self.providers = None;
                self.providers_error = None;
                self.health = None;
                self.last_event = None;
                self.error = None;
                self.dictation_provider = None;
                self.sessions = None;
                self.sessions_error = None;
                Task::none()
            }
            Message::ProbeDaemonPressed => {
                self.daemon = DaemonStatus::Probing;
                self.error = None;
                // Drop the prior probe's diagnostics so the Doctor panel cannot
                // present stale socket/worker details as current while re-probing.
                self.health = None;
                let providers = self.refresh_providers();
                Task::batch([
                    probe_health(self.client.clone(), self.daemon_epoch),
                    providers,
                ])
            }
            Message::DaemonProbed(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::DaemonProbed(_, Ok(health)) => {
                self.daemon = DaemonStatus::Healthy;
                self.health = Some(*health);
                self.error = None;
                Task::none()
            }
            Message::DaemonProbed(_, Err(error)) => {
                self.daemon = DaemonStatus::Unreachable;
                self.error = Some((*error).clone());
                // The daemon is down or unreachable; clear the last good probe so
                // the Doctor panel does not keep rendering its details as current.
                self.health = None;
                Task::none()
            }
            Message::StartDaemonPressed => {
                self.error = None;
                let daemon_epoch = self.daemon_epoch;
                Task::perform(spawn_daemon(), move |result| {
                    Message::DaemonSpawnResult(daemon_epoch, result)
                })
            }
            Message::DaemonSpawnResult(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::DaemonSpawnResult(_, Ok(())) => {
                self.daemon = DaemonStatus::Probing;
                let providers = self.refresh_providers();
                Task::batch([
                    probe_health(self.client.clone(), self.daemon_epoch),
                    providers,
                ])
            }
            Message::DaemonSpawnResult(_, Err(error)) => {
                self.error = Some(format!("Failed to start daemon: {error}"));
                Task::none()
            }
            Message::RefreshProvidersPressed => self.refresh_providers(),
            Message::ProvidersListed(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::ProvidersListed(_, Ok(list)) => {
                reconcile_selected_asr_provider(&mut self.dictation_provider, Some(&list));
                if !dictation_translate_supported(self.dictation_provider.as_deref()) {
                    self.dictation_translate = false;
                }
                self.providers = Some(list);
                self.providers_error = None;
                Task::none()
            }
            Message::ProvidersListed(_, Err(error)) => {
                self.providers_error = Some((*error).clone());
                // Drop the cached list so the Providers panel and the dictation
                // ASR picker stop offering ids from a daemon/socket we can no
                // longer confirm; a successful refresh repopulates it.
                self.providers = None;
                reconcile_selected_asr_provider(&mut self.dictation_provider, None);
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
                // A quit requested while the start was in flight stops the
                // fresh session immediately so the daemon never keeps
                // recording without a UI able to stop it.
                if self.quitting {
                    let client = self.client.clone();
                    let session_id = session.session_id;
                    let stop = Task::perform(
                        async move { client.stop_dictation(session_id).await },
                        |_| Message::QuitNow,
                    );
                    return Task::batch([stop, quit_watchdog()]);
                }
                self.session.mark_listening(session.session_id);
                self.error = None;
                self.sync_hud()
            }
            Message::SessionStarted(Err(error)) => {
                // The start failed, so no session exists to clean up; a pending
                // quit can proceed right away.
                if self.quitting {
                    return iced::exit();
                }
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }
            Message::SessionStopped(_) if self.quitting => {
                // Pre-quit cleanup settled — transcript handling is moot once
                // the operator chose to exit.
                iced::exit()
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
            Message::QuitNow => iced::exit(),
            Message::EventReceived(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::EventReceived(_, event) => {
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
                if !dictation_translate_supported(self.dictation_provider.as_deref()) {
                    self.dictation_translate = false;
                }
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
                if dictation_translate_supported(self.dictation_provider.as_deref()) {
                    self.dictation_translate = !self.dictation_translate;
                } else {
                    self.dictation_translate = false;
                }
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
                self.compose.archetype = archetype;
                Task::none()
            }
            Message::ComposeLanguageEdited(value) => {
                self.compose.edit_language(value);
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
                self.rewrite.edit_language(value);
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
                self.translate.edit_target(value);
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
                    job.set_inject_target(target);
                }
                Task::none()
            }
            Message::InjectAutoSubmitToggled(tab) => {
                if let Some(job) = self.job_mut(tab) {
                    job.toggle_auto_submit();
                }
                Task::none()
            }
            Message::InjectPressed(tab) => self.submit_inject(tab),
            Message::Injected(tab, epoch, result) => {
                if let Some(job) = self.job_mut(tab) {
                    // Drop a reply from an injection a newer submission or inject
                    // superseded: it belongs to a preview no longer on screen.
                    if epoch != job.inject_epoch {
                        return Task::none();
                    }
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
                self.compose.sync_default_language(&value);
                self.rewrite.sync_default_language(&value);
                self.translate.sync_default_target(&value);
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
            Message::HistoryRefreshPressed => self.refresh_sessions(),
            Message::SessionsListed(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::SessionsListed(_, Ok(list)) => {
                self.sessions = Some(list);
                self.sessions_error = None;
                Task::none()
            }
            Message::SessionsListed(_, Err(error)) => {
                self.sessions_error = Some((*error).clone());
                Task::none()
            }
        }
    }

    fn refresh_providers(&mut self) -> Task<Message> {
        self.providers_error = None;
        list_providers(self.client.clone(), self.daemon_epoch)
    }

    fn refresh_sessions(&mut self) -> Task<Message> {
        self.sessions_error = None;
        fetch_sessions(self.client.clone(), self.daemon_epoch)
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
        capture_is_active(self.capture_in_flight, self.session.stage)
    }

    /// A socket retarget cannot safely cross a request whose reply mutates live
    /// workflow state. Read-only daemon caches carry an epoch and can be dropped;
    /// capture and generative jobs instead keep the current target until settled.
    fn daemon_request_active(&self) -> bool {
        self.capture_active()
            || [&self.compose.job, &self.rewrite.job, &self.translate.job]
                .iter()
                .any(|job| job.submitting || job.injecting)
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
        // A one-shot capture already holds the microphone; starting streaming
        // dictation now would run a second recorder against it and race the HUD
        // and transcript state. Serialize: refuse until the capture finishes.
        if self.capture_in_flight {
            self.error = Some("A one-shot capture is in progress; wait for it to finish.".into());
            return Task::none();
        }
        if self.daemon != DaemonStatus::Healthy {
            self.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        self.session.begin_starting();
        self.error = None;
        let client = self.client.clone();
        let translate_to_english = validated_dictation_translate(
            self.dictation_provider.as_deref(),
            self.dictation_translate,
        );
        let request = StartDictationRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            recorder_backend: Some(self.preferences.recorder_backend),
            translate_to_english,
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
        if self.capture_active() {
            self.error = Some("A capture or dictation session is already in progress.".into());
            return Task::none();
        }
        if self.daemon != DaemonStatus::Healthy {
            self.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        self.capture_in_flight = true;
        self.error = None;
        let translate_to_english = validated_dictation_translate(
            self.dictation_provider.as_deref(),
            self.dictation_translate,
        );
        let request = DictationCaptureRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            duration_seconds: self.preferences.capture_seconds,
            recorder_backend: Some(self.preferences.recorder_backend),
            translate_to_english,
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
            output_language: self.compose.output_language(),
        };
        self.compose
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
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
            output_language: self.rewrite.output_language(),
        };
        self.rewrite
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
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
        if source.is_empty() {
            self.translate.job.error = Some("Enter text to translate.".into());
            return Task::none();
        }
        let Some(target) = self.translate.target_language() else {
            self.translate.job.error = Some("Enter a target language.".into());
            return Task::none();
        };
        let request = TranslateRequest {
            source_text: source.to_owned(),
            target_language: target,
        };
        self.translate
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
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
        // Tag this request's reply so a newer submission or inject can supersede
        // it; a late reply with a stale epoch is dropped in `Message::Injected`.
        job.inject_epoch = job.inject_epoch.wrapping_add(1);
        let epoch = job.inject_epoch;
        let client = self.client.clone();
        Task::perform(
            async move { client.inject(request).await.map(Box::new) },
            move |result| Message::Injected(tab, epoch, result),
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

    /// Quit the shell without orphaning the microphone. A streaming dictation
    /// session is owned by the daemon, not by this process, so a bare
    /// `iced::exit()` would leave the recorder running with no UI left to stop
    /// it. Defer the exit until the active — or still-starting — session has
    /// been stopped; a second quit request forces the exit immediately, and
    /// the cleanup reply (success or failure) ends the wait either way. A
    /// one-shot capture in flight is fixed-duration and self-terminating, so
    /// it does not hold the quit.
    fn request_quit(&mut self) -> Task<Message> {
        match quit_plan(self.quitting, self.session.stage, self.session.id) {
            QuitPlan::Exit => iced::exit(),
            QuitPlan::StopThenExit(session_id) => {
                self.quitting = true;
                self.session.begin_stopping();
                let client = self.client.clone();
                let stop = Task::perform(
                    async move { client.stop_dictation(session_id).await },
                    |_| Message::QuitNow,
                );
                Task::batch([stop, quit_watchdog()])
            }
            QuitPlan::AwaitPendingReply => {
                self.quitting = true;
                quit_watchdog()
            }
        }
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
            // Keyed by both socket and generation so every accepted retarget
            // replaces the live stream and late events identify their origin.
            Subscription::run_with(
                (self.client.socket_path().to_path_buf(), self.daemon_epoch),
                |target: &(PathBuf, u64)| sse_stream(target.0.clone(), target.1),
            ),
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

fn probe_health(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(
        async move { client.health().await.map(Box::new) },
        move |result| Message::DaemonProbed(daemon_epoch, result),
    )
}

fn list_providers(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(async move { client.providers().await }, move |result| {
        Message::ProvidersListed(daemon_epoch, result)
    })
}

fn fetch_sessions(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(async move { client.sessions().await }, move |result| {
        Message::SessionsListed(daemon_epoch, result)
    })
}

/// Whether an advertised provider is executable through the daemon's current
/// transcription dispatch. The catalog also contains forward-looking entries,
/// so `ProviderKind::Asr` alone is not sufficient for an actionable picker.
pub(crate) fn is_dispatchable_asr_provider(provider: &ProviderDescriptor) -> bool {
    provider.kind == ProviderKind::Asr
        && is_supported_transcribe_provider_id(Some(provider.id.as_str()))
}

/// Translation is an explicit provider capability. Automatic dispatch and the
/// whisper.cpp provider support it today; new ASR providers remain disabled
/// until their worker path deliberately opts in.
pub(crate) fn dictation_translate_supported(provider_id: Option<&str>) -> bool {
    matches!(provider_id, None | Some("whisper_cpp"))
}

fn validated_dictation_translate(provider_id: Option<&str>, requested: bool) -> bool {
    requested && dictation_translate_supported(provider_id)
}

fn reconcile_selected_asr_provider(
    selected: &mut Option<String>,
    providers: Option<&[ProviderDescriptor]>,
) {
    let should_clear = match selected.as_deref() {
        None => false,
        Some(selected_id) => !providers.is_some_and(|providers| {
            providers.iter().any(|provider| {
                is_dispatchable_asr_provider(provider) && provider.id == selected_id
            })
        }),
    };
    if should_clear {
        *selected = None;
    }
}

fn capture_is_active(capture_in_flight: bool, stage: SessionStage) -> bool {
    capture_in_flight
        || matches!(
            stage,
            SessionStage::Starting | SessionStage::Listening | SessionStage::Stopping
        )
}

/// What a quit request must do about the microphone before exiting. Pure so
/// the deferral policy is unit-testable without driving iced tasks.
#[derive(Debug, PartialEq, Eq)]
enum QuitPlan {
    /// Exit immediately: nothing holds the microphone, the stage has no
    /// session id to stop, or this is a forced second request.
    Exit,
    /// Stop this session first; its reply (success or failure) then exits.
    StopThenExit(Uuid),
    /// A start or stop is already in flight; that reply finishes the quit.
    AwaitPendingReply,
}

fn quit_plan(quitting: bool, stage: SessionStage, session_id: Option<Uuid>) -> QuitPlan {
    if quitting {
        return QuitPlan::Exit;
    }
    match stage {
        SessionStage::Listening => match session_id {
            Some(id) => QuitPlan::StopThenExit(id),
            None => QuitPlan::Exit,
        },
        SessionStage::Starting | SessionStage::Stopping => QuitPlan::AwaitPendingReply,
        // A one-shot capture never moves the stage off a terminal value; it is
        // fixed-duration and self-terminating, so it does not hold the quit.
        _ => QuitPlan::Exit,
    }
}

/// Bound the pre-quit wait. The cleanup request has no daemon-side timeout, so
/// a wedged daemon (accepted the socket, never replies) would otherwise leave
/// the shell lingering as an invisible, windowless process — this is an
/// `iced::daemon`, which deliberately does not exit on last-window-close, and
/// the forced-second-request escape is unreachable once the main window is
/// gone. After the watchdog fires, `Message::QuitNow` forces the exit; the
/// session remains stoppable through the daemon's stop API (e.g. the `vl`
/// CLI) instead of holding the shell process hostage indefinitely.
fn quit_watchdog() -> Task<Message> {
    Task::perform(
        async {
            tokio::time::sleep(Duration::from_secs(5)).await;
        },
        |_| Message::QuitNow,
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
fn sse_stream(
    socket_path: PathBuf,
    daemon_epoch: u64,
) -> impl iced::futures::Stream<Item = Message> {
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
                let reader_guard = AbortOnDrop(reader);
                while let Some(event) = rx.recv().await {
                    if output
                        .send(Message::EventReceived(daemon_epoch, event))
                        .await
                        .is_err()
                    {
                        return;
                    }
                }
                drop(reader_guard);
                // The daemon closed the stream or the socket was unreachable; pause
                // before reconnecting so a down daemon does not spin.
                tokio::time::sleep(Duration::from_millis(1500)).await;
            }
        },
    )
}

struct AbortOnDrop(tokio::task::JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;
    use voicelayer_core::{LanguageProfile, ProviderKind, SessionMode};

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
    fn provider_refresh_keeps_only_a_selected_asr_that_is_still_available() {
        let providers = vec![
            provider("whisper_cpp", ProviderKind::Asr),
            provider("voxtral_realtime", ProviderKind::Asr),
            provider("writer", ProviderKind::Llm),
        ];

        let mut selected = Some("whisper_cpp".to_owned());
        reconcile_selected_asr_provider(&mut selected, Some(&providers));
        assert_eq!(selected.as_deref(), Some("whisper_cpp"));

        let mut advertised_only = Some("voxtral_realtime".to_owned());
        reconcile_selected_asr_provider(&mut advertised_only, Some(&providers));
        assert_eq!(advertised_only, None);

        let mut missing = Some("missing".to_owned());
        reconcile_selected_asr_provider(&mut missing, Some(&providers));
        assert_eq!(missing, None);

        let mut wrong_kind = Some("writer".to_owned());
        reconcile_selected_asr_provider(&mut wrong_kind, Some(&providers));
        assert_eq!(wrong_kind, None);
    }

    #[test]
    fn provider_refresh_error_clears_the_unverifiable_selection() {
        let mut selected = Some("whisper_cpp".to_owned());
        reconcile_selected_asr_provider(&mut selected, None);
        assert_eq!(selected, None);
    }

    #[test]
    fn provider_failures_are_scoped_and_request_start_clears_them() {
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
        assert_eq!(app.error.as_deref(), Some("unrelated daemon error"));
    }

    #[test]
    fn provider_success_clears_the_scoped_error() {
        let mut app = test_app();
        app.providers_error = Some("provider refresh failed".to_owned());

        drop(app.update(Message::ProvidersListed(
            app.daemon_epoch,
            Ok(vec![provider("whisper_cpp", ProviderKind::Asr)]),
        )));

        assert!(app.providers_error.is_none());
        assert_eq!(app.providers.as_ref().map(Vec::len), Some(1));
    }

    #[test]
    fn history_failures_preserve_cached_sessions_and_are_scoped() {
        let mut app = test_app();
        app.sessions = Some(vec![CaptureSession::new(
            SessionMode::Dictation,
            TriggerKind::TrayButton,
            LanguageProfile::default(),
        )]);
        app.error = Some("unrelated daemon error".to_owned());

        drop(app.update(Message::SessionsListed(
            app.daemon_epoch,
            Err(Arc::new("history refresh failed".to_owned())),
        )));

        assert_eq!(app.sessions.as_ref().map(Vec::len), Some(1));
        assert_eq!(
            app.sessions_error.as_deref(),
            Some("history refresh failed")
        );
        assert_eq!(app.error.as_deref(), Some("unrelated daemon error"));

        drop(app.update(Message::HistoryRefreshPressed));
        assert_eq!(app.sessions.as_ref().map(Vec::len), Some(1));
        assert!(app.sessions_error.is_none());
    }

    #[test]
    fn history_failure_without_cache_has_an_error_instead_of_loading_forever() {
        let mut app = test_app();

        drop(app.update(Message::SessionsListed(
            app.daemon_epoch,
            Err(Arc::new("history unavailable".to_owned())),
        )));

        assert!(app.sessions.is_none());
        assert_eq!(app.sessions_error.as_deref(), Some("history unavailable"));
    }

    #[test]
    fn history_success_clears_the_scoped_error() {
        let mut app = test_app();
        app.sessions_error = Some("history refresh failed".to_owned());

        drop(app.update(Message::SessionsListed(app.daemon_epoch, Ok(Vec::new()))));

        assert!(app.sessions_error.is_none());
        assert_eq!(app.sessions.as_ref().map(Vec::len), Some(0));
    }

    #[test]
    fn translation_is_enabled_only_for_compatible_dictation_providers() {
        assert!(dictation_translate_supported(None));
        assert!(dictation_translate_supported(Some("whisper_cpp")));
        assert!(!dictation_translate_supported(Some("mimo_v2_5_asr")));
        assert!(!dictation_translate_supported(Some("qwen3_asr_1_7b")));
        assert!(!dictation_translate_supported(Some("future_asr")));

        assert!(validated_dictation_translate(None, true));
        assert!(!validated_dictation_translate(Some("mimo_v2_5_asr"), true));
        assert!(!validated_dictation_translate(Some("whisper_cpp"), false));
    }

    #[test]
    fn selecting_an_incompatible_provider_disables_translation() {
        let mut app = test_app();
        app.dictation_translate = true;

        drop(app.update(Message::DictationProviderSelected(Some(
            "mimo_v2_5_asr".to_owned(),
        ))));
        assert!(!app.dictation_translate);

        drop(app.update(Message::DictationTranslateToggled));
        assert!(!app.dictation_translate);

        drop(app.update(Message::DictationProviderSelected(Some(
            "whisper_cpp".to_owned(),
        ))));
        drop(app.update(Message::DictationTranslateToggled));
        assert!(app.dictation_translate);
    }

    #[test]
    fn socket_retarget_clears_daemon_scoped_state_and_advances_epoch() {
        let mut app = test_app();
        app.providers = Some(vec![provider("whisper_cpp", ProviderKind::Asr)]);
        app.providers_error = Some("old provider error".to_owned());
        app.dictation_provider = Some("whisper_cpp".to_owned());
        app.sessions = Some(Vec::new());
        app.sessions_error = Some("old history error".to_owned());
        app.last_event = Some(EventEnvelope::new("ready", None, "old daemon"));
        app.session.stage = SessionStage::Completed;
        app.session.transcript = Some("old transcript".to_owned());

        drop(app.update(Message::SocketPathEdited("/tmp/new.sock".to_owned())));

        assert_eq!(app.client.socket_path(), PathBuf::from("/tmp/new.sock"));
        assert_eq!(app.socket_input, "/tmp/new.sock");
        assert_eq!(app.daemon_epoch, 1);
        assert_eq!(app.daemon, DaemonStatus::Unknown);
        assert!(app.providers.is_none());
        assert!(app.providers_error.is_none());
        assert!(app.dictation_provider.is_none());
        assert!(app.sessions.is_none());
        assert!(app.sessions_error.is_none());
        assert!(app.last_event.is_none());
        assert_eq!(app.session.stage, SessionStage::Idle);
        assert!(app.session.transcript.is_none());
    }

    #[test]
    fn same_effective_socket_path_preserves_daemon_state_and_epoch() {
        let mut app = test_app();
        app.providers = Some(vec![provider("whisper_cpp", ProviderKind::Asr)]);

        drop(app.update(Message::SocketPathEdited("  /tmp/old.sock  ".to_owned())));

        assert_eq!(app.daemon_epoch, 0);
        assert_eq!(app.daemon, DaemonStatus::Healthy);
        assert!(app.providers.is_some());
        assert_eq!(app.socket_input, "  /tmp/old.sock  ");
    }

    #[test]
    fn socket_retarget_is_rejected_while_capture_is_active() {
        let mut app = test_app();
        app.capture_in_flight = true;

        drop(app.update(Message::SocketPathEdited("/tmp/new.sock".to_owned())));

        assert_eq!(app.client.socket_path(), PathBuf::from("/tmp/old.sock"));
        assert_eq!(app.socket_input, "/tmp/old.sock");
        assert_eq!(app.daemon_epoch, 0);
        assert!(
            app.error
                .as_deref()
                .is_some_and(|error| error.contains("socket path"))
        );
    }

    #[test]
    fn stale_daemon_replies_are_ignored_after_socket_retarget() {
        let mut app = test_app();
        drop(app.update(Message::SocketPathEdited("/tmp/new.sock".to_owned())));
        drop(app.update(Message::SocketPathEdited("/tmp/old.sock".to_owned())));
        assert_eq!(
            app.daemon_epoch, 2,
            "A -> B -> A remains a new target generation"
        );

        drop(app.update(Message::DaemonProbed(
            0,
            Err(Arc::new("old daemon failed".to_owned())),
        )));
        drop(app.update(Message::DaemonSpawnResult(
            0,
            Err(Arc::new("old daemon spawn failed".to_owned())),
        )));
        drop(app.update(Message::ProvidersListed(
            0,
            Ok(vec![provider("whisper_cpp", ProviderKind::Asr)]),
        )));
        drop(app.update(Message::SessionsListed(
            0,
            Ok(vec![CaptureSession::new(
                SessionMode::Dictation,
                TriggerKind::TrayButton,
                LanguageProfile::default(),
            )]),
        )));
        drop(app.update(Message::EventReceived(
            0,
            EventEnvelope::new("stale", None, "old daemon"),
        )));

        assert_eq!(app.daemon, DaemonStatus::Unknown);
        assert!(app.error.is_none());
        assert!(app.providers.is_none());
        assert!(app.sessions.is_none());
        assert!(app.last_event.is_none());
    }

    #[test]
    fn capture_is_active_for_one_shot_and_streaming_microphone_states() {
        assert!(capture_is_active(true, SessionStage::Idle));
        for stage in [
            SessionStage::Starting,
            SessionStage::Listening,
            SessionStage::Stopping,
        ] {
            assert!(capture_is_active(false, stage), "stage {stage:?}");
        }
        for stage in [
            SessionStage::Idle,
            SessionStage::Completed,
            SessionStage::Failed,
        ] {
            assert!(!capture_is_active(false, stage), "stage {stage:?}");
        }
    }

    #[test]
    fn quit_plan_stops_active_streaming_dictation_before_exiting() {
        let id = Uuid::new_v4();
        assert_eq!(
            quit_plan(false, SessionStage::Listening, Some(id)),
            QuitPlan::StopThenExit(id),
        );
        // A Listening stage without an id has nothing the daemon can stop.
        assert_eq!(
            quit_plan(false, SessionStage::Listening, None),
            QuitPlan::Exit
        );
    }

    #[test]
    fn quit_plan_defers_to_the_pending_start_or_stop_reply() {
        for stage in [SessionStage::Starting, SessionStage::Stopping] {
            assert_eq!(
                quit_plan(false, stage, None),
                QuitPlan::AwaitPendingReply,
                "stage {stage:?}",
            );
        }
    }

    #[test]
    fn quit_plan_exits_immediately_when_idle_or_forced() {
        for stage in [
            SessionStage::Idle,
            SessionStage::Completed,
            SessionStage::Failed,
        ] {
            assert_eq!(
                quit_plan(false, stage, None),
                QuitPlan::Exit,
                "stage {stage:?}"
            );
        }
        // A second quit request forces the exit even mid-cleanup.
        assert_eq!(
            quit_plan(true, SessionStage::Listening, Some(Uuid::new_v4())),
            QuitPlan::Exit,
        );
        assert_eq!(
            quit_plan(true, SessionStage::Starting, None),
            QuitPlan::Exit,
        );
    }

    #[test]
    fn request_quit_marks_quitting_and_moves_listening_into_stopping() {
        let mut app = test_app();
        app.session.mark_listening(Uuid::new_v4());

        drop(app.request_quit());

        assert!(app.quitting, "the quit is now pending cleanup");
        assert_eq!(app.session.stage, SessionStage::Stopping);
    }

    #[test]
    fn request_quit_while_starting_only_marks_quitting() {
        let mut app = test_app();
        app.session.begin_starting();

        drop(app.request_quit());

        assert!(app.quitting);
        assert_eq!(
            app.session.stage,
            SessionStage::Starting,
            "the in-flight start reply finishes the quit",
        );
    }

    #[tokio::test]
    async fn abort_on_drop_cancels_a_pending_reader_task() {
        let reader = tokio::spawn(std::future::pending::<()>());
        let abort_handle = reader.abort_handle();
        let guard = AbortOnDrop(reader);

        drop(guard);
        tokio::time::timeout(Duration::from_millis(100), async {
            while !abort_handle.is_finished() {
                tokio::task::yield_now().await;
            }
        })
        .await
        .expect("dropping the subscription guard should abort its reader task");
    }
}
