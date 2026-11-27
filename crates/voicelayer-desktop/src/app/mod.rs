//! Desktop application controller, message taxonomy, and pure workflow policies.

mod dictation;
mod handlers;
mod subscriptions;
#[cfg(test)]
mod tests;
mod workflows;

use std::time::{Duration, Instant};

use iced::widget::text_editor;
use iced::{Element, Subscription, Task, keyboard, window};
use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, CompositionArchetype, CompositionReceipt, DictationCaptureResult,
    EventEnvelope, HealthResponse, InjectTarget, InjectionPlan, ProviderDescriptor, ProviderKind,
    RewriteStyle, is_supported_transcribe_provider_id, provider_supports_asr_translation,
};
use voicelayer_ui::accessibility::Accessibility;

use crate::accessibility;
use crate::config;
use crate::daemon_client::{self, Client};
use crate::forms::{ComposeForm, RewriteForm, TranslateForm};
use crate::hotkeys::{self, PortalProbe};
use crate::state::{
    DaemonStatus, HotkeyStatus, Preferences, SegChoice, Session, SessionStage, SharedError,
    SystemA11y, WorkflowTab, resolve_accessibility,
};
use crate::views;

use self::handlers::{list_providers, probe_health};

pub(super) fn main_window_settings() -> window::Settings {
    window::Settings {
        size: iced::Size::new(960.0, 680.0),
        min_size: Some(iced::Size::new(720.0, 520.0)),
        position: window::Position::Centered,
        transparent: true,
        ..window::Settings::default()
    }
}

pub(super) fn hud_window_settings() -> window::Settings {
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

/// Top-level `iced::daemon` state shared across the main window and capture HUD.
pub struct App {
    pub(crate) main_window: Option<window::Id>,
    pub(crate) hud_window: Option<window::Id>,
    pub(super) client: Client,
    pub(crate) socket_input: String,
    pub(super) daemon_epoch: u64,
    pub(crate) daemon: DaemonStatus,
    pub(crate) tab: WorkflowTab,
    pub(crate) session: Session,
    pub(crate) providers: Option<Vec<ProviderDescriptor>>,
    pub(crate) providers_error: Option<String>,
    pub(crate) health: Option<HealthResponse>,
    pub(crate) hotkey: HotkeyStatus,
    pub(crate) last_event: Option<EventEnvelope>,
    pub(crate) error: Option<String>,
    pub(crate) elapsed: f32,
    pub(super) last_frame: Option<Instant>,
    pub(crate) preferences: Preferences,
    pub(crate) system_a11y: SystemA11y,
    pub(crate) dictation_provider: Option<String>,
    pub(crate) dictation_language: String,
    pub(crate) dictation_segmentation: SegChoice,
    pub(crate) dictation_translate: bool,
    pub(crate) capture_in_flight: bool,
    pub(crate) compose: ComposeForm,
    pub(crate) rewrite: RewriteForm,
    pub(crate) translate: TranslateForm,
    pub(crate) sessions: Option<Vec<CaptureSession>>,
    pub(crate) sessions_error: Option<String>,
    pub(super) quitting: bool,
}

#[derive(Debug, Clone)]
pub enum Message {
    MainWindowOpened(window::Id),
    HudWindowOpened,
    WindowClosed(window::Id),
    #[cfg(feature = "tray")]
    TrayShowRequested,
    #[cfg(feature = "tray")]
    TrayQuitRequested,
    TabSelected(WorkflowTab),
    SocketPathEdited(String),
    ProbeDaemonPressed,
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
    Tick(Instant),

    DictationProviderSelected(Option<String>),
    DictationLanguageEdited(String),
    DictationSegmentationSelected(SegChoice),
    DictationTranslateToggled,
    CapturePressed,
    Captured(Result<Box<DictationCaptureResult>, SharedError>),

    ComposePromptEdited(text_editor::Action),
    ComposeArchetypeSelected(Option<CompositionArchetype>),
    ComposeLanguageEdited(String),
    ComposeSubmitPressed,
    Composed(Result<Box<CompositionReceipt>, SharedError>),

    RewriteSourceEdited(text_editor::Action),
    RewriteStyleSelected(RewriteStyle),
    RewriteLanguageEdited(String),
    RewriteSubmitPressed,
    Rewritten(Result<Box<CompositionReceipt>, SharedError>),

    TranslateSourceEdited(text_editor::Action),
    TranslateTargetEdited(String),
    TranslateSubmitPressed,
    Translated(Result<Box<CompositionReceipt>, SharedError>),

    InjectTargetSelected(WorkflowTab, InjectTarget),
    InjectAutoSubmitToggled(WorkflowTab),
    InjectPressed(WorkflowTab),
    Injected(WorkflowTab, u64, Result<Box<InjectionPlan>, SharedError>),

    PrefOutputLanguageEdited(String),
    PrefInjectTargetSelected(InjectTarget),
    PrefCaptureSecondsSelected(u32),
    PrefGlassOpacityChanged(f32),
    PrefReduceTransparencyToggled,
    SystemA11yChanged(SystemA11y),

    HistoryRefreshPressed,
    SessionsListed(u64, Result<Vec<CaptureSession>, SharedError>),
    QuitNow,
}

impl App {
    pub fn boot() -> (Self, Task<Message>) {
        let client = Client::new(daemon_client::desktop_socket_path());
        let socket_input = client.socket_path().display().to_string();
        let (main_id, open_main) = window::open(main_window_settings());
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
            Task::perform(hotkeys::probe(), Message::PortalProbed),
            Task::perform(accessibility::probe(), Message::SystemA11yChanged),
        ]);
        (app, startup)
    }

    pub(crate) fn accessibility(&self) -> Accessibility {
        resolve_accessibility(&self.preferences, self.system_a11y)
    }

    pub fn view(&self, window: window::Id) -> Element<'_, Message> {
        views::window_view(self, window)
    }

    pub fn subscription(&self) -> Subscription<Message> {
        let keys = keyboard::listen().filter_map(|event| match event {
            keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(keyboard::key::Named::F9),
                ..
            } => Some(Message::HotkeyReceived),
            _ => None,
        });
        let mut subs = vec![
            keys,
            Subscription::run(subscriptions::hotkey_stream),
            Subscription::run(subscriptions::accessibility_stream),
            Subscription::run_with(
                (self.client.socket_path().to_path_buf(), self.daemon_epoch),
                |target: &(std::path::PathBuf, u64)| {
                    subscriptions::sse_stream(target.0.clone(), target.1)
                },
            ),
            window::close_events().map(Message::WindowClosed),
        ];
        if self.accessibility().animations_enabled() {
            subs.push(window::frames().map(Message::Tick));
        }
        #[cfg(feature = "tray")]
        subs.push(Subscription::run(subscriptions::tray_stream));
        Subscription::batch(subs)
    }
}

/// Returns `true` when `provider` is an ASR backend supported by runtime dispatch.
pub(crate) fn is_dispatchable_asr_provider(provider: &ProviderDescriptor) -> bool {
    provider.kind == ProviderKind::Asr
        && is_supported_transcribe_provider_id(Some(provider.id.as_str()))
}

/// Returns `true` when `provider_id` supports `translate_to_english`.
pub(crate) fn dictation_translate_supported(provider_id: Option<&str>) -> bool {
    provider_supports_asr_translation(provider_id)
}

pub(super) fn validated_dictation_translate(provider_id: Option<&str>, requested: bool) -> bool {
    requested && dictation_translate_supported(provider_id)
}

pub(super) fn reconcile_selected_asr_provider(
    selected: &mut Option<String>,
    providers: Option<&[ProviderDescriptor]>,
) {
    let should_clear = match selected.as_deref() {
        None => false,
        Some(selected_id) => !providers.is_some_and(|list| {
            list.iter().any(|provider| {
                is_dispatchable_asr_provider(provider) && provider.id == selected_id
            })
        }),
    };
    if should_clear {
        *selected = None;
    }
}

pub(super) fn capture_is_active(capture_in_flight: bool, stage: SessionStage) -> bool {
    capture_in_flight
        || matches!(
            stage,
            SessionStage::Starting | SessionStage::Listening | SessionStage::Stopping
        )
}

#[derive(Debug, PartialEq, Eq)]
pub(super) enum QuitPlan {
    Exit,
    StopThenExit(Uuid),
    AwaitPendingReply,
}

pub(super) fn quit_plan(quitting: bool, stage: SessionStage, session_id: Option<Uuid>) -> QuitPlan {
    if quitting {
        return QuitPlan::Exit;
    }
    match stage {
        SessionStage::Listening => match session_id {
            Some(id) => QuitPlan::StopThenExit(id),
            None => QuitPlan::Exit,
        },
        SessionStage::Starting | SessionStage::Stopping => QuitPlan::AwaitPendingReply,
        _ => QuitPlan::Exit,
    }
}

pub(super) fn quit_watchdog() -> Task<Message> {
    Task::perform(
        async {
            tokio::time::sleep(Duration::from_secs(5)).await;
        },
        |_| Message::QuitNow,
    )
}
