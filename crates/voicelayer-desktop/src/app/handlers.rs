//! Top-level message dispatch and lifecycle, daemon, preference, and history handlers.

use std::path::PathBuf;

use iced::Task;
#[cfg(feature = "tray")]
use iced::window;

use crate::config;
use crate::daemon_client::Client;
use crate::hotkeys::PortalProbe;
use crate::launcher::spawn_daemon;
use crate::state::{DaemonStatus, HotkeyStatus, Session, WorkflowTab};

use super::{
    App, Message, QuitPlan, dictation_translate_supported, quit_plan, quit_watchdog,
    reconcile_selected_asr_provider,
};

impl App {
    pub fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::MainWindowOpened(_)
            | Message::HudWindowOpened
            | Message::WindowClosed(_)
            | Message::TabSelected(_)
            | Message::Tick(_)
            | Message::QuitNow => self.handle_lifecycle(message),
            #[cfg(feature = "tray")]
            Message::TrayShowRequested | Message::TrayQuitRequested => {
                self.handle_lifecycle(message)
            }
            Message::SocketPathEdited(_)
            | Message::ProbeDaemonPressed
            | Message::DaemonProbed(_, _)
            | Message::StartDaemonPressed
            | Message::DaemonSpawnResult(_, _)
            | Message::RefreshProvidersPressed
            | Message::ProvidersListed(_, _)
            | Message::PortalProbed(_)
            | Message::EventReceived(_, _) => self.handle_daemon(message),
            Message::HotkeyReceived
            | Message::StartPressed
            | Message::StopPressed
            | Message::SessionStarted(_)
            | Message::SessionStopped(_)
            | Message::DictationProviderSelected(_)
            | Message::DictationLanguageEdited(_)
            | Message::DictationSegmentationSelected(_)
            | Message::DictationTranslateToggled
            | Message::CapturePressed
            | Message::Captured(_) => self.handle_dictation(message),
            Message::ComposePromptEdited(_)
            | Message::ComposeArchetypeSelected(_)
            | Message::ComposeLanguageEdited(_)
            | Message::ComposeSubmitPressed
            | Message::Composed(_)
            | Message::RewriteSourceEdited(_)
            | Message::RewriteStyleSelected(_)
            | Message::RewriteLanguageEdited(_)
            | Message::RewriteSubmitPressed
            | Message::Rewritten(_)
            | Message::TranslateSourceEdited(_)
            | Message::TranslateTargetEdited(_)
            | Message::TranslateSubmitPressed
            | Message::Translated(_)
            | Message::InjectTargetSelected(_, _)
            | Message::InjectAutoSubmitToggled(_)
            | Message::InjectPressed(_)
            | Message::Injected(_, _, _) => self.handle_workflows(message),
            Message::PrefOutputLanguageEdited(_)
            | Message::PrefInjectTargetSelected(_)
            | Message::PrefCaptureSecondsSelected(_)
            | Message::PrefGlassOpacityChanged(_)
            | Message::PrefReduceTransparencyToggled
            | Message::SystemA11yChanged(_) => self.handle_preferences(message),
            Message::HistoryRefreshPressed | Message::SessionsListed(_, _) => {
                self.handle_history(message)
            }
        }
    }

    fn handle_lifecycle(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::MainWindowOpened(id) => {
                self.main_window = Some(id);
                Task::none()
            }
            Message::HudWindowOpened => Task::none(),
            Message::WindowClosed(id) => {
                if self.main_window == Some(id) {
                    self.request_quit()
                } else {
                    if self.hud_window == Some(id) {
                        self.hud_window = None;
                    }
                    Task::none()
                }
            }
            #[cfg(feature = "tray")]
            Message::TrayShowRequested => match self.main_window {
                Some(id) => window::gain_focus(id),
                None => Task::none(),
            },
            #[cfg(feature = "tray")]
            Message::TrayQuitRequested => self.request_quit(),
            Message::TabSelected(tab) => {
                self.tab = tab;
                match tab {
                    WorkflowTab::Providers => self.refresh_providers(),
                    WorkflowTab::Doctor => probe_health(self.client.clone(), self.daemon_epoch),
                    WorkflowTab::History => self.refresh_sessions(),
                    WorkflowTab::Dictation if self.providers.is_none() => self.refresh_providers(),
                    _ => Task::none(),
                }
            }
            Message::Tick(now) => {
                if let Some(prev) = self.last_frame {
                    self.elapsed += now.saturating_duration_since(prev).as_secs_f32();
                }
                self.last_frame = Some(now);
                Task::none()
            }
            Message::QuitNow => iced::exit(),
            _ => Task::none(),
        }
    }

    fn handle_daemon(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::SocketPathEdited(next) => self.retarget_socket(next),
            Message::ProbeDaemonPressed => {
                self.daemon = DaemonStatus::Probing;
                self.error = None;
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
            Message::EventReceived(epoch, _) if epoch != self.daemon_epoch => Task::none(),
            Message::EventReceived(_, event) => {
                self.last_event = Some(event);
                Task::none()
            }
            _ => Task::none(),
        }
    }

    fn handle_preferences(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::PrefOutputLanguageEdited(value) => {
                self.compose.language.sync_default(&value);
                self.rewrite.language.sync_default(&value);
                self.translate.target.sync_default(&value);
                self.preferences.default_output_language = value;
                Task::none()
            }
            Message::PrefInjectTargetSelected(target) => {
                self.preferences.default_inject_target = target;
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
            _ => Task::none(),
        }
    }

    fn handle_history(&mut self, message: Message) -> Task<Message> {
        match message {
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
            _ => Task::none(),
        }
    }

    fn retarget_socket(&mut self, next: String) -> Task<Message> {
        let socket_path = PathBuf::from(next.trim());
        if socket_path == self.client.socket_path() {
            self.socket_input = next;
            return Task::none();
        }
        if self.daemon_request_active() {
            self.error = Some(
                "Finish the active daemon request before changing the socket path.".to_owned(),
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

    fn refresh_providers(&mut self) -> Task<Message> {
        self.providers_error = None;
        list_providers(self.client.clone(), self.daemon_epoch)
    }

    fn refresh_sessions(&mut self) -> Task<Message> {
        self.sessions_error = None;
        fetch_sessions(self.client.clone(), self.daemon_epoch)
    }

    fn daemon_request_active(&self) -> bool {
        self.capture_active()
            || [&self.compose.job, &self.rewrite.job, &self.translate.job]
                .iter()
                .any(|job| job.submitting || job.injecting)
    }

    pub(super) fn request_quit(&mut self) -> Task<Message> {
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

    fn persist_preferences(&self) {
        config::save(&config::DesktopConfig::from_preferences(&self.preferences));
    }
}

pub(super) fn probe_health(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(
        async move { client.health().await.map(Box::new) },
        move |result| Message::DaemonProbed(daemon_epoch, result),
    )
}

pub(super) fn list_providers(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(async move { client.providers().await }, move |result| {
        Message::ProvidersListed(daemon_epoch, result)
    })
}

fn fetch_sessions(client: Client, daemon_epoch: u64) -> Task<Message> {
    Task::perform(async move { client.sessions().await }, move |result| {
        Message::SessionsListed(daemon_epoch, result)
    })
}
