//! Dictation session, one-shot capture, and capture HUD handlers for [`App`].

use iced::{Task, window};
use voicelayer_core::{DictationCaptureRequest, StartDictationRequest, TriggerKind};

use crate::state::{DaemonStatus, SessionStage, language_profile_from_input};

use super::{
    App, Message, capture_is_active, dictation_translate_supported, hud_window_settings,
    quit_watchdog, validated_dictation_translate,
};

impl App {
    pub(super) fn handle_dictation(&mut self, message: Message) -> Task<Message> {
        match message {
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
                if self.quitting {
                    return iced::exit();
                }
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }
            Message::SessionStopped(_) if self.quitting => iced::exit(),
            Message::SessionStopped(Ok(result)) => {
                self.error = self.session.apply_capture(*result);
                self.sync_hud()
            }
            Message::SessionStopped(Err(error)) => {
                self.session.mark_failed();
                self.error = Some((*error).clone());
                self.sync_hud()
            }
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
                self.dictation_translate =
                    dictation_translate_supported(self.dictation_provider.as_deref())
                        && !self.dictation_translate;
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
            _ => Task::none(),
        }
    }

    pub(super) fn capture_active(&self) -> bool {
        capture_is_active(self.capture_in_flight, self.session.stage)
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

    fn sync_hud(&mut self) -> Task<Message> {
        match (self.capture_active(), self.hud_window) {
            (true, None) => {
                let (id, open) = window::open(hud_window_settings());
                self.hud_window = Some(id);
                open.map(|_| Message::HudWindowOpened)
            }
            (false, Some(id)) => {
                self.hud_window = None;
                window::close(id)
            }
            _ => Task::none(),
        }
    }

    fn start_session(&mut self) -> Task<Message> {
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
        let request = StartDictationRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            translate_to_english: validated_dictation_translate(
                self.dictation_provider.as_deref(),
                self.dictation_translate,
            ),
            keep_audio: false,
            segmentation: self.dictation_segmentation.to_mode(),
            provider_id: self.dictation_provider.clone(),
        };
        Task::perform(
            async move { client.start_dictation(request).await },
            Message::SessionStarted,
        )
    }

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
        let request = DictationCaptureRequest {
            trigger: TriggerKind::TrayButton,
            language_profile: language_profile_from_input(&self.dictation_language),
            duration_seconds: self.preferences.capture_seconds,
            translate_to_english: validated_dictation_translate(
                self.dictation_provider.as_deref(),
                self.dictation_translate,
            ),
            keep_audio: false,
            provider_id: self.dictation_provider.clone(),
        };
        let client = self.client.clone();
        Task::perform(
            async move { client.capture(request).await.map(Box::new) },
            Message::Captured,
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
}
