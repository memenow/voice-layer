//! Provider catalog and daemon health diagnostic panels.

use iced::widget::{Column, column, row, text};
use iced::{Element, Length};
use voicelayer_core::{ProviderDescriptor, ProviderKind};
use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::components::{self, Capsule, Tone};
use crate::state::DaemonStatus;
use crate::theme;

use super::{danger, error_text, heading, info_row, primary, secondary, section_title};

pub(super) fn providers_panel(app: &App) -> Element<'_, Message> {
    let header = row![
        heading("Providers"),
        components::capsule("Refresh", Capsule::Ghost).on_press(Message::RefreshProvidersPressed),
    ]
    .spacing(tokens::space::MD);

    let listing: Element<'_, Message> =
        match provider_empty_state(app.providers.as_deref(), app.providers_error.as_deref()) {
            None => {
                let mut rows = Column::new().spacing(tokens::space::SM);
                for descriptor in app.providers.as_deref().unwrap_or_default() {
                    rows = rows.push(provider_row(descriptor));
                }
                rows.into()
            }
            Some(ProviderEmptyState::Empty) => text("No providers registered.")
                .size(tokens::text::BODY)
                .color(secondary())
                .into(),
            Some(ProviderEmptyState::Loading) => text("Loading providers...")
                .size(tokens::text::BODY)
                .color(secondary())
                .into(),
            Some(ProviderEmptyState::Failed(error)) => error_text(&error),
        };

    column![
        components::card(
            column![header, listing].spacing(tokens::space::LG),
            app.accessibility(),
        )
        .width(Length::Fill)
    ]
    .into()
}

pub(super) fn doctor_panel(app: &App) -> Element<'_, Message> {
    let header = row![
        heading("Doctor"),
        components::capsule("Re-probe", Capsule::Ghost).on_press(Message::ProbeDaemonPressed),
    ]
    .spacing(tokens::space::MD);

    let body: Element<'_, Message> = match &app.health {
        Some(health) => {
            let worker = &health.worker;
            column![
                info_row("Status", health.status.clone(), 168.0),
                info_row("Version", health.version.clone(), 168.0),
                info_row("Socket", health.socket_path.clone(), 168.0),
                section_title("Worker"),
                info_row("Worker status", worker.status.clone(), 168.0),
                info_row("ASR configured", yes_no(worker.asr_configured), 168.0),
                info_row(
                    "ASR model",
                    optional(worker.asr_model_path.as_deref()),
                    168.0
                ),
                info_row("LLM configured", yes_no(worker.llm_configured), 168.0),
                info_row(
                    "LLM endpoint",
                    optional(worker.llm_endpoint.as_deref()),
                    168.0
                ),
                info_row("LLM reachable", yes_no(worker.llm_reachable), 168.0),
                info_row(
                    "Global hotkeys",
                    yes_no(worker.global_hotkeys_available),
                    168.0,
                ),
            ]
            .spacing(tokens::space::SM)
            .into()
        }
        None => match doctor_empty_state(app.daemon, app.error.as_deref()) {
            DoctorEmptyState::Probing => text("Probing daemon...")
                .size(tokens::text::BODY)
                .color(secondary())
                .into(),
            DoctorEmptyState::Failed(message) => text(message)
                .size(tokens::text::BODY)
                .color(danger())
                .into(),
        },
    };

    column![
        components::card(
            column![header, body].spacing(tokens::space::LG),
            app.accessibility(),
        )
        .width(Length::Fill)
    ]
    .into()
}

#[derive(Debug, PartialEq, Eq)]
enum ProviderEmptyState {
    Loading,
    Empty,
    Failed(String),
}

fn provider_empty_state(
    providers: Option<&[ProviderDescriptor]>,
    error: Option<&str>,
) -> Option<ProviderEmptyState> {
    if let Some(error) = error {
        return Some(ProviderEmptyState::Failed(error.to_owned()));
    }
    match providers {
        Some([]) => Some(ProviderEmptyState::Empty),
        Some(_) => None,
        None => Some(ProviderEmptyState::Loading),
    }
}

#[derive(Debug, PartialEq, Eq)]
enum DoctorEmptyState {
    Probing,
    Failed(String),
}

fn doctor_empty_state(daemon: DaemonStatus, error: Option<&str>) -> DoctorEmptyState {
    if daemon != DaemonStatus::Unreachable {
        return DoctorEmptyState::Probing;
    }
    let message = error
        .filter(|message| !message.trim().is_empty())
        .map(|message| format!("Daemon unreachable: {message}"))
        .unwrap_or_else(|| "Daemon unreachable.".to_owned());
    DoctorEmptyState::Failed(message)
}

fn provider_row(descriptor: &ProviderDescriptor) -> Element<'static, Message> {
    let locality = if descriptor.local {
        components::badge("local", Tone::Success)
    } else {
        components::badge("remote", Tone::Warning)
    };
    let kind_label = match descriptor.kind {
        ProviderKind::Asr => "ASR",
        ProviderKind::Llm => "LLM",
        ProviderKind::Tts => "TTS",
        ProviderKind::HostAdapter => "Host",
    };
    row![
        text(descriptor.id.clone())
            .font(theme::font(Weight::Semibold))
            .size(tokens::text::BODY)
            .color(primary())
            .width(Length::Fill),
        components::badge(kind_label, Tone::Neutral),
        text(descriptor.transport.clone())
            .size(tokens::text::CAPTION)
            .color(secondary()),
        locality,
    ]
    .spacing(tokens::space::SM)
    .into()
}

fn yes_no(value: bool) -> String {
    if value { "yes" } else { "no" }.to_owned()
}

fn optional(value: Option<&str>) -> String {
    value.unwrap_or("-").to_owned()
}

#[cfg(test)]
mod tests {
    use crate::state::DaemonStatus;

    use super::{DoctorEmptyState, ProviderEmptyState, doctor_empty_state, provider_empty_state};

    #[test]
    fn doctor_empty_state_distinguishes_probing_from_failure() {
        assert_eq!(
            doctor_empty_state(DaemonStatus::Probing, None),
            DoctorEmptyState::Probing,
        );
        assert_eq!(
            doctor_empty_state(DaemonStatus::Unreachable, Some("socket missing")),
            DoctorEmptyState::Failed("Daemon unreachable: socket missing".to_owned()),
        );
        assert_eq!(
            doctor_empty_state(DaemonStatus::Unreachable, None),
            DoctorEmptyState::Failed("Daemon unreachable.".to_owned()),
        );
    }

    #[test]
    fn provider_empty_state_shows_scoped_error_instead_of_loading() {
        assert_eq!(
            provider_empty_state(None, Some("provider refresh failed")),
            Some(ProviderEmptyState::Failed(
                "provider refresh failed".to_owned()
            )),
        );
        assert_eq!(
            provider_empty_state(None, None),
            Some(ProviderEmptyState::Loading),
        );
    }
}
