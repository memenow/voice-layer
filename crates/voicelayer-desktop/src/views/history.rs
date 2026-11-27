//! Daemon session history panel.

use iced::widget::{Column, column, row, text};
use iced::{Element, Length};
use voicelayer_core::{CaptureSession, SessionMode, SessionState, TriggerKind};
use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::components::{self, Capsule, Tone};
use crate::theme;

use super::{error_text, heading, primary, secondary};

pub(super) fn panel(app: &App) -> Element<'_, Message> {
    let header = row![
        heading("History"),
        components::capsule("Refresh", Capsule::Ghost).on_press(Message::HistoryRefreshPressed),
    ]
    .spacing(tokens::space::MD);

    let listing: Element<'_, Message> =
        match history_empty_state(app.sessions.as_deref(), app.sessions_error.as_deref()) {
            None => {
                let mut rows = Column::new().spacing(tokens::space::SM);
                for session in app.sessions.as_deref().unwrap_or_default() {
                    rows = rows.push(session_row(session));
                }
                rows.into()
            }
            Some(HistoryEmptyState::Empty) => text("No capture sessions yet.")
                .size(tokens::text::BODY)
                .color(secondary())
                .into(),
            Some(HistoryEmptyState::Loading) => text("Loading sessions...")
                .size(tokens::text::BODY)
                .color(secondary())
                .into(),
            Some(HistoryEmptyState::Failed(error)) => error_text(&error),
        };

    let mut content = column![header, listing].spacing(tokens::space::LG);
    if app.sessions.is_some()
        && let Some(error) = &app.sessions_error
    {
        content = content.push(error_text(error));
    }

    column![components::card(content, app.accessibility()).width(Length::Fill)].into()
}

#[derive(Debug, PartialEq, Eq)]
enum HistoryEmptyState {
    Loading,
    Empty,
    Failed(String),
}

fn history_empty_state(
    sessions: Option<&[CaptureSession]>,
    error: Option<&str>,
) -> Option<HistoryEmptyState> {
    match sessions {
        Some([]) => Some(HistoryEmptyState::Empty),
        Some(_) => None,
        None => error
            .map(|error| HistoryEmptyState::Failed(error.to_owned()))
            .or(Some(HistoryEmptyState::Loading)),
    }
}

fn session_row(session: &CaptureSession) -> Element<'static, Message> {
    let id = session.session_id.to_string();
    let short: String = id.chars().take(8).collect();
    let state_tone = match session.state {
        SessionState::Completed => Tone::Success,
        SessionState::Failed => Tone::Danger,
        SessionState::Listening
        | SessionState::Transcribing
        | SessionState::Previewing
        | SessionState::AwaitingConfirmation => Tone::Accent,
        SessionState::Idle => Tone::Neutral,
    };
    row![
        text(short)
            .font(theme::font(Weight::Semibold))
            .size(tokens::text::BODY)
            .color(primary())
            .width(Length::Fixed(96.0)),
        components::badge(session_mode_label(&session.mode), Tone::Neutral),
        components::badge(session_state_label(&session.state), state_tone),
        text(trigger_label(&session.trigger))
            .size(tokens::text::CAPTION)
            .color(secondary()),
    ]
    .spacing(tokens::space::SM)
    .into()
}

fn session_mode_label(mode: &SessionMode) -> &'static str {
    match mode {
        SessionMode::Dictation => "dictation",
        SessionMode::Compose => "compose",
        SessionMode::Rewrite => "rewrite",
        SessionMode::Translate => "translate",
    }
}

fn session_state_label(state: &SessionState) -> &'static str {
    match state {
        SessionState::Idle => "idle",
        SessionState::Listening => "listening",
        SessionState::Transcribing => "transcribing",
        SessionState::Previewing => "previewing",
        SessionState::AwaitingConfirmation => "awaiting confirmation",
        SessionState::Completed => "completed",
        SessionState::Failed => "failed",
    }
}

fn trigger_label(trigger: &TriggerKind) -> &'static str {
    match trigger {
        TriggerKind::PushToTalk => "push-to-talk",
        TriggerKind::Toggle => "toggle",
        TriggerKind::Cli => "cli",
        TriggerKind::Tui => "tui",
        TriggerKind::TrayButton => "tray",
    }
}

#[cfg(test)]
mod tests {
    use super::{HistoryEmptyState, history_empty_state};

    #[test]
    fn history_empty_state_shows_scoped_error_instead_of_loading() {
        assert_eq!(
            history_empty_state(None, Some("history unavailable")),
            Some(HistoryEmptyState::Failed("history unavailable".to_owned())),
        );
        assert_eq!(
            history_empty_state(None, None),
            Some(HistoryEmptyState::Loading),
        );
    }
}
