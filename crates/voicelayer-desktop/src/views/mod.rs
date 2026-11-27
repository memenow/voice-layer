//! Desktop shell views: main window frame, sidebar navigation, and shared UI helpers.

mod compose;
mod diagnostics;
mod dictation;
mod history;
mod settings;

use iced::widget::{Column, Space, column, container, row, scrollable, stack, text};
use iced::{Element, Length, window};
use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::backdrop;
use crate::components::{self, Capsule, Tone};
use crate::hud;
use crate::state::{DaemonStatus, WorkflowTab, render_daemon_status};
use crate::theme;

/// Route a window id to the main navigation shell or floating capture HUD.
pub(crate) fn window_view(app: &App, window: window::Id) -> Element<'_, Message> {
    if app.main_window == Some(window) {
        main_view(app)
    } else if app.hud_window == Some(window) {
        hud::view(app)
    } else {
        container(Space::new().width(Length::Fill).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| theme::backdrop())
            .into()
    }
}

fn main_view(app: &App) -> Element<'_, Message> {
    let shell = row![sidebar(app), content(app)]
        .spacing(tokens::space::LG)
        .height(Length::Fill);
    stack![
        container(Space::new().width(Length::Fill).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| theme::backdrop()),
        backdrop::background(app.elapsed, app.accessibility().opacity),
        container(shell)
            .padding(theme::pad(tokens::space::XL, tokens::space::XL))
            .width(Length::Fill)
            .height(Length::Fill),
    ]
    .width(Length::Fill)
    .height(Length::Fill)
    .into()
}

fn sidebar(app: &App) -> Element<'_, Message> {
    let wordmark = text("VoiceLayer")
        .font(theme::font(Weight::Bold))
        .size(tokens::text::TITLE)
        .color(primary());

    let daemon_tone = match app.daemon {
        DaemonStatus::Healthy => Tone::Success,
        DaemonStatus::Unreachable => Tone::Danger,
        DaemonStatus::Probing | DaemonStatus::Unknown => Tone::Warning,
    };
    let daemon_badge = components::badge(
        &format!("Daemon · {}", render_daemon_status(app.daemon)),
        daemon_tone,
    );

    let mut nav = Column::new().spacing(tokens::space::XS);
    for tab in WorkflowTab::ALL {
        nav = nav.push(nav_button(app, tab));
    }

    let body = column![
        wordmark,
        daemon_badge,
        nav,
        Space::new().height(Length::Fill),
        connection_controls(app),
    ]
    .spacing(tokens::space::MD)
    .height(Length::Fill);

    components::card(body, app.accessibility())
        .width(Length::Fixed(232.0))
        .height(Length::Fill)
        .into()
}

fn nav_button(app: &App, tab: WorkflowTab) -> Element<'_, Message> {
    let kind = if app.tab == tab {
        Capsule::Secondary
    } else {
        Capsule::Ghost
    };
    components::capsule(tab.label(), kind)
        .width(Length::Fill)
        .on_press(Message::TabSelected(tab))
        .into()
}

fn connection_controls(app: &App) -> Element<'_, Message> {
    let field = components::field("Socket path", &app.socket_input, Message::SocketPathEdited);
    let control: Element<'_, Message> = match app.daemon {
        DaemonStatus::Unknown => components::capsule("Probe daemon", Capsule::Ghost)
            .width(Length::Fill)
            .on_press(Message::ProbeDaemonPressed)
            .into(),
        DaemonStatus::Probing => components::capsule("Probing...", Capsule::Ghost)
            .width(Length::Fill)
            .into(),
        DaemonStatus::Healthy => components::capsule("Re-probe", Capsule::Ghost)
            .width(Length::Fill)
            .on_press(Message::ProbeDaemonPressed)
            .into(),
        DaemonStatus::Unreachable => column![
            components::capsule("Start daemon", Capsule::Secondary)
                .width(Length::Fill)
                .on_press(Message::StartDaemonPressed),
            components::capsule("Retry probe", Capsule::Ghost)
                .width(Length::Fill)
                .on_press(Message::ProbeDaemonPressed),
        ]
        .spacing(tokens::space::XS)
        .into(),
    };
    column![field, control].spacing(tokens::space::SM).into()
}

fn content(app: &App) -> Element<'_, Message> {
    let panel = match app.tab {
        WorkflowTab::Dictation => dictation::panel(app),
        WorkflowTab::Compose => compose::compose_panel(app),
        WorkflowTab::Rewrite => compose::rewrite_panel(app),
        WorkflowTab::Translate => compose::translate_panel(app),
        WorkflowTab::Providers => diagnostics::providers_panel(app),
        WorkflowTab::Doctor => diagnostics::doctor_panel(app),
        WorkflowTab::History => history::panel(app),
        WorkflowTab::Settings => settings::panel(app),
    };
    scrollable(panel)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

pub(super) fn heading(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .font(theme::font(Weight::Bold))
        .size(tokens::text::LARGE_TITLE)
        .color(primary())
}

pub(super) fn section_title(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .font(theme::font(Weight::Semibold))
        .size(tokens::text::TITLE)
        .color(primary())
}

pub(super) fn caption(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .size(tokens::text::CAPTION)
        .color(secondary())
}

pub(super) fn field_group<'a>(
    label: &str,
    control: impl Into<Element<'a, Message>>,
) -> Element<'a, Message> {
    column![caption(label), control.into()]
        .spacing(tokens::space::XS)
        .into()
}

pub(super) fn info_row(label: &str, value: String, label_width: f32) -> Element<'static, Message> {
    row![
        text(label.to_owned())
            .size(tokens::text::CAPTION)
            .color(secondary())
            .width(Length::Fixed(label_width)),
        text(value).size(tokens::text::CAPTION).color(primary()),
    ]
    .spacing(tokens::space::SM)
    .into()
}

pub(super) fn submit_capsule(
    label: &str,
    busy: bool,
    message: Message,
) -> Element<'static, Message> {
    if busy {
        components::capsule("Working...", Capsule::Ghost).into()
    } else {
        components::capsule(label, Capsule::Primary)
            .on_press(message)
            .into()
    }
}

pub(super) fn error_text(message: &str) -> Element<'static, Message> {
    text(format!("Error: {message}"))
        .size(tokens::text::CAPTION)
        .color(danger())
        .into()
}

pub(super) fn primary() -> iced::Color {
    theme::color(theme::palette().text_primary)
}

pub(super) fn secondary() -> iced::Color {
    theme::color(theme::palette().text_secondary)
}

pub(super) fn danger() -> iced::Color {
    theme::color(theme::palette().danger)
}
