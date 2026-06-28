//! Rendering for the desktop shell's main window: the navigation shell and the
//! per-workflow content panels.
//!
//! These are free functions over `&App` rather than `App` methods so the view
//! layer stays a separate, replaceable concern from the [`crate::app`]
//! controller — the seam the plan opens up for the full `views/` set in P3.
//! The two-layer Liquid Glass rule holds throughout: the backdrop is the only
//! opaque surface, glass cards float directly on it as siblings (the sidebar
//! and each content card), and never nest glass-on-glass — interactive
//! materials (capsules, fields) are the only glass allowed to rest on a card.

use iced::widget::{Column, Space, column, container, row, scrollable, stack, text};
use iced::{Element, Length, window};

use voicelayer_core::{ProviderDescriptor, ProviderKind};
use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::components::{self, Capsule, ProviderChoice, Tone};
use crate::glass;
use crate::hud;
use crate::state::{
    DaemonStatus, SegChoice, SessionStage, WorkflowTab, render_daemon_status, render_session_stage,
};
use crate::theme;
use crate::workflows;

/// Route a window id to its content: the main navigation window, the capture HUD
/// overlay, or — for an id we do not recognize (e.g. a settings window before its
/// view lands) — a bare backdrop so nothing flashes.
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
    // Three layers, bottom to top: an opaque gradient backdrop (the graceful
    // fallback tier, and what shows if the GPU shader cannot draw), the animated
    // wgpu glass material, then the transparent content shell whose glass cards
    // lens the material behind them.
    stack![
        container(Space::new().width(Length::Fill).height(Length::Fill))
            .width(Length::Fill)
            .height(Length::Fill)
            .style(|_theme| theme::backdrop()),
        glass::background(app.elapsed, app.accessibility().opacity),
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
    let p = theme::palette();
    let wordmark = text("VoiceLayer")
        .font(theme::font(Weight::Bold))
        .size(tokens::text::TITLE)
        .color(theme::color(p.text_primary));

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
        DaemonStatus::Unknown | DaemonStatus::Probing => {
            components::capsule("Probing…", Capsule::Ghost)
                .width(Length::Fill)
                .into()
        }
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
        WorkflowTab::Dictation => dictation_panel(app),
        WorkflowTab::Compose => workflows::compose_panel(app),
        WorkflowTab::Rewrite => workflows::rewrite_panel(app),
        WorkflowTab::Translate => workflows::translate_panel(app),
        WorkflowTab::Providers => providers_panel(app),
        WorkflowTab::Doctor => doctor_panel(app),
        WorkflowTab::History => workflows::history_panel(app),
        WorkflowTab::Settings => workflows::settings_panel(app),
    };
    scrollable(panel)
        .width(Length::Fill)
        .height(Length::Fill)
        .into()
}

fn dictation_panel(app: &App) -> Element<'_, Message> {
    let p = theme::palette();
    let session_tone = match app.session.stage {
        SessionStage::Listening => Tone::Accent,
        SessionStage::Completed => Tone::Success,
        SessionStage::Failed => Tone::Danger,
        SessionStage::Idle | SessionStage::Starting | SessionStage::Stopping => Tone::Neutral,
    };
    let header = row![
        text("Dictation")
            .font(theme::font(Weight::Bold))
            .size(tokens::text::LARGE_TITLE)
            .color(theme::color(p.text_primary)),
        components::badge(
            &format!("Session · {}", render_session_stage(app.session.stage)),
            session_tone,
        ),
    ]
    .spacing(tokens::space::MD);

    let hotkey_line: Element<'_, Message> = if app.hotkey.portal_available {
        text("Global shortcut registered via the XDG portal — VoiceLayer: Toggle dictation")
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_secondary))
            .into()
    } else {
        text(format!(
            "XDG global shortcuts unavailable ({}). The F9 hotkey works only while this \
             window is focused.",
            app.hotkey
                .portal_error
                .as_deref()
                .unwrap_or("unknown reason"),
        ))
        .size(tokens::text::CAPTION)
        .color(theme::color(p.text_secondary))
        .into()
    };

    let primary: Element<'_, Message> = match app.session.stage {
        SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => {
            components::capsule("Start dictation  ·  F9", Capsule::Primary)
                .on_press(Message::StartPressed)
                .into()
        }
        SessionStage::Listening => components::capsule("Stop dictation  ·  F9", Capsule::Primary)
            .on_press(Message::StopPressed)
            .into(),
        SessionStage::Starting | SessionStage::Stopping => {
            components::capsule(render_session_stage(app.session.stage), Capsule::Ghost).into()
        }
    };

    let mut card_body = column![header, hotkey_line, primary].spacing(tokens::space::LG);

    if let Some(language) = &app.session.detected_language {
        card_body = card_body.push(components::badge(
            &format!("Detected language · {language}"),
            Tone::Neutral,
        ));
    }
    if let Some(event) = &app.last_event {
        card_body = card_body.push(
            text(format!(
                "Last event · {} — {}",
                event.event_type, event.message
            ))
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_secondary)),
        );
    }
    if let Some(error) = &app.error {
        card_body = card_body.push(
            text(format!("Error: {error}"))
                .size(tokens::text::CAPTION)
                .color(theme::color(p.danger)),
        );
    }
    if !app.session.notes.is_empty() {
        let notes = app
            .session
            .notes
            .iter()
            .map(|note| format!("• {note}"))
            .collect::<Vec<_>>()
            .join("\n");
        card_body = card_body.push(
            text(notes)
                .size(tokens::text::CAPTION)
                .color(theme::color(p.text_secondary)),
        );
    }

    let mut panel = column![components::card(card_body, app.accessibility()).width(Length::Fill)]
        .spacing(tokens::space::LG);

    panel = panel.push(capture_options(app));

    if let Some(transcript) = app.session.transcript.as_deref().filter(|t| !t.is_empty()) {
        panel = panel.push(
            components::card(
                column![
                    text("Transcript")
                        .font(theme::font(Weight::Semibold))
                        .size(tokens::text::TITLE)
                        .color(theme::color(p.text_primary)),
                    text(transcript.to_owned())
                        .size(tokens::text::BODY)
                        .color(theme::color(p.text_primary)),
                ]
                .spacing(tokens::space::SM),
                app.accessibility(),
            )
            .width(Length::Fill),
        );
    }

    panel.into()
}

fn providers_panel(app: &App) -> Element<'_, Message> {
    let p = theme::palette();
    let header = row![
        text("Providers")
            .font(theme::font(Weight::Bold))
            .size(tokens::text::LARGE_TITLE)
            .color(theme::color(p.text_primary)),
        components::capsule("Refresh", Capsule::Ghost).on_press(Message::RefreshProvidersPressed),
    ]
    .spacing(tokens::space::MD);

    let listing: Element<'_, Message> = match &app.providers {
        Some(list) if !list.is_empty() => {
            let mut rows = Column::new().spacing(tokens::space::SM);
            for descriptor in list {
                rows = rows.push(provider_row(descriptor));
            }
            rows.into()
        }
        Some(_) => text("No providers registered.")
            .size(tokens::text::BODY)
            .color(theme::color(p.text_secondary))
            .into(),
        None => text("Loading providers…")
            .size(tokens::text::BODY)
            .color(theme::color(p.text_secondary))
            .into(),
    };

    column![
        components::card(
            column![header, listing].spacing(tokens::space::LG),
            app.accessibility()
        )
        .width(Length::Fill)
    ]
    .into()
}

fn doctor_panel(app: &App) -> Element<'_, Message> {
    let p = theme::palette();
    let header = row![
        text("Doctor")
            .font(theme::font(Weight::Bold))
            .size(tokens::text::LARGE_TITLE)
            .color(theme::color(p.text_primary)),
        components::capsule("Re-probe", Capsule::Ghost).on_press(Message::ProbeDaemonPressed),
    ]
    .spacing(tokens::space::MD);

    let body: Element<'_, Message> = match &app.health {
        Some(health) => {
            let worker = &health.worker;
            column![
                info_row("Status", health.status.clone()),
                info_row("Version", health.version.clone()),
                info_row("Socket", health.socket_path.clone()),
                text("Worker")
                    .font(theme::font(Weight::Semibold))
                    .size(tokens::text::TITLE)
                    .color(theme::color(p.text_primary)),
                info_row("Worker status", worker.status.clone()),
                info_row("ASR configured", yes_no(worker.asr_configured)),
                info_row("ASR model", optional(worker.asr_model_path.as_deref())),
                info_row("LLM configured", yes_no(worker.llm_configured)),
                info_row("LLM endpoint", optional(worker.llm_endpoint.as_deref())),
                info_row("LLM reachable", yes_no(worker.llm_reachable)),
                info_row(
                    "Shortcuts portal",
                    yes_no(worker.global_shortcuts_portal_available),
                ),
            ]
            .spacing(tokens::space::SM)
            .into()
        }
        None => text("Probing daemon…")
            .size(tokens::text::BODY)
            .color(theme::color(p.text_secondary))
            .into(),
    };

    column![
        components::card(
            column![header, body].spacing(tokens::space::LG),
            app.accessibility()
        )
        .width(Length::Fill)
    ]
    .into()
}

fn provider_row(descriptor: &ProviderDescriptor) -> Element<'static, Message> {
    let p = theme::palette();
    let locality = if descriptor.local {
        components::badge("local", Tone::Success)
    } else {
        components::badge("remote", Tone::Warning)
    };
    row![
        text(descriptor.id.clone())
            .font(theme::font(Weight::Semibold))
            .size(tokens::text::BODY)
            .color(theme::color(p.text_primary))
            .width(Length::Fill),
        components::badge(provider_kind_label(&descriptor.kind), Tone::Neutral),
        text(descriptor.transport.clone())
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_secondary)),
        locality,
    ]
    .spacing(tokens::space::SM)
    .into()
}

fn provider_kind_label(kind: &ProviderKind) -> &'static str {
    match kind {
        ProviderKind::Asr => "ASR",
        ProviderKind::Llm => "LLM",
        ProviderKind::Tts => "TTS",
        ProviderKind::HostAdapter => "Host",
    }
}

/// The dictation "Capture options" card: per-session ASR provider, input
/// language, segmentation, and translate toggle, plus the one-shot fixed-window
/// capture. A sibling card to the streaming-status card above it — never nested.
fn capture_options(app: &App) -> Element<'_, Message> {
    let p = theme::palette();
    let asr_ids: Vec<String> = app
        .providers
        .as_deref()
        .unwrap_or(&[])
        .iter()
        .filter(|descriptor| matches!(descriptor.kind, ProviderKind::Asr))
        .map(|descriptor| descriptor.id.clone())
        .collect();

    let provider_picker = components::picker(
        "Automatic (daemon default)",
        components::provider_choices(&asr_ids),
        Some(ProviderChoice(app.dictation_provider.clone())),
        |choice| Message::DictationProviderSelected(choice.0),
    );
    let language = components::field(
        "Input language — blank auto-detects",
        &app.dictation_language,
        Message::DictationLanguageEdited,
    );
    let segmentation = components::picker(
        "Segmentation",
        SegChoice::ALL.to_vec(),
        Some(app.dictation_segmentation),
        Message::DictationSegmentationSelected,
    );
    let translate = components::capsule(
        if app.dictation_translate {
            "Translate to English: on"
        } else {
            "Translate to English: off"
        },
        if app.dictation_translate {
            Capsule::Secondary
        } else {
            Capsule::Ghost
        },
    )
    .on_press(Message::DictationTranslateToggled);

    // One-shot capture is offered only when no streaming session is mid-flight.
    let capturable = matches!(
        app.session.stage,
        SessionStage::Idle | SessionStage::Completed | SessionStage::Failed
    ) && !app.capture_in_flight;
    let capture_label = format!("One-shot capture  ·  {} s", app.preferences.capture_seconds);
    let capture: Element<'_, Message> = if app.capture_in_flight {
        components::capsule("Capturing…", Capsule::Ghost).into()
    } else if capturable {
        components::capsule(&capture_label, Capsule::Secondary)
            .on_press(Message::CapturePressed)
            .into()
    } else {
        components::capsule(&capture_label, Capsule::Ghost).into()
    };

    let header = text("Capture options")
        .font(theme::font(Weight::Semibold))
        .size(tokens::text::TITLE)
        .color(theme::color(p.text_primary));

    components::card(
        column![
            header,
            field_group("ASR provider", provider_picker),
            field_group("Input language", language),
            field_group("Segmentation", segmentation),
            translate,
            capture,
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill)
    .into()
}

/// A captioned control: the label sits above its input.
fn field_group<'a>(label: &str, control: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    let p = theme::palette();
    column![
        text(label.to_owned())
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_secondary)),
        control.into(),
    ]
    .spacing(tokens::space::XS)
    .into()
}

fn info_row(label: &str, value: String) -> Element<'static, Message> {
    let p = theme::palette();
    row![
        text(label.to_owned())
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_secondary))
            .width(Length::Fixed(168.0)),
        text(value)
            .size(tokens::text::CAPTION)
            .color(theme::color(p.text_primary)),
    ]
    .spacing(tokens::space::SM)
    .into()
}

fn yes_no(value: bool) -> String {
    if value { "yes" } else { "no" }.to_owned()
}

fn optional(value: Option<&str>) -> String {
    value.unwrap_or("—").to_owned()
}
