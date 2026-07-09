//! Content panels for the generative workflows (compose / rewrite / translate),
//! plus History and Settings.
//!
//! These render the daemon's full `/v1` surface that the dictation / providers /
//! doctor panels in [`crate::view`] do not. The two-layer Liquid Glass rule
//! holds: every panel is a column of sibling glass cards on the backdrop, and
//! the only glass nested on a card is an interactive material (field, picker,
//! capsule). The generative flows share one preview → inject sequence: a job
//! yields a [`PreviewArtifact`]; the operator reviews it, picks a target, and
//! asks the daemon to plan the injection — which prepares a paste, it does not
//! type into the focused application.

use iced::widget::{Column, column, row, text};
use iced::{Element, Length};

use voicelayer_core::{
    CaptureSession, InjectionPlan, PreviewArtifact, PreviewStatus, SessionMode, SessionState,
    TriggerKind,
};
use voicelayer_ui::a11y::Accessibility;
use voicelayer_ui::tokens::{self, Weight};

use crate::app::{App, Message};
use crate::components::{
    self, ArchetypeChoice, Capsule, RecorderChoice, StyleChoice, TargetChoice, Tone,
    archetype_choices, recorder_choices, style_choices, target_choices,
};
use crate::forms::JobStage;
use crate::state::{SystemA11y, WorkflowTab};
use crate::theme;

pub(crate) fn compose_panel(app: &App) -> Element<'_, Message> {
    let form = &app.compose;
    let card = components::card(
        column![
            labeled(
                "Spoken prompt",
                components::editor(
                    "Describe what to write — e.g. \"a polite email declining the meeting\"",
                    &form.prompt,
                    Message::ComposePromptEdited,
                ),
            ),
            labeled(
                "Archetype (optional)",
                components::picker(
                    "Automatic",
                    archetype_choices(),
                    Some(ArchetypeChoice(form.archetype.clone())),
                    |choice| Message::ComposeArchetypeSelected(choice.0),
                ),
            ),
            labeled(
                "Output language (optional)",
                components::field(
                    "Blank keeps the prompt's language",
                    &form.language,
                    Message::ComposeLanguageEdited,
                ),
            ),
            submit_capsule(
                "Compose",
                form.job.submitting,
                Message::ComposeSubmitPressed
            ),
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill);

    let panel = column![heading("Compose"), card].spacing(tokens::space::LG);
    push_job(panel, WorkflowTab::Compose, &form.job, app.accessibility()).into()
}

pub(crate) fn rewrite_panel(app: &App) -> Element<'_, Message> {
    let form = &app.rewrite;
    let card = components::card(
        column![
            labeled(
                "Source text",
                components::editor(
                    "Paste the text to rewrite…",
                    &form.source,
                    Message::RewriteSourceEdited,
                ),
            ),
            labeled(
                "Restyle",
                components::picker(
                    "Choose a style",
                    style_choices(),
                    Some(StyleChoice(form.style.clone())),
                    |choice| Message::RewriteStyleSelected(choice.0),
                ),
            ),
            labeled(
                "Output language (optional)",
                components::field(
                    "Blank keeps the source language",
                    &form.language,
                    Message::RewriteLanguageEdited,
                ),
            ),
            submit_capsule(
                "Rewrite",
                form.job.submitting,
                Message::RewriteSubmitPressed
            ),
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill);

    let panel = column![heading("Rewrite"), card].spacing(tokens::space::LG);
    push_job(panel, WorkflowTab::Rewrite, &form.job, app.accessibility()).into()
}

pub(crate) fn translate_panel(app: &App) -> Element<'_, Message> {
    let form = &app.translate;
    let card = components::card(
        column![
            labeled(
                "Source text",
                components::editor(
                    "Paste the text to translate…",
                    &form.source,
                    Message::TranslateSourceEdited,
                ),
            ),
            labeled(
                "Target language",
                components::field(
                    "e.g. Spanish, French, ja",
                    &form.target,
                    Message::TranslateTargetEdited,
                ),
            ),
            submit_capsule(
                "Translate",
                form.job.submitting,
                Message::TranslateSubmitPressed,
            ),
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill);

    let panel = column![heading("Translate"), card].spacing(tokens::space::LG);
    push_job(
        panel,
        WorkflowTab::Translate,
        &form.job,
        app.accessibility(),
    )
    .into()
}

pub(crate) fn history_panel(app: &App) -> Element<'_, Message> {
    let header = row![
        heading("History"),
        components::capsule("Refresh", Capsule::Ghost).on_press(Message::HistoryRefreshPressed),
    ]
    .spacing(tokens::space::MD);

    let listing: Element<'_, Message> = match &app.sessions {
        Some(list) if !list.is_empty() => {
            let mut rows = Column::new().spacing(tokens::space::SM);
            for session in list {
                rows = rows.push(session_row(session));
            }
            rows.into()
        }
        Some(_) => text("No capture sessions yet.")
            .size(tokens::text::BODY)
            .color(secondary())
            .into(),
        None => text("Loading sessions…")
            .size(tokens::text::BODY)
            .color(secondary())
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

pub(crate) fn settings_panel(app: &App) -> Element<'_, Message> {
    let prefs = &app.preferences;
    let a11y = app.accessibility();
    let form_card = components::card(
        column![
            labeled(
                "Default output language",
                components::field(
                    "Blank keeps the source language",
                    &prefs.default_output_language,
                    Message::PrefOutputLanguageEdited,
                ),
            ),
            labeled(
                "Default inject target",
                components::picker(
                    "Inject target",
                    target_choices(),
                    Some(TargetChoice(prefs.default_inject_target.clone())),
                    |choice| Message::PrefInjectTargetSelected(choice.0),
                ),
            ),
            labeled(
                "Recorder backend",
                components::picker(
                    "Recorder backend",
                    recorder_choices(),
                    Some(RecorderChoice(prefs.recorder_backend)),
                    |choice| Message::PrefRecorderSelected(choice.0),
                ),
            ),
            labeled(
                "One-shot capture window",
                components::picker(
                    "Capture seconds",
                    seconds_choices(),
                    Some(SecondsChoice(prefs.capture_seconds)),
                    |choice| Message::PrefCaptureSecondsSelected(choice.0),
                ),
            ),
        ]
        .spacing(tokens::space::MD),
        a11y,
    )
    .width(Length::Fill);

    let opacity_pct = (prefs.glass_opacity.clamp(0.0, 1.0) * 100.0).round() as u32;
    let transparency = components::capsule(
        if prefs.reduce_transparency {
            "Reduce transparency: on"
        } else {
            "Reduce transparency: off"
        },
        if prefs.reduce_transparency {
            Capsule::Secondary
        } else {
            Capsule::Ghost
        },
    )
    .on_press(Message::PrefReduceTransparencyToggled);
    let appearance_card = components::card(
        column![
            section_title("Appearance & accessibility"),
            labeled(
                &format!("Glass opacity · {opacity_pct}%"),
                components::opacity_slider(prefs.glass_opacity, Message::PrefGlassOpacityChanged),
            ),
            transparency,
            system_a11y_line(app.system_a11y),
        ]
        .spacing(tokens::space::MD),
        a11y,
    )
    .width(Length::Fill);

    let note = components::card(
        text(
            "Glass opacity and Reduce Transparency are saved to this shell's desktop.toml and \
             apply immediately. Increase Contrast and Reduce Motion follow the system \
             accessibility settings read from the XDG portal; GNOME exposes no reduce-transparency \
             signal, so that one stays a manual toggle. The remaining preferences are session \
             defaults; daemon-side configuration remains the CLI's domain.",
        )
        .size(tokens::text::CAPTION)
        .color(secondary()),
        a11y,
    )
    .width(Length::Fill);

    column![heading("Settings"), form_card, appearance_card, note]
        .spacing(tokens::space::LG)
        .into()
}

/// A read-only line reflecting the OS accessibility state the shell mirrors from
/// the XDG portal (Increase Contrast / Reduce Motion). Shown so the user can see
/// what was detected; these follow the system settings and are not edited here.
fn system_a11y_line(system: SystemA11y) -> Element<'static, Message> {
    let contrast = components::badge(
        if system.increase_contrast {
            "Increase contrast: on"
        } else {
            "Increase contrast: off"
        },
        if system.increase_contrast {
            Tone::Accent
        } else {
            Tone::Neutral
        },
    );
    let motion = components::badge(
        if system.reduce_motion {
            "Reduce motion: on"
        } else {
            "Reduce motion: off"
        },
        if system.reduce_motion {
            Tone::Accent
        } else {
            Tone::Neutral
        },
    );
    row![contrast, motion].spacing(tokens::space::SM).into()
}

/// Append a generative job's preview, inject controls, and plan to its panel.
fn push_job<'a>(
    mut panel: Column<'a, Message>,
    tab: WorkflowTab,
    job: &'a JobStage,
    a11y: Accessibility,
) -> Column<'a, Message> {
    if let Some(message) = &job.error {
        panel = panel.push(error_text(message));
    }
    if let Some(preview) = &job.preview {
        panel = panel.push(preview_card(preview, a11y));
        if preview
            .generated_text
            .as_deref()
            .is_some_and(|text| !text.is_empty())
        {
            panel = panel.push(inject_card(tab, job, a11y));
        }
    }
    if let Some(plan) = &job.plan {
        panel = panel.push(plan_card(plan, a11y));
    }
    panel
}

fn preview_card(preview: &PreviewArtifact, a11y: Accessibility) -> Element<'_, Message> {
    let (label, tone) = preview_status(&preview.status);
    let mut body = column![
        row![
            section_title(&preview.title),
            components::badge(label, tone)
        ]
        .spacing(tokens::space::MD)
    ]
    .spacing(tokens::space::SM);
    match preview.generated_text.as_deref().filter(|t| !t.is_empty()) {
        Some(generated) => {
            body = body.push(
                text(generated.to_owned())
                    .size(tokens::text::BODY)
                    .color(primary()),
            );
        }
        None => {
            body = body.push(
                text("No text was generated for this preview.")
                    .size(tokens::text::BODY)
                    .color(secondary()),
            );
        }
    }
    for note in &preview.notes {
        body = body.push(
            text(format!("• {note}"))
                .size(tokens::text::CAPTION)
                .color(secondary()),
        );
    }
    components::card(body, a11y).width(Length::Fill).into()
}

fn inject_card(tab: WorkflowTab, job: &JobStage, a11y: Accessibility) -> Element<'_, Message> {
    let target_picker = components::picker(
        "Inject target",
        target_choices(),
        Some(TargetChoice(job.inject_target.clone())),
        move |choice| Message::InjectTargetSelected(tab, choice.0),
    );
    let auto = components::capsule(
        if job.auto_submit {
            "Auto-submit: on"
        } else {
            "Auto-submit: off"
        },
        if job.auto_submit {
            Capsule::Secondary
        } else {
            Capsule::Ghost
        },
    )
    .on_press(Message::InjectAutoSubmitToggled(tab));
    let plan_button: Element<'_, Message> = if job.injecting {
        components::capsule("Planning…", Capsule::Ghost).into()
    } else {
        components::capsule("Plan injection", Capsule::Primary)
            .on_press(Message::InjectPressed(tab))
            .into()
    };

    components::card(
        column![
            labeled("Inject target", target_picker),
            row![auto, plan_button].spacing(tokens::space::SM),
        ]
        .spacing(tokens::space::MD),
        a11y,
    )
    .width(Length::Fill)
    .into()
}

fn plan_card(plan: &InjectionPlan, a11y: Accessibility) -> Element<'_, Message> {
    let body = column![
        section_title("Injection plan prepared"),
        detail_row("Target", TargetChoice(plan.target.clone()).to_string()),
        detail_row(
            "Auto-submit",
            (if plan.auto_submit { "yes" } else { "no" }).to_owned(),
        ),
        labeled(
            "Payload",
            text(plan.payload.clone())
                .size(tokens::text::BODY)
                .color(primary()),
        ),
        text(
            "VoiceLayer staged this paste for the selected target. Nothing was typed into the \
             focused application automatically.",
        )
        .size(tokens::text::CAPTION)
        .color(secondary()),
    ]
    .spacing(tokens::space::SM);
    components::card(body, a11y).width(Length::Fill).into()
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

fn preview_status(status: &PreviewStatus) -> (&'static str, Tone) {
    match status {
        PreviewStatus::Ready => ("ready", Tone::Success),
        PreviewStatus::NeedsProvider => ("needs provider", Tone::Warning),
        PreviewStatus::Rejected => ("rejected", Tone::Danger),
    }
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

/// Preset one-shot capture windows for the Settings picker. `u32` alone would
/// render as a bare number; this pairs the value with a "N s" label.
#[derive(Debug, Clone, PartialEq, Eq)]
struct SecondsChoice(u32);

impl std::fmt::Display for SecondsChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{} s", self.0)
    }
}

fn seconds_choices() -> Vec<SecondsChoice> {
    [5, 8, 12, 20, 30].map(SecondsChoice).to_vec()
}

// --- small shared builders ---

fn heading(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .font(theme::font(Weight::Bold))
        .size(tokens::text::LARGE_TITLE)
        .color(primary())
}

fn section_title(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .font(theme::font(Weight::Semibold))
        .size(tokens::text::TITLE)
        .color(primary())
}

fn caption(label: &str) -> iced::widget::Text<'static> {
    text(label.to_owned())
        .size(tokens::text::CAPTION)
        .color(secondary())
}

/// A captioned form control: the label sits above its input.
fn labeled<'a>(label: &str, control: impl Into<Element<'a, Message>>) -> Element<'a, Message> {
    column![caption(label), control.into()]
        .spacing(tokens::space::XS)
        .into()
}

/// A label/value detail line, used by the injection plan card.
fn detail_row(label: &str, value: String) -> Element<'static, Message> {
    row![
        text(label.to_owned())
            .size(tokens::text::CAPTION)
            .color(secondary())
            .width(Length::Fixed(140.0)),
        text(value).size(tokens::text::CAPTION).color(primary()),
    ]
    .spacing(tokens::space::SM)
    .into()
}

fn submit_capsule(label: &str, busy: bool, message: Message) -> Element<'static, Message> {
    if busy {
        components::capsule("Working…", Capsule::Ghost).into()
    } else {
        components::capsule(label, Capsule::Primary)
            .on_press(message)
            .into()
    }
}

fn error_text(message: &str) -> Element<'static, Message> {
    text(format!("Error: {message}"))
        .size(tokens::text::CAPTION)
        .color(danger())
        .into()
}

fn primary() -> iced::Color {
    theme::color(theme::palette().text_primary)
}

fn secondary() -> iced::Color {
    theme::color(theme::palette().text_secondary)
}

fn danger() -> iced::Color {
    theme::color(theme::palette().danger)
}
