//! Compose, rewrite, and translate workflow panels with shared preview-to-inject cards.

use iced::widget::{Column, column, row, text};
use iced::{Element, Length};
use voicelayer_core::{InjectionPlan, PreviewArtifact, PreviewStatus};
use voicelayer_ui::accessibility::Accessibility;
use voicelayer_ui::tokens;

use crate::app::{App, Message};
use crate::components::{
    self, ArchetypeChoice, Capsule, StyleChoice, TargetChoice, Tone, archetype_choices,
    style_choices, target_choices,
};
use crate::forms::JobStage;
use crate::state::WorkflowTab;

use super::{
    error_text, field_group, heading, info_row, primary, secondary, section_title, submit_capsule,
};

pub(super) fn compose_panel(app: &App) -> Element<'_, Message> {
    let form = &app.compose;
    let card = components::card(
        column![
            field_group(
                "Spoken prompt",
                components::editor(
                    "Describe what to write (e.g. a polite email declining a meeting)",
                    &form.prompt,
                    Message::ComposePromptEdited,
                ),
            ),
            field_group(
                "Archetype (optional)",
                components::picker(
                    "Automatic",
                    archetype_choices(),
                    Some(ArchetypeChoice(form.archetype.clone())),
                    |choice| Message::ComposeArchetypeSelected(choice.0),
                ),
            ),
            field_group(
                "Output language (optional)",
                components::field(
                    "Blank keeps the prompt language",
                    form.language.as_str(),
                    Message::ComposeLanguageEdited,
                ),
            ),
            submit_capsule(
                "Compose",
                form.job.submitting,
                Message::ComposeSubmitPressed,
            ),
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill);

    let panel = column![heading("Compose"), card].spacing(tokens::space::LG);
    push_job(panel, WorkflowTab::Compose, &form.job, app.accessibility()).into()
}

pub(super) fn rewrite_panel(app: &App) -> Element<'_, Message> {
    let form = &app.rewrite;
    let card = components::card(
        column![
            field_group(
                "Source text",
                components::editor(
                    "Paste the text to rewrite...",
                    &form.source,
                    Message::RewriteSourceEdited,
                ),
            ),
            field_group(
                "Style",
                components::picker(
                    "Choose a style",
                    style_choices(),
                    Some(StyleChoice(form.style.clone())),
                    |choice| Message::RewriteStyleSelected(choice.0),
                ),
            ),
            field_group(
                "Output language (optional)",
                components::field(
                    "Blank keeps the source language",
                    form.language.as_str(),
                    Message::RewriteLanguageEdited,
                ),
            ),
            submit_capsule(
                "Rewrite",
                form.job.submitting,
                Message::RewriteSubmitPressed,
            ),
        ]
        .spacing(tokens::space::MD),
        app.accessibility(),
    )
    .width(Length::Fill);

    let panel = column![heading("Rewrite"), card].spacing(tokens::space::LG);
    push_job(panel, WorkflowTab::Rewrite, &form.job, app.accessibility()).into()
}

pub(super) fn translate_panel(app: &App) -> Element<'_, Message> {
    let form = &app.translate;
    let card = components::card(
        column![
            field_group(
                "Source text",
                components::editor(
                    "Paste the text to translate...",
                    &form.source,
                    Message::TranslateSourceEdited,
                ),
            ),
            field_group(
                "Target language",
                components::field(
                    "e.g. Spanish, French, ja",
                    form.target.as_str(),
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
    let (label, tone) = match preview.status {
        PreviewStatus::Ready => ("ready", Tone::Success),
        PreviewStatus::NeedsProvider => ("needs provider", Tone::Warning),
        PreviewStatus::Rejected => ("rejected", Tone::Danger),
    };
    let mut body = column![
        row![
            section_title(&preview.title),
            components::badge(label, tone),
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
        components::capsule("Planning...", Capsule::Ghost).into()
    } else {
        components::capsule("Plan injection", Capsule::Primary)
            .on_press(Message::InjectPressed(tab))
            .into()
    };

    components::card(
        column![
            field_group("Inject target", target_picker),
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
        info_row(
            "Target",
            TargetChoice(plan.target.clone()).to_string(),
            140.0
        ),
        info_row(
            "Auto-submit",
            if plan.auto_submit { "yes" } else { "no" }.to_owned(),
            140.0,
        ),
        field_group(
            "Payload",
            text(plan.payload.clone())
                .size(tokens::text::BODY)
                .color(primary()),
        ),
        text("Prepared paste payload for the selected target without typing automatically.")
            .size(tokens::text::CAPTION)
            .color(secondary()),
    ]
    .spacing(tokens::space::SM);
    components::card(body, a11y).width(Length::Fill).into()
}
