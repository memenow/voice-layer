//! Local desktop shell preferences and accessibility status panel.

use iced::widget::{column, row, text};
use iced::{Element, Length};
use voicelayer_ui::tokens;

use crate::app::{App, Message};
use crate::components::{self, Capsule, TargetChoice, Tone, target_choices};
use crate::state::SystemA11y;

use super::{field_group, heading, secondary, section_title};

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

pub(super) fn panel(app: &App) -> Element<'_, Message> {
    let prefs = &app.preferences;
    let a11y = app.accessibility();
    let form_card = components::card(
        column![
            field_group(
                "Default output language",
                components::field(
                    "Blank keeps the source language",
                    &prefs.default_output_language,
                    Message::PrefOutputLanguageEdited,
                ),
            ),
            field_group(
                "Default inject target",
                components::picker(
                    "Inject target",
                    target_choices(),
                    Some(TargetChoice(prefs.default_inject_target.clone())),
                    |choice| Message::PrefInjectTargetSelected(choice.0),
                ),
            ),
            field_group(
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
            section_title("Appearance and accessibility"),
            field_group(
                &format!("Surface opacity · {opacity_pct}%"),
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
            "Surface opacity and Reduce Transparency are saved to desktop.toml. \
             Increase Contrast and Reduce Motion follow the XDG Settings portal on Linux. \
             Daemon settings live in config.toml and VOICELAYER_* environment variables.",
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
