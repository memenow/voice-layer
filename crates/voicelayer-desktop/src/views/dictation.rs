//! Dictation status, capture controls, and transcript view.

use iced::widget::{column, row, text};
use iced::{Element, Length};
use voicelayer_core::ProviderDescriptor;
use voicelayer_ui::tokens;

use crate::app::{App, Message, dictation_translate_supported, is_dispatchable_asr_provider};
use crate::components::{self, Capsule, ProviderChoice, Tone};
use crate::state::{SegChoice, SessionStage, render_session_stage};

use super::{error_text, field_group, heading, primary, secondary, section_title};

pub(super) fn panel(app: &App) -> Element<'_, Message> {
    let mut cards = column![status_card(app), capture_options_card(app)].spacing(tokens::space::LG);

    if let Some(transcript) = app.session.transcript.as_deref().filter(|t| !t.is_empty()) {
        cards = cards.push(
            components::card(
                column![
                    section_title("Transcript"),
                    text(transcript.to_owned())
                        .size(tokens::text::BODY)
                        .color(primary()),
                ]
                .spacing(tokens::space::SM),
                app.accessibility(),
            )
            .width(Length::Fill),
        );
    }

    cards.into()
}

fn status_card(app: &App) -> Element<'_, Message> {
    let session_tone = match app.session.stage {
        SessionStage::Listening => Tone::Accent,
        SessionStage::Completed => Tone::Success,
        SessionStage::Failed => Tone::Danger,
        SessionStage::Idle | SessionStage::Starting | SessionStage::Stopping => Tone::Neutral,
    };
    let header = row![
        heading("Dictation"),
        components::badge(
            &format!("Session · {}", render_session_stage(app.session.stage)),
            session_tone,
        ),
    ]
    .spacing(tokens::space::MD);

    let hotkey_text = if app.hotkey.portal_available {
        "Global shortcut registered: VoiceLayer: Toggle dictation".to_owned()
    } else {
        format!(
            "Global shortcuts unavailable ({}). F9 works while this window is focused.",
            app.hotkey.portal_error.as_deref().unwrap_or("unknown"),
        )
    };
    let hotkey_line = text(hotkey_text)
        .size(tokens::text::CAPTION)
        .color(secondary());

    let primary_button = primary_control(app);
    let mut body = column![header, hotkey_line, primary_button].spacing(tokens::space::LG);

    if let Some(language) = &app.session.detected_language {
        body = body.push(components::badge(
            &format!("Detected language · {language}"),
            Tone::Neutral,
        ));
    }
    if let Some(event) = &app.last_event {
        body = body.push(
            text(format!("Last event · {}", event.event.name()))
                .size(tokens::text::CAPTION)
                .color(secondary()),
        );
    }
    if let Some(error) = &app.error {
        body = body.push(error_text(error));
    }
    if !app.session.notes.is_empty() {
        let notes = app
            .session
            .notes
            .iter()
            .map(|note| format!("• {note}"))
            .collect::<Vec<_>>()
            .join("\n");
        body = body.push(text(notes).size(tokens::text::CAPTION).color(secondary()));
    }

    components::card(body, app.accessibility())
        .width(Length::Fill)
        .into()
}

fn primary_control(app: &App) -> Element<'_, Message> {
    if app.capture_in_flight {
        return components::capsule("Start dictation  ·  F9", Capsule::Ghost).into();
    }
    match app.session.stage {
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
    }
}

fn capture_options_card(app: &App) -> Element<'_, Message> {
    let asr_ids = dispatchable_asr_ids(app.providers.as_deref().unwrap_or(&[]));
    let provider_picker = components::picker(
        "Automatic (daemon default)",
        components::provider_choices(&asr_ids),
        Some(ProviderChoice(app.dictation_provider.clone())),
        |choice| Message::DictationProviderSelected(choice.0),
    );
    let language = components::field(
        "Blank auto-detects",
        &app.dictation_language,
        Message::DictationLanguageEdited,
    );
    let segmentation = components::picker(
        "Segmentation",
        SegChoice::ALL.to_vec(),
        Some(app.dictation_segmentation),
        Message::DictationSegmentationSelected,
    );
    let translate: Element<'_, Message> =
        if dictation_translate_supported(app.dictation_provider.as_deref()) {
            let label = if app.dictation_translate {
                "Translate to English: on"
            } else {
                "Translate to English: off"
            };
            let style = if app.dictation_translate {
                Capsule::Secondary
            } else {
                Capsule::Ghost
            };
            components::capsule(label, style)
                .on_press(Message::DictationTranslateToggled)
                .into()
        } else {
            components::capsule("Translate to English: unavailable", Capsule::Ghost).into()
        };

    let capturable = matches!(
        app.session.stage,
        SessionStage::Idle | SessionStage::Completed | SessionStage::Failed
    ) && !app.capture_in_flight;
    let capture_label = format!("One-shot capture  ·  {} s", app.preferences.capture_seconds);
    let capture: Element<'_, Message> = if app.capture_in_flight {
        components::capsule("Capturing...", Capsule::Ghost).into()
    } else if capturable {
        components::capsule(&capture_label, Capsule::Secondary)
            .on_press(Message::CapturePressed)
            .into()
    } else {
        components::capsule(&capture_label, Capsule::Ghost).into()
    };

    let mut content = column![
        section_title("Capture options"),
        field_group("ASR provider", provider_picker),
        field_group("Input language", language),
        field_group("Segmentation", segmentation),
        translate,
        capture,
    ]
    .spacing(tokens::space::MD);
    if let Some(error) = &app.providers_error {
        content = content.push(error_text(error));
    }

    components::card(content, app.accessibility())
        .width(Length::Fill)
        .into()
}

fn dispatchable_asr_ids(providers: &[ProviderDescriptor]) -> Vec<String> {
    providers
        .iter()
        .filter(|provider| is_dispatchable_asr_provider(provider))
        .map(|provider| provider.id.clone())
        .collect()
}

#[cfg(test)]
mod tests {
    use voicelayer_core::{ProviderDescriptor, ProviderKind};

    use super::dispatchable_asr_ids;

    fn provider(id: &str, kind: ProviderKind) -> ProviderDescriptor {
        ProviderDescriptor {
            id: id.to_owned(),
            kind,
            transport: "test".to_owned(),
            local: true,
            default_enabled: true,
            experimental: false,
            license: "Apache-2.0".to_owned(),
        }
    }

    #[test]
    fn dictation_picker_lists_only_dispatchable_asr_providers() {
        let providers = vec![
            provider("whisper_cpp", ProviderKind::Asr),
            provider("mimo_v2_5_asr", ProviderKind::Asr),
            provider("qwen3_asr_1_7b", ProviderKind::Asr),
            provider("voxtral_realtime", ProviderKind::Asr),
            provider("writer", ProviderKind::Llm),
        ];
        assert_eq!(
            dispatchable_asr_ids(&providers),
            vec![
                "whisper_cpp".to_owned(),
                "mimo_v2_5_asr".to_owned(),
                "qwen3_asr_1_7b".to_owned(),
            ],
        );
    }
}
