//! Reusable Liquid Glass widgets, built on [`crate::theme`].
//!
//! Views compose these instead of hand-styling primitives, so the design system
//! stays in one place and the look is consistent across every workflow. The
//! controls here (capsule buttons, fields) are the interactive glass materials
//! that may sit on a glass card; cards and panels are the floating surfaces and
//! are never nested glass-on-glass.

use iced::widget::{button, container, pick_list, row, slider, text, text_editor, text_input};
use iced::{Background, Border, Color, Element, Length, Shadow, Vector};

use voicelayer_core::{CompositionArchetype, InjectTarget, RewriteStyle};
use voicelayer_ui::a11y::Accessibility;
use voicelayer_ui::color::Rgba;
use voicelayer_ui::tokens::{self, Surface, Weight};

use crate::theme;

/// Visual weight of a capsule (pill) button.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Capsule {
    /// Tinted fill — the single primary action on a surface.
    Primary,
    /// Glass fill — secondary actions.
    Secondary,
    /// Transparent until hovered — tertiary / low-emphasis actions.
    Ghost,
}

/// A pill-shaped button carrying the Liquid Glass styling. Wire `.on_press(..)`
/// on the returned widget at the call site.
pub fn capsule<'a, Message: 'a>(label: &str, kind: Capsule) -> button::Button<'a, Message> {
    let content = text(label.to_owned())
        .font(theme::font(Weight::Semibold))
        .size(tokens::text::BODY);
    button(content)
        .padding(theme::pad(tokens::space::SM, tokens::space::LG))
        .style(move |_theme, status| capsule_style(kind, status))
}

fn capsule_style(kind: Capsule, status: button::Status) -> button::Style {
    let p = theme::palette();
    let pill = Border {
        color: Color::TRANSPARENT,
        width: 0.0,
        radius: tokens::radius::PILL.into(),
    };
    let base = button::Style {
        text_color: theme::color(p.text_primary),
        border: pill,
        ..button::Style::default()
    };

    // Pointer interaction uses a subdued amplitude — this is a desktop app, not
    // a touch surface, so hover/press shift brightness modestly.
    match kind {
        Capsule::Primary => {
            let fill = match status {
                button::Status::Hovered => lighten(p.accent, 0.08),
                button::Status::Pressed => darken(p.accent, 0.08),
                button::Status::Disabled => p.accent.with_alpha(0.4),
                button::Status::Active => p.accent,
            };
            button::Style {
                background: Some(Background::Color(theme::color(fill))),
                text_color: theme::color(on_accent(p.accent)),
                shadow: Shadow {
                    color: theme::color(p.accent.with_alpha(0.45)),
                    offset: Vector::new(0.0, 4.0),
                    blur_radius: 16.0,
                },
                ..base
            }
        }
        Capsule::Secondary => {
            let film = tokens::glass(&p, Surface::Small).tint;
            let fill = match status {
                button::Status::Hovered => raise_alpha(film, 0.06),
                button::Status::Pressed => raise_alpha(film, 0.10),
                button::Status::Disabled => film.with_alpha(film.a * 0.5),
                button::Status::Active => film,
            };
            button::Style {
                background: Some(Background::Color(theme::color(fill))),
                border: Border {
                    color: theme::color(p.glass_highlight),
                    width: 1.0,
                    ..pill
                },
                ..base
            }
        }
        Capsule::Ghost => {
            let fill = match status {
                button::Status::Hovered => Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.06),
                button::Status::Pressed => Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.10),
                _ => Rgba::TRANSPARENT,
            };
            button::Style {
                background: Some(Background::Color(theme::color(fill))),
                text_color: theme::color(p.text_secondary),
                ..base
            }
        }
    }
}

/// Tone of a status badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Tone {
    Neutral,
    Accent,
    Success,
    Warning,
    Danger,
}

/// A small glass pill with a colored status dot and a caption — used for daemon
/// health, session stage, and similar at-a-glance state.
pub fn badge<'a, Message: 'a>(label: &str, tone: Tone) -> Element<'a, Message> {
    let p = theme::palette();
    let accent = tone_color(tone);
    let dot = text("●")
        .size(tokens::text::CAPTION)
        .color(theme::color(accent));
    let caption = text(label.to_owned())
        .size(tokens::text::CAPTION)
        .color(theme::color(p.text_secondary));
    container(row![dot, caption].spacing(tokens::space::XS))
        .padding(theme::pad(tokens::space::XS, tokens::space::SM))
        .style(move |_theme| badge_style(accent))
        .into()
}

fn badge_style(tone: Rgba) -> container::Style {
    container::Style {
        background: Some(Background::Color(theme::color(tone.with_alpha(0.14)))),
        border: Border {
            color: theme::color(tone.with_alpha(0.40)),
            width: 1.0,
            radius: tokens::radius::PILL.into(),
        },
        text_color: Some(theme::color(theme::palette().text_primary)),
        ..container::Style::default()
    }
}

/// Wrap content in a glass card that floats on the backdrop. The caller sets
/// width/height on the returned container and passes the live [`Accessibility`]
/// so the card honors the opacity slider and the Reduce Transparency / Increase
/// Contrast fallbacks. `Accessibility` is `Copy`, so the style closure captures
/// it by value.
pub fn card<'a, Message: 'a>(
    content: impl Into<Element<'a, Message>>,
    a11y: Accessibility,
) -> container::Container<'a, Message> {
    container(content)
        .padding(theme::pad(tokens::space::LG, tokens::space::LG))
        .style(move |_theme| theme::glass_card(&a11y))
}

/// A slider for the Liquid Glass opacity preference, fixed to the `0.0..=1.0`
/// contract range with a coarse step so the value reads cleanly as a percentage.
pub fn opacity_slider<'a, Message: Clone + 'a>(
    value: f32,
    on_change: impl Fn(f32) -> Message + 'a,
) -> Element<'a, Message> {
    slider(0.0..=1.0, value, on_change).step(0.05_f32).into()
}

/// A glass text field.
pub fn field<'a, Message: 'a + Clone>(
    placeholder: &'a str,
    value: &'a str,
    on_input: impl Fn(String) -> Message + 'a,
) -> text_input::TextInput<'a, Message> {
    text_input(placeholder, value)
        .on_input(on_input)
        .padding(theme::pad(tokens::space::SM, tokens::space::MD))
        .size(tokens::text::BODY)
        .style(field_style)
}

fn field_style(t: &iced::Theme, status: text_input::Status) -> text_input::Style {
    let p = theme::palette();
    let film = tokens::glass(&p, Surface::Small).tint;
    let mut style = text_input::default(t, status);
    style.background = Background::Color(theme::color(film.with_alpha(0.08)));
    style.border = Border {
        color: theme::color(match status {
            text_input::Status::Focused { .. } => p.accent.with_alpha(0.8),
            _ => p.glass_highlight,
        }),
        width: 1.0,
        radius: tokens::radius::CARD.into(),
    };
    style.value = theme::color(p.text_primary);
    style.placeholder = theme::color(p.text_secondary);
    style.icon = theme::color(p.text_secondary);
    style.selection = theme::color(p.accent.with_alpha(0.35));
    style
}

/// A glass multi-line editor for longer composition input. The caller owns the
/// [`text_editor::Content`]; edits arrive as [`text_editor::Action`] to apply
/// back onto that content.
pub fn editor<'a, Message: Clone + 'a>(
    placeholder: &'a str,
    content: &'a text_editor::Content,
    on_action: impl Fn(text_editor::Action) -> Message + 'a,
) -> Element<'a, Message> {
    text_editor(content)
        .placeholder(placeholder)
        .padding(theme::pad(tokens::space::SM, tokens::space::MD))
        .height(Length::Fixed(148.0))
        .on_action(on_action)
        .style(editor_style)
        .into()
}

fn editor_style(t: &iced::Theme, status: text_editor::Status) -> text_editor::Style {
    let p = theme::palette();
    let film = tokens::glass(&p, Surface::Small).tint;
    let focused = matches!(status, text_editor::Status::Focused { .. });
    let mut style = text_editor::default(t, status);
    style.background = Background::Color(theme::color(film.with_alpha(0.08)));
    style.border = Border {
        color: theme::color(if focused {
            p.accent.with_alpha(0.8)
        } else {
            p.glass_highlight
        }),
        width: 1.0,
        radius: tokens::radius::CARD.into(),
    };
    style.value = theme::color(p.text_primary);
    style.placeholder = theme::color(p.text_secondary);
    style.selection = theme::color(p.accent.with_alpha(0.35));
    style
}

/// A glass dropdown for selecting one option from a fixed set. `T` is usually
/// one of the `*Choice` wrappers below, which pair a wire enum with a label.
pub fn picker<'a, T, Message>(
    placeholder: &'a str,
    options: Vec<T>,
    selected: Option<T>,
    on_select: impl Fn(T) -> Message + 'a,
) -> Element<'a, Message>
where
    T: ToString + PartialEq + Clone + 'a,
    Message: Clone + 'a,
{
    pick_list(options, selected, on_select)
        .placeholder(placeholder)
        .padding(theme::pad(tokens::space::SM, tokens::space::MD))
        .text_size(tokens::text::BODY)
        .style(picker_style)
        .into()
}

fn picker_style(t: &iced::Theme, status: pick_list::Status) -> pick_list::Style {
    let p = theme::palette();
    let film = tokens::glass(&p, Surface::Small).tint;
    let active = matches!(status, pick_list::Status::Active);
    let mut style = pick_list::default(t, status);
    style.background = Background::Color(theme::color(film.with_alpha(0.08)));
    style.border = Border {
        color: theme::color(if active {
            p.glass_highlight
        } else {
            p.accent.with_alpha(0.8)
        }),
        width: 1.0,
        radius: tokens::radius::CARD.into(),
    };
    style.text_color = theme::color(p.text_primary);
    style.placeholder_color = theme::color(p.text_secondary);
    style.handle_color = theme::color(p.text_secondary);
    style
}

/// Presentation wrappers pairing a wire enum with its human label so it can fill
/// a [`picker`] (which requires `ToString`). Each `*_choices()` helper returns
/// the options in display order; unwrap `.0` to recover the wire value.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ArchetypeChoice(pub Option<CompositionArchetype>);

impl std::fmt::Display for ArchetypeChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0.as_ref() {
            None => "Automatic",
            Some(CompositionArchetype::Email) => "Email",
            Some(CompositionArchetype::CoverLetter) => "Cover letter",
            Some(CompositionArchetype::DailyReport) => "Daily report",
            Some(CompositionArchetype::Issue) => "Issue",
            Some(CompositionArchetype::PullRequestDescription) => "Pull request description",
            Some(CompositionArchetype::Prompt) => "Prompt",
            Some(CompositionArchetype::TechnicalSummary) => "Technical summary",
            Some(CompositionArchetype::Custom) => "Custom",
        })
    }
}

pub fn archetype_choices() -> Vec<ArchetypeChoice> {
    let mut choices = vec![ArchetypeChoice(None)];
    choices.extend(
        [
            CompositionArchetype::Email,
            CompositionArchetype::CoverLetter,
            CompositionArchetype::DailyReport,
            CompositionArchetype::Issue,
            CompositionArchetype::PullRequestDescription,
            CompositionArchetype::Prompt,
            CompositionArchetype::TechnicalSummary,
            CompositionArchetype::Custom,
        ]
        .map(Some)
        .map(ArchetypeChoice),
    );
    choices
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StyleChoice(pub RewriteStyle);

impl std::fmt::Display for StyleChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            RewriteStyle::MoreFormal => "More formal",
            RewriteStyle::Shorter => "Shorter",
            RewriteStyle::Politer => "Politer",
            RewriteStyle::MoreTechnical => "More technical",
            RewriteStyle::Translate => "Translate",
        })
    }
}

pub fn style_choices() -> Vec<StyleChoice> {
    [
        RewriteStyle::MoreFormal,
        RewriteStyle::Shorter,
        RewriteStyle::Politer,
        RewriteStyle::MoreTechnical,
        RewriteStyle::Translate,
    ]
    .map(StyleChoice)
    .to_vec()
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TargetChoice(pub InjectTarget);

impl std::fmt::Display for TargetChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(match self.0 {
            InjectTarget::GuiAccessible => "GUI · accessible text",
            InjectTarget::GuiClipboard => "GUI · clipboard",
            InjectTarget::TerminalBracketedPaste => "Terminal · bracketed paste",
            InjectTarget::TerminalKittyRemote => "Terminal · kitty remote",
        })
    }
}

pub fn target_choices() -> Vec<TargetChoice> {
    [
        InjectTarget::GuiAccessible,
        InjectTarget::GuiClipboard,
        InjectTarget::TerminalBracketedPaste,
        InjectTarget::TerminalKittyRemote,
    ]
    .map(TargetChoice)
    .to_vec()
}

/// The ASR provider selection: `None` keeps the daemon's default. `provider_choices`
/// prepends the automatic option ahead of the supplied provider ids.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProviderChoice(pub Option<String>);

impl std::fmt::Display for ProviderChoice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.0 {
            None => f.write_str("Automatic (daemon default)"),
            Some(id) => f.write_str(id),
        }
    }
}

pub fn provider_choices(ids: &[String]) -> Vec<ProviderChoice> {
    let mut choices = vec![ProviderChoice(None)];
    choices.extend(ids.iter().cloned().map(Some).map(ProviderChoice));
    choices
}

fn tone_color(tone: Tone) -> Rgba {
    let p = theme::palette();
    match tone {
        Tone::Neutral => p.text_secondary,
        Tone::Accent => p.accent,
        Tone::Success => p.success,
        Tone::Warning => p.warning,
        Tone::Danger => p.danger,
    }
}

/// Near-black or near-white, whichever reads on `accent`. The accent is bright
/// enough on dark that a dark glyph is more legible than white.
fn on_accent(accent: Rgba) -> Rgba {
    let luminance = 0.2126 * accent.r + 0.7152 * accent.g + 0.0722 * accent.b;
    if luminance > 0.55 {
        Rgba::rgb8(0x0a, 0x0c, 0x10)
    } else {
        Rgba::rgb8(0xff, 0xff, 0xff)
    }
}

fn lighten(c: Rgba, t: f32) -> Rgba {
    c.lerp(Rgba::new(1.0, 1.0, 1.0, c.a), t)
}

fn darken(c: Rgba, t: f32) -> Rgba {
    c.lerp(Rgba::new(0.0, 0.0, 0.0, c.a), t)
}

fn raise_alpha(c: Rgba, delta: f32) -> Rgba {
    c.with_alpha((c.a + delta).min(1.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn on_accent_is_dark_for_a_bright_accent() {
        let dark = on_accent(Rgba::rgb8(0x58, 0xa6, 0xff));
        assert!(dark.r < 0.2 && dark.g < 0.2 && dark.b < 0.2);
    }

    #[test]
    fn on_accent_is_light_for_a_deep_accent() {
        let light = on_accent(Rgba::rgb8(0x10, 0x2a, 0x6f));
        assert!(light.r > 0.9 && light.g > 0.9 && light.b > 0.9);
    }

    #[test]
    fn raise_alpha_saturates_at_one() {
        assert_eq!(raise_alpha(Rgba::new(1.0, 1.0, 1.0, 0.95), 0.2).a, 1.0);
    }

    #[test]
    fn archetype_choices_include_a_selectable_automatic_option() {
        let choices = archetype_choices();
        assert_eq!(choices.first(), Some(&ArchetypeChoice(None)));
        assert_eq!(
            choices.first().map(ToString::to_string).as_deref(),
            Some("Automatic")
        );
        assert!(choices.contains(&ArchetypeChoice(Some(CompositionArchetype::Email))));
    }
}
