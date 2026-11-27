//! `iced` styling adapters for the shared [`voicelayer_ui`] design tokens.

use std::f32::consts::PI;

use iced::widget::container;
use iced::{Background, Border, Color, Font, Radians, Shadow, Theme, Vector, gradient};

use voicelayer_ui::accessibility::{Accessibility, resolve_glass};
use voicelayer_ui::color::Rgba;
use voicelayer_ui::tokens::{self, Appearance, Glass, Palette, Surface, Weight};

const APPEARANCE: Appearance = Appearance::Dark;

pub fn palette() -> Palette {
    tokens::palette(APPEARANCE)
}

pub fn color(c: Rgba) -> Color {
    Color::from_rgba(c.r, c.g, c.b, c.a)
}

pub const fn pad(vertical: f32, horizontal: f32) -> iced::Padding {
    iced::Padding {
        top: vertical,
        right: horizontal,
        bottom: vertical,
        left: horizontal,
    }
}

/// Bundled UI font family (Inter static faces, SIL OFL 1.1; see `assets/fonts/OFL.txt`).
/// Static faces are used instead of a variable font because fontdb resolves weights
/// per face from OS/2 metadata.
pub const FONT_FAMILY: &str = "Inter";

pub fn font(weight: Weight) -> Font {
    Font {
        family: iced::font::Family::Name(FONT_FAMILY),
        weight: font_weight(weight),
        ..Font::DEFAULT
    }
}

fn font_weight(weight: Weight) -> iced::font::Weight {
    match weight {
        Weight::Regular => iced::font::Weight::Normal,
        Weight::Medium => iced::font::Weight::Medium,
        Weight::Semibold => iced::font::Weight::Semibold,
        Weight::Bold => iced::font::Weight::Bold,
    }
}

pub fn app_theme() -> Theme {
    let p = palette();
    Theme::custom(
        "VoiceLayer",
        iced::theme::Palette {
            background: color(p.bg_base),
            text: color(p.text_primary),
            primary: color(p.accent),
            success: color(p.success),
            warning: color(p.warning),
            danger: color(p.danger),
        },
    )
}

/// Window clear style. Transparent so the borderless HUD window keeps rounded
/// corners see-through; the main window paints an opaque [`backdrop`] beneath it.
pub fn app_style() -> iced::theme::Style {
    let p = palette();
    iced::theme::Style {
        background_color: Color::TRANSPARENT,
        text_color: color(p.text_primary),
    }
}

/// Fallback opaque vertical gradient rendered beneath the GPU shader backdrop.
pub fn backdrop() -> container::Style {
    let p = palette();
    let fill = gradient::Linear::new(Radians(PI))
        .add_stop(0.0, color(p.bg_elevated))
        .add_stop(1.0, color(p.bg_base));
    container::Style {
        background: Some(Background::Gradient(gradient::Gradient::Linear(fill))),
        text_color: Some(color(p.text_primary)),
        ..container::Style::default()
    }
}

pub fn glass_card(a11y: &Accessibility) -> container::Style {
    glass(Surface::Small, a11y)
}

pub fn glass(surface: Surface, a11y: &Accessibility) -> container::Style {
    let p = palette();
    let resolved = resolve_glass(tokens::glass(&p, surface), a11y, p.text_primary);
    glass_style(&p, &resolved)
}

fn glass_style(p: &Palette, g: &Glass) -> container::Style {
    // `resolve_glass` sets blur_radius to 0 when reduced transparency or high contrast is active.
    let frosted = g.blur_radius == 0.0;

    let background = if frosted {
        Background::Color(color(g.tint))
    } else {
        let top = color(g.tint.with_alpha((g.tint.a + 0.10).min(1.0)));
        let bottom = color(g.tint.with_alpha(g.tint.a * 0.6));
        let fill = gradient::Linear::new(Radians(PI))
            .add_stop(0.0, top)
            .add_stop(1.0, bottom);
        Background::Gradient(gradient::Gradient::Linear(fill))
    };

    let border = if g.border_width > 0.0 {
        Border {
            color: color(g.border_color),
            width: g.border_width,
            radius: g.corner_radius.into(),
        }
    } else {
        Border {
            color: color(g.highlight),
            width: 1.0,
            radius: g.corner_radius.into(),
        }
    };

    let shadow = if frosted {
        Shadow::default()
    } else {
        Shadow {
            color: color(g.dark_edge),
            offset: Vector::new(0.0, 6.0),
            blur_radius: 24.0,
        }
    };

    container::Style {
        background: Some(background),
        text_color: Some(color(p.text_primary)),
        border,
        shadow,
        ..container::Style::default()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn color_preserves_channels() {
        let c = color(Rgba::new(0.1, 0.2, 0.3, 0.4));
        assert!((c.r - 0.1).abs() < 1e-6);
        assert!((c.g - 0.2).abs() < 1e-6);
        assert!((c.b - 0.3).abs() < 1e-6);
        assert!((c.a - 0.4).abs() < 1e-6);
    }

    #[test]
    fn default_glass_floats_with_gradient_rim_and_shadow() {
        let style = glass_card(&Accessibility::default());
        assert!(matches!(
            style.background,
            Some(Background::Gradient(gradient::Gradient::Linear(_)))
        ));
        assert_eq!(style.border.width, 1.0);
        assert!(style.shadow.blur_radius > 0.0);
    }

    #[test]
    fn increase_contrast_glass_is_flat_opaque_with_hard_border() {
        let a11y = Accessibility {
            increase_contrast: true,
            ..Default::default()
        };
        let style = glass(Surface::Small, &a11y);
        match style.background {
            Some(Background::Color(c)) => assert_eq!(c.a, 1.0),
            other => panic!("expected an opaque flat fill, got {other:?}"),
        }
        assert_eq!(style.border.width, 1.0);
        assert_eq!(style.shadow.blur_radius, 0.0);
    }

    #[test]
    fn inter_static_faces_resolve_each_weight_under_one_family() {
        let mut db = fontdb::Database::new();
        db.load_font_data(include_bytes!("../assets/fonts/Inter-Regular.ttf").to_vec());
        db.load_font_data(include_bytes!("../assets/fonts/Inter-Medium.ttf").to_vec());
        db.load_font_data(include_bytes!("../assets/fonts/Inter-SemiBold.ttf").to_vec());
        db.load_font_data(include_bytes!("../assets/fonts/Inter-Bold.ttf").to_vec());

        for (weight, expected) in [
            (Weight::Regular, fontdb::Weight::NORMAL),
            (Weight::Medium, fontdb::Weight::MEDIUM),
            (Weight::Semibold, fontdb::Weight::SEMIBOLD),
            (Weight::Bold, fontdb::Weight::BOLD),
        ] {
            assert_eq!(font(weight).weight, font_weight(weight));
            let query = fontdb::Query {
                families: &[fontdb::Family::Name(FONT_FAMILY)],
                weight: expected,
                stretch: fontdb::Stretch::Normal,
                style: fontdb::Style::Normal,
            };
            let id = db
                .query(&query)
                .expect("Inter family should resolve for this weight");
            assert_eq!(
                db.face(id).expect("face info").weight,
                expected,
                "weight {expected:?} selected the wrong face",
            );
        }
    }
}
