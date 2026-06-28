//! The `iced` rendering edge for the shared Liquid Glass tokens.
//!
//! [`voicelayer_ui`] owns the backend-agnostic design tokens; this module is the
//! single place that turns them into `iced` style values. Every other GUI module
//! builds views from the semantic helpers here ([`glass_card`], [`backdrop`],
//! [`app_theme`], …) and never hand-codes a color or radius.
//!
//! Pure styling only — no GPU. The true lensed specular edge and animated sheen
//! arrive with the wgpu shader stage; here the glass identity is approximated
//! with a translucent fill, a top-weighted highlight rim, a soft separation
//! shadow, and the app-drawn gradient backdrop the glass floats above. On GNOME
//! Wayland that backdrop — not the real desktop — is what shows through the
//! translucent surfaces.

use std::f32::consts::PI;

use iced::widget::container;
use iced::{Background, Border, Color, Font, Radians, Shadow, Theme, Vector, gradient};

use voicelayer_ui::a11y::{Accessibility, resolve_glass};
use voicelayer_ui::color::Rgba;
use voicelayer_ui::tokens::{self, Appearance, Glass, Palette, Surface, Weight};

/// The appearance VoiceLayer renders in. Dark is primary — glass reads best on a
/// deep base and a dictation tool is commonly dark. A light/auto toggle (later)
/// only needs to change this resolver.
const APPEARANCE: Appearance = Appearance::Dark;

/// The active palette.
pub fn palette() -> Palette {
    tokens::palette(APPEARANCE)
}

/// Shared token color → `iced::Color`. The token already stores straight-alpha
/// sRGB in `0..=1`, which is exactly `iced::Color`'s representation.
pub fn color(c: Rgba) -> Color {
    Color::from_rgba(c.r, c.g, c.b, c.a)
}

/// Symmetric padding (vertical, horizontal) → `iced::Padding`.
pub const fn pad(vertical: f32, horizontal: f32) -> iced::Padding {
    iced::Padding {
        top: vertical,
        right: horizontal,
        bottom: vertical,
        left: horizontal,
    }
}

/// The bundled UI family. Inter (SIL OFL 1.1) static instances are registered in
/// `main`; `assets/fonts/OFL.txt` carries the license, independent of the code's
/// Apache-2.0. The static set is deliberate: fontdb resolves a weight from each
/// face's OS/2 table, whereas a single variable file collapses to one weight
/// (cosmic-text #406).
pub const FONT_FAMILY: &str = "Inter";

/// The font for a token weight, in the bundled [`FONT_FAMILY`].
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

/// The custom iced theme, seeded from the shared palette. Widgets we do not
/// explicitly style (menus, tooltips, the text cursor) inherit coherent colors
/// from this instead of iced's stock palette.
pub fn app_theme() -> Theme {
    let p = palette();
    Theme::custom(
        "VoiceLayer Liquid Glass",
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

/// Window-level base fill — shows at the window edges and is the fallback when
/// transparency is unavailable. The rich layer the glass floats above is drawn
/// by [`backdrop`].
pub fn app_style() -> iced::theme::Style {
    let p = palette();
    iced::theme::Style {
        background_color: color(p.bg_base),
        text_color: color(p.text_primary),
    }
}

/// The app-drawn background panel. A near-vertical gradient from the elevated
/// tone down to the base gives depth without a shader, and is what translucent
/// glass surfaces lens on GNOME Wayland.
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

/// Glass card style (small floating surface): toolbars, control groups, list
/// rows. Accessibility-resolved.
pub fn glass_card(a11y: &Accessibility) -> container::Style {
    glass(Surface::Small, a11y)
}

/// Glass container style for a surface size, with the accessibility contract
/// applied. `a11y` carries the live preferences (Reduce Transparency / Increase
/// Contrast and the opacity slider); until the settings stage feeds real values
/// the 2026 baseline default is used.
pub fn glass(surface: Surface, a11y: &Accessibility) -> container::Style {
    let p = palette();
    let resolved = resolve_glass(tokens::glass(&p, surface), a11y, p.text_primary);
    glass_style(&p, &resolved)
}

fn glass_style(p: &Palette, g: &Glass) -> container::Style {
    // `resolve_glass` zeroes the blur whenever the material is frosted (Reduce
    // Transparency or Increase Contrast). Use that as the signal to drop the
    // lensing approximations and render a flat, maximally legible surface.
    let frosted = g.blur_radius == 0.0;

    let background = if frosted {
        Background::Color(color(g.tint))
    } else {
        // Light pools at the top edge (brighter, more opaque) and thins toward
        // the bottom — a flat fill can't suggest the lensed highlight, a
        // gradient can. calibrated.
        let top = color(g.tint.with_alpha((g.tint.a + 0.10).min(1.0)));
        let bottom = color(g.tint.with_alpha(g.tint.a * 0.6));
        let fill = gradient::Linear::new(Radians(PI))
            .add_stop(0.0, top)
            .add_stop(1.0, bottom);
        Background::Gradient(gradient::Gradient::Linear(fill))
    };

    let border = if g.border_width > 0.0 {
        // Increase Contrast: a hard contrasting hairline.
        Border {
            color: color(g.border_color),
            width: g.border_width,
            radius: g.corner_radius.into(),
        }
    } else {
        // Default: a 1px specular rim approximates the lensed highlight edge.
        Border {
            color: color(g.highlight),
            width: 1.0,
            radius: g.corner_radius.into(),
        }
    };

    let shadow = if frosted {
        Shadow::default()
    } else {
        // The 2026 darkened edge, rendered as a soft separation shadow so the
        // glass reads as floating above the backdrop.
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

    /// The GUI can't be driven headlessly, so verify against the same font db
    /// iced uses at runtime that each bundled weight resolves to a distinct face
    /// under the single "Inter" family. Guards the variable-vs-static decision
    /// and a future asset swap that might break weight selection.
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
            // `font(weight)` carries the Inter family; the db query proves a real
            // face of `expected` weight backs it.
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
