//! Liquid Glass design tokens (2026 baseline).
//!
//! Numbers marked `calibrated` are NOT Apple spec — Apple ships the material as
//! opaque system rendering and publishes almost no quantities. The only
//! Apple-published number here is [`CLEAR_DIM`] (the clear-variant 35% dimming
//! layer). Treat everything else as our own calibration against the published
//! qualitative model (translucency, lensed edge, concentric corners, two-layer
//! separation, monochrome adaptive content color).

use crate::color::Rgba;

/// 8pt spacing scale.
pub mod space {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const XL: f32 = 24.0;
    pub const XXL: f32 = 32.0;
}

/// Continuous (squircle) corner radii. Nested shapes are concentric:
/// `inner = outer - padding` (Apple's `ConcentricRectangle` rule).
pub mod radius {
    /// Capsule default control shape (a large radius the renderer clamps to
    /// half the height).
    pub const PILL: f32 = 999.0;
    /// Card / small floating surface. calibrated.
    pub const CARD: f32 = 20.0;
    /// Large panel / sidebar / sheet. calibrated.
    pub const PANEL: f32 = 28.0;

    /// Concentric inner radius for content padded inside an `outer`-radius
    /// surface; never negative.
    pub fn concentric_inner(outer: f32, padding: f32) -> f32 {
        (outer - padding).max(0.0)
    }
}

/// Type weight ramp. SF Pro is proprietary; map these onto Inter / the system
/// font at the rendering edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    Regular,
    Medium,
    Semibold,
    Bold,
}

/// Point sizes for the type ramp. calibrated to a desktop dictation tool.
pub mod text {
    pub const CAPTION: f32 = 12.0;
    pub const BODY: f32 = 14.0;
    pub const TITLE: f32 = 17.0;
    pub const LARGE_TITLE: f32 = 28.0;
}

/// Motion tokens. The concrete easing curves live in the GUI crate (mapped to
/// `iced::animation::Easing`); here we expose the design intent and durations.
pub mod motion {
    /// Animation intent. Liquid Glass favors elastic/springy transitions; a
    /// pointer-driven desktop app uses the subdued amplitude.
    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub enum Curve {
        Standard,
        Elastic,
        Back,
        Linear,
    }

    pub const FAST_MS: u64 = 160;
    pub const BASE_MS: u64 = 240;
    pub const SLOW_MS: u64 = 420;
}

/// Color palette for one appearance (dark or light).
#[derive(Debug, Clone, Copy)]
pub struct Palette {
    /// App-drawn window background (deepest layer). On GNOME Wayland this — not
    /// the real desktop — is what shows through translucent glass.
    pub bg_base: Rgba,
    pub bg_elevated: Rgba,
    pub text_primary: Rgba,
    pub text_secondary: Rgba,
    /// Accent used only to emphasize primary actions (tint-as-emphasis).
    pub accent: Rgba,
    /// Semantic status colors for badges and feedback. calibrated to the same
    /// GitHub Primer family the accent is drawn from, so the docs, GUI, and TUI
    /// agree on what "healthy / attention / error" looks like.
    pub success: Rgba,
    pub warning: Rgba,
    pub danger: Rgba,
    /// Base glass film drawn atop the background. Alpha is the small-element
    /// baseline; large surfaces scale up via [`glass`].
    pub glass_tint: Rgba,
    /// Top / inner specular stroke (the lensed highlight edge).
    pub glass_highlight: Rgba,
    /// 2026 darkened outer edge for depth and separation.
    pub glass_dark_edge: Rgba,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Appearance {
    Dark,
    Light,
}

/// Resolve the palette for an appearance. Dark is the primary target — glass
/// reads best on deep backgrounds and a dictation tool is commonly dark.
pub fn palette(appearance: Appearance) -> Palette {
    match appearance {
        Appearance::Dark => dark(),
        Appearance::Light => light(),
    }
}

fn dark() -> Palette {
    Palette {
        bg_base: Rgba::rgb8(0x0b, 0x0d, 0x12),
        bg_elevated: Rgba::rgb8(0x12, 0x15, 0x1c),
        text_primary: Rgba::rgb8(0xf5, 0xf7, 0xfa),
        text_secondary: Rgba::rgb8(0xaa, 0xb2, 0xc0),
        // docs dark accent (#58a6ff), kept in sync with the documentation theme.
        accent: Rgba::rgb8(0x58, 0xa6, 0xff),
        // Primer dark status colors (success/attention/danger). calibrated.
        success: Rgba::rgb8(0x3f, 0xb9, 0x50),
        warning: Rgba::rgb8(0xd2, 0x99, 0x22),
        danger: Rgba::rgb8(0xf8, 0x51, 0x49),
        // A light film lifts the glass off a dark base; the glass identity comes
        // from this film + the highlight edge + the app-drawn background.
        glass_tint: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.10),
        glass_highlight: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.18),
        glass_dark_edge: Rgba::rgb8(0x00, 0x00, 0x00).with_alpha(0.28),
    }
}

fn light() -> Palette {
    Palette {
        bg_base: Rgba::rgb8(0xf4, 0xf6, 0xf9),
        bg_elevated: Rgba::rgb8(0xff, 0xff, 0xff),
        text_primary: Rgba::rgb8(0x1c, 0x1e, 0x22),
        text_secondary: Rgba::rgb8(0x5b, 0x63, 0x70),
        // docs light accent (#1f6feb).
        accent: Rgba::rgb8(0x1f, 0x6f, 0xeb),
        // Primer light status colors. calibrated.
        success: Rgba::rgb8(0x1a, 0x7f, 0x37),
        warning: Rgba::rgb8(0x9a, 0x67, 0x00),
        danger: Rgba::rgb8(0xcf, 0x22, 0x2e),
        glass_tint: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.55),
        glass_highlight: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.65),
        glass_dark_edge: Rgba::rgb8(0x00, 0x00, 0x00).with_alpha(0.10),
    }
}

/// Apple-published: the clear glass variant adds a 35% black dimming layer over
/// bright content to preserve legibility.
pub const CLEAR_DIM: f32 = 0.35;

/// Surface size class. Larger surfaces are more opaque to preserve legibility
/// over complex backgrounds (HIG → Color).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Pills, toolbars, badges.
    Small,
    /// Sidebars, sheets, cards.
    Large,
}

/// Glass material parameters resolved for a given surface size, before the
/// accessibility contract is applied (see [`crate::a11y::resolve_glass`]).
#[derive(Debug, Clone, Copy)]
pub struct Glass {
    pub tint: Rgba,
    pub highlight: Rgba,
    pub dark_edge: Rgba,
    /// Backdrop blur radius in px. Used by the shader; in styling-only mode it
    /// is a semantic hint for how "frosted" the film looks.
    pub blur_radius: f32,
    /// Backdrop saturation boost approximating vibrancy.
    pub saturation: f32,
    pub corner_radius: f32,
    /// Normally 0; raised to a hard hairline by Increase Contrast.
    pub border_width: f32,
    /// Border color; transparent unless Increase Contrast sets a contrasting
    /// hairline.
    pub border_color: Rgba,
}

/// Build the base glass material for a surface size from a palette.
pub fn glass(palette: &Palette, surface: Surface) -> Glass {
    // Large surfaces bump alpha up (more opaque) and blur more. calibrated.
    let (alpha_bump, blur, corner) = match surface {
        Surface::Small => (0.0, 22.0, radius::CARD),
        Surface::Large => (0.06, 34.0, radius::PANEL),
    };
    Glass {
        tint: palette
            .glass_tint
            .with_alpha((palette.glass_tint.a + alpha_bump).min(1.0)),
        highlight: palette.glass_highlight,
        dark_edge: palette.glass_dark_edge,
        blur_radius: blur,
        saturation: 1.7, // calibrated vibrancy approximation
        corner_radius: corner,
        border_width: 0.0,
        border_color: Rgba::TRANSPARENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn large_glass_is_more_opaque_than_small() {
        let p = palette(Appearance::Dark);
        assert!(glass(&p, Surface::Large).tint.a > glass(&p, Surface::Small).tint.a);
    }

    #[test]
    fn concentric_inner_never_negative_and_subtracts_padding() {
        assert_eq!(radius::concentric_inner(10.0, 16.0), 0.0);
        assert_eq!(radius::concentric_inner(28.0, 8.0), 20.0);
    }

    #[test]
    fn clear_dim_is_the_apple_published_value() {
        assert_eq!(CLEAR_DIM, 0.35);
    }

    #[test]
    fn status_colors_are_distinct_per_appearance() {
        for appearance in [Appearance::Dark, Appearance::Light] {
            let p = palette(appearance);
            assert_ne!(p.success, p.warning);
            assert_ne!(p.warning, p.danger);
            assert_ne!(p.success, p.danger);
        }
    }

    #[test]
    fn base_glass_has_no_border_until_a11y_adds_one() {
        let g = glass(&palette(Appearance::Dark), Surface::Small);
        assert_eq!(g.border_width, 0.0);
        assert_eq!(g.border_color, Rgba::TRANSPARENT);
    }
}
