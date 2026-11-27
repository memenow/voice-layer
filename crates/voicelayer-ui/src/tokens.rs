//! Shared spacing, typography, radius, motion, palette, and translucent surface tokens.

use crate::color::Rgba;

/// 4pt-base spacing scale.
pub mod space {
    pub const XS: f32 = 4.0;
    pub const SM: f32 = 8.0;
    pub const MD: f32 = 12.0;
    pub const LG: f32 = 16.0;
    pub const XL: f32 = 24.0;
    pub const XXL: f32 = 32.0;
}

/// Corner radii for pills, cards, and panels.
pub mod radius {
    /// Large radius clamped by the renderer to half the control height.
    pub const PILL: f32 = 999.0;
    pub const CARD: f32 = 20.0;
    pub const PANEL: f32 = 28.0;

    /// Concentric inner radius for content inset by `padding` inside `outer`.
    pub fn concentric_inner(outer: f32, padding: f32) -> f32 {
        (outer - padding).max(0.0)
    }
}

/// Font weight tokens mapped onto Inter static faces by the desktop frontend.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Weight {
    Regular,
    Medium,
    Semibold,
    Bold,
}

/// Point sizes for the type scale.
pub mod text {
    pub const CAPTION: f32 = 12.0;
    pub const BODY: f32 = 14.0;
    pub const TITLE: f32 = 17.0;
    pub const LARGE_TITLE: f32 = 28.0;
}

/// Transition curves and durations in milliseconds.
pub mod motion {
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

/// Color palette for a light or dark appearance.
#[derive(Debug, Clone, Copy)]
pub struct Palette {
    pub bg_base: Rgba,
    pub bg_elevated: Rgba,
    pub text_primary: Rgba,
    pub text_secondary: Rgba,
    pub accent: Rgba,
    pub success: Rgba,
    pub warning: Rgba,
    pub danger: Rgba,
    /// Base translucent fill drawn above the window backdrop.
    pub glass_tint: Rgba,
    /// Inner top specular highlight stroke.
    pub glass_highlight: Rgba,
    /// Outer darkened edge for depth separation.
    pub glass_dark_edge: Rgba,
}

/// Light or dark color scheme selector.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Appearance {
    Dark,
    Light,
}

/// Resolve the palette for `appearance`.
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
        accent: Rgba::rgb8(0x58, 0xa6, 0xff),
        success: Rgba::rgb8(0x3f, 0xb9, 0x50),
        warning: Rgba::rgb8(0xd2, 0x99, 0x22),
        danger: Rgba::rgb8(0xf8, 0x51, 0x49),
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
        accent: Rgba::rgb8(0x1f, 0x6f, 0xeb),
        success: Rgba::rgb8(0x1a, 0x7f, 0x37),
        warning: Rgba::rgb8(0x9a, 0x67, 0x00),
        danger: Rgba::rgb8(0xcf, 0x22, 0x2e),
        glass_tint: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.55),
        glass_highlight: Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.65),
        glass_dark_edge: Rgba::rgb8(0x00, 0x00, 0x00).with_alpha(0.10),
    }
}

/// Dimming factor applied over bright content under clear translucent surfaces.
pub const CLEAR_DIM: f32 = 0.35;

/// Surface size class. Larger surfaces use higher opacity for text legibility.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Surface {
    /// Pills, toolbars, and badges.
    Small,
    /// Sidebars, sheets, and cards.
    Large,
}

/// Resolved translucent surface parameters prior to accessibility overrides.
#[derive(Debug, Clone, Copy)]
pub struct Glass {
    pub tint: Rgba,
    pub highlight: Rgba,
    pub dark_edge: Rgba,
    pub blur_radius: f32,
    pub saturation: f32,
    pub corner_radius: f32,
    pub border_width: f32,
    pub border_color: Rgba,
}

/// Build the base surface parameters for `surface` from `palette`.
pub fn glass(palette: &Palette, surface: Surface) -> Glass {
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
        saturation: 1.7,
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
    fn clear_dim_matches_expected_factor() {
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
    fn base_glass_has_no_border_until_accessibility_adds_one() {
        let g = glass(&palette(Appearance::Dark), Surface::Small);
        assert_eq!(g.border_width, 0.0);
        assert_eq!(g.border_color, Rgba::TRANSPARENT);
    }
}
