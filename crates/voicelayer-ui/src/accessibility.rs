//! Accessibility overrides for translucent UI surfaces.
//!
//! Pure functions so both desktop and terminal frontends resolve and test the
//! same degradation rules:
//!
//! - Reduce Transparency raises surface opacity and disables blur.
//! - Increase Contrast switches to an opaque fill with a 1 px contrasting border.
//! - Reduce Motion disables non-essential transitions and shader animation.

use crate::color::Rgba;
use crate::tokens::Glass;

/// Alpha ceiling used when opacity is maximized or Reduce Transparency is active.
const FROSTY_ALPHA: f32 = 0.92;

/// Accessibility and surface opacity preferences.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accessibility {
    pub reduce_transparency: bool,
    pub increase_contrast: bool,
    pub reduce_motion: bool,
    /// User surface opacity slider in `0.0..=1.0` (`0.0` = base tint, `1.0` = frosted).
    pub opacity: f32,
}

impl Default for Accessibility {
    fn default() -> Self {
        Self {
            reduce_transparency: false,
            increase_contrast: false,
            reduce_motion: false,
            opacity: 0.5,
        }
    }
}

impl Accessibility {
    /// Returns `true` when non-essential UI animations are allowed.
    pub fn animations_enabled(&self) -> bool {
        !self.reduce_motion
    }
}

/// Apply accessibility preferences and user opacity to a base surface material.
///
/// `contrast_border` is used for the 1 px hairline when Increase Contrast is enabled.
pub fn resolve_glass(base: Glass, a11y: &Accessibility, contrast_border: Rgba) -> Glass {
    let mut surface = base;

    let opacity = a11y.opacity.clamp(0.0, 1.0);
    let target_alpha = surface.tint.a + (FROSTY_ALPHA - surface.tint.a) * opacity;
    surface.tint = surface.tint.with_alpha(target_alpha);

    if a11y.reduce_transparency {
        surface.tint = surface.tint.with_alpha(surface.tint.a.max(FROSTY_ALPHA));
        surface.blur_radius = 0.0;
    }

    if a11y.increase_contrast {
        surface.tint = surface.tint.with_alpha(1.0);
        surface.blur_radius = 0.0;
        surface.border_width = 1.0;
        surface.border_color = contrast_border;
    }

    surface
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tokens::{Appearance, Surface, glass, palette};

    fn base() -> Glass {
        glass(&palette(Appearance::Dark), Surface::Large)
    }

    fn white() -> Rgba {
        Rgba::rgb8(0xff, 0xff, 0xff)
    }

    #[test]
    fn opacity_zero_keeps_base_alpha() {
        let base = base();
        let a11y = Accessibility {
            opacity: 0.0,
            ..Default::default()
        };
        let resolved = resolve_glass(base, &a11y, white());
        assert!((resolved.tint.a - base.tint.a).abs() < 1e-6);
    }

    #[test]
    fn opacity_one_approaches_frosty() {
        let a11y = Accessibility {
            opacity: 1.0,
            ..Default::default()
        };
        let resolved = resolve_glass(base(), &a11y, white());
        assert!(resolved.tint.a >= 0.9);
    }

    #[test]
    fn reduce_transparency_frosts_and_drops_blur() {
        let a11y = Accessibility {
            reduce_transparency: true,
            opacity: 0.0,
            ..Default::default()
        };
        let resolved = resolve_glass(base(), &a11y, white());
        assert_eq!(resolved.blur_radius, 0.0);
        assert!(resolved.tint.a >= FROSTY_ALPHA);
    }

    #[test]
    fn increase_contrast_adds_opaque_contrasting_border() {
        let border = white();
        let a11y = Accessibility {
            increase_contrast: true,
            ..Default::default()
        };
        let resolved = resolve_glass(base(), &a11y, border);
        assert_eq!(resolved.tint.a, 1.0);
        assert_eq!(resolved.blur_radius, 0.0);
        assert_eq!(resolved.border_width, 1.0);
        assert_eq!(resolved.border_color, border);
    }

    #[test]
    fn reduce_motion_disables_animations() {
        let a11y = Accessibility {
            reduce_motion: true,
            ..Default::default()
        };
        assert!(!a11y.animations_enabled());
    }
}
