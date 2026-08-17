//! The Liquid Glass accessibility degradation contract.
//!
//! Apple applies these fallbacks automatically inside the system material;
//! off-platform we must wire each token to the OS flags ourselves. Everything
//! here is a pure function so both the GUI and TUI can resolve and unit-test
//! the final material:
//!
//! - **Reduce Transparency** → frostier, blur dropped.
//! - **Increase Contrast** → opaque fill + contrasting hard border.
//! - **Reduce Motion** → elastic/morph effects disabled.
//!
//! Plus a user **opacity** control (the 2026 Settings slider) that doubles as an
//! accessibility escape hatch.

use crate::color::Rgba;
use crate::tokens::Glass;

/// Calibrated near-opaque ceiling reached when the user opacity slider is at
/// max or Reduce Transparency is on.
const FROSTY_ALPHA: f32 = 0.92;

/// Current accessibility / appearance preferences, typically read from the
/// desktop environment (GNOME a11y settings / the XDG settings portal) with a
/// user override exposed in the settings UI.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Accessibility {
    pub reduce_transparency: bool,
    pub increase_contrast: bool,
    pub reduce_motion: bool,
    /// User opacity control. `0.0` = clearest allowed, `1.0` = fully frosted.
    /// The 2026 baseline ships such a slider and defaults more opaque than the
    /// 2025 reveal.
    pub opacity: f32,
}

impl Default for Accessibility {
    fn default() -> Self {
        // 2026 baseline sits noticeably more opaque than the 2025 reveal.
        Self {
            reduce_transparency: false,
            increase_contrast: false,
            reduce_motion: false,
            opacity: 0.5,
        }
    }
}

impl Accessibility {
    /// Whether elastic / morph / highlight-travel animation should run.
    pub fn animations_enabled(&self) -> bool {
        !self.reduce_motion
    }
}

/// Resolve the final glass material after applying the accessibility contract.
///
/// `contrast_border` is the color used for the hard hairline when Increase
/// Contrast is on (typically the surface's primary text color).
pub fn resolve_glass(base: Glass, a11y: &Accessibility, contrast_border: Rgba) -> Glass {
    let mut glass = base;

    // User opacity slider: raise the tint alpha toward frosty as opacity → 1.
    let opacity = a11y.opacity.clamp(0.0, 1.0);
    let target_alpha = glass.tint.a + (FROSTY_ALPHA - glass.tint.a) * opacity;
    glass.tint = glass.tint.with_alpha(target_alpha);

    if a11y.reduce_transparency {
        // "Makes Liquid Glass frostier and obscures more of the content behind."
        glass.tint = glass.tint.with_alpha(glass.tint.a.max(FROSTY_ALPHA));
        glass.blur_radius = 0.0;
    }

    if a11y.increase_contrast {
        // "Predominantly black or white ... highlighted with a contrasting
        // border." Opaque fill, no blur, hard hairline.
        glass.tint = glass.tint.with_alpha(1.0);
        glass.blur_radius = 0.0;
        glass.border_width = 1.0;
        glass.border_color = contrast_border;
    }

    glass
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
