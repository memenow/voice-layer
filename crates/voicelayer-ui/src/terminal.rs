//! Terminal (TUI) edge for the shared tokens.
//!
//! A terminal cell has no true translucency or blur, so the TUI "fakes" glass
//! by compositing the translucent film over the base background at design time
//! and emitting the resulting opaque color. The `vl` crate converts the
//! returned `(r, g, b)` to a `crossterm::style::Color` — keeping crossterm out
//! of this shared crate.

use crate::color::Rgba;

/// Flatten a straight-alpha foreground over an opaque background → opaque
/// 8-bit RGB. This is how the TUI approximates a translucent glass film: the
/// composite happens here because the terminal cell is always opaque.
pub fn flatten_over(fg: Rgba, bg: Rgba) -> (u8, u8, u8) {
    let a = fg.a.clamp(0.0, 1.0);
    let mix = |f: f32, b: f32| f * a + b * (1.0 - a);
    Rgba::new(mix(fg.r, bg.r), mix(fg.g, bg.g), mix(fg.b, bg.b), 1.0).to_rgb8()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn opaque_foreground_is_identity() {
        assert_eq!(
            flatten_over(Rgba::rgb8(10, 20, 30), Rgba::rgb8(0, 0, 0)),
            (10, 20, 30)
        );
    }

    #[test]
    fn faint_white_film_over_black_is_dark_gray() {
        let film = Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.10);
        let (r, g, b) = flatten_over(film, Rgba::rgb8(0, 0, 0));
        // 0.10 * 255 ≈ 25.5 → ~26; allow ±1 for f32 rounding.
        assert!((25..=26).contains(&r));
        assert_eq!(r, g);
        assert_eq!(g, b);
    }

    #[test]
    fn fully_transparent_film_shows_background() {
        let film = Rgba::rgb8(0xff, 0xff, 0xff).with_alpha(0.0);
        assert_eq!(flatten_over(film, Rgba::rgb8(7, 8, 9)), (7, 8, 9));
    }
}
