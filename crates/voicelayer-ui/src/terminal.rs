//! Terminal color conversion helpers.
//!
//! Terminal cells are always opaque, so translucent surface tints are
//! alpha-composited over the base background color into 8-bit RGB triples.

use crate::color::Rgba;

/// Composite a straight-alpha foreground color over an opaque background into 8-bit RGB.
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
