//! Backend-agnostic sRGBA color primitive.
//!
//! Channels are straight (non-premultiplied) alpha in `0.0..=1.0`. Both
//! `iced::Color` (GUI) and terminal colors (TUI) are derived from this type at
//! the edge so the design tokens never pull a rendering backend into this
//! crate.

/// An sRGB color with straight alpha. Channels are `0.0..=1.0`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Rgba {
    pub r: f32,
    pub g: f32,
    pub b: f32,
    pub a: f32,
}

impl Rgba {
    /// Fully transparent — the default "no border" color for glass.
    pub const TRANSPARENT: Rgba = Rgba::new(0.0, 0.0, 0.0, 0.0);

    pub const fn new(r: f32, g: f32, b: f32, a: f32) -> Self {
        Self { r, g, b, a }
    }

    /// Opaque color from 8-bit channels — the ergonomic way to transcribe hex.
    pub fn rgb8(r: u8, g: u8, b: u8) -> Self {
        Self::new(r as f32 / 255.0, g as f32 / 255.0, b as f32 / 255.0, 1.0)
    }

    /// The same color at a new alpha.
    pub const fn with_alpha(self, a: f32) -> Self {
        Self {
            r: self.r,
            g: self.g,
            b: self.b,
            a,
        }
    }

    /// Channel-wise linear interpolation (`t` clamped to `0..=1`).
    pub fn lerp(self, other: Rgba, t: f32) -> Rgba {
        let t = t.clamp(0.0, 1.0);
        Rgba {
            r: self.r + (other.r - self.r) * t,
            g: self.g + (other.g - self.g) * t,
            b: self.b + (other.b - self.b) * t,
            a: self.a + (other.a - self.a) * t,
        }
    }

    /// Straight-alpha → premultiplied. Wayland surfaces on Mesa require
    /// `CompositeAlphaMode::PreMultiplied`, so the transparent-window path must
    /// pre-multiply RGB by alpha before handing colors to wgpu.
    pub fn premultiplied(self) -> Rgba {
        Rgba {
            r: self.r * self.a,
            g: self.g * self.a,
            b: self.b * self.a,
            a: self.a,
        }
    }

    /// Quantize to opaque 8-bit RGB (alpha dropped). Used by the TUI edge.
    pub fn to_rgb8(self) -> (u8, u8, u8) {
        let q = |c: f32| (c.clamp(0.0, 1.0) * 255.0).round() as u8;
        (q(self.r), q(self.g), q(self.b))
    }
}

#[cfg(test)]
mod tests {
    use super::Rgba;

    #[test]
    fn rgb8_roundtrips_through_to_rgb8() {
        assert_eq!(Rgba::rgb8(0x58, 0xa6, 0xff).to_rgb8(), (0x58, 0xa6, 0xff));
    }

    #[test]
    fn lerp_midpoint_averages_channels() {
        let m = Rgba::new(0.0, 0.0, 0.0, 0.0).lerp(Rgba::new(1.0, 1.0, 1.0, 1.0), 0.5);
        assert!((m.r - 0.5).abs() < 1e-6);
        assert!((m.a - 0.5).abs() < 1e-6);
    }

    #[test]
    fn lerp_clamps_t() {
        let a = Rgba::new(0.0, 0.0, 0.0, 0.0);
        let b = Rgba::new(1.0, 1.0, 1.0, 1.0);
        assert_eq!(a.lerp(b, 2.0), b);
        assert_eq!(a.lerp(b, -1.0), a);
    }

    #[test]
    fn premultiplied_scales_rgb_by_alpha() {
        let c = Rgba::new(1.0, 0.5, 0.0, 0.5).premultiplied();
        assert!((c.r - 0.5).abs() < 1e-6);
        assert!((c.g - 0.25).abs() < 1e-6);
        assert!((c.a - 0.5).abs() < 1e-6);
    }
}
