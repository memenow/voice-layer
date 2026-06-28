//! VoiceLayer shared UI design tokens — the single source of truth for the
//! Apple "Liquid Glass" design language across the desktop GUI (`vl-desktop`)
//! and the terminal UI (`vl`).
//!
//! Dependency-free by design: rendering backends (iced, crossterm) convert
//! these tokens at their own edge, so neither backend leaks into the shared
//! layer. The GUI maps [`color::Rgba`] to `iced::Color`; the TUI flattens it
//! to an opaque terminal cell via [`terminal::flatten_over`].
//!
//! # Liquid Glass baseline
//!
//! Tokens target the **2026 baseline** (iOS 27 / macOS "Golden Gate"), which is
//! deliberately more opaque than the 2025 reveal. Apple ships the material as
//! system rendering and publishes almost no quantities, so every numeric token
//! marked `calibrated` is our own value, NOT Apple spec. The single
//! Apple-published number is the clear-variant 35% dimming layer
//! ([`tokens::CLEAR_DIM`]).
//!
//! # Honest platform note
//!
//! On GNOME Wayland a client cannot blur the real desktop behind its window, so
//! "glass" here means an app-drawn background with translucent overlay panels —
//! never a true backdrop blur. The accessibility degradation contract
//! ([`a11y`]) is part of the Liquid Glass identity and is wired as pure
//! functions so both targets can resolve and test the final material.

pub mod a11y;
pub mod color;
pub mod terminal;
pub mod tokens;

pub use a11y::Accessibility;
pub use color::Rgba;
pub use tokens::{Appearance, Glass, Palette, Surface};
