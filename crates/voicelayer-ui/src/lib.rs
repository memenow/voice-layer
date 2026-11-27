//! Shared UI design tokens for the desktop shell (`vl-desktop`) and terminal
//! panel (`vl`).
//!
//! This crate has no rendering dependencies. Each frontend maps [`color::Rgba`]
//! and [`tokens::Palette`] into its own primitives (iced colors in the GUI,
//! flattened opaque RGB via [`terminal::flatten_over`] in the terminal).

pub mod accessibility;
pub mod color;
pub mod terminal;
pub mod tokens;

pub use accessibility::Accessibility;
pub use color::Rgba;
pub use tokens::{Appearance, Glass, Palette, Surface};
