//! Liquid Glass styling for the terminal UI.
//!
//! A terminal cell has no true translucency, blur, or refraction, so this is an
//! honest *approximation* of the Liquid Glass language built only from what a
//! TUI can actually render: a single rounded "glass card" whose translucent
//! film is flattened to one opaque cell color, a brighter top "lensed" edge
//! over a darker 2026 outer edge, accent-tinted titles, and colored status
//! badges. There is deliberately no faked blur — the depth cue is color, not a
//! sampled backdrop.
//!
//! The backend-agnostic design tokens live in `voicelayer-ui`; this module is
//! the crossterm edge that turns those `Rgba` tokens into
//! [`crossterm::style::Color`], keeping crossterm out of the shared crate (as
//! [`voicelayer_ui::terminal`] documents).

use std::io::Write;

use crossterm::queue;
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use voicelayer_ui::terminal::flatten_over;
use voicelayer_ui::tokens::{self, Appearance, Surface};
use voicelayer_ui::{Palette, Rgba};

/// Flatten a straight-alpha film over an opaque base into an opaque color — how
/// the TUI fakes a translucent glass layer in a cell that cannot be see-through.
fn over(film: Rgba, base: Rgba) -> Rgba {
    let (r, g, b) = flatten_over(film, base);
    Rgba::rgb8(r, g, b)
}

/// Quantize an opaque token color into a crossterm truecolor cell.
fn rgb(color: Rgba) -> Color {
    let (r, g, b) = color.to_rgb8();
    Color::Rgb { r, g, b }
}

/// Crossterm colors resolved once from the dark Liquid Glass palette.
#[derive(Debug, Clone, Copy)]
pub struct GlassTheme {
    /// Glass-card interior: the translucent film flattened over the app-drawn
    /// background, emitted as one opaque cell color.
    pub panel: Color,
    /// Brighter top / inner "lensed" highlight edge.
    pub edge_top: Color,
    /// Darker 2026 outer edge (sides, dividers, and bottom).
    pub edge_bottom: Color,
    /// Accent for titles and control affordances (tint-as-emphasis).
    pub accent: Color,
    pub text: Color,
    pub muted: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
}

impl GlassTheme {
    /// Resolve the dark theme — the primary target, since glass reads best on a
    /// deep background and a dictation tool is commonly dark.
    pub fn dark() -> Self {
        Self::from_palette(tokens::palette(Appearance::Dark))
    }

    fn from_palette(palette: Palette) -> Self {
        let base = palette.bg_base;
        // The whole card is a large surface, so it uses the most opaque film.
        let film = tokens::glass(&palette, Surface::Large).tint;
        let panel = over(film, base);
        GlassTheme {
            panel: rgb(panel),
            edge_top: rgb(over(palette.glass_highlight, panel)),
            edge_bottom: rgb(over(palette.glass_dark_edge, panel)),
            accent: rgb(palette.accent),
            text: rgb(palette.text_primary),
            muted: rgb(palette.text_secondary),
            success: rgb(palette.success),
            warning: rgb(palette.warning),
            danger: rgb(palette.danger),
        }
    }

    /// The badge color for a session status tone.
    pub fn tone(&self, tone: StatusTone) -> Color {
        match tone {
            StatusTone::Active => self.accent,
            StatusTone::Working => self.warning,
            StatusTone::Done => self.success,
            StatusTone::Error => self.danger,
            StatusTone::Idle => self.muted,
        }
    }
}

/// Semantic tone for a session status label, decoupling the status strings the
/// event loop maintains from the concrete palette colors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusTone {
    Idle,
    Active,
    Working,
    Done,
    Error,
}

/// Classify the status label that [`crate::foreground_ptt`] sets into a tone.
/// The labels mirror `voicelayer_core::SessionState` rendered as text.
pub fn classify_status(status: &str) -> StatusTone {
    match status {
        "Listening" => StatusTone::Active,
        "Transcribing" | "Previewing" | "AwaitingConfirmation" => StatusTone::Working,
        "Completed" => StatusTone::Done,
        "Failed" => StatusTone::Error,
        _ => StatusTone::Idle,
    }
}

/// A horizontal border split into colorable segments: the rounded rule
/// (`lead` + `trail`, drawn in an edge color) and an optional inline `label`
/// (drawn in the accent). Concatenated, the three span `inner + 2` columns.
#[derive(Debug, Clone)]
pub struct Rule {
    pub lead: String,
    pub label: String,
    pub trail: String,
}

/// Build a plain or titled horizontal border between the `left` and `right`
/// corner glyphs, spanning `inner` rule columns between them. An empty `label`
/// yields a plain rule; a non-empty one is inlined and truncated to fit while
/// always leaving at least one trailing dash.
pub fn rule(inner: usize, left: char, label: &str, right: char) -> Rule {
    if label.is_empty() {
        return Rule {
            lead: left.to_string(),
            label: String::new(),
            trail: format!("{}{right}", "─".repeat(inner)),
        };
    }
    let label = truncate_cols(label, inner.saturating_sub(4));
    let fill = inner.saturating_sub(label.chars().count() + 3);
    Rule {
        lead: format!("{left}─ "),
        label,
        trail: format!(" {}{right}", "─".repeat(fill)),
    }
}

/// Truncate `text` to at most `max` columns. The labels here are plain status
/// and section strings, so a char count is a faithful column count.
pub fn truncate_cols(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        return text.to_owned();
    }
    text.chars().take(max).collect()
}

/// Draw a glass-card border row (top, divider, or bottom) in the `edge` color,
/// the optional inline label in the accent, all over the panel film.
pub fn border(
    out: &mut impl Write,
    theme: &GlassTheme,
    r: Rule,
    edge: Color,
) -> std::io::Result<()> {
    queue!(
        out,
        SetBackgroundColor(theme.panel),
        SetForegroundColor(edge),
        Print(r.lead)
    )?;
    if r.label.is_empty() {
        queue!(out, Print(r.trail))?;
    } else {
        queue!(out, SetForegroundColor(theme.accent), Print(r.label))?;
        queue!(out, SetForegroundColor(edge), Print(r.trail))?;
    }
    queue!(out, ResetColor, Print("\n"))?;
    Ok(())
}

/// Draw one interior content row of the glass card: the side bars in the edge
/// color and the `segments` laid left-to-right, each in its own color, over the
/// panel film. Content is truncated to the card width and the remainder padded,
/// so every row is exactly `inner + 2` columns wide.
pub fn row(
    out: &mut impl Write,
    theme: &GlassTheme,
    inner: usize,
    segments: &[(Color, &str)],
) -> std::io::Result<()> {
    let area = inner.saturating_sub(2);
    queue!(
        out,
        SetBackgroundColor(theme.panel),
        SetForegroundColor(theme.edge_bottom),
        Print("│ ")
    )?;
    let mut used = 0usize;
    for (color, text) in segments {
        let remaining = area.saturating_sub(used);
        if remaining == 0 {
            break;
        }
        let piece = truncate_cols(text, remaining);
        used += piece.chars().count();
        queue!(out, SetForegroundColor(*color), Print(piece))?;
    }
    if used < area {
        queue!(out, Print(" ".repeat(area - used)))?;
    }
    queue!(
        out,
        SetForegroundColor(theme.edge_bottom),
        Print(" │"),
        ResetColor,
        Print("\n")
    )?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rule_width(r: &Rule) -> usize {
        r.lead.chars().count() + r.label.chars().count() + r.trail.chars().count()
    }

    #[test]
    fn titled_rule_spans_inner_plus_two_corners() {
        let r = rule(40, '╭', "Session", '╮');
        assert_eq!(rule_width(&r), 42);
        assert!(r.lead.starts_with('╭'));
        assert!(r.trail.ends_with('╮'));
        assert_eq!(r.label, "Session");
    }

    #[test]
    fn plain_rule_has_no_label_and_spans_width() {
        let r = rule(10, '╰', "", '╯');
        assert!(r.label.is_empty());
        assert_eq!(rule_width(&r), 12);
        assert!(r.lead.starts_with('╰'));
        assert!(r.trail.ends_with('╯'));
    }

    #[test]
    fn overlong_label_is_truncated_but_keeps_a_trailing_dash() {
        let r = rule(12, '├', "an overly long section title", '┤');
        assert_eq!(rule_width(&r), 14);
        assert!(r.trail.chars().filter(|&c| c == '─').count() >= 1);
        assert!(r.trail.ends_with('┤'));
    }

    #[test]
    fn truncate_cols_caps_and_passes_through() {
        assert_eq!(truncate_cols("hello", 5), "hello");
        assert_eq!(truncate_cols("hello world", 5), "hello");
        assert_eq!(truncate_cols("hi", 5), "hi");
    }

    #[test]
    fn classify_status_maps_lifecycle_labels() {
        assert_eq!(classify_status("Listening"), StatusTone::Active);
        assert_eq!(classify_status("Transcribing"), StatusTone::Working);
        assert_eq!(classify_status("AwaitingConfirmation"), StatusTone::Working);
        assert_eq!(classify_status("Completed"), StatusTone::Done);
        assert_eq!(classify_status("Failed"), StatusTone::Error);
        assert_eq!(classify_status("Idle"), StatusTone::Idle);
        assert_eq!(classify_status("anything else"), StatusTone::Idle);
    }

    #[test]
    fn row_emits_two_side_bars_and_its_content() {
        let theme = GlassTheme::dark();
        let mut buf: Vec<u8> = Vec::new();
        row(&mut buf, &theme, 20, &[(theme.text, "hello")]).unwrap();
        let rendered = String::from_utf8(buf).unwrap();
        assert_eq!(rendered.matches('│').count(), 2);
        assert!(rendered.contains("hello"));
    }
}
