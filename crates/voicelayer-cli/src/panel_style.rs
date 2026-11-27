//! Terminal card theme and box-drawing helpers derived from `voicelayer-ui` tokens.

use std::io::Write;

use crossterm::queue;
use crossterm::style::{Color, Print, ResetColor, SetBackgroundColor, SetForegroundColor};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use voicelayer_ui::terminal::flatten_over;
use voicelayer_ui::tokens::{self, Appearance, Surface};
use voicelayer_ui::{Palette, Rgba};

fn composite(film: Rgba, base: Rgba) -> Rgba {
    let (r, g, b) = flatten_over(film, base);
    Rgba::rgb8(r, g, b)
}

fn to_crossterm_color(color: Rgba) -> Color {
    let (r, g, b) = color.to_rgb8();
    Color::Rgb { r, g, b }
}

/// Crossterm truecolor palette resolved from [`voicelayer_ui`] tokens.
#[derive(Debug, Clone, Copy)]
pub struct PanelTheme {
    pub panel: Color,
    pub edge_top: Color,
    pub edge_bottom: Color,
    pub accent: Color,
    pub text: Color,
    pub muted: Color,
    pub success: Color,
    pub warning: Color,
    pub danger: Color,
}

impl PanelTheme {
    pub fn dark() -> Self {
        Self::from_palette(tokens::palette(Appearance::Dark))
    }

    fn from_palette(palette: Palette) -> Self {
        let base = palette.bg_base;
        let film = tokens::glass(&palette, Surface::Large).tint;
        let panel = composite(film, base);
        Self {
            panel: to_crossterm_color(panel),
            edge_top: to_crossterm_color(composite(palette.glass_highlight, panel)),
            edge_bottom: to_crossterm_color(composite(palette.glass_dark_edge, panel)),
            accent: to_crossterm_color(palette.accent),
            text: to_crossterm_color(palette.text_primary),
            muted: to_crossterm_color(palette.text_secondary),
            success: to_crossterm_color(palette.success),
            warning: to_crossterm_color(palette.warning),
            danger: to_crossterm_color(palette.danger),
        }
    }

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

/// Semantic tone for a session status badge.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StatusTone {
    Idle,
    Active,
    Working,
    Done,
    Error,
}

/// Map a rendered [`voicelayer_core::SessionState`] label onto a [`StatusTone`].
pub fn classify_status(status: &str) -> StatusTone {
    match status {
        "Listening" => StatusTone::Active,
        "Transcribing" | "Previewing" | "AwaitingConfirmation" => StatusTone::Working,
        "Completed" => StatusTone::Done,
        "Failed" => StatusTone::Error,
        _ => StatusTone::Idle,
    }
}

/// Horizontal card border split into colorable segments spanning `inner + 2` columns.
#[derive(Debug, Clone)]
pub struct Rule {
    pub lead: String,
    pub label: String,
    pub trail: String,
}

/// Build a plain or titled horizontal border spanning `inner` columns between corner glyphs.
pub fn rule(inner: usize, left: char, label: &str, right: char) -> Rule {
    if label.is_empty() {
        return Rule {
            lead: left.to_string(),
            label: String::new(),
            trail: format!("{}{right}", "─".repeat(inner)),
        };
    }
    let label = truncate_cols(label, inner.saturating_sub(4));
    let fill = inner.saturating_sub(UnicodeWidthStr::width(label.as_str()) + 3);
    Rule {
        lead: format!("{left}─ "),
        label,
        trail: format!(" {}{right}", "─".repeat(fill)),
    }
}

/// Truncate `text` to at most `max` display columns without splitting grapheme clusters.
pub fn truncate_cols(text: &str, max: usize) -> String {
    if UnicodeWidthStr::width(text) <= max {
        return text.to_owned();
    }
    let mut end = 0;
    for (index, grapheme) in text.grapheme_indices(true) {
        let candidate_end = index + grapheme.len();
        // Check every boundary because ligatures can make a longer prefix narrower.
        if UnicodeWidthStr::width(&text[..candidate_end]) <= max {
            end = candidate_end;
        }
    }
    text[..end].to_owned()
}

/// Draw a top, divider, or bottom border row.
pub fn border(
    out: &mut impl Write,
    theme: &PanelTheme,
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

/// Draw one content row padded to exactly `inner + 2` columns.
pub fn row(
    out: &mut impl Write,
    theme: &PanelTheme,
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
        used += UnicodeWidthStr::width(piece.as_str());
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
    use unicode_width::UnicodeWidthStr;

    fn rule_width(r: &Rule) -> usize {
        UnicodeWidthStr::width(r.lead.as_str())
            + UnicodeWidthStr::width(r.label.as_str())
            + UnicodeWidthStr::width(r.trail.as_str())
    }

    fn strip_ansi_sequences(input: &str) -> String {
        let bytes = input.as_bytes();
        let mut output = String::new();
        let mut index = 0;
        while index < bytes.len() {
            if bytes[index] == 0x1b && bytes.get(index + 1) == Some(&b'[') {
                index += 2;
                while index < bytes.len() {
                    let byte = bytes[index];
                    index += 1;
                    if (0x40..=0x7e).contains(&byte) {
                        break;
                    }
                }
                continue;
            }
            let character = input[index..].chars().next().expect("valid char boundary");
            output.push(character);
            index += character.len_utf8();
        }
        output
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
    }

    #[test]
    fn overlong_label_is_truncated_but_keeps_a_trailing_dash() {
        let r = rule(12, '├', "an overly long section title", '┤');
        assert_eq!(rule_width(&r), 14);
        assert!(r.trail.chars().filter(|&c| c == '─').count() >= 1);
    }

    #[test]
    fn truncate_cols_respects_unicode_display_width() {
        assert_eq!(truncate_cols("hello", 5), "hello");
        assert_eq!(truncate_cols("hello world", 5), "hello");
        assert_eq!(truncate_cols("你好世界", 4), "你好");
        assert_eq!(truncate_cols("e\u{301}x", 1), "e\u{301}");
        assert_eq!(truncate_cols("👩‍💻!", 2), "👩‍💻");
        assert_eq!(truncate_cols("✈️!", 1), "");
        assert_eq!(truncate_cols("ⵏ\u{2D7F}ⴾ!", 1), "ⵏ\u{2D7F}ⴾ");
    }

    #[test]
    fn wide_rule_label_respects_unicode_display_width() {
        let r = rule(12, '├', "状态正常", '┤');
        assert_eq!(rule_width(&r), 14);
        assert!(r.trail.contains('─'));
        assert!(r.trail.ends_with('┤'));
    }

    #[test]
    fn classify_status_maps_lifecycle_labels() {
        assert_eq!(classify_status("Listening"), StatusTone::Active);
        assert_eq!(classify_status("Transcribing"), StatusTone::Working);
        assert_eq!(classify_status("AwaitingConfirmation"), StatusTone::Working);
        assert_eq!(classify_status("Completed"), StatusTone::Done);
        assert_eq!(classify_status("Failed"), StatusTone::Error);
        assert_eq!(classify_status("Idle"), StatusTone::Idle);
        assert_eq!(classify_status("unknown"), StatusTone::Idle);
    }

    #[test]
    fn row_respects_unicode_display_width_across_segments() {
        let theme = PanelTheme::dark();
        let mut buf: Vec<u8> = Vec::new();
        row(
            &mut buf,
            &theme,
            8,
            &[(theme.text, "你好"), (theme.muted, "abc")],
        )
        .unwrap();
        let rendered = String::from_utf8(buf).unwrap();
        let visible = strip_ansi_sequences(&rendered);
        let visible = visible.trim_end_matches('\n');
        assert_eq!(visible, "│ 你好ab │");
        assert_eq!(UnicodeWidthStr::width(visible), 10);
    }
}
