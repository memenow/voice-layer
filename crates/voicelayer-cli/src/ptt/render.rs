//! Alternate-screen panel state, layout math, and rendering for foreground PTT.

use std::io::Write;
use std::path::PathBuf;

use crossterm::{
    cursor::MoveTo,
    queue,
    terminal::{Clear, ClearType, size},
};
use unicode_segmentation::UnicodeSegmentation;
use unicode_width::UnicodeWidthStr;
use voicelayer_core::{DictationCaptureResult, StopAction};

use crate::panel_style::{self, PanelTheme};
use crate::terminal_targets::ForegroundInjectionTarget;

#[derive(Debug, Clone)]
pub(super) struct PttUiState {
    pub key_label: String,
    pub target_label: String,
    pub status_label: String,
    pub default_stop_action: StopAction,
    pub save_dir: Option<PathBuf>,
    pub restore_clipboard_on_exit: bool,
    pub active_session_id: Option<String>,
    pub last_session_id: Option<String>,
    pub last_result: Option<DictationCaptureResult>,
    pub last_transcript_full: Option<String>,
    pub last_transcript_preview: Option<String>,
    pub transcript_scroll: usize,
    pub clipboard_backup_text: Option<String>,
    pub last_injection_status: Option<String>,
    pub last_error: Option<String>,
    pub recent_events: Vec<String>,
}

impl PttUiState {
    pub(super) fn new(
        key_label: String,
        target_label: String,
        default_stop_action: StopAction,
        save_dir: Option<PathBuf>,
        restore_clipboard_on_exit: bool,
    ) -> Self {
        Self {
            key_label,
            target_label,
            status_label: "Idle".to_owned(),
            default_stop_action,
            save_dir,
            restore_clipboard_on_exit,
            active_session_id: None,
            last_session_id: None,
            last_result: None,
            last_transcript_full: None,
            last_transcript_preview: None,
            transcript_scroll: 0,
            clipboard_backup_text: None,
            last_injection_status: None,
            last_error: None,
            recent_events: Vec::new(),
        }
    }

    pub(super) fn push_event(&mut self, event: String) {
        self.recent_events.push(event);
        if self.recent_events.len() > 6 {
            let _ = self.recent_events.remove(0);
        }
    }

    pub(super) fn discard_last_transcript(&mut self) {
        self.last_result = None;
        self.last_session_id = None;
        self.last_transcript_full = None;
        self.last_transcript_preview = None;
        self.transcript_scroll = 0;
        self.last_injection_status = None;
        self.last_error = None;
        self.push_event("Discarded last transcript from the panel.".to_owned());
    }
}

pub(super) fn describe_foreground_target(
    target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
) -> String {
    match target {
        ForegroundInjectionTarget::Tmux => tmux_target_pane
            .map(|pane| format!("tmux pane {pane}"))
            .unwrap_or_else(|| "tmux disabled or no target pane".to_owned()),
        ForegroundInjectionTarget::Wezterm { pane_id } => format!("wezterm pane {pane_id}"),
        ForegroundInjectionTarget::Kitty { r#match: selector } => {
            format!("kitty target {selector}")
        }
        ForegroundInjectionTarget::None => "stdout panel only".to_owned(),
    }
}

pub(super) fn truncate_for_panel(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = chars.into_iter().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

pub(super) fn render_panel(ui: &PttUiState) -> Result<(), Box<dyn std::error::Error>> {
    let theme = PanelTheme::dark();
    let event_rows = ui.recent_events.len().max(1);
    let (inner, text_area, transcript_height) = transcript_geometry(event_rows);

    let transcript_lines =
        wrap_for_panel(ui.last_transcript_full.as_deref().unwrap_or("-"), text_area);
    let max_scroll = transcript_lines.len().saturating_sub(transcript_height);
    let scroll = ui.transcript_scroll.min(max_scroll);

    let mut out = std::io::stdout();
    queue!(out, MoveTo(0, 0), Clear(ClearType::All))?;
    render_header_and_fields(&mut out, &theme, inner, ui)?;
    render_transcript(
        &mut out,
        &theme,
        inner,
        &transcript_lines,
        scroll,
        transcript_height,
    )?;
    render_events_and_controls(&mut out, &theme, inner, ui)?;
    out.flush()?;
    Ok(())
}

fn render_header_and_fields(
    out: &mut impl Write,
    theme: &PanelTheme,
    inner: usize,
    ui: &PttUiState,
) -> std::io::Result<()> {
    panel_style::border(
        out,
        theme,
        panel_style::rule(inner, '╭', "VoiceLayer · Foreground PTT", '╮'),
        theme.edge_top,
    )?;

    let tone = panel_style::classify_status(&ui.status_label);
    let status_text = format!(" {}", ui.status_label);
    panel_style::row(
        out,
        theme,
        inner,
        &[(theme.tone(tone), "●"), (theme.text, status_text.as_str())],
    )?;

    let (error_text, error_color) = match ui.last_error.as_deref() {
        Some(error) => (error, theme.danger),
        None => ("-", theme.muted),
    };
    let fields = [
        ("Key", ui.key_label.as_str(), theme.text),
        ("Target", ui.target_label.as_str(), theme.text),
        (
            "Active",
            ui.active_session_id.as_deref().unwrap_or("-"),
            theme.text,
        ),
        (
            "Last",
            ui.last_session_id.as_deref().unwrap_or("-"),
            theme.text,
        ),
        (
            "Injection",
            ui.last_injection_status.as_deref().unwrap_or("-"),
            theme.text,
        ),
        ("Error", error_text, error_color),
    ];
    for (name, value, color) in fields {
        let label = format!("{name:<11}");
        panel_style::row(
            out,
            theme,
            inner,
            &[(theme.muted, label.as_str()), (color, value)],
        )?;
    }
    Ok(())
}

fn render_transcript(
    out: &mut impl Write,
    theme: &PanelTheme,
    inner: usize,
    lines: &[String],
    scroll: usize,
    height: usize,
) -> std::io::Result<()> {
    let title = format!("Transcript · line {} / {}", scroll + 1, lines.len().max(1));
    panel_style::border(
        out,
        theme,
        panel_style::rule(inner, '├', &title, '┤'),
        theme.edge_bottom,
    )?;
    for line in lines.iter().skip(scroll).take(height) {
        panel_style::row(out, theme, inner, &[(theme.text, line.as_str())])?;
    }
    Ok(())
}

fn render_events_and_controls(
    out: &mut impl Write,
    theme: &PanelTheme,
    inner: usize,
    ui: &PttUiState,
) -> std::io::Result<()> {
    panel_style::border(
        out,
        theme,
        panel_style::rule(inner, '├', "Recent events", '┤'),
        theme.edge_bottom,
    )?;
    if ui.recent_events.is_empty() {
        panel_style::row(
            out,
            theme,
            inner,
            &[(theme.muted, "(waiting for activity)")],
        )?;
    } else {
        for event in &ui.recent_events {
            panel_style::row(out, theme, inner, &[(theme.muted, event.as_str())])?;
        }
    }

    panel_style::border(
        out,
        theme,
        panel_style::rule(inner, '├', "Controls", '┤'),
        theme.edge_bottom,
    )?;
    let key = ui.key_label.as_str();
    let controls = [
        format!("{key}: start · release or {key} again: stop · Esc: exit"),
        "j/k or Up/Down: scroll · PgUp/PgDn: jump".to_owned(),
        "c: copy · r: restore · i: inject · s: save · d: discard".to_owned(),
    ];
    for line in &controls {
        panel_style::row(
            out,
            theme,
            inner,
            &[(theme.accent, "▸ "), (theme.muted, line.as_str())],
        )?;
    }
    panel_style::border(
        out,
        theme,
        panel_style::rule(inner, '╰', "", '╯'),
        theme.edge_bottom,
    )?;
    Ok(())
}

fn transcript_geometry(event_rows: usize) -> (usize, usize, usize) {
    let (term_width, term_height) = size().unwrap_or((100, 30));
    transcript_geometry_for(term_width as usize, term_height as usize, event_rows)
}

fn transcript_geometry_for(
    term_width: usize,
    term_height: usize,
    event_rows: usize,
) -> (usize, usize, usize) {
    let total = term_width.clamp(2, 120);
    let inner = total.saturating_sub(2);
    let text_area = inner.saturating_sub(2);
    let chrome = 15 + event_rows;
    let transcript_height = term_height.saturating_sub(chrome).max(3);
    (inner, text_area, transcript_height)
}

fn wrap_for_panel(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    for raw_line in text.lines() {
        if raw_line.is_empty() {
            lines.push(String::new());
            continue;
        }
        let mut current = String::new();
        for grapheme in raw_line.graphemes(true) {
            let mut candidate = current.clone();
            candidate.push_str(grapheme);
            if !current.is_empty() && UnicodeWidthStr::width(candidate.as_str()) > width {
                lines.push(std::mem::take(&mut current));
                current.push_str(grapheme);
            } else {
                current = candidate;
            }
        }
        if !current.is_empty() {
            lines.push(current);
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

pub(super) fn scroll_transcript(ui: &mut PttUiState, delta: isize) -> bool {
    let text = ui.last_transcript_full.as_deref().unwrap_or("-");
    let (_inner, text_area, transcript_height) = transcript_geometry(ui.recent_events.len().max(1));
    let lines = wrap_for_panel(text, text_area);
    let max_scroll = lines.len().saturating_sub(transcript_height) as isize;
    let current = ui.transcript_scroll as isize;
    let next = (current + delta).clamp(0, max_scroll);
    let changed = next != current;
    ui.transcript_scroll = next as usize;
    changed
}

#[cfg(test)]
mod tests {
    use unicode_width::UnicodeWidthStr;

    use super::{transcript_geometry_for, wrap_for_panel};

    #[test]
    fn wrap_for_panel_respects_unicode_display_width() {
        for (text, width, expected) in [
            ("你好世界", 4, vec!["你好", "世界"]),
            ("e\u{301}x", 1, vec!["e\u{301}", "x"]),
            ("👩‍💻ok", 2, vec!["👩‍💻", "ok"]),
        ] {
            let lines = wrap_for_panel(text, width);
            assert_eq!(lines, expected);
            assert!(
                lines
                    .iter()
                    .all(|line| UnicodeWidthStr::width(line.as_str()) <= width)
            );
            assert_eq!(lines.concat(), text);
        }
    }

    #[test]
    fn transcript_geometry_never_exceeds_the_live_terminal_width() {
        for term_width in [2usize, 5, 20, 27, 28, 80, 120, 200] {
            let (inner, text_area, _height) = transcript_geometry_for(term_width, 30, 1);
            let total = inner + 2;
            assert!(total <= term_width.max(2));
            assert!(total <= 120);
            assert!(text_area <= inner);
        }
    }
}
