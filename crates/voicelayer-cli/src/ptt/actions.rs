//! Stop-action side effects for foreground PTT: clipboard backup/copy, transcript save, and injection.
//!
//! On completion, side effects run in fixed order:
//! 1. Copy to clipboard if `--copy-on-stop` or `default_stop_action == copy` (snapshotting prior clipboard once).
//! 2. Save transcript if `default_stop_action == save`.
//! 3. Inject into configured terminal target if `default_stop_action == inject`.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use voicelayer_core::{DictationCaptureResult, SessionState, StopAction};

use crate::terminal_targets::{
    ForegroundInjectionTarget, paste_into_kitty_target, paste_into_tmux_pane,
    paste_into_wezterm_pane,
};

use super::render::{PttUiState, truncate_for_panel};

pub(super) fn apply_dictation_result(
    ui: &mut PttUiState,
    result: &DictationCaptureResult,
    foreground_target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
    copy_on_stop: bool,
) {
    ui.active_session_id = None;
    ui.last_session_id = Some(result.session.session_id.to_string());
    ui.last_result = Some(result.clone());
    ui.status_label = session_state_label(result.session.state).to_owned();
    ui.last_transcript_full = Some(result.transcription.text.clone());
    ui.last_transcript_preview = Some(truncate_for_panel(&result.transcription.text, 220));
    ui.transcript_scroll = 0;
    ui.last_error = None;
    ui.last_injection_status = None;

    if copy_on_stop || matches!(ui.default_stop_action, StopAction::Copy) {
        copy_result_to_clipboard(ui, &result.transcription.text);
    }

    if matches!(ui.default_stop_action, StopAction::Save)
        && let Some(text) = ui.last_transcript_full.clone()
    {
        match save_transcript_to_target_dir(&text, ui.save_dir.as_deref()) {
            Ok(path) => {
                ui.last_injection_status = Some(format!("Saved transcript to {}.", path.display()));
            }
            Err(error) => {
                ui.last_error = Some(format!("Failed to save transcript: {error}"));
            }
        }
    }

    apply_target_injection(
        ui,
        foreground_target,
        tmux_target_pane,
        &result.transcription.text,
    );

    if result.session.state == SessionState::Failed && ui.last_error.is_none() {
        ui.last_error = result.transcription.notes.first().cloned();
    }
}

fn apply_target_injection(
    ui: &mut PttUiState,
    target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
    text: &str,
) {
    let should_inject = matches!(ui.default_stop_action, StopAction::Inject);
    match target {
        ForegroundInjectionTarget::Tmux => {
            if let Some(pane) = tmux_target_pane.filter(|_| should_inject) {
                record_injection(
                    ui,
                    paste_into_tmux_pane(pane, text),
                    &format!("Pasted into tmux pane {pane}."),
                    &format!("tmux pane {pane}"),
                );
            }
        }
        ForegroundInjectionTarget::Wezterm { pane_id } if should_inject => {
            record_injection(
                ui,
                paste_into_wezterm_pane(pane_id, text),
                &format!("Sent to WezTerm pane {pane_id}."),
                &format!("WezTerm pane {pane_id}"),
            );
        }
        ForegroundInjectionTarget::Kitty { r#match: selector } if should_inject => {
            record_injection(
                ui,
                paste_into_kitty_target(selector, text),
                &format!("Sent to Kitty target {selector}."),
                &format!("Kitty target {selector}"),
            );
        }
        ForegroundInjectionTarget::None if ui.last_injection_status.is_none() => {
            ui.last_injection_status = Some(if should_inject {
                "No foreground injection target configured.".to_owned()
            } else {
                "No default injection target configured.".to_owned()
            });
        }
        _ => {}
    }
}

fn record_injection(
    ui: &mut PttUiState,
    outcome: Result<(), Box<dyn std::error::Error>>,
    success_detail: &str,
    failure_label: &str,
) {
    match outcome {
        Ok(()) => {
            let prefix = ui
                .last_injection_status
                .clone()
                .unwrap_or_else(|| "Done.".to_owned());
            ui.last_injection_status = Some(format!("{prefix} {success_detail}"));
        }
        Err(error) => {
            ui.last_error = Some(format!("[injection-failed] {failure_label}: {error}"));
        }
    }
}

fn session_state_label(state: SessionState) -> &'static str {
    match state {
        SessionState::Completed => "Completed",
        SessionState::Failed => "Failed",
        SessionState::Listening => "Listening",
        SessionState::Transcribing => "Transcribing",
        SessionState::Idle => "Idle",
        SessionState::Previewing => "Previewing",
        SessionState::AwaitingConfirmation => "AwaitingConfirmation",
    }
}

pub(super) fn copy_result_to_clipboard(ui: &mut PttUiState, text: &str) {
    if ui.clipboard_backup_text.is_none() {
        ui.clipboard_backup_text = snapshot_clipboard_text();
    }
    match copy_transcript_to_clipboard(text) {
        Ok(()) => {
            ui.last_injection_status = Some("Copied transcript to system clipboard.".to_owned());
            ui.last_error = None;
        }
        Err(error) => {
            ui.last_error = Some(format!("Clipboard copy failed: {error}"));
        }
    }
}

pub(super) fn restore_clipboard_backup(ui: &mut PttUiState) -> Result<(), String> {
    let Some(text) = ui.clipboard_backup_text.clone() else {
        return Err("No text clipboard backup is available to restore.".to_owned());
    };
    copy_transcript_to_clipboard(&text).map_err(|error| error.to_string())?;
    ui.last_injection_status = Some("Restored the saved clipboard text.".to_owned());
    ui.last_error = None;
    Ok(())
}

pub(super) fn save_transcript_to_target_dir(
    text: &str,
    save_dir: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let base_dir = match save_dir {
        Some(dir) => dir.to_path_buf(),
        None => default_save_dir(),
    };
    std::fs::create_dir_all(&base_dir)?;
    let path = unique_transcript_path(&base_dir, timestamp);
    std::fs::write(&path, text)?;
    Ok(path)
}

fn copy_transcript_to_clipboard(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text.to_owned())?;
    Ok(())
}

fn snapshot_clipboard_text() -> Option<String> {
    let mut clipboard = arboard::Clipboard::new().ok()?;
    clipboard.get_text().ok()
}

fn default_save_dir() -> PathBuf {
    if let Some(state) = std::env::var_os("XDG_STATE_HOME") {
        let path = PathBuf::from(state);
        if path.is_absolute() {
            return path.join("voicelayer").join("transcripts");
        }
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home)
            .join(".local")
            .join("state")
            .join("voicelayer")
            .join("transcripts");
    }
    std::env::temp_dir().join("voicelayer").join("transcripts")
}

/// Choose a non-colliding filename `voicelayer-transcript-<timestamp>[-N].txt`.
fn unique_transcript_path(base_dir: &Path, timestamp: u64) -> PathBuf {
    let primary = base_dir.join(format!("voicelayer-transcript-{timestamp}.txt"));
    if !primary.exists() {
        return primary;
    }
    for suffix in 1u32.. {
        let candidate = base_dir.join(format!("voicelayer-transcript-{timestamp}-{suffix}.txt"));
        if !candidate.exists() {
            return candidate;
        }
    }
    unreachable!("u32 suffix space exhausted while saving transcript")
}

#[cfg(test)]
mod tests {
    use std::fs;
    use std::path::PathBuf;

    use voicelayer_testkit::EnvGuard;

    use super::{default_save_dir, unique_transcript_path};

    #[test]
    fn unique_transcript_path_is_primary_when_unused() {
        let dir = tempfile::tempdir().expect("tempdir");
        let path = unique_transcript_path(dir.path(), 1_700_000_000);
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("voicelayer-transcript-1700000000.txt")
        );
    }

    #[test]
    fn unique_transcript_path_appends_suffix_on_collision() {
        let dir = tempfile::tempdir().expect("tempdir");
        let timestamp = 1_700_000_000u64;
        fs::write(
            dir.path()
                .join(format!("voicelayer-transcript-{timestamp}.txt")),
            b"",
        )
        .unwrap();
        fs::write(
            dir.path()
                .join(format!("voicelayer-transcript-{timestamp}-1.txt")),
            b"",
        )
        .unwrap();
        let path = unique_transcript_path(dir.path(), timestamp);
        assert_eq!(
            path.file_name().and_then(|s| s.to_str()),
            Some("voicelayer-transcript-1700000000-2.txt")
        );
    }

    #[test]
    fn default_save_dir_honors_xdg_state_home() {
        let mut env = EnvGuard::lock();
        env.set("XDG_STATE_HOME", "/tmp/vl-xdg-state-probe");
        assert_eq!(
            default_save_dir(),
            PathBuf::from("/tmp/vl-xdg-state-probe/voicelayer/transcripts"),
        );
    }

    #[test]
    fn default_save_dir_falls_back_to_home_state_spec() {
        let mut env = EnvGuard::lock();
        env.remove("XDG_STATE_HOME");
        env.set("HOME", "/tmp/vl-home-probe");
        assert_eq!(
            default_save_dir(),
            PathBuf::from("/tmp/vl-home-probe/.local/state/voicelayer/transcripts"),
        );
    }

    #[test]
    fn default_save_dir_falls_back_to_temp_dir_when_home_unset() {
        let mut env = EnvGuard::lock();
        env.remove("XDG_STATE_HOME");
        env.remove("HOME");
        assert_eq!(
            default_save_dir(),
            std::env::temp_dir().join("voicelayer").join("transcripts"),
        );
    }
}
