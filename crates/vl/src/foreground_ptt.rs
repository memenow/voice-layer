//! Foreground push-to-talk event loop and the side-effect policies that govern it.
//!
//! # Stop-action ordering
//!
//! When a dictation session transitions to `Completed` (either via key-release or a toggle
//! press), [`apply_dictation_result_to_ui`] performs side effects in a deterministic order:
//!
//! 1. Copy-on-stop: if the CLI flag `--copy-on-stop` is set or the effective
//!    `default_stop_action` is `copy`, the transcript is placed on the system clipboard.
//!    Before the first such copy in a session, the pre-existing clipboard contents are
//!    snapshotted into `clipboard_backup_text`.
//! 2. Save-on-stop: if the effective `default_stop_action` is `save`, the transcript is
//!    written to disk via [`save_transcript_to_target_dir`].
//! 3. Inject-on-stop: if the effective `default_stop_action` is `inject`, the transcript is
//!    pushed into the configured foreground target (tmux / wezterm / kitty).
//!
//! `default_stop_action` is a single enum; setting it to `none` disables the save and inject
//! branches entirely. The copy branch is independent — `--copy-on-stop` is additive and can
//! combine with any non-`none` stop action (for example, `default_stop_action=save` together
//! with `--copy-on-stop=true` will both save and copy).
//!
//! # Clipboard-backup lifecycle
//!
//! `clipboard_backup_text` is populated lazily the first time [`copy_result_to_clipboard`] or
//! the manual `c` key is triggered during a session, so the user can recover the text that
//! VoiceLayer replaced. `r` restores the snapshot on demand at any point.
//!
//! When `--restore-clipboard-on-exit` is set, the loop calls [`restore_clipboard_backup`]
//! just before breaking out of the event loop (including the Esc path after a final stop),
//! so the pre-session clipboard text is put back. With both `--copy-on-stop` and
//! `--restore-clipboard-on-exit`, the restore happens only at exit — during the session the
//! transcript stays on the clipboard so the user can paste it.
//!
//! # Save-filename collision policy
//!
//! [`save_transcript_to_target_dir`] writes to `voicelayer-transcript-<epoch_secs>.txt`.
//! If the base name already exists (for example, two stops in the same second), the writer
//! appends `-1`, `-2`, … until an unused name is found. The first free candidate wins; we do
//! not overwrite existing transcripts.

use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{
        Event, KeyCode, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags, read,
    },
    execute,
    style::Print,
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode, size,
    },
};
use voicelayer_core::{
    DictationCaptureResult, DictationFailureKind, LanguageProfile, LanguageStrategy,
    SegmentationMode, SessionState, StartDictationRequest, StopDictationRequest, TriggerKind,
};

use crate::config::{CliPttKey, CliRecorderBackend, StopAction};
use crate::terminal_targets::{
    ForegroundInjectionTarget, paste_into_kitty_target, paste_into_tmux_pane,
    paste_into_wezterm_pane, resolve_foreground_injection_target, resolve_tmux_target_pane,
};
use crate::uds::{cli_socket_path, uds_post_json};

#[allow(clippy::too_many_arguments)]
pub(crate) async fn run_foreground_ptt(
    language: Option<String>,
    backend: CliRecorderBackend,
    translate_to_english: bool,
    keep_audio: bool,
    key: CliPttKey,
    tmux_target_pane: Option<String>,
    wezterm_target_pane_id: Option<String>,
    kitty_match: Option<String>,
    copy_on_stop: bool,
    default_stop_action: StopAction,
    restore_clipboard_on_exit: bool,
    save_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let foreground_target = resolve_foreground_injection_target(
        tmux_target_pane.clone(),
        wezterm_target_pane_id,
        kitty_match,
    )?;
    let tmux_target_pane = resolve_tmux_target_pane(tmux_target_pane)?;
    let _raw_mode = RawTerminalMode::enter()?;
    let mut ui = ForegroundPttUiState::new(
        key.label().to_owned(),
        describe_foreground_target(&foreground_target, tmux_target_pane.as_deref()),
        default_stop_action,
        save_dir,
        restore_clipboard_on_exit,
    );
    ui.push_event(format!(
        "Press {} to start. Release to stop when supported; otherwise press {} again. Esc exits.",
        key.label(),
        key.label()
    ));
    render_foreground_ptt_ui(&ui)?;
    let mut active_session: Option<uuid::Uuid> = None;

    loop {
        let event = read()?;
        let Event::Key(key_event) = event else {
            continue;
        };

        if key_event.code == KeyCode::Esc {
            if let Some(session_id) = active_session.take() {
                let result: DictationCaptureResult = uds_post_json(
                    &cli_socket_path(),
                    "/v1/sessions/dictation/stop",
                    &StopDictationRequest { session_id },
                )
                .await?;
                apply_dictation_result_to_ui(
                    &mut ui,
                    &result,
                    &foreground_target,
                    tmux_target_pane.as_deref(),
                    copy_on_stop,
                );
                ui.push_event("Stopped active session before exit.".to_owned());
                render_foreground_ptt_ui(&ui)?;
            }
            if ui.restore_clipboard_on_exit {
                let _ = restore_clipboard_backup(&mut ui);
            }
            ui.push_event("Foreground PTT exited.".to_owned());
            break;
        }

        if key_event.code != key.as_key_code() {
            if key_event.kind == KeyEventKind::Press {
                match key_event.code {
                    KeyCode::Char('c') => {
                        if let Some(text) = ui.last_transcript_full.clone() {
                            if ui.clipboard_backup_text.is_none() {
                                ui.clipboard_backup_text = snapshot_clipboard_text();
                            }
                            match copy_transcript_to_clipboard(&text) {
                                Ok(()) => {
                                    ui.last_injection_status = Some(
                                        "Copied last transcript to system clipboard.".to_owned(),
                                    );
                                    ui.last_error = None;
                                    ui.push_event(
                                        "Copied last transcript to clipboard.".to_owned(),
                                    );
                                }
                                Err(error) => {
                                    ui.last_error = Some(format!("Clipboard copy failed: {error}"));
                                }
                            }
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('r') => {
                        match restore_clipboard_backup(&mut ui) {
                            Ok(()) => {
                                ui.push_event("Restored clipboard from saved backup.".to_owned());
                            }
                            Err(error) => {
                                ui.last_error = Some(error);
                            }
                        }
                        render_foreground_ptt_ui(&ui)?;
                    }
                    KeyCode::Char('i') => {
                        if let Some(result) = ui.last_result.clone() {
                            apply_dictation_result_to_ui(
                                &mut ui,
                                &result,
                                &foreground_target,
                                tmux_target_pane.as_deref(),
                                false,
                            );
                            ui.push_event("Re-applied last transcript injection.".to_owned());
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('s') => {
                        if let Some(text) = ui.last_transcript_full.clone() {
                            match save_transcript_to_file(&text) {
                                Ok(path) => {
                                    ui.last_injection_status =
                                        Some(format!("Saved transcript to {}.", path.display()));
                                    ui.last_error = None;
                                    ui.push_event(format!(
                                        "Saved last transcript to {}.",
                                        path.display()
                                    ));
                                }
                                Err(error) => {
                                    ui.last_error =
                                        Some(format!("Failed to save transcript: {error}"));
                                }
                            }
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('d') => {
                        ui.last_result = None;
                        ui.last_session_id = None;
                        ui.last_transcript_full = None;
                        ui.last_transcript_preview = None;
                        ui.transcript_scroll = 0;
                        ui.last_injection_status = None;
                        ui.last_error = None;
                        ui.push_event("Discarded last transcript from the panel.".to_owned());
                        render_foreground_ptt_ui(&ui)?;
                    }
                    KeyCode::Char('j') | KeyCode::Down => {
                        if scroll_transcript(&mut ui, 1) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        if scroll_transcript(&mut ui, -1) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::PageDown => {
                        if scroll_transcript(&mut ui, 6) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::PageUp => {
                        if scroll_transcript(&mut ui, -6) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    _ => {}
                }
            }
            continue;
        }

        match key_event.kind {
            KeyEventKind::Press => {
                if active_session.is_none() {
                    let request = StartDictationRequest {
                        trigger: TriggerKind::Cli,
                        language_profile: language.clone().map(|language| LanguageProfile {
                            strategy: LanguageStrategy::Locked,
                            input_languages: vec![language],
                            output_language: None,
                        }),
                        recorder_backend: Some(backend.into()),
                        translate_to_english,
                        keep_audio,
                        segmentation: SegmentationMode::default(),
                    };
                    let session: voicelayer_core::CaptureSession =
                        uds_post_json(&cli_socket_path(), "/v1/sessions/dictation", &request)
                            .await?;
                    active_session = Some(session.session_id);
                    ui.status_label = "Listening".to_owned();
                    ui.active_session_id = Some(session.session_id.to_string());
                    ui.last_error = None;
                    ui.push_event(format!("Dictation started: {}", session.session_id));
                    render_foreground_ptt_ui(&ui)?;
                } else if let Some(session_id) = active_session.take() {
                    ui.status_label = "Transcribing".to_owned();
                    ui.push_event(format!(
                        "Stopping session {session_id} via key press toggle.",
                    ));
                    render_foreground_ptt_ui(&ui)?;
                    let result: DictationCaptureResult = uds_post_json(
                        &cli_socket_path(),
                        "/v1/sessions/dictation/stop",
                        &StopDictationRequest { session_id },
                    )
                    .await?;
                    apply_dictation_result_to_ui(
                        &mut ui,
                        &result,
                        &foreground_target,
                        tmux_target_pane.as_deref(),
                        copy_on_stop,
                    );
                    ui.push_event("Dictation stopped via key press toggle.".to_owned());
                    render_foreground_ptt_ui(&ui)?;
                }
            }
            KeyEventKind::Release => {
                if let Some(session_id) = active_session.take() {
                    ui.status_label = "Transcribing".to_owned();
                    ui.push_event(format!("Stopping session {session_id} via key release."));
                    render_foreground_ptt_ui(&ui)?;
                    let result: DictationCaptureResult = uds_post_json(
                        &cli_socket_path(),
                        "/v1/sessions/dictation/stop",
                        &StopDictationRequest { session_id },
                    )
                    .await?;
                    apply_dictation_result_to_ui(
                        &mut ui,
                        &result,
                        &foreground_target,
                        tmux_target_pane.as_deref(),
                        copy_on_stop,
                    );
                    ui.push_event("Dictation stopped via key release.".to_owned());
                    render_foreground_ptt_ui(&ui)?;
                }
            }
            KeyEventKind::Repeat => {}
        }
    }

    Ok(())
}

struct RawTerminalMode;

impl RawTerminalMode {
    fn enter() -> Result<Self, Box<dyn std::error::Error>> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            EnterAlternateScreen,
            Hide,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
        );
        Ok(Self)
    }
}

impl Drop for RawTerminalMode {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            PopKeyboardEnhancementFlags,
            Show,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
    }
}

fn apply_dictation_result_to_ui(
    ui: &mut ForegroundPttUiState,
    result: &DictationCaptureResult,
    foreground_target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
    copy_on_stop: bool,
) {
    ui.active_session_id = None;
    ui.last_session_id = Some(result.session.session_id.to_string());
    ui.last_result = Some(result.clone());
    ui.status_label = match result.session.state {
        SessionState::Completed => "Completed".to_owned(),
        SessionState::Failed => "Failed".to_owned(),
        SessionState::Listening => "Listening".to_owned(),
        SessionState::Transcribing => "Transcribing".to_owned(),
        SessionState::Idle => "Idle".to_owned(),
        SessionState::Previewing => "Previewing".to_owned(),
        SessionState::AwaitingConfirmation => "AwaitingConfirmation".to_owned(),
    };
    ui.last_transcript_full = Some(result.transcription.text.clone());
    ui.last_transcript_preview = Some(truncate_for_panel(&result.transcription.text, 220));
    ui.transcript_scroll = 0;
    ui.last_error = None;
    ui.last_injection_status = None;

    let should_copy = copy_on_stop || matches!(ui.default_stop_action, StopAction::Copy);
    if should_copy {
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

    match foreground_target {
        ForegroundInjectionTarget::Tmux => {
            if let Some(target_pane) =
                tmux_target_pane.filter(|_| matches!(ui.default_stop_action, StopAction::Inject))
            {
                match paste_into_tmux_pane(target_pane, &result.transcription.text) {
                    Ok(()) => {
                        ui.last_injection_status = Some(format!(
                            "{} Pasted into tmux pane {target_pane}.",
                            ui.last_injection_status
                                .clone()
                                .unwrap_or_else(|| "Done.".to_owned())
                        ))
                    }
                    Err(error) => {
                        ui.last_error = Some(format!(
                            "[injection-failed] tmux pane {target_pane}: {error}"
                        ))
                    }
                }
            }
        }
        ForegroundInjectionTarget::Wezterm { pane_id } => {
            if matches!(ui.default_stop_action, StopAction::Inject) {
                match paste_into_wezterm_pane(pane_id, &result.transcription.text) {
                    Ok(()) => {
                        ui.last_injection_status = Some(format!(
                            "{} Sent to WezTerm pane {pane_id}.",
                            ui.last_injection_status
                                .clone()
                                .unwrap_or_else(|| "Done.".to_owned())
                        ))
                    }
                    Err(error) => {
                        ui.last_error = Some(format!(
                            "[injection-failed] WezTerm pane {pane_id}: {error}"
                        ))
                    }
                }
            }
        }
        ForegroundInjectionTarget::Kitty { r#match: selector } => {
            if matches!(ui.default_stop_action, StopAction::Inject) {
                match paste_into_kitty_target(selector, &result.transcription.text) {
                    Ok(()) => {
                        let prefix = ui
                            .last_injection_status
                            .clone()
                            .unwrap_or_else(|| "Done.".to_owned());
                        ui.last_injection_status =
                            Some(format!("{prefix} Sent to Kitty target {selector}."))
                    }
                    Err(error) => {
                        ui.last_error = Some(format!(
                            "[injection-failed] Kitty target {selector}: {error}"
                        ))
                    }
                }
            }
        }
        ForegroundInjectionTarget::None => {
            if ui.last_injection_status.is_none() {
                ui.last_injection_status =
                    Some(if matches!(ui.default_stop_action, StopAction::Inject) {
                        "No foreground injection target configured.".to_owned()
                    } else {
                        "No default injection target configured.".to_owned()
                    })
            }
        }
    }
    if let Some(kind) = result.failure_kind {
        let tag = failure_kind_tag(kind);
        let detail = result
            .transcription
            .notes
            .first()
            .cloned()
            .unwrap_or_else(|| "dictation failure reported without detail.".to_owned());
        let prefixed = format!("{tag} {detail}");
        if ui
            .last_error
            .as_deref()
            .is_none_or(|existing| !existing.starts_with("[injection-failed]"))
        {
            ui.last_error = Some(prefixed.clone());
        }
        ui.push_event(prefixed);
    } else if result.session.state == SessionState::Failed && ui.last_error.is_none() {
        ui.last_error = result.transcription.notes.first().cloned();
    }
}

fn failure_kind_tag(kind: DictationFailureKind) -> &'static str {
    match kind {
        DictationFailureKind::RecordingFailed => "[recording-failed]",
        DictationFailureKind::AsrFailed => "[asr-failed]",
        DictationFailureKind::InjectionFailed => "[injection-failed]",
    }
}

#[derive(Debug, Clone)]
struct ForegroundPttUiState {
    key_label: String,
    target_label: String,
    status_label: String,
    default_stop_action: StopAction,
    save_dir: Option<PathBuf>,
    restore_clipboard_on_exit: bool,
    active_session_id: Option<String>,
    last_session_id: Option<String>,
    last_result: Option<DictationCaptureResult>,
    last_transcript_full: Option<String>,
    last_transcript_preview: Option<String>,
    transcript_scroll: usize,
    clipboard_backup_text: Option<String>,
    last_injection_status: Option<String>,
    last_error: Option<String>,
    recent_events: Vec<String>,
}

impl ForegroundPttUiState {
    fn new(
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

    fn push_event(&mut self, event: String) {
        self.recent_events.push(event);
        if self.recent_events.len() > 6 {
            let _ = self.recent_events.remove(0);
        }
    }
}

fn describe_foreground_target(
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

fn truncate_for_panel(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = chars.into_iter().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

fn render_foreground_ptt_ui(ui: &ForegroundPttUiState) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = size().unwrap_or((100, 30));
    let transcript_width = width.saturating_sub(4).max(20) as usize;
    let transcript_lines = wrap_for_panel(
        ui.last_transcript_full.as_deref().unwrap_or("-"),
        transcript_width,
    );
    let transcript_height = height.saturating_sub(18).max(6) as usize;
    let max_scroll = transcript_lines.len().saturating_sub(transcript_height);
    let scroll = ui.transcript_scroll.min(max_scroll);
    let visible_lines = transcript_lines
        .iter()
        .skip(scroll)
        .take(transcript_height)
        .cloned()
        .collect::<Vec<_>>();

    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        MoveTo(0, 0),
        Clear(ClearType::All),
        Print("VoiceLayer Foreground PTT\n"),
        Print("==========================\n"),
        Print(format!("Status: {}\n", ui.status_label)),
        Print(format!("Key: {}\n", ui.key_label)),
        Print(format!("Target: {}\n", ui.target_label)),
        Print(format!(
            "Active session: {}\n",
            ui.active_session_id.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last session: {}\n",
            ui.last_session_id.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last injection: {}\n",
            ui.last_injection_status.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last error: {}\n",
            ui.last_error.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "\nTranscript view (line {} of {}):\n",
            scroll + 1,
            transcript_lines.len().max(1)
        )),
    )?;

    for line in visible_lines {
        execute!(stdout, Print(format!("  {line}\n")))?;
    }

    execute!(stdout, Print("\nRecent events:\n"),)?;

    for event in &ui.recent_events {
        execute!(stdout, Print(format!("  - {event}\n")))?;
    }

    execute!(
        stdout,
        Print("\nControls:\n"),
        Print("  press selected key to start\n"),
        Print("  release key to stop when supported\n"),
        Print("  otherwise press selected key again to stop\n"),
        Print("  j/k or Up/Down scroll transcript, PageUp/PageDown jump\n"),
        Print("  c copies the last transcript to the clipboard\n"),
        Print("  r restores the saved clipboard text backup\n"),
        Print("  i re-applies the last injection\n"),
        Print("  s saves the last transcript to a file\n"),
        Print("  d discards the last transcript from the panel\n"),
        Print("  Esc exits\n"),
    )?;
    Ok(())
}

fn wrap_for_panel(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    for raw_line in text.lines() {
        let chars: Vec<char> = raw_line.chars().collect();
        if chars.is_empty() {
            lines.push(String::new());
            continue;
        }
        for chunk in chars.chunks(width) {
            lines.push(chunk.iter().collect());
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn scroll_transcript(ui: &mut ForegroundPttUiState, delta: isize) -> bool {
    let text = ui.last_transcript_full.as_deref().unwrap_or("-");
    let lines = wrap_for_panel(text, 80);
    let max_scroll = lines.len().saturating_sub(6) as isize;
    let current = ui.transcript_scroll as isize;
    let next = (current + delta).clamp(0, max_scroll);
    let changed = next != current;
    ui.transcript_scroll = next as usize;
    changed
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

fn restore_clipboard_backup(ui: &mut ForegroundPttUiState) -> Result<(), String> {
    let Some(text) = ui.clipboard_backup_text.clone() else {
        return Err("No text clipboard backup is available to restore.".to_owned());
    };
    copy_transcript_to_clipboard(&text).map_err(|error| error.to_string())?;
    ui.last_injection_status = Some("Restored the saved clipboard text.".to_owned());
    ui.last_error = None;
    Ok(())
}

fn save_transcript_to_file(text: &str) -> Result<PathBuf, Box<dyn std::error::Error>> {
    save_transcript_to_target_dir(text, None)
}

fn save_transcript_to_target_dir(
    text: &str,
    save_dir: Option<&Path>,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let timestamp = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let base_dir = match save_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?,
    };
    std::fs::create_dir_all(&base_dir)?;
    let path = unique_transcript_path(&base_dir, timestamp);
    std::fs::write(&path, text)?;
    Ok(path)
}

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

fn copy_result_to_clipboard(ui: &mut ForegroundPttUiState, text: &str) {
    if ui.clipboard_backup_text.is_none() {
        ui.clipboard_backup_text = snapshot_clipboard_text();
    }
    match copy_transcript_to_clipboard(text) {
        Ok(()) => {
            ui.last_injection_status = Some("Copied transcript to system clipboard.".to_owned())
        }
        Err(error) => {
            ui.last_error = Some(format!("Clipboard copy failed: {error}"));
        }
    }
}

#[cfg(test)]
mod tests {
    use super::unique_transcript_path;
    use std::fs;

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
}
