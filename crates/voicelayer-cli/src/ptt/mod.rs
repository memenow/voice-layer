//! Foreground push-to-talk terminal event loop.

mod actions;
mod render;

use std::path::PathBuf;

use crossterm::{
    cursor::{Hide, Show},
    event::{
        Event, KeyCode, KeyEvent, KeyEventKind, KeyboardEnhancementFlags,
        PopKeyboardEnhancementFlags, PushKeyboardEnhancementFlags, read,
    },
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use voicelayer_core::client::DaemonClient;
use voicelayer_core::{
    CaptureSession, DictationCaptureResult, LanguageProfile, LanguageStrategy, SegmentationMode,
    StartDictationRequest, StopAction, StopDictationRequest, TriggerKind,
};

use crate::config::CliPttKey;
use crate::terminal_targets::{
    ForegroundInjectionTarget, resolve_foreground_injection_target, resolve_tmux_target_pane,
};

use self::actions::{
    apply_dictation_result, copy_result_to_clipboard, restore_clipboard_backup,
    save_transcript_to_target_dir,
};
use self::render::{PttUiState, describe_foreground_target, render_panel, scroll_transcript};

/// Effective runtime options for `vl dictation foreground-ptt`.
pub(crate) struct PttOptions {
    pub language: Option<String>,
    pub translate_to_english: bool,
    pub keep_audio: bool,
    pub key: CliPttKey,
    pub tmux_target_pane: Option<String>,
    pub wezterm_target_pane_id: Option<String>,
    pub kitty_match: Option<String>,
    pub copy_on_stop: bool,
    pub default_stop_action: StopAction,
    pub restore_clipboard_on_exit: bool,
    pub save_dir: Option<PathBuf>,
    pub provider_id: Option<String>,
}

struct SessionTarget {
    foreground: ForegroundInjectionTarget,
    tmux_pane: Option<String>,
}

pub(crate) async fn run_foreground_ptt(
    client: &DaemonClient,
    options: PttOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    let target = SessionTarget {
        foreground: resolve_foreground_injection_target(
            options.tmux_target_pane.clone(),
            options.wezterm_target_pane_id.clone(),
            options.kitty_match.clone(),
        )?,
        tmux_pane: resolve_tmux_target_pane(options.tmux_target_pane.clone())?,
    };
    let _raw_mode = RawTerminalMode::enter()?;
    let mut ui = PttUiState::new(
        options.key.label().to_owned(),
        describe_foreground_target(&target.foreground, target.tmux_pane.as_deref()),
        options.default_stop_action,
        options.save_dir.clone(),
        options.restore_clipboard_on_exit,
    );
    ui.push_event(format!(
        "Press {} to start. Release to stop when supported; otherwise press {} again. Esc exits.",
        options.key.label(),
        options.key.label()
    ));
    render_panel(&ui)?;

    let mut active_session: Option<uuid::Uuid> = None;
    loop {
        let Event::Key(key_event) = read()? else {
            continue;
        };

        if key_event.code == KeyCode::Esc {
            stop_and_exit(client, &mut active_session, &mut ui, &target, &options).await?;
            break;
        }
        if key_event.code != options.key.as_key_code() {
            handle_panel_key(key_event, &mut ui, &target)?;
            continue;
        }
        handle_ptt_key(
            client,
            key_event.kind,
            &mut active_session,
            &mut ui,
            &target,
            &options,
        )
        .await?;
    }

    Ok(())
}

async fn stop_and_exit(
    client: &DaemonClient,
    active_session: &mut Option<uuid::Uuid>,
    ui: &mut PttUiState,
    target: &SessionTarget,
    options: &PttOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    if let Some(session_id) = active_session.take() {
        let result = stop_dictation(client, session_id).await?;
        apply_dictation_result(
            ui,
            &result,
            &target.foreground,
            target.tmux_pane.as_deref(),
            options.copy_on_stop,
        );
        ui.push_event("Stopped active session before exit.".to_owned());
        render_panel(ui)?;
    }
    if ui.restore_clipboard_on_exit {
        let _ = restore_clipboard_backup(ui);
    }
    ui.push_event("Foreground PTT exited.".to_owned());
    Ok(())
}

fn handle_panel_key(
    key_event: KeyEvent,
    ui: &mut PttUiState,
    target: &SessionTarget,
) -> Result<(), Box<dyn std::error::Error>> {
    if key_event.kind != KeyEventKind::Press {
        return Ok(());
    }
    match key_event.code {
        KeyCode::Char('c') => {
            if let Some(text) = ui.last_transcript_full.clone() {
                copy_result_to_clipboard(ui, &text);
                if ui.last_error.is_none() {
                    ui.push_event("Copied last transcript to clipboard.".to_owned());
                }
                render_panel(ui)?;
            }
        }
        KeyCode::Char('r') => {
            match restore_clipboard_backup(ui) {
                Ok(()) => ui.push_event("Restored clipboard from saved backup.".to_owned()),
                Err(error) => ui.last_error = Some(error),
            }
            render_panel(ui)?;
        }
        KeyCode::Char('i') => {
            if let Some(result) = ui.last_result.clone() {
                apply_dictation_result(
                    ui,
                    &result,
                    &target.foreground,
                    target.tmux_pane.as_deref(),
                    false,
                );
                ui.push_event("Re-applied last transcript injection.".to_owned());
                render_panel(ui)?;
            }
        }
        KeyCode::Char('s') => {
            if let Some(text) = ui.last_transcript_full.clone() {
                match save_transcript_to_target_dir(&text, ui.save_dir.as_deref()) {
                    Ok(path) => {
                        ui.last_injection_status =
                            Some(format!("Saved transcript to {}.", path.display()));
                        ui.last_error = None;
                        ui.push_event(format!("Saved last transcript to {}.", path.display()));
                    }
                    Err(error) => {
                        ui.last_error = Some(format!("Failed to save transcript: {error}"));
                    }
                }
                render_panel(ui)?;
            }
        }
        KeyCode::Char('d') => {
            ui.discard_last_transcript();
            render_panel(ui)?;
        }
        KeyCode::Char('j') | KeyCode::Down if scroll_transcript(ui, 1) => render_panel(ui)?,
        KeyCode::Char('k') | KeyCode::Up if scroll_transcript(ui, -1) => render_panel(ui)?,
        KeyCode::PageDown if scroll_transcript(ui, 6) => render_panel(ui)?,
        KeyCode::PageUp if scroll_transcript(ui, -6) => render_panel(ui)?,
        _ => {}
    }
    Ok(())
}

async fn handle_ptt_key(
    client: &DaemonClient,
    kind: KeyEventKind,
    active_session: &mut Option<uuid::Uuid>,
    ui: &mut PttUiState,
    target: &SessionTarget,
    options: &PttOptions,
) -> Result<(), Box<dyn std::error::Error>> {
    match kind {
        KeyEventKind::Press if active_session.is_none() => {
            let request = StartDictationRequest {
                trigger: TriggerKind::Cli,
                language_profile: options.language.clone().map(|lang| LanguageProfile {
                    strategy: LanguageStrategy::Locked,
                    input_languages: vec![lang],
                    output_language: None,
                }),
                translate_to_english: options.translate_to_english,
                keep_audio: options.keep_audio,
                segmentation: SegmentationMode::default(),
                provider_id: options.provider_id.clone(),
            };
            let session: CaptureSession =
                client.post_json("/v1/sessions/dictation", &request).await?;
            *active_session = Some(session.session_id);
            ui.status_label = "Listening".to_owned();
            ui.active_session_id = Some(session.session_id.to_string());
            ui.last_error = None;
            ui.push_event(format!("Dictation started: {}", session.session_id));
            render_panel(ui)?;
        }
        KeyEventKind::Press => {
            if let Some(session_id) = active_session.take() {
                finish_session(
                    client,
                    session_id,
                    "key press toggle",
                    ui,
                    target,
                    options.copy_on_stop,
                )
                .await?;
            }
        }
        KeyEventKind::Release => {
            if let Some(session_id) = active_session.take() {
                finish_session(
                    client,
                    session_id,
                    "key release",
                    ui,
                    target,
                    options.copy_on_stop,
                )
                .await?;
            }
        }
        KeyEventKind::Repeat => {}
    }
    Ok(())
}

async fn finish_session(
    client: &DaemonClient,
    session_id: uuid::Uuid,
    reason: &str,
    ui: &mut PttUiState,
    target: &SessionTarget,
    copy_on_stop: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    ui.status_label = "Transcribing".to_owned();
    ui.push_event(format!("Stopping session {session_id} via {reason}."));
    render_panel(ui)?;
    let result = stop_dictation(client, session_id).await?;
    apply_dictation_result(
        ui,
        &result,
        &target.foreground,
        target.tmux_pane.as_deref(),
        copy_on_stop,
    );
    ui.push_event(format!("Dictation stopped via {reason}."));
    render_panel(ui)?;
    Ok(())
}

async fn stop_dictation(
    client: &DaemonClient,
    session_id: uuid::Uuid,
) -> Result<DictationCaptureResult, Box<dyn std::error::Error>> {
    Ok(client
        .post_json(
            "/v1/sessions/dictation/stop",
            &StopDictationRequest { session_id },
        )
        .await?)
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
