//! Minimal VoiceLayer desktop shell.
//!
//! This is the Phase 4 prototype: a single-window iced application that
//! probes the daemon, exposes start/stop buttons for a dictation session,
//! and shows the resulting transcript. Global shortcuts are registered
//! through the XDG `GlobalShortcuts` portal when available; otherwise the
//! app falls back to an in-window F9 hotkey.
//!
//! Out of scope for this shell (per the Phase 4 plan): tray, settings
//! center, auto-injection into external apps, multi-window, themes.

mod daemon_client;
mod overlay;
mod portal;

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use iced::futures::SinkExt;
use iced::futures::channel::mpsc as futures_mpsc;
use iced::widget::{button, column, container, row, scrollable, text, text_input};
use iced::{Element, Length, Subscription, Task, keyboard, stream};
use tokio::sync::mpsc;

use daemon_client::{desktop_socket_path, uds_get_json, uds_post_json};
use overlay::{
    CaptureSession, DaemonStatus, DictationCaptureResult, HealthResponse, HotkeyStatus,
    Segmentation, SessionStage, SharedError, StartDictationRequest, State, StopDictationRequest,
    render_daemon_status, render_session_stage,
};
use portal::{PortalProbe, SHORTCUT_TOGGLE};
use uuid::Uuid;

#[derive(Debug, Clone)]
enum Message {
    DaemonProbed(Result<HealthResponse, SharedError>),
    ProbeDaemonPressed,
    StartDaemonPressed,
    DaemonSpawnResult(Result<(), SharedError>),
    PortalProbed(PortalProbe),
    HotkeyReceived,
    StartPressed,
    StopPressed,
    SessionStarted(Result<CaptureSession, SharedError>),
    SessionStopped(Result<DictationCaptureResult, SharedError>),
    SocketPathEdited(String),
}

struct App {
    state: State,
    socket_path: PathBuf,
    socket_path_input: String,
}

impl App {
    fn boot() -> (Self, Task<Message>) {
        let socket_path = desktop_socket_path();
        let app = Self {
            state: State {
                daemon: DaemonStatus::Probing,
                ..State::default()
            },
            socket_path: socket_path.clone(),
            socket_path_input: socket_path.display().to_string(),
        };
        let startup = Task::batch(vec![
            probe_health_task(socket_path),
            Task::perform(portal::probe(), Message::PortalProbed),
        ]);
        (app, startup)
    }

    fn update(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ProbeDaemonPressed => {
                self.state.daemon = DaemonStatus::Probing;
                self.state.error = None;
                probe_health_task(self.socket_path.clone())
            }
            Message::DaemonProbed(Ok(_)) => {
                self.state.daemon = DaemonStatus::Healthy;
                self.state.error = None;
                Task::none()
            }
            Message::DaemonProbed(Err(error)) => {
                self.state.daemon = DaemonStatus::Unreachable;
                self.state.error = Some((*error).clone());
                Task::none()
            }
            Message::StartDaemonPressed => {
                self.state.error = None;
                Task::perform(spawn_daemon(), Message::DaemonSpawnResult)
            }
            Message::DaemonSpawnResult(Ok(())) => {
                self.state.daemon = DaemonStatus::Probing;
                probe_health_task(self.socket_path.clone())
            }
            Message::DaemonSpawnResult(Err(error)) => {
                self.state.error = Some(format!("Failed to start daemon: {error}"));
                Task::none()
            }
            Message::PortalProbed(PortalProbe::Available) => {
                self.state.hotkey = HotkeyStatus {
                    portal_available: true,
                    portal_error: None,
                };
                Task::none()
            }
            Message::PortalProbed(PortalProbe::Unavailable(reason)) => {
                self.state.hotkey = HotkeyStatus {
                    portal_available: false,
                    portal_error: Some(reason),
                };
                Task::none()
            }
            Message::HotkeyReceived => self.toggle_session(),
            Message::StartPressed => self.start_session(),
            Message::StopPressed => self.stop_session(),
            Message::SessionStarted(Ok(session)) => {
                self.state.session_stage = SessionStage::Listening;
                self.state.session_id = Some(session.session_id);
                self.state.error = None;
                Task::none()
            }
            Message::SessionStarted(Err(error)) => {
                self.state.session_stage = SessionStage::Failed;
                self.state.error = Some((*error).clone());
                Task::none()
            }
            Message::SessionStopped(Ok(result)) => {
                if result.failure_kind.is_some() {
                    self.state.session_stage = SessionStage::Failed;
                } else {
                    self.state.session_stage = SessionStage::Completed;
                }
                self.state.session_id = None;
                self.state.transcript = Some(result.transcription.text);
                self.state.detected_language = result.transcription.detected_language;
                self.state.last_notes = result.transcription.notes;
                if let Some(kind) = result.failure_kind {
                    self.state.error = Some(format!("[{kind}] capture did not complete cleanly"));
                }
                Task::none()
            }
            Message::SessionStopped(Err(error)) => {
                self.state.session_stage = SessionStage::Failed;
                self.state.session_id = None;
                self.state.error = Some((*error).clone());
                Task::none()
            }
            Message::SocketPathEdited(next) => {
                self.socket_path = PathBuf::from(next.trim());
                self.socket_path_input = next;
                Task::none()
            }
        }
    }

    fn toggle_session(&mut self) -> Task<Message> {
        match self.state.session_stage {
            SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => {
                self.start_session()
            }
            SessionStage::Listening => self.stop_session(),
            SessionStage::Starting | SessionStage::Stopping => Task::none(),
        }
    }

    fn start_session(&mut self) -> Task<Message> {
        if self.state.daemon != DaemonStatus::Healthy {
            self.state.error = Some("Daemon is not healthy; start it first.".into());
            return Task::none();
        }
        self.state.session_stage = SessionStage::Starting;
        self.state.error = None;
        let socket = self.socket_path.clone();
        Task::perform(
            async move { start_session_request(socket).await },
            Message::SessionStarted,
        )
    }

    fn stop_session(&mut self) -> Task<Message> {
        let Some(session_id) = self.state.session_id else {
            self.state.error = Some("No active session to stop.".into());
            return Task::none();
        };
        self.state.session_stage = SessionStage::Stopping;
        self.state.error = None;
        let socket = self.socket_path.clone();
        Task::perform(
            async move { stop_session_request(socket, session_id).await },
            Message::SessionStopped,
        )
    }

    fn view(&self) -> Element<'_, Message> {
        let status_row = row![
            text("Daemon:").size(14),
            text(render_daemon_status(self.state.daemon)).size(14),
            text("  |  Session:").size(14),
            text(render_session_stage(self.state.session_stage)).size(14),
        ]
        .spacing(8);

        let hotkey_line: Element<'_, Message> = if self.state.hotkey.portal_available {
            text("Global shortcut registered via XDG portal (VoiceLayer: Toggle dictation)")
                .size(12)
                .into()
        } else {
            text(format!(
                "XDG global_shortcuts portal unavailable ({}). \
                 Hotkey only works while this window is focused (F9).",
                self.state
                    .hotkey
                    .portal_error
                    .as_deref()
                    .unwrap_or("unknown reason")
            ))
            .size(12)
            .into()
        };

        let socket_row = row![
            text("Socket path:").size(14),
            text_input("", &self.socket_path_input)
                .on_input(Message::SocketPathEdited)
                .width(Length::Fill),
        ]
        .spacing(8);

        let daemon_controls: Element<'_, Message> = match self.state.daemon {
            DaemonStatus::Unknown | DaemonStatus::Probing => {
                button(text("Probing daemon...")).into()
            }
            DaemonStatus::Healthy => button(text("Re-probe daemon"))
                .on_press(Message::ProbeDaemonPressed)
                .into(),
            DaemonStatus::Unreachable => row![
                button(text("Start daemon")).on_press(Message::StartDaemonPressed),
                button(text("Retry probe")).on_press(Message::ProbeDaemonPressed),
            ]
            .spacing(8)
            .into(),
        };

        let session_controls: Element<'_, Message> = match self.state.session_stage {
            SessionStage::Idle | SessionStage::Completed | SessionStage::Failed => {
                button(text("Start dictation (F9)"))
                    .on_press(Message::StartPressed)
                    .into()
            }
            SessionStage::Listening => button(text("Stop dictation (F9)"))
                .on_press(Message::StopPressed)
                .into(),
            SessionStage::Starting | SessionStage::Stopping => {
                button(text(render_session_stage(self.state.session_stage))).into()
            }
        };

        let mut layout = column![
            status_row,
            hotkey_line,
            socket_row,
            daemon_controls,
            session_controls,
        ]
        .spacing(12)
        .padding(16);

        if let Some(language) = &self.state.detected_language {
            layout = layout.push(text(format!("Detected language: {language}")).size(12));
        }

        if let Some(error) = &self.state.error {
            layout = layout.push(text(format!("Error: {error}")).size(12));
        }

        if !self.state.last_notes.is_empty() {
            let notes_lines: Vec<String> = self
                .state
                .last_notes
                .iter()
                .map(|n| format!("• {n}"))
                .collect();
            layout = layout.push(text(notes_lines.join("\n")).size(12));
        }

        if let Some(transcript) = self.state.transcript.as_deref().filter(|t| !t.is_empty()) {
            layout = layout.push(scrollable(text(transcript).size(16)).height(Length::Fill));
        }

        container(layout)
            .width(Length::Fill)
            .height(Length::Fill)
            .into()
    }

    fn subscription(&self) -> Subscription<Message> {
        let keys = keyboard::listen().filter_map(|event| match event {
            keyboard::Event::KeyPressed {
                key: keyboard::Key::Named(keyboard::key::Named::F9),
                ..
            } => Some(Message::HotkeyReceived),
            _ => None,
        });
        Subscription::batch(vec![keys, Subscription::run(portal_stream)])
    }
}

/// Best-effort stream for portal-activated hotkeys. The subscription starts a
/// tokio task that opens a `GlobalShortcuts` session and forwards each
/// activation through the iced futures mpsc sender handed in by
/// [`iced::stream::channel`]. When the portal is unavailable the task exits
/// immediately and the stream stays silent.
fn portal_stream() -> impl iced::futures::Stream<Item = Message> {
    stream::channel(8, async |mut output: futures_mpsc::Sender<Message>| {
        let (tx, mut rx) = mpsc::unbounded_channel::<String>();
        let registration = tokio::spawn(async move {
            if let Err(error) = portal::run_listener(tx).await {
                tracing::warn!(error = %error, "portal listener returned error");
            }
        });
        while let Some(shortcut_id) = rx.recv().await {
            if shortcut_id != SHORTCUT_TOGGLE {
                continue;
            }
            let _ = shortcut_id;
            if output.send(Message::HotkeyReceived).await.is_err() {
                break;
            }
        }
        registration.abort();
    })
}

fn probe_health_task(socket_path: PathBuf) -> Task<Message> {
    Task::perform(
        async move {
            uds_get_json::<HealthResponse>(&socket_path, "/v1/health")
                .await
                .map_err(|error| Arc::new(error.to_string()))
        },
        Message::DaemonProbed,
    )
}

async fn start_session_request(socket_path: PathBuf) -> Result<CaptureSession, SharedError> {
    let payload = StartDictationRequest {
        trigger: "tray_button",
        segmentation: Segmentation { mode: "one_shot" },
    };
    uds_post_json::<_, CaptureSession>(&socket_path, "/v1/sessions/dictation", &payload)
        .await
        .map_err(|error| Arc::new(error.to_string()))
}

async fn stop_session_request(
    socket_path: PathBuf,
    session_id: Uuid,
) -> Result<DictationCaptureResult, SharedError> {
    let payload = StopDictationRequest { session_id };
    uds_post_json::<_, DictationCaptureResult>(
        &socket_path,
        "/v1/sessions/dictation/stop",
        &payload,
    )
    .await
    .map_err(|error| Arc::new(error.to_string()))
}

fn resolve_vl_binary() -> PathBuf {
    // Prefer an explicit override for development setups, then the install.sh
    // target, then $PATH. Falling back to the bare name lets `Command::spawn`
    // surface the usual "program not found" error with a clear hint.
    if let Some(explicit) = std::env::var_os("VOICELAYER_VL_BIN") {
        return PathBuf::from(explicit);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".local/bin/vl");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("vl")
}

async fn spawn_daemon() -> Result<(), SharedError> {
    let binary = resolve_vl_binary();
    let mut command = Command::new(&binary);
    command
        .args(["daemon", "run"])
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Detach from the GUI's process group so closing the desktop shell does
    // not SIGHUP the daemon. `process_group(0)` makes the child the leader
    // of a new group; on non-Unix platforms this degrades to the default
    // attached behavior.
    #[cfg(unix)]
    command.process_group(0);

    match command.spawn() {
        Ok(_) => {
            // Give the daemon a moment to open its socket before the caller
            // re-probes health; without this the probe often loses the race.
            tokio::time::sleep(std::time::Duration::from_millis(500)).await;
            Ok(())
        }
        Err(error) => Err(Arc::new(format!(
            "could not execute `{}`: {error}. Set VOICELAYER_VL_BIN or add \
             ~/.local/bin to PATH.",
            binary.display(),
        ))),
    }
}

fn init_tracing() {
    let filter = std::env::var("VOICELAYER_LOG").unwrap_or_else(|_| "vl_desktop=info".to_owned());
    let _ = tracing_subscriber::fmt()
        .with_env_filter(filter)
        .with_target(false)
        .try_init();
}

pub fn main() -> iced::Result {
    init_tracing();
    iced::application(App::boot, App::update, App::view)
        .title("VoiceLayer")
        .subscription(App::subscription)
        .window_size((520.0, 420.0))
        .run()
}

#[cfg(test)]
mod tests {
    use std::ffi::OsString;
    use std::sync::Mutex;

    use super::resolve_vl_binary;

    /// Serializes env mutation across every test in this module; see
    /// `crates/voicelayerd/src/worker.rs` for the rationale.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Snapshots an env var at construction and restores it on drop so
    /// an assertion panic never leaks state into the next test's view.
    struct EnvGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: ENV_LOCK is held by the surrounding test for the
            // lifetime of the guard, so no other thread can observe the
            // intermediate state.
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    fn set_env(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        // SAFETY: ENV_LOCK must be held by the caller.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn unset_env(key: &str) {
        // SAFETY: ENV_LOCK must be held by the caller.
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn resolve_vl_binary_honors_explicit_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_VL_BIN");
        set_env("VOICELAYER_VL_BIN", "/custom/vl");
        assert_eq!(resolve_vl_binary().to_str(), Some("/custom/vl"));
    }

    #[test]
    fn resolve_vl_binary_falls_back_to_local_bin_when_present() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_VL_BIN");
        let _home_guard = EnvGuard::capture("HOME");
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        let fake_local = tmp.path().join(".local/bin");
        std::fs::create_dir_all(&fake_local).expect("create fake ~/.local/bin");
        let vl_path = fake_local.join("vl");
        std::fs::write(&vl_path, b"#!/bin/sh\n").expect("write fake vl binary");
        unset_env("VOICELAYER_VL_BIN");
        set_env("HOME", tmp.path());
        assert_eq!(resolve_vl_binary(), vl_path);
    }

    #[test]
    fn resolve_vl_binary_falls_back_to_path_lookup_as_last_resort() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_VL_BIN");
        let _home_guard = EnvGuard::capture("HOME");
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        unset_env("VOICELAYER_VL_BIN");
        // Point HOME at an empty directory so `~/.local/bin/vl` misses.
        set_env("HOME", tmp.path());
        assert_eq!(resolve_vl_binary().to_str(), Some("vl"));
    }
}
