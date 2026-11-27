//! Global hotkey registration for the desktop shell.
//!
//! Uses the XDG `GlobalShortcuts` portal on Linux and `global-hotkey` (Carbon
//! `RegisterEventHotKey`) on macOS. When unavailable, the shell falls back to
//! focused-window F9 handling inside iced.

pub const SHORTCUT_TOGGLE: &str = "voicelayer.dictation_toggle";

#[derive(Debug, Clone)]
pub enum PortalProbe {
    Available,
    Unavailable(String),
}

#[cfg(target_os = "linux")]
mod backend {
    use std::time::Duration;

    use ashpd::desktop::CreateSessionOptions;
    use ashpd::desktop::global_shortcuts::{BindShortcutsOptions, GlobalShortcuts, NewShortcut};
    use futures_util::StreamExt;
    use tokio::sync::mpsc;

    use super::{PortalProbe, SHORTCUT_TOGGLE};

    pub async fn probe() -> PortalProbe {
        match GlobalShortcuts::new().await {
            Ok(_) => PortalProbe::Available,
            Err(error) => PortalProbe::Unavailable(error.to_string()),
        }
    }

    /// Bind the dictation toggle shortcut and forward activations to `sender`.
    pub async fn run_listener(sender: mpsc::UnboundedSender<String>) -> Result<(), String> {
        let shortcuts = GlobalShortcuts::new().await.map_err(|e| e.to_string())?;
        let session = shortcuts
            .create_session(CreateSessionOptions::default())
            .await
            .map_err(|e| e.to_string())?;

        let entry = NewShortcut::new(SHORTCUT_TOGGLE, "VoiceLayer: Toggle dictation");
        shortcuts
            .bind_shortcuts(&session, &[entry], None, BindShortcutsOptions::default())
            .await
            .map_err(|e| format!("bind_shortcuts failed: {e}"))?
            .response()
            .map_err(|e| format!("bind_shortcuts response failed: {e}"))?;

        let mut activated = shortcuts
            .receive_activated()
            .await
            .map_err(|e| e.to_string())?;

        while let Some(signal) = activated.next().await {
            if sender.send(signal.shortcut_id().to_owned()).is_err() {
                break;
            }
            // Debounce held shortcuts while an HTTP round trip is in flight.
            tokio::time::sleep(Duration::from_millis(150)).await;
        }

        Ok(())
    }
}

#[cfg(target_os = "macos")]
mod backend {
    use std::time::Duration;

    use global_hotkey::{
        GlobalHotKeyEvent, GlobalHotKeyManager,
        hotkey::{Code, HotKey},
    };
    use tokio::sync::mpsc;

    use super::{PortalProbe, SHORTCUT_TOGGLE};

    pub async fn probe() -> PortalProbe {
        match GlobalHotKeyManager::new() {
            Ok(_) => PortalProbe::Available,
            Err(error) => PortalProbe::Unavailable(error.to_string()),
        }
    }

    /// Register F9 globally and poll the hotkey queue. Requires Input
    /// Monitoring permission in macOS System Settings.
    pub async fn run_listener(sender: mpsc::UnboundedSender<String>) -> Result<(), String> {
        let manager = GlobalHotKeyManager::new().map_err(|e| e.to_string())?;
        let hotkey = HotKey::new(None, Code::F9);
        manager
            .register(hotkey)
            .map_err(|e| format!("register failed (Input Monitoring permission?): {e}"))?;

        loop {
            while let Ok(_event) = GlobalHotKeyEvent::receiver().try_recv() {
                if sender.send(SHORTCUT_TOGGLE.to_owned()).is_err() {
                    return Ok(());
                }
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod backend {
    use tokio::sync::mpsc;

    use super::PortalProbe;

    pub async fn probe() -> PortalProbe {
        PortalProbe::Unavailable("unsupported platform".to_owned())
    }

    pub async fn run_listener(_sender: mpsc::UnboundedSender<String>) -> Result<(), String> {
        Err("global hotkeys are not supported on this platform".to_owned())
    }
}

pub use backend::{probe, run_listener};
