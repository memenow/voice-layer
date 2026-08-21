//! Global hotkey integration for the desktop shell.
//!
//! Linux uses the XDG `GlobalShortcuts` portal; macOS uses the
//! `global-hotkey` crate (Carbon `RegisterEventHotKey`). When the backend is
//! unavailable the app still runs — it falls back to window-focused key
//! handling inside the iced window. Activations are forwarded through a
//! [`tokio::sync::mpsc`] channel so the iced subscription can relay them as
//! `Message::HotkeyReceived` events.

pub const SHORTCUT_TOGGLE: &str = "voicelayer.dictation_toggle";

#[derive(Debug, Clone)]
pub enum HotkeyProbe {
    Available(&'static str),
    Unavailable(String),
}

#[cfg(target_os = "linux")]
mod backend {
    use std::time::Duration;

    use ashpd::desktop::CreateSessionOptions;
    use ashpd::desktop::global_shortcuts::{BindShortcutsOptions, GlobalShortcuts, NewShortcut};
    use futures_util::StreamExt;
    use tokio::sync::mpsc;

    use super::{HotkeyProbe, SHORTCUT_TOGGLE};

    pub async fn probe() -> HotkeyProbe {
        match GlobalShortcuts::new().await {
            Ok(_) => HotkeyProbe::Available("xdg_portal"),
            Err(error) => HotkeyProbe::Unavailable(error.to_string()),
        }
    }

    /// Register a single "toggle dictation" shortcut and forward activations
    /// to `sender`. Lives for the duration of the registration.
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
            .map_err(|e| format!("bind_shortcuts failed: {e}"))?;

        let mut activated = shortcuts
            .receive_activated()
            .await
            .map_err(|e| e.to_string())?;

        while let Some(signal) = activated.next().await {
            if sender.send(signal.shortcut_id().to_owned()).is_err() {
                break;
            }
            // Cheap debounce: prevent a held shortcut from flooding the app
            // with back-to-back toggles while an HTTP round trip is in flight.
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

    use super::{HotkeyProbe, SHORTCUT_TOGGLE};

    pub async fn probe() -> HotkeyProbe {
        match GlobalHotKeyManager::new() {
            Ok(_) => HotkeyProbe::Available("global_hotkey"),
            Err(error) => HotkeyProbe::Unavailable(error.to_string()),
        }
    }

    /// Register F9 as the toggle hotkey and poll the event queue. Registering
    /// requires the user to grant the app Input Monitoring permission in
    /// System Settings; a registration failure surfaces as an error here.
    pub async fn run_listener(sender: mpsc::UnboundedSender<String>) -> Result<(), String> {
        let manager = GlobalHotKeyManager::new().map_err(|e| e.to_string())?;
        let hotkey = HotKey::new(None, Code::F9);
        manager
            .register(hotkey)
            .map_err(|e| format!("register failed (Input Monitoring permission?): {e}"))?;

        loop {
            while let Ok(event) = GlobalHotKeyEvent::receiver().try_recv() {
                if sender.send(SHORTCUT_TOGGLE.to_owned()).is_err() {
                    return Ok(());
                }
                let _ = event;
                // Debounce held keys while an HTTP round trip is in flight.
                tokio::time::sleep(Duration::from_millis(150)).await;
            }
            tokio::time::sleep(Duration::from_millis(50)).await;
        }
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
mod backend {
    use tokio::sync::mpsc;

    use super::HotkeyProbe;

    pub async fn probe() -> HotkeyProbe {
        HotkeyProbe::Unavailable("unsupported platform".to_owned())
    }

    pub async fn run_listener(_sender: mpsc::UnboundedSender<String>) -> Result<(), String> {
        Err("global hotkeys are not supported on this platform".to_owned())
    }
}

pub use backend::{probe, run_listener};
