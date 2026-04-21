//! XDG `GlobalShortcuts` portal integration.
//!
//! The desktop shell tries to register a short list of shortcut identifiers
//! via the portal on startup. When the portal is unavailable the app still
//! runs — it just falls back to window-focused key handling inside the
//! iced window. Registered shortcut activations are forwarded to the app
//! through a [`tokio::sync::mpsc`] channel so the iced subscription can
//! relay them as `Message::HotkeyReceived` events.

use std::time::Duration;

use ashpd::desktop::CreateSessionOptions;
use ashpd::desktop::global_shortcuts::{BindShortcutsOptions, GlobalShortcuts, NewShortcut};
use futures_util::StreamExt;
use tokio::sync::mpsc;

pub const SHORTCUT_TOGGLE: &str = "voicelayer.dictation_toggle";

#[derive(Debug, Clone)]
pub enum PortalProbe {
    Available,
    Unavailable(String),
}

pub async fn probe() -> PortalProbe {
    match GlobalShortcuts::new().await {
        Ok(_) => PortalProbe::Available,
        Err(error) => PortalProbe::Unavailable(error.to_string()),
    }
}

/// Register a single "toggle dictation" shortcut and forward activations to
/// `sender`. The returned future lives for the duration of the registration:
/// it terminates when the portal stream ends or the receiver is dropped.
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
        // Cheap debounce: prevent a held shortcut from flooding the app with
        // back-to-back toggles while an HTTP round trip is in flight.
        tokio::time::sleep(Duration::from_millis(150)).await;
    }

    Ok(())
}
