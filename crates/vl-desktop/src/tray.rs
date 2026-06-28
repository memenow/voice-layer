//! Optional `StatusNotifierItem` system tray (the `tray` feature).
//!
//! VoiceLayer is a background voice layer, so a tray indicator is a natural
//! affordance — but it is strictly best-effort and is never the only path to a
//! function (everything here is also reachable from the main window and the
//! CLI). GNOME Shell has no built-in tray: a StatusNotifierItem renders only
//! when a StatusNotifierHost is present, which on GNOME means the user installed
//! the "AppIndicator and KStatusNotifierItem Support" extension. When no host is
//! reachable, [`ksni`]'s `spawn` returns an error immediately — it does not
//! panic or block — so the shell logs once and runs on without a tray.
//!
//! `ksni` is used because it speaks the StatusNotifier D-Bus protocol in pure
//! Rust with no GTK/libappindicator dependency, keeping the runtime path free of
//! GLib and copyleft. Menu activations are forwarded to the iced runtime over an
//! unbounded channel and surface as `Message`s in [`crate::app`]; the tray holds
//! no application state and does not reflect live session status — its menu
//! items are stateless actions.

use ksni::menu::{MenuItem, StandardItem};
use ksni::{Tray, TrayMethods};
use tokio::sync::mpsc::UnboundedSender;

/// A tray menu activation, forwarded to the iced runtime.
#[derive(Debug, Clone, Copy)]
pub enum TrayCommand {
    /// Start or stop streaming dictation, matching the global hotkey.
    ToggleDictation,
    /// Raise the main navigation window (best-effort on Wayland).
    ShowMain,
    /// Quit the desktop shell.
    Quit,
}

/// The StatusNotifierItem model. Each menu item's `activate` closure forwards a
/// [`TrayCommand`] over the channel; the receiver lives in the iced subscription.
struct VoiceLayerTray {
    sender: UnboundedSender<TrayCommand>,
}

impl Tray for VoiceLayerTray {
    fn id(&self) -> String {
        "voicelayer".to_owned()
    }

    fn title(&self) -> String {
        "VoiceLayer".to_owned()
    }

    /// A themed freedesktop icon name (the microphone glyph), resolved by the
    /// user's icon theme — no bundled asset, so nothing to ship or license.
    fn icon_name(&self) -> String {
        "audio-input-microphone".to_owned()
    }

    fn menu(&self) -> Vec<MenuItem<Self>> {
        vec![
            StandardItem {
                label: "Toggle dictation".to_owned(),
                activate: Box::new(|tray: &mut Self| {
                    let _ = tray.sender.send(TrayCommand::ToggleDictation);
                }),
                ..Default::default()
            }
            .into(),
            MenuItem::Separator,
            StandardItem {
                label: "Show VoiceLayer".to_owned(),
                activate: Box::new(|tray: &mut Self| {
                    let _ = tray.sender.send(TrayCommand::ShowMain);
                }),
                ..Default::default()
            }
            .into(),
            StandardItem {
                label: "Quit VoiceLayer".to_owned(),
                icon_name: "application-exit".to_owned(),
                activate: Box::new(|tray: &mut Self| {
                    let _ = tray.sender.send(TrayCommand::Quit);
                }),
                ..Default::default()
            }
            .into(),
        ]
    }
}

/// Register the tray and keep its D-Bus service alive.
///
/// Returns `Err` when no StatusNotifierHost/Watcher is reachable (e.g. GNOME
/// without the AppIndicator extension) or the session bus is unavailable; the
/// caller logs once and continues without a tray. On success this future never
/// resolves — it holds the [`ksni::Handle`] so the background service keeps
/// dispatching menu clicks until the iced subscription cancels it (at shutdown).
pub async fn run(sender: UnboundedSender<TrayCommand>) -> Result<(), String> {
    let tray = VoiceLayerTray { sender };
    let _handle = tray.spawn().await.map_err(|error| error.to_string())?;
    std::future::pending::<()>().await;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_tray() -> VoiceLayerTray {
        let (sender, _receiver) = tokio::sync::mpsc::unbounded_channel();
        VoiceLayerTray { sender }
    }

    #[test]
    fn menu_exposes_core_actions_in_order() {
        let labels: Vec<String> = sample_tray()
            .menu()
            .into_iter()
            .filter_map(|item| match item {
                MenuItem::Standard(item) => Some(item.label),
                _ => None,
            })
            .collect();
        assert_eq!(
            labels,
            [
                "Toggle dictation".to_owned(),
                "Show VoiceLayer".to_owned(),
                "Quit VoiceLayer".to_owned(),
            ],
        );
    }

    #[test]
    fn menu_separates_quit_from_the_actions() {
        let has_separator = sample_tray()
            .menu()
            .iter()
            .any(|item| matches!(item, MenuItem::Separator));
        assert!(has_separator, "quit should be divided from the actions");
    }
}
