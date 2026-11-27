//! Optional `StatusNotifierItem` system tray enabled by the `tray` feature.
//!
//! Uses `ksni` over D-Bus without GTK or libappindicator. Registration is
//! best-effort: when no StatusNotifierHost is available (for example, GNOME
//! without an AppIndicator extension), [`run`] returns an error that the caller
//! logs once before continuing without a tray.

use ksni::menu::{MenuItem, StandardItem};
use ksni::{Tray, TrayMethods};
use tokio::sync::mpsc::UnboundedSender;

/// Menu action forwarded from the tray service to the iced runtime.
#[derive(Debug, Clone, Copy)]
pub enum TrayCommand {
    ToggleDictation,
    ShowMain,
    Quit,
}

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

/// Register the tray icon and hold its D-Bus service alive until cancelled.
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
