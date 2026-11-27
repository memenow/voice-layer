//! macOS global hotkey capability probe.

use super::GlobalHotkeysStatus;

/// Reports Carbon `global-hotkey` support; Input Monitoring permission is
/// checked by the desktop shell when registering the shortcut.
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    GlobalHotkeysStatus {
        available: true,
        backend: "global_hotkey".to_owned(),
        detail: Some(
            "macOS Carbon global hotkeys; requires Input Monitoring permission.".to_owned(),
        ),
    }
}
