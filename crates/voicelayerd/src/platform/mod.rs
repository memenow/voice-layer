//! Platform capability probes isolated behind OS-specific submodules.

#[cfg(target_os = "linux")]
mod linux;
#[cfg(target_os = "macos")]
mod macos;

use serde::Serialize;

/// Availability status for desktop global hotkeys.
#[derive(Debug, Clone, Serialize)]
pub struct GlobalHotkeysStatus {
    pub available: bool,
    pub backend: String,
    pub detail: Option<String>,
}

/// Probe whether the current desktop session exposes a global hotkey backend.
#[cfg(target_os = "linux")]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    linux::probe_global_hotkeys().await
}

/// Probe whether the current desktop session exposes a global hotkey backend.
#[cfg(target_os = "macos")]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    macos::probe_global_hotkeys().await
}

/// Probe whether the current desktop session exposes a global hotkey backend.
#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    GlobalHotkeysStatus {
        available: false,
        backend: "unsupported".to_owned(),
        detail: Some("Global hotkeys are not supported on this platform.".to_owned()),
    }
}
