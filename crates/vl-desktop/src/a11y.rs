//! XDG Settings portal integration for the accessibility contract.
//!
//! GNOME publishes appearance/accessibility preferences through the
//! `org.freedesktop.portal.Settings` interface (namespace
//! `org.freedesktop.appearance`). The shell reads two of them and maps them onto
//! [`SystemA11y`]:
//!
//! - **Increase Contrast** ← `contrast` (`u`; `1` = high). Native, stable.
//! - **Reduce Motion** ← `reduced-motion` (`u`; `1` = reduce). New in GNOME 50 /
//!   xdg-desktop-portal 1.21; on older systems the key is absent, so we fall back
//!   to `org.gnome.desktop.interface` `enable-animations` (`b`, inverted).
//!
//! There is deliberately no Reduce Transparency read: GNOME exposes no such
//! signal (unlike macOS), so that stays a manual user toggle (see
//! [`crate::state::Preferences`]).
//!
//! Everything is best-effort: when the portal is unavailable the shell keeps its
//! defaults and simply does not react to system accessibility changes.

use ashpd::desktop::settings::{
    APPEARANCE_NAMESPACE, CONTRAST_KEY, Contrast, REDUCED_MOTION_KEY, ReducedMotion, Settings,
};
use futures_util::StreamExt;
use tokio::sync::mpsc::UnboundedSender;

use crate::state::SystemA11y;

/// The GNOME interface namespace carrying the legacy `enable-animations` boolean
/// used as the reduced-motion fallback on systems without the portal key.
const GNOME_INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";
/// The legacy boolean: animations enabled. `false` means motion is reduced.
const ENABLE_ANIMATIONS_KEY: &str = "enable-animations";

/// Read the current accessibility preferences once. A missing portal yields the
/// full default (no preference); any single failing read leaves that field at
/// its default. Best-effort by design — used to seed state at startup.
pub async fn probe() -> SystemA11y {
    match Settings::new().await {
        Ok(proxy) => read_state(&proxy).await,
        Err(_) => SystemA11y::default(),
    }
}

/// Watch for live accessibility changes, sending a fresh [`SystemA11y`] on each
/// relevant portal `SettingChanged`. Emits the initial state immediately so a
/// subscriber that started after [`probe`] still converges. Long-lived: returns
/// when the portal stream ends or the receiver drops. Best-effort — an
/// unavailable portal returns an error the caller logs and ignores.
pub async fn watch(sender: UnboundedSender<SystemA11y>) -> Result<(), String> {
    let proxy = Settings::new().await.map_err(|e| e.to_string())?;
    if sender.send(read_state(&proxy).await).is_err() {
        return Ok(());
    }
    let mut changed = proxy
        .receive_setting_changed()
        .await
        .map_err(|e| e.to_string())?;
    while let Some(setting) = changed.next().await {
        if is_relevant(setting.namespace(), setting.key())
            && sender.send(read_state(&proxy).await).is_err()
        {
            break;
        }
    }
    Ok(())
}

/// Read both tracked preferences from an open portal proxy.
async fn read_state(proxy: &Settings) -> SystemA11y {
    SystemA11y {
        increase_contrast: read_contrast(proxy).await,
        reduce_motion: read_reduced_motion(proxy).await,
    }
}

async fn read_contrast(proxy: &Settings) -> bool {
    matches!(proxy.contrast().await, Ok(Contrast::High))
}

/// Reduced motion, preferring the modern portal key and falling back to the
/// older `enable-animations` boolean when it is absent (GNOME < 50). Any error
/// resolves to "motion not reduced".
async fn read_reduced_motion(proxy: &Settings) -> bool {
    if let Ok(motion) = proxy.reduced_motion().await {
        return matches!(motion, ReducedMotion::ReducedMotion);
    }
    // `enable-animations == false` means motion is reduced.
    matches!(
        proxy
            .read::<bool>(GNOME_INTERFACE_NAMESPACE, ENABLE_ANIMATIONS_KEY)
            .await,
        Ok(false)
    )
}

/// Whether a `SettingChanged` for `namespace.key` affects what we track, so we
/// only re-read on relevant changes rather than every appearance tweak.
fn is_relevant(namespace: &str, key: &str) -> bool {
    (namespace == APPEARANCE_NAMESPACE && (key == CONTRAST_KEY || key == REDUCED_MOTION_KEY))
        || (namespace == GNOME_INTERFACE_NAMESPACE && key == ENABLE_ANIMATIONS_KEY)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Only the contrast, reduced-motion, and animations keys should trigger a
    /// re-read; an unrelated appearance change (e.g. color scheme) must not.
    #[test]
    fn relevant_keys_are_recognized() {
        assert!(is_relevant(APPEARANCE_NAMESPACE, CONTRAST_KEY));
        assert!(is_relevant(APPEARANCE_NAMESPACE, REDUCED_MOTION_KEY));
        assert!(is_relevant(
            GNOME_INTERFACE_NAMESPACE,
            ENABLE_ANIMATIONS_KEY
        ));
        assert!(!is_relevant(APPEARANCE_NAMESPACE, "color-scheme"));
        assert!(!is_relevant("org.example.other", CONTRAST_KEY));
    }
}
