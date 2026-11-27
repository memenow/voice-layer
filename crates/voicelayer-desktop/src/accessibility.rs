//! OS accessibility preference integration via the XDG Settings portal.
//!
//! On Linux, reads `org.freedesktop.appearance`:
//! - `contrast` (`1` = high contrast)
//! - `reduced-motion` (`1` = reduce motion), falling back to
//!   `org.gnome.desktop.interface` `enable-animations` (`false`) on older portals
//!
//! Reduce transparency is not exposed by GNOME and remains a manual toggle in
//! [`crate::state::Preferences`]. All reads are best-effort and fall back to
//! defaults when the portal is unavailable.

#[cfg(target_os = "linux")]
mod backend {
    use ashpd::desktop::settings::{
        APPEARANCE_NAMESPACE, CONTRAST_KEY, Contrast, REDUCED_MOTION_KEY, ReducedMotion, Settings,
    };
    use futures_util::StreamExt;
    use tokio::sync::mpsc::UnboundedSender;

    use crate::state::SystemA11y;

    const GNOME_INTERFACE_NAMESPACE: &str = "org.gnome.desktop.interface";
    const ENABLE_ANIMATIONS_KEY: &str = "enable-animations";

    /// Read the current accessibility preferences once at startup.
    pub async fn probe() -> SystemA11y {
        match Settings::new().await {
            Ok(proxy) => read_state(&proxy).await,
            Err(_) => SystemA11y::default(),
        }
    }

    /// Watch for live accessibility updates and send snapshots on `sender`.
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

    async fn read_state(proxy: &Settings) -> SystemA11y {
        SystemA11y {
            increase_contrast: read_contrast(proxy).await,
            reduce_motion: read_reduced_motion(proxy).await,
        }
    }

    async fn read_contrast(proxy: &Settings) -> bool {
        matches!(proxy.contrast().await, Ok(Contrast::High))
    }

    async fn read_reduced_motion(proxy: &Settings) -> bool {
        if let Ok(motion) = proxy.reduced_motion().await {
            return matches!(motion, ReducedMotion::ReducedMotion);
        }
        matches!(
            proxy
                .read::<bool>(GNOME_INTERFACE_NAMESPACE, ENABLE_ANIMATIONS_KEY)
                .await,
            Ok(false)
        )
    }

    fn is_relevant(namespace: &str, key: &str) -> bool {
        (namespace == APPEARANCE_NAMESPACE && (key == CONTRAST_KEY || key == REDUCED_MOTION_KEY))
            || (namespace == GNOME_INTERFACE_NAMESPACE && key == ENABLE_ANIMATIONS_KEY)
    }
}

#[cfg(not(target_os = "linux"))]
mod backend {
    use tokio::sync::mpsc::UnboundedSender;

    use crate::state::SystemA11y;

    pub async fn probe() -> SystemA11y {
        SystemA11y::default()
    }

    pub async fn watch(_sender: UnboundedSender<SystemA11y>) -> Result<(), String> {
        Err("XDG Settings portal is not available on this platform".to_owned())
    }
}

pub use backend::{probe, watch};
