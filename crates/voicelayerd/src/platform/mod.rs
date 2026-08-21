//! Platform-specific capabilities.
//!
//! All `cfg(target_os)` differences in the daemon live in this module;
//! business code consumes the platform-neutral surface defined here.

use serde::Serialize;

/// Whether the desktop session can deliver global hotkeys to VoiceLayer.
#[derive(Debug, Clone, Serialize)]
pub struct GlobalHotkeysStatus {
    pub available: bool,
    pub backend: String,
    pub detail: Option<String>,
}

#[cfg(target_os = "linux")]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    use std::convert::TryInto;
    use zbus::{
        Connection,
        fdo::{DBusProxy, PropertiesProxy},
        names::{BusName, InterfaceName},
    };

    fn unavailable(detail: String) -> GlobalHotkeysStatus {
        GlobalHotkeysStatus {
            available: false,
            backend: "xdg_portal".to_owned(),
            detail: Some(detail),
        }
    }

    let connection = match Connection::session().await {
        Ok(connection) => connection,
        Err(error) => {
            return unavailable(format!(
                "Unable to connect to the D-Bus session bus: {error}"
            ));
        }
    };

    let dbus_proxy = match DBusProxy::new(&connection).await {
        Ok(proxy) => proxy,
        Err(error) => return unavailable(format!("Unable to create a DBusProxy: {error}")),
    };

    let portal_bus_name = match BusName::try_from("org.freedesktop.portal.Desktop") {
        Ok(name) => name,
        Err(error) => return unavailable(format!("Invalid portal bus name constant: {error}")),
    };

    match dbus_proxy.name_has_owner(portal_bus_name).await {
        Ok(false) => {
            return unavailable(
                "org.freedesktop.portal.Desktop is not owned on the current session bus."
                    .to_owned(),
            );
        }
        Err(error) => {
            return unavailable(format!("Failed to check portal bus ownership: {error}"));
        }
        Ok(true) => {}
    }

    let properties = match PropertiesProxy::new(
        &connection,
        "org.freedesktop.portal.Desktop",
        "/org/freedesktop/portal/desktop",
    )
    .await
    {
        Ok(proxy) => proxy,
        Err(error) => {
            return unavailable(format!(
                "Unable to create the portal PropertiesProxy: {error}"
            ));
        }
    };

    let interface_name = match InterfaceName::try_from("org.freedesktop.portal.GlobalShortcuts") {
        Ok(name) => name,
        Err(error) => {
            return unavailable(format!(
                "Invalid GlobalShortcuts interface constant: {error}"
            ));
        }
    };

    match properties.get(interface_name, "version").await {
        Ok(value) => match TryInto::<u32>::try_into(value) {
            Ok(version) => GlobalHotkeysStatus {
                available: true,
                backend: "xdg_portal".to_owned(),
                detail: Some(format!("GlobalShortcuts portal version {version}")),
            },
            Err(error) => unavailable(format!(
                "GlobalShortcuts portal returned an unexpected `version` type: {error}"
            )),
        },
        Err(error) => unavailable(format!(
            "Unable to read org.freedesktop.portal.GlobalShortcuts.version: {error}"
        )),
    }
}

#[cfg(target_os = "macos")]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    // macOS global hotkeys are driven through the `global-hotkey` crate
    // (Carbon RegisterEventHotKey). Binding requires the user to grant the
    // host app Input Monitoring permission; that is verified at bind time
    // by the desktop shell, so here we report the backend as present.
    GlobalHotkeysStatus {
        available: true,
        backend: "global_hotkey".to_owned(),
        detail: Some(
            "macOS Carbon global hotkeys; requires Input Monitoring permission.".to_owned(),
        ),
    }
}

#[cfg(not(any(target_os = "linux", target_os = "macos")))]
pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    GlobalHotkeysStatus {
        available: false,
        backend: "unsupported".to_owned(),
        detail: Some("Global hotkeys are not supported on this platform.".to_owned()),
    }
}
