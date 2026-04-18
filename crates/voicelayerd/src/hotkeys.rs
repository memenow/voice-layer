use std::convert::TryInto;

use serde::Serialize;
use zbus::{
    Connection,
    fdo::{DBusProxy, PropertiesProxy},
    names::{BusName, InterfaceName},
};

#[derive(Debug, Clone, Serialize)]
pub struct GlobalShortcutsPortalStatus {
    pub available: bool,
    pub version: Option<u32>,
    pub error: Option<String>,
}

pub async fn probe_global_shortcuts_portal() -> GlobalShortcutsPortalStatus {
    let connection = match Connection::session().await {
        Ok(connection) => connection,
        Err(error) => {
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!(
                    "Unable to connect to the D-Bus session bus: {error}"
                )),
            };
        }
    };

    let dbus_proxy = match DBusProxy::new(&connection).await {
        Ok(proxy) => proxy,
        Err(error) => {
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!("Unable to create a DBusProxy: {error}")),
            };
        }
    };

    let portal_bus_name = match BusName::try_from("org.freedesktop.portal.Desktop") {
        Ok(name) => name,
        Err(error) => {
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!("Invalid portal bus name constant: {error}")),
            };
        }
    };

    let has_portal_owner = match dbus_proxy.name_has_owner(portal_bus_name).await {
        Ok(has_owner) => has_owner,
        Err(error) => {
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!("Failed to check portal bus ownership: {error}")),
            };
        }
    };

    if !has_portal_owner {
        return GlobalShortcutsPortalStatus {
            available: false,
            version: None,
            error: Some(
                "org.freedesktop.portal.Desktop is not owned on the current session bus."
                    .to_owned(),
            ),
        };
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
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!(
                    "Unable to create the portal PropertiesProxy: {error}"
                )),
            };
        }
    };

    let interface_name = match InterfaceName::try_from("org.freedesktop.portal.GlobalShortcuts") {
        Ok(name) => name,
        Err(error) => {
            return GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!(
                    "Invalid GlobalShortcuts interface constant: {error}"
                )),
            };
        }
    };

    match properties.get(interface_name, "version").await {
        Ok(value) => match TryInto::<u32>::try_into(value) {
            Ok(version) => GlobalShortcutsPortalStatus {
                available: true,
                version: Some(version),
                error: None,
            },
            Err(error) => GlobalShortcutsPortalStatus {
                available: false,
                version: None,
                error: Some(format!(
                    "GlobalShortcuts portal returned an unexpected `version` type: {error}"
                )),
            },
        },
        Err(error) => GlobalShortcutsPortalStatus {
            available: false,
            version: None,
            error: Some(format!(
                "Unable to read org.freedesktop.portal.GlobalShortcuts.version: {error}"
            )),
        },
    }
}
