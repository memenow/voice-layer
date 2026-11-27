//! Linux XDG GlobalShortcuts portal probe over the session D-Bus (`zbus` 6).

use std::convert::TryInto;

use zbus::{
    Connection,
    fdo::{DBusProxy, PropertiesProxy},
    names::{BusName, InterfaceName},
};

use super::GlobalHotkeysStatus;

const PORTAL_BUS_NAME: &str = "org.freedesktop.portal.Desktop";
const PORTAL_OBJECT_PATH: &str = "/org/freedesktop/portal/desktop";
const GLOBAL_SHORTCUTS_INTERFACE: &str = "org.freedesktop.portal.GlobalShortcuts";

fn unavailable(detail: String) -> GlobalHotkeysStatus {
    GlobalHotkeysStatus {
        available: false,
        backend: "xdg_portal".to_owned(),
        detail: Some(detail),
    }
}

pub async fn probe_global_hotkeys() -> GlobalHotkeysStatus {
    let connection = match Connection::session().await {
        Ok(connection) => connection,
        Err(error) => {
            return unavailable(format!(
                "Unable to connect to the D-Bus session bus: {error}"
            ));
        }
    };

    if let Err(status) = verify_portal_owner(&connection).await {
        return status;
    }
    read_portal_version(&connection).await
}

async fn verify_portal_owner(connection: &Connection) -> Result<(), GlobalHotkeysStatus> {
    let dbus_proxy = DBusProxy::new(connection)
        .await
        .map_err(|error| unavailable(format!("Unable to create a DBusProxy: {error}")))?;
    let portal_bus_name = BusName::try_from(PORTAL_BUS_NAME)
        .map_err(|error| unavailable(format!("Invalid portal bus name constant: {error}")))?;
    match dbus_proxy.name_has_owner(portal_bus_name).await {
        Ok(true) => Ok(()),
        Ok(false) => Err(unavailable(format!(
            "{PORTAL_BUS_NAME} is not owned on the current session bus."
        ))),
        Err(error) => Err(unavailable(format!(
            "Failed to check portal bus ownership: {error}"
        ))),
    }
}

async fn read_portal_version(connection: &Connection) -> GlobalHotkeysStatus {
    let properties =
        match PropertiesProxy::new(connection, PORTAL_BUS_NAME, PORTAL_OBJECT_PATH).await {
            Ok(proxy) => proxy,
            Err(error) => {
                return unavailable(format!(
                    "Unable to create the portal PropertiesProxy: {error}"
                ));
            }
        };

    let interface_name = match InterfaceName::try_from(GLOBAL_SHORTCUTS_INTERFACE) {
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
            "Unable to read {GLOBAL_SHORTCUTS_INTERFACE}.version: {error}"
        )),
    }
}
