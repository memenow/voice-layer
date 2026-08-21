use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Asr,
    Llm,
    Tts,
    HostAdapter,
}

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderDescriptor {
    pub id: String,
    pub kind: ProviderKind,
    pub transport: String,
    pub local: bool,
    pub default_enabled: bool,
    pub experimental: bool,
    pub license: String,
}

pub fn default_host_adapter_catalog() -> Vec<ProviderDescriptor> {
    let mut catalog = vec![
        ProviderDescriptor {
            id: "global_shortcuts".to_owned(),
            kind: ProviderKind::HostAdapter,
            transport: if cfg!(target_os = "linux") {
                "xdg_portal".to_owned()
            } else {
                "global_hotkey".to_owned()
            },
            local: true,
            default_enabled: true,
            experimental: false,
            license: "system".to_owned(),
        },
        ProviderDescriptor {
            id: "terminal_bracketed_paste".to_owned(),
            kind: ProviderKind::HostAdapter,
            transport: "stdout_payload".to_owned(),
            local: true,
            default_enabled: true,
            experimental: false,
            license: "n/a".to_owned(),
        },
    ];
    if cfg!(target_os = "linux") {
        catalog.insert(
            0,
            ProviderDescriptor {
                id: "atspi_accessible_text".to_owned(),
                kind: ProviderKind::HostAdapter,
                transport: "desktop_bus".to_owned(),
                local: true,
                default_enabled: true,
                experimental: false,
                license: "system".to_owned(),
            },
        );
    } else {
        catalog.insert(
            0,
            ProviderDescriptor {
                id: "macos_clipboard_paste".to_owned(),
                kind: ProviderKind::HostAdapter,
                transport: "core_graphics".to_owned(),
                local: true,
                default_enabled: true,
                experimental: false,
                license: "system".to_owned(),
            },
        );
    }
    catalog
}

#[cfg(test)]
mod tests {
    use super::default_host_adapter_catalog;

    #[test]
    fn host_adapter_catalog_contains_terminal_path() {
        let catalog = default_host_adapter_catalog();
        assert!(
            catalog
                .iter()
                .any(|provider| provider.id == "terminal_bracketed_paste")
        );
    }

    #[test]
    fn host_adapter_catalog_is_platform_specific() {
        let catalog = default_host_adapter_catalog();
        if cfg!(target_os = "linux") {
            assert!(
                catalog
                    .iter()
                    .any(|provider| provider.id == "atspi_accessible_text")
            );
        } else {
            assert!(
                catalog
                    .iter()
                    .any(|provider| provider.id == "macos_clipboard_paste")
            );
        }
    }
}
