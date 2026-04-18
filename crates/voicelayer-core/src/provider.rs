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

pub fn default_provider_catalog() -> Vec<ProviderDescriptor> {
    vec![
        ProviderDescriptor {
            id: "whisper_cpp".to_owned(),
            kind: ProviderKind::Asr,
            transport: "local_process".to_owned(),
            local: true,
            default_enabled: true,
            experimental: false,
            license: "MIT".to_owned(),
        },
        ProviderDescriptor {
            id: "voxtral_realtime".to_owned(),
            kind: ProviderKind::Asr,
            transport: "local_process".to_owned(),
            local: true,
            default_enabled: false,
            experimental: true,
            license: "Apache-2.0".to_owned(),
        },
        ProviderDescriptor {
            id: "gemma_4_local".to_owned(),
            kind: ProviderKind::Llm,
            transport: "local_process".to_owned(),
            local: true,
            default_enabled: true,
            experimental: false,
            license: "Apache-2.0".to_owned(),
        },
    ]
}

pub fn default_host_adapter_catalog() -> Vec<ProviderDescriptor> {
    vec![
        ProviderDescriptor {
            id: "atspi_accessible_text".to_owned(),
            kind: ProviderKind::HostAdapter,
            transport: "desktop_bus".to_owned(),
            local: true,
            default_enabled: true,
            experimental: false,
            license: "system".to_owned(),
        },
        ProviderDescriptor {
            id: "global_shortcuts_portal".to_owned(),
            kind: ProviderKind::HostAdapter,
            transport: "desktop_bus".to_owned(),
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
    ]
}

#[cfg(test)]
mod tests {
    use super::{ProviderKind, default_host_adapter_catalog, default_provider_catalog};

    #[test]
    fn catalog_contains_required_local_defaults() {
        let catalog = default_provider_catalog();
        assert!(catalog.iter().any(|provider| provider.id == "whisper_cpp"));
        assert!(
            catalog
                .iter()
                .any(|provider| provider.id == "gemma_4_local")
        );
        assert!(
            catalog
                .iter()
                .any(|provider| provider.kind == ProviderKind::Asr && provider.default_enabled)
        );
    }

    #[test]
    fn host_adapter_catalog_contains_terminal_path() {
        let catalog = default_host_adapter_catalog();
        assert!(
            catalog
                .iter()
                .any(|provider| provider.id == "terminal_bracketed_paste")
        );
    }
}
