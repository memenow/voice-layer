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

    /// Every entry in either catalog must have a non-empty `id`,
    /// `transport`, and `license`. Empty strings would silently pass
    /// the existing `id == "..."` checks but break the `vl providers`
    /// listing (empty rows) and the OpenAPI contract's
    /// `ProviderDescriptor` schema (`required: [id, kind, transport,
    /// ...]` doesn't constrain string emptiness).
    #[test]
    fn provider_catalog_entries_have_non_empty_required_string_fields() {
        for provider in default_provider_catalog()
            .iter()
            .chain(default_host_adapter_catalog().iter())
        {
            assert!(
                !provider.id.is_empty(),
                "every catalog entry must have a non-empty id; got: {provider:?}",
            );
            assert!(
                !provider.transport.is_empty(),
                "every catalog entry must have a non-empty transport; got: {provider:?}",
            );
            assert!(
                !provider.license.is_empty(),
                "every catalog entry must have a non-empty license; got: {provider:?}",
            );
        }
    }

    /// `id` is the operator-facing handle used by `vl providers`,
    /// `vl doctor`, and the worker's RPC dispatch. A duplicate id
    /// would silently route every selection to whichever entry
    /// happens to be enumerated first; the cross-catalog check
    /// catches duplicates introduced by accidentally reusing a host
    /// adapter id under the regular provider catalog.
    #[test]
    fn provider_catalog_ids_are_unique_across_both_catalogs() {
        let mut seen = std::collections::HashSet::new();
        for provider in default_provider_catalog()
            .iter()
            .chain(default_host_adapter_catalog().iter())
        {
            assert!(
                seen.insert(provider.id.clone()),
                "duplicate id `{}` across catalogs; ids must be globally unique",
                provider.id,
            );
        }
    }

    /// The host adapter catalog is reserved for `HostAdapter`-kind
    /// entries; mixing in an `Asr`/`Llm`/`Tts` entry would mislead
    /// the desktop integration about which providers are
    /// host-side automation surfaces.
    #[test]
    fn host_adapter_catalog_entries_are_all_host_adapter_kind() {
        for provider in default_host_adapter_catalog() {
            assert_eq!(
                provider.kind,
                ProviderKind::HostAdapter,
                "host_adapter_catalog entry `{}` has kind {:?}; expected HostAdapter",
                provider.id,
                provider.kind,
            );
        }
    }

    /// Symmetric invariant: the regular provider catalog must not
    /// include `HostAdapter`-kind entries; those belong in
    /// `default_host_adapter_catalog()`. Catches the inverse drift
    /// of the previous test.
    #[test]
    fn regular_provider_catalog_does_not_include_host_adapter_kind() {
        for provider in default_provider_catalog() {
            assert_ne!(
                provider.kind,
                ProviderKind::HostAdapter,
                "default_provider_catalog entry `{}` is HostAdapter-kind; \
                 move it to default_host_adapter_catalog",
                provider.id,
            );
        }
    }
}
