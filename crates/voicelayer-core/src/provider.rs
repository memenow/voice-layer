//! Static provider and host-adapter catalogs plus ASR dispatch validation.

use serde::{Deserialize, Serialize};

/// Capability category served by a provider.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ProviderKind {
    Asr,
    Llm,
    Tts,
    HostAdapter,
}

/// Static catalog entry returned by `GET /v1/providers` and `vl providers`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProviderDescriptor {
    /// Identifier accepted by `TranscribeRequest.provider_id` and RPC dispatch.
    pub id: String,
    pub kind: ProviderKind,
    /// Transport mechanism (for example `local_process`, `stdio_worker`, `xdg_portal`).
    pub transport: String,
    /// `true` when execution stays on the local host.
    pub local: bool,
    /// `true` when active without explicit opt-in.
    pub default_enabled: bool,
    /// `true` for experimental or advertised-only entries.
    pub experimental: bool,
    /// SPDX license identifier, or `system` / `n/a` for host-provided surfaces.
    pub license: String,
}

/// ASR provider ids supported by runtime transcription dispatch.
///
/// Keep aligned with `python/voicelayer_worker/pipeline.py` and the `provider_id`
/// enum in `openapi/voicelayerd.v1.yaml`.
pub const SUPPORTED_TRANSCRIBE_PROVIDER_IDS: &[&str] =
    &["whisper_cpp", "mimo_v2_5_asr", "qwen3_asr_1_7b"];

/// Returns `true` if `provider_id` is `None` (default whisper.cpp) or listed in
/// [`SUPPORTED_TRANSCRIBE_PROVIDER_IDS`].
pub fn is_supported_transcribe_provider_id(provider_id: Option<&str>) -> bool {
    match provider_id {
        None => true,
        Some(id) => SUPPORTED_TRANSCRIBE_PROVIDER_IDS.contains(&id),
    }
}

/// Returns `true` if the ASR provider supports `translate_to_english`.
/// Only the default `whisper_cpp` backend implements speech-to-English translation.
pub fn provider_supports_asr_translation(provider_id: Option<&str>) -> bool {
    matches!(provider_id, None | Some("whisper_cpp"))
}

/// Static catalog of ASR and LLM providers.
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
            id: "mimo_v2_5_asr".to_owned(),
            kind: ProviderKind::Asr,
            transport: "stdio_worker".to_owned(),
            local: true,
            default_enabled: false,
            experimental: true,
            license: "MIT".to_owned(),
        },
        ProviderDescriptor {
            id: "qwen3_asr_1_7b".to_owned(),
            kind: ProviderKind::Asr,
            transport: "stdio_worker".to_owned(),
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

/// Static catalog of platform host adapters for text insertion and global hotkeys.
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
    use super::{
        ProviderKind, SUPPORTED_TRANSCRIBE_PROVIDER_IDS, default_host_adapter_catalog,
        default_provider_catalog, is_supported_transcribe_provider_id,
        provider_supports_asr_translation,
    };

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
    fn catalog_contains_mimo_v2_5_asr_descriptor_as_optional_experimental() {
        let catalog = default_provider_catalog();
        let mimo = catalog
            .iter()
            .find(|provider| provider.id == "mimo_v2_5_asr")
            .expect("mimo_v2_5_asr descriptor missing");
        assert_eq!(mimo.kind, ProviderKind::Asr);
        assert!(!mimo.default_enabled);
        assert!(mimo.experimental);
        assert_eq!(mimo.license, "MIT");
    }

    #[test]
    fn catalog_contains_qwen3_asr_1_7b_descriptor_as_optional_experimental() {
        let catalog = default_provider_catalog();
        let qwen3 = catalog
            .iter()
            .find(|provider| provider.id == "qwen3_asr_1_7b")
            .expect("qwen3_asr_1_7b descriptor missing");
        assert_eq!(qwen3.kind, ProviderKind::Asr);
        assert!(!qwen3.default_enabled);
        assert!(qwen3.experimental);
        assert_eq!(qwen3.license, "Apache-2.0");
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

    #[test]
    fn provider_catalog_entries_have_non_empty_required_string_fields() {
        for provider in default_provider_catalog()
            .iter()
            .chain(default_host_adapter_catalog().iter())
        {
            assert!(!provider.id.is_empty());
            assert!(!provider.transport.is_empty());
            assert!(!provider.license.is_empty());
        }
    }

    #[test]
    fn provider_catalog_ids_are_unique_across_both_catalogs() {
        let mut seen = std::collections::HashSet::new();
        for provider in default_provider_catalog()
            .iter()
            .chain(default_host_adapter_catalog().iter())
        {
            assert!(
                seen.insert(provider.id.clone()),
                "duplicate provider id `{}`",
                provider.id,
            );
        }
    }

    #[test]
    fn host_adapter_catalog_entries_are_all_host_adapter_kind() {
        for provider in default_host_adapter_catalog() {
            assert_eq!(provider.kind, ProviderKind::HostAdapter);
        }
    }

    #[test]
    fn regular_provider_catalog_does_not_include_host_adapter_kind() {
        for provider in default_provider_catalog() {
            assert_ne!(provider.kind, ProviderKind::HostAdapter);
        }
    }

    #[test]
    fn every_supported_transcribe_provider_id_appears_in_catalog_as_asr() {
        let catalog = default_provider_catalog();
        for id in SUPPORTED_TRANSCRIBE_PROVIDER_IDS {
            let descriptor = catalog
                .iter()
                .find(|provider| provider.id == *id)
                .unwrap_or_else(|| panic!("missing catalog entry for `{id}`"));
            assert_eq!(descriptor.kind, ProviderKind::Asr);
        }
    }

    #[test]
    fn is_supported_transcribe_provider_id_accepts_listed_ids_and_none() {
        assert!(is_supported_transcribe_provider_id(None));
        for id in SUPPORTED_TRANSCRIBE_PROVIDER_IDS {
            assert!(is_supported_transcribe_provider_id(Some(*id)));
        }
        assert!(!is_supported_transcribe_provider_id(Some("not_a_provider")));
        assert!(!is_supported_transcribe_provider_id(Some("mimo_v2_5")));
        assert!(!is_supported_transcribe_provider_id(Some("qwen3_asr")));
    }

    #[test]
    fn translation_support_is_limited_to_whisper_cpp() {
        assert!(provider_supports_asr_translation(None));
        assert!(provider_supports_asr_translation(Some("whisper_cpp")));
        assert!(!provider_supports_asr_translation(Some("mimo_v2_5_asr")));
        assert!(!provider_supports_asr_translation(Some("qwen3_asr_1_7b")));
    }
}
