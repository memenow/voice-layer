//! Persisted desktop-only visual preferences (`desktop.toml`).
//!
//! Stored beside `config.toml` in the platform config directory and intentionally
//! decoupled from CLI/daemon settings. Load and save are best-effort so a missing
//! or read-only config directory never prevents the desktop shell from running.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::state::{DEFAULT_GLASS_OPACITY, Preferences};

/// Persisted subset of [`Preferences`].
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopConfig {
    /// Translucent surface opacity in `0.0..=1.0`.
    pub glass_opacity: f32,
    /// Manual Reduce Transparency override (GNOME exposes no system signal).
    pub reduce_transparency: bool,
}

impl Default for DesktopConfig {
    fn default() -> Self {
        Self {
            glass_opacity: DEFAULT_GLASS_OPACITY,
            reduce_transparency: false,
        }
    }
}

impl DesktopConfig {
    pub fn from_preferences(prefs: &Preferences) -> Self {
        Self {
            glass_opacity: prefs.glass_opacity.clamp(0.0, 1.0),
            reduce_transparency: prefs.reduce_transparency,
        }
    }

    pub fn apply_to(&self, prefs: &mut Preferences) {
        prefs.glass_opacity = self.glass_opacity.clamp(0.0, 1.0);
        prefs.reduce_transparency = self.reduce_transparency;
    }
}

/// Platform path for `desktop.toml`, or `None` if no user config directory exists.
pub fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .map(|dirs| dirs.config_dir().join("desktop.toml"))
}

/// Load `desktop.toml`, falling back to defaults on missing or invalid files.
pub fn load() -> DesktopConfig {
    let Some(path) = config_path() else {
        return DesktopConfig::default();
    };
    if !path.is_file() {
        return DesktopConfig::default();
    }
    match std::fs::read_to_string(&path) {
        Ok(contents) => match toml::from_str::<DesktopConfig>(&contents) {
            Ok(config) => config,
            Err(error) => {
                tracing::warn!(
                    error = %error,
                    path = %path.display(),
                    "ignoring malformed desktop.toml",
                );
                DesktopConfig::default()
            }
        },
        Err(error) => {
            tracing::warn!(
                error = %error,
                path = %path.display(),
                "could not read desktop.toml",
            );
            DesktopConfig::default()
        }
    }
}

/// Write `desktop.toml`, logging and ignoring I/O errors.
pub fn save(config: &DesktopConfig) -> bool {
    let Some(path) = config_path() else {
        return false;
    };
    let result = (|| -> std::io::Result<()> {
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)?;
        }
        let body = toml::to_string_pretty(config)
            .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
        std::fs::write(&path, body)
    })();
    match result {
        Ok(()) => true,
        Err(error) => {
            tracing::warn!(
                error = %error,
                path = %path.display(),
                "could not write desktop.toml",
            );
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_preferences_clamps_opacity_out_of_range() {
        let prefs = Preferences {
            glass_opacity: 1.7,
            reduce_transparency: true,
            ..Default::default()
        };
        let cfg = DesktopConfig::from_preferences(&prefs);
        assert_eq!(cfg.glass_opacity, 1.0);
        assert!(cfg.reduce_transparency);
    }

    #[test]
    fn apply_to_restores_persisted_fields_and_clamps() {
        let cfg = DesktopConfig {
            glass_opacity: -0.5,
            reduce_transparency: true,
        };
        let mut prefs = Preferences::default();
        cfg.apply_to(&mut prefs);
        assert_eq!(prefs.glass_opacity, 0.0);
        assert!(prefs.reduce_transparency);
    }

    #[test]
    fn roundtrip_through_toml_preserves_values() {
        let cfg = DesktopConfig {
            glass_opacity: 0.75,
            reduce_transparency: true,
        };
        let text = toml::to_string_pretty(&cfg).expect("serialize");
        let back: DesktopConfig = toml::from_str(&text).expect("deserialize");
        assert_eq!(back, cfg);
    }

    #[test]
    fn empty_file_falls_back_to_defaults() {
        let cfg: DesktopConfig = toml::from_str("").expect("empty toml should load defaults");
        assert_eq!(cfg, DesktopConfig::default());
    }
}
