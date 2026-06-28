//! The desktop shell's own persisted preferences (`desktop.toml`).
//!
//! The CLI owns daemon-side configuration (`config.toml`, in the `vl` crate);
//! this is a separate, GUI-only file holding the two preferences the shell can
//! change at runtime: the Liquid Glass opacity slider and the manual Reduce
//! Transparency toggle. It lives beside the CLI config in the platform config
//! directory but is intentionally decoupled — the shell never edits the CLI's
//! keys, and these never appear in the CLI's `SUPPORTED_CONFIG_KEYS`.
//!
//! Loading and saving are best-effort: any error falls back to defaults and is
//! logged, never fatal, since a missing or unreadable preference file must not
//! stop the shell from opening or a read-only home from running it.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

use crate::state::{DEFAULT_GLASS_OPACITY, Preferences};

/// The persisted slice of [`Preferences`]. Only the two user-controlled glass
/// settings are stored; the workflow defaults are session-scoped.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
#[serde(default)]
pub struct DesktopConfig {
    /// Liquid Glass opacity, `0.0` (clearest) .. `1.0` (frosted).
    pub glass_opacity: f32,
    /// Manual Reduce Transparency (GNOME has no system signal for it).
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
    /// Snapshot the persisted fields out of the live [`Preferences`], clamping
    /// opacity into range so a hand-edited file can't push it out of bounds.
    pub fn from_preferences(prefs: &Preferences) -> Self {
        Self {
            glass_opacity: prefs.glass_opacity.clamp(0.0, 1.0),
            reduce_transparency: prefs.reduce_transparency,
        }
    }

    /// Apply the persisted fields onto a [`Preferences`], leaving the session
    /// defaults untouched and clamping opacity for the same reason.
    pub fn apply_to(&self, prefs: &mut Preferences) {
        prefs.glass_opacity = self.glass_opacity.clamp(0.0, 1.0);
        prefs.reduce_transparency = self.reduce_transparency;
    }
}

/// `desktop.toml` beside the CLI config in the platform config directory
/// (`~/.config/voicelayer/` on Linux). `None` when the platform has no config
/// directory, in which case preferences are simply not persisted.
pub fn config_path() -> Option<PathBuf> {
    directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .map(|dirs| dirs.config_dir().join("desktop.toml"))
}

/// Load persisted preferences, best-effort. A missing file yields defaults; a
/// malformed file logs a warning and yields defaults so a bad edit can't wedge
/// the shell.
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

/// Persist preferences, best-effort. Returns whether the write succeeded; a
/// failure is logged and swallowed (a read-only home must not crash the shell).
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

    /// `#[serde(default)]` lets an empty or partial file load with baseline
    /// values rather than failing — a forward-compatible config surface.
    #[test]
    fn empty_file_falls_back_to_defaults() {
        let cfg: DesktopConfig = toml::from_str("").expect("empty toml is all-defaults");
        assert_eq!(cfg, DesktopConfig::default());
    }
}
