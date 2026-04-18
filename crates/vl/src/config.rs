use std::path::PathBuf;

use clap::ValueEnum;
use crossterm::event::KeyCode;
use serde::{Deserialize, Serialize};
use voicelayer_core::RecorderBackend;

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub(crate) enum CliRecorderBackend {
    Auto,
    Pipewire,
    Alsa,
}

impl From<CliRecorderBackend> for RecorderBackend {
    fn from(value: CliRecorderBackend) -> Self {
        match value {
            CliRecorderBackend::Auto => Self::Auto,
            CliRecorderBackend::Pipewire => Self::Pipewire,
            CliRecorderBackend::Alsa => Self::Alsa,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub(crate) enum CliPttKey {
    Space,
    Enter,
    Tab,
    F8,
    F9,
    F10,
}

impl CliPttKey {
    pub(crate) fn as_key_code(self) -> KeyCode {
        match self {
            CliPttKey::Space => KeyCode::Char(' '),
            CliPttKey::Enter => KeyCode::Enter,
            CliPttKey::Tab => KeyCode::Tab,
            CliPttKey::F8 => KeyCode::F(8),
            CliPttKey::F9 => KeyCode::F(9),
            CliPttKey::F10 => KeyCode::F(10),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            CliPttKey::Space => "Space",
            CliPttKey::Enter => "Enter",
            CliPttKey::Tab => "Tab",
            CliPttKey::F8 => "F8",
            CliPttKey::F9 => "F9",
            CliPttKey::F10 => "F10",
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, Deserialize)]
pub(crate) enum StopAction {
    None,
    Copy,
    Inject,
    Save,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub(crate) struct ForegroundPttConfig {
    pub(crate) language: Option<String>,
    pub(crate) backend: CliRecorderBackend,
    pub(crate) translate_to_english: bool,
    pub(crate) keep_audio: bool,
    pub(crate) key: CliPttKey,
    pub(crate) tmux_target_pane: Option<String>,
    pub(crate) wezterm_target_pane_id: Option<String>,
    pub(crate) kitty_match: Option<String>,
    pub(crate) copy_on_stop: bool,
    pub(crate) default_stop_action: StopAction,
    pub(crate) restore_clipboard_on_exit: bool,
    pub(crate) save_dir: Option<PathBuf>,
}

impl Default for ForegroundPttConfig {
    fn default() -> Self {
        Self {
            language: None,
            backend: CliRecorderBackend::Auto,
            translate_to_english: false,
            keep_audio: false,
            key: CliPttKey::Space,
            tmux_target_pane: None,
            wezterm_target_pane_id: None,
            kitty_match: None,
            copy_on_stop: false,
            default_stop_action: StopAction::None,
            restore_clipboard_on_exit: false,
            save_dir: None,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub(crate) struct VlConfig {
    #[serde(default)]
    pub(crate) foreground_ptt: ForegroundPttConfig,
}

pub(crate) fn vl_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let project_dirs = directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .ok_or("Unable to determine the platform config directory for VoiceLayer")?;
    Ok(project_dirs.config_dir().join("config.toml"))
}

pub(crate) fn load_vl_config() -> Result<VlConfig, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    if !path.is_file() {
        return Ok(VlConfig::default());
    }
    let contents = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&contents)?)
}

pub(crate) fn write_vl_config(config: &VlConfig) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, toml::to_string_pretty(config)?)?;
    Ok(path)
}

const SUPPORTED_CONFIG_KEYS: &[&str] = &[
    "foreground_ptt.language",
    "foreground_ptt.backend",
    "foreground_ptt.translate_to_english",
    "foreground_ptt.keep_audio",
    "foreground_ptt.key",
    "foreground_ptt.tmux_target_pane",
    "foreground_ptt.wezterm_target_pane_id",
    "foreground_ptt.kitty_match",
    "foreground_ptt.copy_on_stop",
    "foreground_ptt.default_stop_action",
    "foreground_ptt.restore_clipboard_on_exit",
    "foreground_ptt.save_dir",
];

pub(crate) fn set_config_value(
    config: &mut VlConfig,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    match key {
        "foreground_ptt.default_stop_action" => {
            config.foreground_ptt.default_stop_action = match value {
                "none" => StopAction::None,
                "copy" => StopAction::Copy,
                "inject" => StopAction::Inject,
                "save" => StopAction::Save,
                _ => return Err("expected one of: none, copy, inject, save".into()),
            };
        }
        "foreground_ptt.copy_on_stop" => {
            config.foreground_ptt.copy_on_stop = parse_bool(value)?;
        }
        "foreground_ptt.restore_clipboard_on_exit" => {
            config.foreground_ptt.restore_clipboard_on_exit = parse_bool(value)?;
        }
        "foreground_ptt.save_dir" => {
            config.foreground_ptt.save_dir = parse_optional_path(value);
        }
        "foreground_ptt.backend" => {
            config.foreground_ptt.backend = match value {
                "auto" => CliRecorderBackend::Auto,
                "pipewire" => CliRecorderBackend::Pipewire,
                "alsa" => CliRecorderBackend::Alsa,
                _ => return Err("expected one of: auto, pipewire, alsa".into()),
            };
        }
        "foreground_ptt.key" => {
            config.foreground_ptt.key = match value {
                "space" => CliPttKey::Space,
                "enter" => CliPttKey::Enter,
                "tab" => CliPttKey::Tab,
                "f8" => CliPttKey::F8,
                "f9" => CliPttKey::F9,
                "f10" => CliPttKey::F10,
                _ => return Err("expected one of: space, enter, tab, f8, f9, f10".into()),
            };
        }
        "foreground_ptt.language" => {
            config.foreground_ptt.language = parse_optional_string(value);
        }
        "foreground_ptt.translate_to_english" => {
            config.foreground_ptt.translate_to_english = parse_bool(value)?;
        }
        "foreground_ptt.keep_audio" => {
            config.foreground_ptt.keep_audio = parse_bool(value)?;
        }
        "foreground_ptt.tmux_target_pane" => {
            config.foreground_ptt.tmux_target_pane = parse_optional_string(value);
        }
        "foreground_ptt.wezterm_target_pane_id" => {
            config.foreground_ptt.wezterm_target_pane_id = parse_optional_string(value);
        }
        "foreground_ptt.kitty_match" => {
            config.foreground_ptt.kitty_match = parse_optional_string(value);
        }
        _ => {
            return Err(format!(
                "unsupported config key `{key}`; supported keys: {}",
                SUPPORTED_CONFIG_KEYS.join(", ")
            )
            .into());
        }
    }
    Ok(())
}

fn parse_optional_string(value: &str) -> Option<String> {
    if value.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(value.to_owned())
    }
}

fn parse_optional_path(value: &str) -> Option<PathBuf> {
    if value.eq_ignore_ascii_case("none") {
        None
    } else {
        Some(PathBuf::from(value))
    }
}

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err("expected a boolean value: true/false".into()),
    }
}

#[cfg(test)]
mod tests {
    use super::{CliPttKey, CliRecorderBackend, StopAction, VlConfig, set_config_value};
    use crossterm::event::KeyCode;
    use voicelayer_core::RecorderBackend;

    #[test]
    fn cli_backend_maps_to_domain_backend() {
        assert_eq!(
            RecorderBackend::from(CliRecorderBackend::Auto),
            RecorderBackend::Auto
        );
    }

    #[test]
    fn ptt_key_maps_to_terminal_key_code() {
        assert_eq!(CliPttKey::Space.as_key_code(), KeyCode::Char(' '));
        assert_eq!(CliPttKey::F9.as_key_code(), KeyCode::F(9));
    }

    #[test]
    fn set_config_value_updates_default_stop_action() {
        let mut config = VlConfig::default();
        set_config_value(&mut config, "foreground_ptt.default_stop_action", "inject").unwrap();
        assert_eq!(
            config.foreground_ptt.default_stop_action,
            StopAction::Inject
        );
    }

    #[test]
    fn set_config_value_rejects_unknown_key() {
        let mut config = VlConfig::default();
        let error = set_config_value(&mut config, "foreground_ptt.unknown", "x").unwrap_err();
        assert!(error.to_string().contains("unsupported config key"));
    }

    #[test]
    fn set_config_value_accepts_language_and_clears_with_none() {
        let mut config = VlConfig::default();
        set_config_value(&mut config, "foreground_ptt.language", "en").unwrap();
        assert_eq!(config.foreground_ptt.language.as_deref(), Some("en"));
        set_config_value(&mut config, "foreground_ptt.language", "none").unwrap();
        assert!(config.foreground_ptt.language.is_none());
    }

    #[test]
    fn set_config_value_accepts_translate_and_keep_audio_booleans() {
        let mut config = VlConfig::default();
        set_config_value(&mut config, "foreground_ptt.translate_to_english", "true").unwrap();
        set_config_value(&mut config, "foreground_ptt.keep_audio", "yes").unwrap();
        assert!(config.foreground_ptt.translate_to_english);
        assert!(config.foreground_ptt.keep_audio);
    }

    #[test]
    fn set_config_value_accepts_target_selectors() {
        let mut config = VlConfig::default();
        set_config_value(&mut config, "foreground_ptt.tmux_target_pane", "%3").unwrap();
        set_config_value(&mut config, "foreground_ptt.wezterm_target_pane_id", "12").unwrap();
        set_config_value(&mut config, "foreground_ptt.kitty_match", "title:editor").unwrap();
        assert_eq!(
            config.foreground_ptt.tmux_target_pane.as_deref(),
            Some("%3")
        );
        assert_eq!(
            config.foreground_ptt.wezterm_target_pane_id.as_deref(),
            Some("12")
        );
        assert_eq!(
            config.foreground_ptt.kitty_match.as_deref(),
            Some("title:editor")
        );
        set_config_value(&mut config, "foreground_ptt.tmux_target_pane", "none").unwrap();
        set_config_value(&mut config, "foreground_ptt.wezterm_target_pane_id", "None").unwrap();
        set_config_value(&mut config, "foreground_ptt.kitty_match", "NONE").unwrap();
        assert!(config.foreground_ptt.tmux_target_pane.is_none());
        assert!(config.foreground_ptt.wezterm_target_pane_id.is_none());
        assert!(config.foreground_ptt.kitty_match.is_none());
    }
}
