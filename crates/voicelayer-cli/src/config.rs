//! CLI configuration helpers backed by [`voicelayer_core::VoiceLayerConfig`].

use std::path::PathBuf;

use clap::ValueEnum;
use crossterm::event::KeyCode;
use voicelayer_core::{SegmentationMode, StopAction, VoiceLayerConfig, config_path};

/// Supported keys for `vl config set`.
pub(crate) const SUPPORTED_CONFIG_KEYS: &[&str] = &[
    "foreground_ptt.language",
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
    "foreground_ptt.provider_id",
];

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
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
            Self::Space => KeyCode::Char(' '),
            Self::Enter => KeyCode::Enter,
            Self::Tab => KeyCode::Tab,
            Self::F8 => KeyCode::F(8),
            Self::F9 => KeyCode::F(9),
            Self::F10 => KeyCode::F(10),
        }
    }

    pub(crate) fn label(self) -> &'static str {
        match self {
            Self::Space => "Space",
            Self::Enter => "Enter",
            Self::Tab => "Tab",
            Self::F8 => "F8",
            Self::F9 => "F9",
            Self::F10 => "F10",
        }
    }

    pub(crate) fn config_value(self) -> &'static str {
        match self {
            Self::Space => "space",
            Self::Enter => "enter",
            Self::Tab => "tab",
            Self::F8 => "f8",
            Self::F9 => "f9",
            Self::F10 => "f10",
        }
    }

    pub(crate) fn parse_str(value: &str) -> Option<Self> {
        match value.trim().to_ascii_lowercase().as_str() {
            "space" => Some(Self::Space),
            "enter" => Some(Self::Enter),
            "tab" => Some(Self::Tab),
            "f8" => Some(Self::F8),
            "f9" => Some(Self::F9),
            "f10" => Some(Self::F10),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliStopAction {
    None,
    Copy,
    Inject,
    Save,
}

impl From<CliStopAction> for StopAction {
    fn from(value: CliStopAction) -> Self {
        match value {
            CliStopAction::None => Self::None,
            CliStopAction::Copy => Self::Copy,
            CliStopAction::Inject => Self::Inject,
            CliStopAction::Save => Self::Save,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
pub(crate) enum CliSegmentationMode {
    OneShot,
    Fixed,
    VadGated,
}

/// Convert flattened CLI segmentation flags into [`SegmentationMode`].
pub(crate) fn build_segmentation_mode(
    mode: CliSegmentationMode,
    segment_secs: Option<u32>,
    probe_secs: Option<u32>,
    max_segment_secs: Option<u32>,
    silence_gap_probes: u32,
) -> SegmentationMode {
    match mode {
        CliSegmentationMode::OneShot => SegmentationMode::OneShot,
        CliSegmentationMode::Fixed => SegmentationMode::Fixed {
            segment_secs: segment_secs
                .expect("clap required_if_eq(mode, fixed) must populate segment_secs"),
        },
        CliSegmentationMode::VadGated => SegmentationMode::VadGated {
            probe_secs: probe_secs
                .expect("clap required_if_eq(mode, vad-gated) must populate probe_secs"),
            max_segment_secs: max_segment_secs
                .expect("clap required_if_eq(mode, vad-gated) must populate max_segment_secs"),
            silence_gap_probes,
        },
    }
}

pub(crate) fn vl_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(config_path()?)
}

pub(crate) fn load_vl_config() -> Result<VoiceLayerConfig, Box<dyn std::error::Error>> {
    let path = config_path()?;
    Ok(VoiceLayerConfig::load_file_only(&path)?)
}

pub(crate) fn write_vl_config(
    config: &VoiceLayerConfig,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = config_path()?;
    config.save_to(&path)?;
    Ok(path)
}

pub(crate) fn set_config_value(
    config: &mut VoiceLayerConfig,
    key: &str,
    value: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    let ptt = &mut config.foreground_ptt;
    match key {
        "foreground_ptt.default_stop_action" => {
            ptt.default_stop_action = match value {
                "none" => StopAction::None,
                "copy" => StopAction::Copy,
                "inject" => StopAction::Inject,
                "save" => StopAction::Save,
                _ => return Err("expected one of: none, copy, inject, save".into()),
            };
        }
        "foreground_ptt.copy_on_stop" => ptt.copy_on_stop = parse_bool(value)?,
        "foreground_ptt.restore_clipboard_on_exit" => {
            ptt.restore_clipboard_on_exit = parse_bool(value)?;
        }
        "foreground_ptt.save_dir" => ptt.save_dir = parse_optional_string(value),
        "foreground_ptt.key" => {
            let parsed = CliPttKey::parse_str(value)
                .ok_or("expected one of: space, enter, tab, f8, f9, f10")?;
            ptt.key = parsed.config_value().to_owned();
        }
        "foreground_ptt.language" => ptt.language = parse_optional_string(value),
        "foreground_ptt.translate_to_english" => ptt.translate_to_english = parse_bool(value)?,
        "foreground_ptt.keep_audio" => ptt.keep_audio = parse_bool(value)?,
        "foreground_ptt.tmux_target_pane" => ptt.tmux_target_pane = parse_optional_string(value),
        "foreground_ptt.wezterm_target_pane_id" => {
            ptt.wezterm_target_pane_id = parse_optional_string(value);
        }
        "foreground_ptt.kitty_match" => ptt.kitty_match = parse_optional_string(value),
        "foreground_ptt.provider_id" => ptt.provider_id = parse_optional_string(value),
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

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err("expected a boolean value: true/false".into()),
    }
}

#[cfg(test)]
mod tests {
    use crossterm::event::KeyCode;
    use voicelayer_core::{SegmentationMode, StopAction, VoiceLayerConfig};

    use super::{CliPttKey, CliSegmentationMode, build_segmentation_mode, set_config_value};

    #[test]
    fn ptt_key_maps_to_terminal_key_code() {
        assert_eq!(CliPttKey::Space.as_key_code(), KeyCode::Char(' '));
        assert_eq!(CliPttKey::F9.as_key_code(), KeyCode::F(9));
    }

    #[test]
    fn build_segmentation_mode_one_shot_ignores_numeric_knobs() {
        let mode =
            build_segmentation_mode(CliSegmentationMode::OneShot, Some(8), Some(2), Some(30), 1);
        assert_eq!(mode, SegmentationMode::OneShot);
    }

    #[test]
    fn build_segmentation_mode_fixed_populates_from_required_knob() {
        let mode = build_segmentation_mode(CliSegmentationMode::Fixed, Some(8), None, None, 1);
        assert_eq!(mode, SegmentationMode::Fixed { segment_secs: 8 });
    }

    #[test]
    fn build_segmentation_mode_vad_gated_populates_from_required_knobs() {
        let mode =
            build_segmentation_mode(CliSegmentationMode::VadGated, None, Some(2), Some(30), 2);
        assert_eq!(
            mode,
            SegmentationMode::VadGated {
                probe_secs: 2,
                max_segment_secs: 30,
                silence_gap_probes: 2,
            },
        );
    }

    #[test]
    #[should_panic(expected = "segment_secs")]
    fn build_segmentation_mode_fixed_panics_when_segment_secs_missing() {
        let _ = build_segmentation_mode(CliSegmentationMode::Fixed, None, None, None, 1);
    }

    #[test]
    #[should_panic(expected = "probe_secs")]
    fn build_segmentation_mode_vad_gated_panics_when_probe_secs_missing() {
        let _ = build_segmentation_mode(CliSegmentationMode::VadGated, None, None, Some(30), 1);
    }

    #[test]
    #[should_panic(expected = "max_segment_secs")]
    fn build_segmentation_mode_vad_gated_panics_when_max_segment_secs_missing() {
        let _ = build_segmentation_mode(CliSegmentationMode::VadGated, None, Some(2), None, 1);
    }

    #[test]
    fn set_config_value_updates_default_stop_action() {
        let mut config = VoiceLayerConfig::default();
        set_config_value(&mut config, "foreground_ptt.default_stop_action", "inject").unwrap();
        assert_eq!(
            config.foreground_ptt.default_stop_action,
            StopAction::Inject
        );
    }

    #[test]
    fn set_config_value_rejects_unknown_key() {
        let mut config = VoiceLayerConfig::default();
        let error = set_config_value(&mut config, "foreground_ptt.unknown", "x").unwrap_err();
        assert!(error.to_string().contains("unsupported config key"));
    }

    #[test]
    fn set_config_value_accepts_language_provider_and_clears_with_none() {
        let mut config = VoiceLayerConfig::default();
        set_config_value(&mut config, "foreground_ptt.language", "en").unwrap();
        set_config_value(&mut config, "foreground_ptt.provider_id", "qwen3_asr_1_7b").unwrap();
        assert_eq!(config.foreground_ptt.language.as_deref(), Some("en"));
        assert_eq!(
            config.foreground_ptt.provider_id.as_deref(),
            Some("qwen3_asr_1_7b")
        );
        set_config_value(&mut config, "foreground_ptt.language", "none").unwrap();
        set_config_value(&mut config, "foreground_ptt.provider_id", "none").unwrap();
        assert!(config.foreground_ptt.language.is_none());
        assert!(config.foreground_ptt.provider_id.is_none());
    }

    #[test]
    fn set_config_value_accepts_booleans_and_target_selectors() {
        let mut config = VoiceLayerConfig::default();
        set_config_value(&mut config, "foreground_ptt.translate_to_english", "true").unwrap();
        set_config_value(&mut config, "foreground_ptt.keep_audio", "yes").unwrap();
        set_config_value(&mut config, "foreground_ptt.tmux_target_pane", "%3").unwrap();
        set_config_value(&mut config, "foreground_ptt.wezterm_target_pane_id", "12").unwrap();
        set_config_value(&mut config, "foreground_ptt.kitty_match", "title:editor").unwrap();
        assert!(config.foreground_ptt.translate_to_english);
        assert!(config.foreground_ptt.keep_audio);
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
