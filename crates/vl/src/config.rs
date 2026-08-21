//! CLI-side access to the unified VoiceLayer config file.
//!
//! The schema lives in `voicelayer_core::config`; this module adds the
//! `vl config set` dotted-key writer and the push-to-talk key enum used by
//! clap and crossterm.

use std::path::PathBuf;

use clap::ValueEnum;
use crossterm::event::KeyCode;
use voicelayer_core::{VoiceLayerConfig, config_path};

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
    pub(crate) fn parse(name: &str) -> Result<Self, String> {
        match name.to_ascii_lowercase().as_str() {
            "space" => Ok(Self::Space),
            "enter" => Ok(Self::Enter),
            "tab" => Ok(Self::Tab),
            "f8" => Ok(Self::F8),
            "f9" => Ok(Self::F9),
            "f10" => Ok(Self::F10),
            _ => Err(format!(
                "unknown key `{name}`; expected one of: space, enter, tab, f8, f9, f10"
            )),
        }
    }

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

pub(crate) fn vl_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    Ok(config_path()?)
}

/// Effective config (file + env overrides) for CLI behavior.
pub(crate) fn load_vl_config() -> Result<VoiceLayerConfig, Box<dyn std::error::Error>> {
    Ok(VoiceLayerConfig::load()?)
}

pub(crate) fn write_vl_config(
    config: &VoiceLayerConfig,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, toml::to_string_pretty(config)?)?;
    Ok(path)
}

/// Set a dotted config key (`llm.endpoint`, `foreground_ptt.key`, ...) to a
/// scalar value; `"none"` removes the key. The file is validated against
/// the schema before being written, so unknown keys are rejected.
pub(crate) fn set_config_value(
    key: &str,
    value: &str,
) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    let mut document: toml::Value = if path.is_file() {
        toml::from_str(&std::fs::read_to_string(&path)?)?
    } else {
        toml::Value::Table(toml::map::Map::new())
    };

    let segments: Vec<&str> = key.split('.').collect();
    if segments.len() < 2 || segments.iter().any(|segment| segment.is_empty()) {
        return Err(
            format!("config keys must be dotted paths like `llm.endpoint`, got `{key}`").into(),
        );
    }

    if value.eq_ignore_ascii_case("none") {
        remove_nested(&mut document, &segments);
    } else {
        set_nested(&mut document, &segments, parse_scalar(value))?;
    }

    // Validate against the schema (rejects unknown keys), then write the
    // normalized form.
    let config: VoiceLayerConfig = document
        .try_into()
        .map_err(|error| format!("invalid config key or value for `{key}`: {error}"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, toml::to_string_pretty(&config)?)?;
    Ok(path)
}

fn parse_scalar(value: &str) -> toml::Value {
    match value.to_ascii_lowercase().as_str() {
        "true" => return toml::Value::Boolean(true),
        "false" => return toml::Value::Boolean(false),
        _ => {}
    }
    if let Ok(integer) = value.parse::<i64>() {
        return toml::Value::Integer(integer);
    }
    if let Ok(float) = value.parse::<f64>() {
        return toml::Value::Float(float);
    }
    toml::Value::String(value.to_owned())
}

fn set_nested(
    document: &mut toml::Value,
    segments: &[&str],
    value: toml::Value,
) -> Result<(), Box<dyn std::error::Error>> {
    let mut current = document;
    for segment in &segments[..segments.len() - 1] {
        if !current.is_table() {
            return Err(format!("config path component `{segment}` is not a table").into());
        }
        current = current
            .as_table_mut()
            .expect("checked is_table")
            .entry(segment.to_string())
            .or_insert_with(|| toml::Value::Table(toml::map::Map::new()));
    }
    let table = current
        .as_table_mut()
        .ok_or("config path does not resolve to a table")?;
    table.insert(segments[segments.len() - 1].to_string(), value);
    Ok(())
}

fn remove_nested(document: &mut toml::Value, segments: &[&str]) {
    let mut current = document;
    for segment in &segments[..segments.len() - 1] {
        match current.get_mut(segment) {
            Some(next) if next.is_table() => current = next,
            _ => return,
        }
    }
    if let Some(table) = current.as_table_mut() {
        table.remove(segments[segments.len() - 1]);
    }
}

#[cfg(test)]
mod tests {
    use super::parse_scalar;

    #[test]
    fn scalar_parsing_prefers_bool_then_int_then_float_then_string() {
        assert_eq!(parse_scalar("true"), toml::Value::Boolean(true));
        assert_eq!(parse_scalar("off"), toml::Value::String("off".to_owned()));
        assert_eq!(parse_scalar("42"), toml::Value::Integer(42));
        assert_eq!(parse_scalar("0.5"), toml::Value::Float(0.5));
        assert_eq!(
            parse_scalar("http://127.0.0.1:8080"),
            toml::Value::String("http://127.0.0.1:8080".to_owned())
        );
    }
}
