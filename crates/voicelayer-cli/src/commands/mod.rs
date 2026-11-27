//! CLI argument definitions and top-level command dispatch.

mod handlers;

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use voicelayer_core::{CompositionArchetype, RewriteStyle};

use crate::config::{CliPttKey, CliSegmentationMode, CliStopAction};

#[derive(Debug, Parser)]
#[command(name = "vl")]
#[command(about = "VoiceLayer CLI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
pub(crate) struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
pub(crate) enum Command {
    /// Inspect and edit the local VoiceLayer configuration file.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Manage dictation sessions (start, stop, foreground-ptt, list).
    Dictation {
        #[command(subcommand)]
        command: DictationCommand,
    },
    /// Inspect desktop global hotkey integration.
    Hotkeys {
        #[command(subcommand)]
        command: HotkeysCommand,
    },
    /// Print runtime diagnostics for the daemon, socket, and providers.
    Doctor,
    /// List host adapters and worker provider descriptors as JSON.
    Providers,
    /// Emit bracketed-paste-wrapped text for terminal piping.
    PrintBracketedPaste {
        text: String,
        #[arg(long, default_value_t = false)]
        auto_submit: bool,
    },
    /// Record from the microphone for a fixed duration and transcribe.
    RecordTranscribe {
        #[arg(long, default_value_t = 8)]
        duration_seconds: u32,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        /// ASR provider id (defaults to whisper_cpp).
        #[arg(long)]
        provider_id: Option<String>,
    },
    /// Transcribe an existing audio file through the daemon.
    TranscribeFile {
        audio_file: String,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        /// ASR provider id (defaults to whisper_cpp).
        #[arg(long)]
        provider_id: Option<String>,
    },
    /// Generate text previews via compose, rewrite, or translate.
    Preview {
        #[command(subcommand)]
        command: PreviewCommand,
    },
}

#[derive(Debug, Subcommand)]
pub(crate) enum DictationCommand {
    ForegroundPtt {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        #[arg(long, value_enum, default_value_t = CliPttKey::Space)]
        key: CliPttKey,
        #[arg(long)]
        tmux_target_pane: Option<String>,
        #[arg(long)]
        wezterm_target_pane_id: Option<String>,
        #[arg(long)]
        kitty_match: Option<String>,
        #[arg(long, default_value_t = false)]
        copy_on_stop: bool,
        #[arg(long, value_enum, default_value_t = CliStopAction::None)]
        default_stop_action: CliStopAction,
        #[arg(long, default_value_t = false)]
        restore_clipboard_on_exit: bool,
        #[arg(long)]
        save_dir: Option<PathBuf>,
        /// ASR provider id for transcriptions in this session (`whisper_cpp`, `mimo_v2_5_asr`, `qwen3_asr_1_7b`).
        #[arg(long)]
        provider_id: Option<String>,
    },
    Start {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        /// Segmentation strategy: `one-shot`, `fixed`, or `vad-gated`.
        #[arg(long, value_enum, default_value_t = CliSegmentationMode::OneShot)]
        mode: CliSegmentationMode,
        /// Segment length in whole seconds (required for `--mode fixed`).
        #[arg(
            long,
            required_if_eq("mode", "fixed"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        segment_secs: Option<u32>,
        /// VAD probe interval in whole seconds (required for `--mode vad-gated`).
        #[arg(
            long,
            required_if_eq("mode", "vad-gated"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        probe_secs: Option<u32>,
        /// Maximum buffered speech unit length in whole seconds (required for `--mode vad-gated`).
        #[arg(
            long,
            required_if_eq("mode", "vad-gated"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        max_segment_secs: Option<u32>,
        /// Consecutive silent probes required to flush a buffered speech unit (default 1).
        #[arg(long, default_value_t = 1)]
        silence_gap_probes: u32,
        /// Sleep N seconds after starting, then stop and print the final capture result.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        duration_seconds: Option<u32>,
        /// ASR provider id (`whisper_cpp`, `mimo_v2_5_asr`, `qwen3_asr_1_7b`).
        #[arg(long)]
        provider_id: Option<String>,
    },
    Stop {
        session_id: uuid::Uuid,
    },
    /// List sessions tracked by the daemon as JSON (`GET /v1/sessions`).
    List,
}

#[derive(Debug, Subcommand)]
pub(crate) enum HotkeysCommand {
    /// Report global hotkey backend availability.
    PortalStatus,
}

#[derive(Debug, Subcommand)]
pub(crate) enum ConfigCommand {
    /// Print the resolved path to `config.toml`.
    Path,
    /// Print the effective configuration as TOML.
    Show,
    /// Write the default configuration to disk.
    InitDefaults,
    /// Update a dotted key in `config.toml`.
    Set { key: String, value: String },
}

#[derive(Debug, Subcommand)]
pub(crate) enum PreviewCommand {
    /// Compose structured text from a spoken prompt.
    Compose {
        spoken_prompt: String,
        #[arg(long)]
        archetype: Option<CliArchetype>,
        #[arg(long)]
        output_language: Option<String>,
    },
    /// Rewrite text in a chosen style.
    Rewrite {
        source_text: String,
        #[arg(long)]
        style: CliRewriteStyle,
        #[arg(long)]
        output_language: Option<String>,
    },
    /// Translate text into a target language.
    Translate {
        source_text: String,
        #[arg(long)]
        target_language: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum CliArchetype {
    Email,
    CoverLetter,
    DailyReport,
    Issue,
    PullRequestDescription,
    Prompt,
    TechnicalSummary,
    Custom,
}

impl From<CliArchetype> for CompositionArchetype {
    fn from(value: CliArchetype) -> Self {
        match value {
            CliArchetype::Email => Self::Email,
            CliArchetype::CoverLetter => Self::CoverLetter,
            CliArchetype::DailyReport => Self::DailyReport,
            CliArchetype::Issue => Self::Issue,
            CliArchetype::PullRequestDescription => Self::PullRequestDescription,
            CliArchetype::Prompt => Self::Prompt,
            CliArchetype::TechnicalSummary => Self::TechnicalSummary,
            CliArchetype::Custom => Self::Custom,
        }
    }
}

#[derive(Debug, Clone, ValueEnum)]
pub(crate) enum CliRewriteStyle {
    MoreFormal,
    Shorter,
    Politer,
    MoreTechnical,
    Translate,
}

impl From<CliRewriteStyle> for RewriteStyle {
    fn from(value: CliRewriteStyle) -> Self {
        match value {
            CliRewriteStyle::MoreFormal => Self::MoreFormal,
            CliRewriteStyle::Shorter => Self::Shorter,
            CliRewriteStyle::Politer => Self::Politer,
            CliRewriteStyle::MoreTechnical => Self::MoreTechnical,
            CliRewriteStyle::Translate => Self::Translate,
        }
    }
}

pub(crate) async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    handlers::dispatch(args.command).await
}

#[cfg(test)]
mod tests {
    use clap::Parser;

    use super::{Args, Command, DictationCommand};
    use crate::config::CliSegmentationMode;

    fn try_parse(args: &[&str]) -> Result<Args, clap::Error> {
        Args::try_parse_from(args)
    }

    #[test]
    fn dictation_start_defaults_to_one_shot_mode() {
        let parsed = try_parse(&["vl", "dictation", "start"]).expect("parse default start");
        match parsed.command {
            Command::Dictation {
                command:
                    DictationCommand::Start {
                        mode,
                        segment_secs,
                        probe_secs,
                        max_segment_secs,
                        silence_gap_probes,
                        duration_seconds,
                        ..
                    },
            } => {
                assert_eq!(mode, CliSegmentationMode::OneShot);
                assert!(segment_secs.is_none());
                assert!(probe_secs.is_none());
                assert!(max_segment_secs.is_none());
                assert_eq!(silence_gap_probes, 1);
                assert!(duration_seconds.is_none());
            }
            other => panic!("unexpected command: {other:?}"),
        }
    }

    #[test]
    fn dictation_start_fixed_requires_positive_segment_secs() {
        assert!(try_parse(&["vl", "dictation", "start", "--mode", "fixed"]).is_err());
        assert!(
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "fixed",
                "--segment-secs",
                "0"
            ])
            .is_err()
        );
        assert!(
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "fixed",
                "--segment-secs",
                "8"
            ])
            .is_ok()
        );
    }

    #[test]
    fn dictation_start_vad_gated_requires_probe_and_max_segment_secs() {
        assert!(try_parse(&["vl", "dictation", "start", "--mode", "vad-gated"]).is_err());
        assert!(
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "vad-gated",
                "--probe-secs",
                "2",
                "--max-segment-secs",
                "30",
                "--silence-gap-probes",
                "2",
            ])
            .is_ok()
        );
    }

    #[test]
    fn dictation_start_rejects_zero_duration_seconds() {
        assert!(try_parse(&["vl", "dictation", "start", "--duration-seconds", "0"]).is_err());
        assert!(try_parse(&["vl", "dictation", "start", "--duration-seconds", "5"]).is_ok());
    }
}
