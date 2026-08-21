//! VoiceLayer operator CLI. A pure client of the daemon control socket; it
//! never links daemon internals.

use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use voicelayer_client::Client;
use voicelayer_core::{
    ComposeRequest, CompositionArchetype, DictationCaptureRequest, HealthResponse, InjectRequest,
    InjectTarget, InjectionPlan, LanguageProfile, LanguageStrategy, RewriteRequest, RewriteStyle,
    SegmentationMode, StartDictationRequest, StopDictationRequest, TranscribeRequest,
    TranslateRequest, TriggerKind, VoiceLayerConfig,
};

use crate::config::{CliPttKey, load_vl_config, set_config_value, vl_config_path, write_vl_config};
use crate::foreground_ptt::run_foreground_ptt;

#[derive(Debug, Parser)]
#[command(name = "vl")]
#[command(about = "VoiceLayer operator CLI")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    Dictation {
        #[command(subcommand)]
        command: DictationCommand,
    },
    Hotkeys {
        #[command(subcommand)]
        command: HotkeysCommand,
    },
    Doctor,
    Providers,
    PrintBracketedPaste {
        text: String,
        #[arg(long, default_value_t = false)]
        auto_submit: bool,
    },
    RecordTranscribe {
        #[arg(long, default_value_t = 8)]
        duration_seconds: u32,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
    },
    TranscribeFile {
        audio_file: String,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
    },
    Preview {
        #[command(subcommand)]
        command: PreviewCommand,
    },
}

#[derive(Debug, Subcommand)]
enum DictationCommand {
    ForegroundPtt {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        #[arg(long)]
        key: Option<String>,
        #[arg(long)]
        tmux_target_pane: Option<String>,
        #[arg(long)]
        wezterm_target_pane_id: Option<String>,
        #[arg(long)]
        kitty_match: Option<String>,
        #[arg(long, default_value_t = false)]
        copy_on_stop: bool,
        #[arg(long)]
        default_stop_action: Option<String>,
        #[arg(long, default_value_t = false)]
        restore_clipboard_on_exit: bool,
        #[arg(long)]
        save_dir: Option<PathBuf>,
    },
    Start {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        #[arg(long)]
        segment_secs: Option<u32>,
    },
    Stop {
        session_id: uuid::Uuid,
    },
}

#[derive(Debug, Subcommand)]
enum HotkeysCommand {
    Status,
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    Path,
    Show,
    InitDefaults,
    Set { key: String, value: String },
}

#[derive(Debug, Subcommand)]
enum PreviewCommand {
    Compose {
        spoken_prompt: String,
        #[arg(long)]
        archetype: Option<CliArchetype>,
        #[arg(long)]
        output_language: Option<String>,
    },
    Rewrite {
        source_text: String,
        #[arg(long)]
        style: CliRewriteStyle,
        #[arg(long)]
        output_language: Option<String>,
    },
    Translate {
        source_text: String,
        #[arg(long)]
        target_language: String,
    },
}

#[derive(Debug, Clone, ValueEnum)]
enum CliArchetype {
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
enum CliRewriteStyle {
    MoreFormal,
    Shorter,
    Politer,
    MoreTechnical,
}

impl From<CliRewriteStyle> for RewriteStyle {
    fn from(value: CliRewriteStyle) -> Self {
        match value {
            CliRewriteStyle::MoreFormal => Self::MoreFormal,
            CliRewriteStyle::Shorter => Self::Shorter,
            CliRewriteStyle::Politer => Self::Politer,
            CliRewriteStyle::MoreTechnical => Self::MoreTechnical,
        }
    }
}

fn print_json<T: serde::Serialize>(value: &T) -> Result<(), Box<dyn std::error::Error>> {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}

fn language_profile(language: Option<String>) -> Option<LanguageProfile> {
    language.map(|language| LanguageProfile {
        strategy: LanguageStrategy::Locked,
        input_languages: vec![language],
        output_language: None,
    })
}

pub(crate) async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();
    let client = Client::from_env();

    match args.command {
        Command::Config { command } => match command {
            ConfigCommand::Path => {
                println!("{}", vl_config_path()?.display());
            }
            ConfigCommand::Show => {
                let config = load_vl_config()?;
                println!("{}", toml::to_string_pretty(&config)?);
            }
            ConfigCommand::InitDefaults => {
                let path = write_vl_config(&VoiceLayerConfig::default())?;
                print_json(&serde_json::json!({
                    "status": "ok",
                    "config_path": path,
                }))?;
            }
            ConfigCommand::Set { key, value } => {
                let path = set_config_value(&key, &value)?;
                print_json(&serde_json::json!({
                    "status": "ok",
                    "config_path": path,
                    "updated_key": key,
                }))?;
            }
        },
        Command::Dictation { command } => match command {
            DictationCommand::ForegroundPtt {
                language,
                translate_to_english,
                keep_audio,
                key,
                tmux_target_pane,
                wezterm_target_pane_id,
                kitty_match,
                copy_on_stop,
                default_stop_action,
                restore_clipboard_on_exit,
                save_dir,
            } => {
                let defaults = load_vl_config().unwrap_or_default().foreground_ptt;
                let key_name = key.unwrap_or(defaults.key);
                let key = CliPttKey::parse(&key_name)
                    .map_err(|error| -> Box<dyn std::error::Error> { error.into() })?;
                let stop_action = match default_stop_action {
                    Some(action) => parse_stop_action(&action)?,
                    None => defaults.default_stop_action,
                };
                run_foreground_ptt(
                    client,
                    language.or(defaults.language),
                    translate_to_english || defaults.translate_to_english,
                    keep_audio || defaults.keep_audio,
                    key,
                    tmux_target_pane.or(defaults.tmux_target_pane),
                    wezterm_target_pane_id.or(defaults.wezterm_target_pane_id),
                    kitty_match.or(defaults.kitty_match),
                    copy_on_stop || defaults.copy_on_stop,
                    stop_action,
                    restore_clipboard_on_exit || defaults.restore_clipboard_on_exit,
                    save_dir.or_else(|| defaults.save_dir.map(PathBuf::from)),
                )
                .await?;
            }
            DictationCommand::Start {
                language,
                translate_to_english,
                keep_audio,
                segment_secs,
            } => {
                let segmentation = match segment_secs {
                    Some(segment_secs) => SegmentationMode::Fixed { segment_secs },
                    None => SegmentationMode::OneShot,
                };
                let request = StartDictationRequest {
                    trigger: TriggerKind::Cli,
                    language_profile: language_profile(language),
                    translate_to_english,
                    keep_audio,
                    segmentation,
                };
                let session: voicelayer_core::CaptureSession =
                    client.post("/v1/sessions/dictation", &request).await?;
                print_json(&session)?;
            }
            DictationCommand::Stop { session_id } => {
                let result: voicelayer_core::DictationCaptureResult = client
                    .post(
                        "/v1/sessions/dictation/stop",
                        &StopDictationRequest { session_id },
                    )
                    .await?;
                print_json(&result)?;
            }
        },
        Command::Hotkeys { command } => match command {
            HotkeysCommand::Status => {
                let health: HealthResponse = client.get("/v1/health").await?;
                print_json(&serde_json::json!({
                    "available": health.worker.global_hotkeys_available,
                    "backend": health.worker.global_hotkeys_backend,
                    "detail": health.worker.global_hotkeys_detail,
                }))?;
            }
        },
        Command::Doctor => {
            let socket_path = client.socket_path().display().to_string();
            let config_file = vl_config_path()
                .map(|path| path.display().to_string())
                .unwrap_or_else(|_| "unavailable".to_owned());
            match client.get::<HealthResponse>("/v1/health").await {
                Ok(health) => {
                    print_json(&serde_json::json!({
                        "daemon_reachable": true,
                        "socket_path": socket_path,
                        "config_path": config_file,
                        "os": std::env::consts::OS,
                        "arch": std::env::consts::ARCH,
                        "health": health,
                    }))?;
                }
                Err(error) => {
                    print_json(&serde_json::json!({
                        "daemon_reachable": false,
                        "socket_path": socket_path,
                        "config_path": config_file,
                        "os": std::env::consts::OS,
                        "arch": std::env::consts::ARCH,
                        "error": error.to_string(),
                        "hint": daemon_start_hint(),
                    }))?;
                }
            }
        }
        Command::Providers => {
            let providers: serde_json::Value = client.get("/v1/providers").await?;
            print_json(&providers)?;
        }
        Command::PrintBracketedPaste { text, auto_submit } => {
            let plan = InjectionPlan::from_request(&InjectRequest {
                target: InjectTarget::TerminalBracketedPaste,
                text,
                auto_submit,
            });
            print!("{}", plan.payload);
        }
        Command::RecordTranscribe {
            duration_seconds,
            language,
            translate_to_english,
            keep_audio,
        } => {
            let request = DictationCaptureRequest {
                trigger: TriggerKind::Cli,
                language_profile: language_profile(language),
                duration_seconds,
                translate_to_english,
                keep_audio,
            };
            let result: voicelayer_core::DictationCaptureResult =
                client.post("/v1/dictation/capture", &request).await?;
            print_json(&result)?;
        }
        Command::TranscribeFile {
            audio_file,
            language,
            translate_to_english,
        } => {
            let result: voicelayer_core::TranscriptionResult = client
                .post(
                    "/v1/transcriptions",
                    &TranscribeRequest {
                        audio_file,
                        language,
                        translate_to_english,
                    },
                )
                .await?;
            print_json(&result)?;
        }
        Command::Preview { command } => match command {
            PreviewCommand::Compose {
                spoken_prompt,
                archetype,
                output_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = client
                    .post(
                        "/v1/sessions/compose",
                        &ComposeRequest {
                            spoken_prompt,
                            archetype: archetype.map(Into::into),
                            output_language,
                        },
                    )
                    .await?;
                print_json(&receipt)?;
            }
            PreviewCommand::Rewrite {
                source_text,
                style,
                output_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = client
                    .post(
                        "/v1/rewrites",
                        &RewriteRequest {
                            source_text,
                            style: style.into(),
                            output_language,
                        },
                    )
                    .await?;
                print_json(&receipt)?;
            }
            PreviewCommand::Translate {
                source_text,
                target_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = client
                    .post(
                        "/v1/translations",
                        &TranslateRequest {
                            source_text,
                            target_language,
                        },
                    )
                    .await?;
                print_json(&receipt)?;
            }
        },
    }

    Ok(())
}

fn parse_stop_action(
    value: &str,
) -> Result<voicelayer_core::StopAction, Box<dyn std::error::Error>> {
    use voicelayer_core::StopAction;
    match value.to_ascii_lowercase().as_str() {
        "none" => Ok(StopAction::None),
        "copy" => Ok(StopAction::Copy),
        "inject" => Ok(StopAction::Inject),
        "save" => Ok(StopAction::Save),
        _ => Err(format!(
            "unknown stop action `{value}`; expected one of: none, copy, inject, save"
        )
        .into()),
    }
}

fn daemon_start_hint() -> &'static str {
    if cfg!(target_os = "macos") {
        "start the daemon with `launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memenow.voicelayerd.plist` or run `voicelayerd` directly"
    } else {
        "start the daemon with `systemctl --user start voicelayerd` or run `voicelayerd` directly"
    }
}
