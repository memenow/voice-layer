use std::path::PathBuf;

use clap::{Parser, Subcommand, ValueEnum};
use voicelayer_core::{
    ComposeRequest, CompositionArchetype, DictationCaptureRequest, HealthResponse, InjectRequest,
    InjectTarget, InjectionPlan, LanguageProfile, LanguageStrategy, RewriteRequest, RewriteStyle,
    StartDictationRequest, StopDictationRequest, TranscribeRequest, TranslateRequest, TriggerKind,
};

use crate::config::{
    CliPttKey, CliSegmentationMode, StopAction, VlConfig, build_segmentation_mode, load_vl_config,
    set_config_value, vl_config_path, write_vl_config,
};
use crate::foreground_ptt::run_foreground_ptt;
use crate::uds::{cli_socket_path, uds_get_json, uds_post_json};

#[derive(Debug, Parser)]
#[command(name = "vl")]
#[command(about = "VoiceLayer operator CLI")]
#[command(version = env!("CARGO_PKG_VERSION"))]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    /// Manage the long-running VoiceLayer daemon (start, supervise).
    /// Inspect and edit the operator-local VoiceLayer config file.
    Config {
        #[command(subcommand)]
        command: ConfigCommand,
    },
    /// Drive dictation sessions: start, stop, foreground-PTT, list.
    Dictation {
        #[command(subcommand)]
        command: DictationCommand,
    },
    /// Probe desktop hotkey integration (global shortcuts portal).
    Hotkeys {
        #[command(subcommand)]
        command: HotkeysCommand,
    },
    /// Print runtime diagnostics: socket, providers, recorder, env.
    Doctor,
    /// List host adapters and worker provider descriptors as JSON.
    Providers,
    /// Emit bracketed-paste-wrapped text for piping into a terminal.
    PrintBracketedPaste {
        text: String,
        #[arg(long, default_value_t = false)]
        auto_submit: bool,
    },
    /// Record from the microphone and transcribe in a single call.
    RecordTranscribe {
        #[arg(long, default_value_t = 8)]
        duration_seconds: u32,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
        /// ASR provider for the single transcription this capture
        /// emits. Defaults to the configured whisper.cpp chain when
        /// omitted.
        #[arg(long)]
        provider_id: Option<String>,
    },
    /// Transcribe an existing audio file via the daemon's ASR chain.
    TranscribeFile {
        audio_file: String,
        #[arg(long)]
        language: Option<String>,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        /// ASR provider for this transcription. Defaults to the
        /// configured whisper.cpp chain when omitted.
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
enum DictationCommand {
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
        #[arg(long, value_enum, default_value_t = StopAction::None)]
        default_stop_action: StopAction,
        #[arg(long, default_value_t = false)]
        restore_clipboard_on_exit: bool,
        #[arg(long)]
        save_dir: Option<PathBuf>,
        /// ASR provider for every transcription this PTT loop emits.
        /// Defaults to the configured whisper.cpp chain when omitted;
        /// `mimo_v2_5_asr` opts into Xiaomi MiMo-V2.5-ASR, and
        /// `qwen3_asr_1_7b` opts into Alibaba Qwen3-ASR-1.7B.
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
        /// Segmentation strategy. `one-shot` runs a single recorder for the
        /// whole session (default); `fixed` rolls the recorder every
        /// `--segment-secs` seconds; `vad-gated` rolls at `--probe-secs`
        /// cadence and flushes on silence (requires the worker's silero-vad
        /// to be configured).
        #[arg(long, value_enum, default_value_t = CliSegmentationMode::OneShot)]
        mode: CliSegmentationMode,
        /// Fixed-mode: duration of each segment in whole seconds. Required
        /// when `--mode fixed`; ignored otherwise. Must be >= 1; the
        /// daemon also rejects 0 but catching it at parse time gives a
        /// cleaner error.
        #[arg(
            long,
            required_if_eq("mode", "fixed"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        segment_secs: Option<u32>,
        /// Fixed-mode: reserved for future overlap-based stitching; the
        /// current implementation records each segment back-to-back.
        /// VAD-gated: duration of each classification probe in whole
        /// seconds. Required when `--mode vad-gated`. Must be >= 1.
        #[arg(
            long,
            required_if_eq("mode", "vad-gated"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        probe_secs: Option<u32>,
        /// VAD-gated: upper bound on a buffered speech unit before a
        /// forced flush, in whole seconds. Required when
        /// `--mode vad-gated`. Must be >= 1.
        #[arg(
            long,
            required_if_eq("mode", "vad-gated"),
            value_parser = clap::value_parser!(u32).range(1..)
        )]
        max_segment_secs: Option<u32>,
        /// VAD-gated: number of consecutive silent probes that must
        /// arrive after speech before the pending buffer flushes.
        /// Defaults to 1 (flush on the first silent probe after speech).
        #[arg(long, default_value_t = 1)]
        silence_gap_probes: u32,
        /// When set, the CLI sleeps `N` seconds after the daemon
        /// confirms the session is listening, then issues the matching
        /// stop request and prints the final `DictationCaptureResult`
        /// instead of just the listening session. Useful for benchmarks
        /// and smoke checks where the operator wants a one-command
        /// start-record-stop flow. Must be >= 1; omit for the default
        /// "print listening session, leave running" behavior.
        #[arg(long, value_parser = clap::value_parser!(u32).range(1..))]
        duration_seconds: Option<u32>,
        /// ASR provider for every transcription this session emits.
        /// Defaults to the configured whisper.cpp chain when omitted.
        /// Currently accepted values: `whisper_cpp`, `mimo_v2_5_asr`,
        /// `qwen3_asr_1_7b`. The daemon rejects unknown values rather
        /// than falling back.
        #[arg(long)]
        provider_id: Option<String>,
    },
    Stop {
        session_id: uuid::Uuid,
    },
    /// List every dictation session the daemon currently knows about.
    /// Useful when the operator has lost a `session_id` returned by an
    /// earlier `dictation start` call and needs to recover it before
    /// issuing the matching `dictation stop`. The output is the JSON
    /// array the daemon returns from `GET /v1/sessions`.
    List,
}

#[derive(Debug, Subcommand)]
enum HotkeysCommand {
    /// Report the XDG Global Shortcuts portal availability and version.
    PortalStatus,
}

#[derive(Debug, Subcommand)]
enum ConfigCommand {
    /// Print the resolved path of the VoiceLayer CLI config file.
    Path,
    /// Show the effective CLI config as JSON.
    Show,
    /// Write the default CLI config to disk if no file exists yet.
    InitDefaults,
    /// Set a single dotted-key value in the CLI config file.
    Set { key: String, value: String },
}

#[derive(Debug, Subcommand)]
enum PreviewCommand {
    /// Compose long-form text from a spoken prompt via the LLM worker.
    Compose {
        spoken_prompt: String,
        #[arg(long)]
        archetype: Option<CliArchetype>,
        #[arg(long)]
        output_language: Option<String>,
    },
    /// Rewrite existing text under a chosen style (formal, shorter, ...).
    Rewrite {
        source_text: String,
        #[arg(long)]
        style: CliRewriteStyle,
        #[arg(long)]
        output_language: Option<String>,
    },
    /// Translate source text into the given target language.
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

#[allow(clippy::too_many_lines)]
pub(crate) async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

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
                let config = VlConfig::default();
                let path = write_vl_config(&config)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "status": "ok",
                        "config_path": path,
                    }))?
                );
            }
            ConfigCommand::Set { key, value } => {
                let mut config = load_vl_config()?;
                set_config_value(&mut config, &key, &value)?;
                let path = write_vl_config(&config)?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "status": "ok",
                        "config_path": path,
                        "updated_key": key,
                        "updated_value": value,
                    }))?
                );
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
                provider_id,
            } => {
                let defaults = load_vl_config().unwrap_or_default().foreground_ptt;
                run_foreground_ptt(
                    language.or(defaults.language),
                    translate_to_english || defaults.translate_to_english,
                    keep_audio || defaults.keep_audio,
                    if key == CliPttKey::Space {
                        defaults.key
                    } else {
                        key
                    },
                    tmux_target_pane.or(defaults.tmux_target_pane),
                    wezterm_target_pane_id.or(defaults.wezterm_target_pane_id),
                    kitty_match.or(defaults.kitty_match),
                    copy_on_stop || defaults.copy_on_stop,
                    if default_stop_action == StopAction::None {
                        defaults.default_stop_action
                    } else {
                        default_stop_action
                    },
                    restore_clipboard_on_exit || defaults.restore_clipboard_on_exit,
                    save_dir.or(defaults.save_dir),
                    provider_id.or(defaults.provider_id),
                )
                .await?;
            }
            DictationCommand::Start {
                language,
                translate_to_english,
                keep_audio,
                mode,
                segment_secs,
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
                duration_seconds,
                provider_id,
            } => {
                let segmentation = build_segmentation_mode(
                    mode,
                    segment_secs,
                    probe_secs,
                    max_segment_secs,
                    silence_gap_probes,
                );
                let request = StartDictationRequest {
                    trigger: TriggerKind::Cli,
                    language_profile: language.map(|language| LanguageProfile {
                        strategy: LanguageStrategy::Locked,
                        input_languages: vec![language],
                        output_language: None,
                    }),
                    translate_to_english,
                    keep_audio,
                    segmentation,
                    provider_id,
                };
                let session: voicelayer_core::CaptureSession =
                    uds_post_json(&cli_socket_path(), "/v1/sessions/dictation", &request).await?;

                match duration_seconds {
                    None => {
                        // Original behavior: print the listening session and
                        // leave the orchestrator running. Operator must
                        // call `vl dictation stop <id>` themselves.
                        println!("{}", serde_json::to_string_pretty(&session)?);
                    }
                    Some(seconds) => {
                        // One-command start-record-stop: hold the session
                        // for `seconds`, then issue stop and print the
                        // final result instead of the listening session.
                        // The session id is intentionally not printed here
                        // — the operator has nothing to do with it. Any
                        // unrecoverable error during sleep or stop bubbles
                        // up; callers shouldn't see a half-finished output.
                        eprintln!(
                            "session {} listening; auto-stopping after {seconds}s",
                            session.session_id,
                        );
                        tokio::time::sleep(std::time::Duration::from_secs(u64::from(seconds)))
                            .await;
                        let result: voicelayer_core::DictationCaptureResult = uds_post_json(
                            &cli_socket_path(),
                            "/v1/sessions/dictation/stop",
                            &StopDictationRequest {
                                session_id: session.session_id,
                            },
                        )
                        .await?;
                        println!("{}", serde_json::to_string_pretty(&result)?);
                    }
                }
            }
            DictationCommand::Stop { session_id } => {
                let result: voicelayer_core::DictationCaptureResult = uds_post_json(
                    &cli_socket_path(),
                    "/v1/sessions/dictation/stop",
                    &StopDictationRequest { session_id },
                )
                .await?;
                println!("{}", serde_json::to_string_pretty(&result)?);
            }
            DictationCommand::List => {
                let sessions: Vec<voicelayer_core::CaptureSession> =
                    uds_get_json(&cli_socket_path(), "/v1/sessions").await?;
                println!("{}", serde_json::to_string_pretty(&sessions)?);
            }
        },
        Command::Hotkeys { command } => match command {
            HotkeysCommand::PortalStatus => {
                let health: HealthResponse = uds_get_json(&cli_socket_path(), "/v1/health").await?;
                println!(
                    "{}",
                    serde_json::to_string_pretty(&serde_json::json!({
                        "available": health.worker.global_hotkeys_available,
                        "backend": health.worker.global_hotkeys_backend,
                        "detail": health.worker.global_hotkeys_detail,
                    }))?
                );
            }
        },
        Command::Doctor => {
            let socket_path = cli_socket_path();
            match uds_get_json::<HealthResponse>(&socket_path, "/v1/health").await {
                Ok(health) => {
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "daemon_reachable": true,
                            "socket_path": socket_path.display().to_string(),
                            "os": std::env::consts::OS,
                            "arch": std::env::consts::ARCH,
                            "health": health,
                        }))?
                    );
                }
                Err(error) => {
                    let hint = if cfg!(target_os = "macos") {
                        "start the daemon with `launchctl bootstrap gui/$(id -u) \
                         ~/Library/LaunchAgents/com.memenow.voicelayerd.plist` \
                         or run `voicelayerd` directly"
                    } else {
                        "start the daemon with `systemctl --user start voicelayerd` \
                         or run `voicelayerd` directly"
                    };
                    println!(
                        "{}",
                        serde_json::to_string_pretty(&serde_json::json!({
                            "daemon_reachable": false,
                            "socket_path": socket_path.display().to_string(),
                            "os": std::env::consts::OS,
                            "arch": std::env::consts::ARCH,
                            "error": error.to_string(),
                            "hint": hint,
                        }))?
                    );
                }
            }
        }
        Command::Providers => {
            let providers: serde_json::Value =
                uds_get_json(&cli_socket_path(), "/v1/providers").await?;
            println!("{}", serde_json::to_string_pretty(&providers)?);
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
            provider_id,
        } => {
            let request = DictationCaptureRequest {
                trigger: TriggerKind::Cli,
                language_profile: language.map(|language| LanguageProfile {
                    strategy: LanguageStrategy::Locked,
                    input_languages: vec![language],
                    output_language: None,
                }),
                duration_seconds,
                translate_to_english,
                keep_audio,
                provider_id,
            };
            // The daemon owns capture; the CLI is a pure socket client.
            let result: voicelayer_core::DictationCaptureResult =
                uds_post_json(&cli_socket_path(), "/v1/dictation/capture", &request).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::TranscribeFile {
            audio_file,
            language,
            translate_to_english,
            provider_id,
        } => {
            let request = TranscribeRequest {
                audio_file,
                language,
                translate_to_english,
                provider_id,
            };
            let result: voicelayer_core::TranscriptionResult =
                uds_post_json(&cli_socket_path(), "/v1/transcriptions", &request).await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Preview { command } => match command {
            PreviewCommand::Compose {
                spoken_prompt,
                archetype,
                output_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = uds_post_json(
                    &cli_socket_path(),
                    "/v1/sessions/compose",
                    &ComposeRequest {
                        spoken_prompt,
                        archetype: archetype.map(Into::into),
                        output_language,
                    },
                )
                .await?;
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            PreviewCommand::Rewrite {
                source_text,
                style,
                output_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = uds_post_json(
                    &cli_socket_path(),
                    "/v1/rewrites",
                    &RewriteRequest {
                        source_text,
                        style: style.into(),
                        output_language,
                    },
                )
                .await?;
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            PreviewCommand::Translate {
                source_text,
                target_language,
            } => {
                let receipt: voicelayer_core::CompositionReceipt = uds_post_json(
                    &cli_socket_path(),
                    "/v1/translations",
                    &TranslateRequest {
                        source_text,
                        target_language,
                    },
                )
                .await?;
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {

    /// Pins the XDG branch. Doctor's `systemd_unit_installed` field
    /// drives operator-facing diagnostics ("did the install script
    /// land the unit?"); a regression that ignored `XDG_CONFIG_HOME`
    /// would silently misreport on hosts that override it.
    /// Pins the HOME-fallback branch. The install script targets
    /// `~/.config/systemd/user/` regardless of XDG; this asserts the
    /// helper agrees so doctor cannot disagree with the installer.
    /// Pins the no-anchor case. A degenerate sandbox (no XDG, no HOME)
    /// must yield `None` so doctor reports the unit as not installed
    /// instead of crashing or guessing a path that does not exist.
    /// Pins the affirmative branch of `systemd_unit_installed` and
    /// proves the unit-name parameter is honoured: the same XDG
    /// directory holds `voicelayerd.service` but not
    /// `voicelayer-whisper-server.service`, and the helper must
    /// report exactly one as installed.
    mod dictation_parsing {
        use clap::Parser;

        use super::super::Args;

        fn try_parse(args: &[&str]) -> Result<Args, clap::Error> {
            Args::try_parse_from(args)
        }

        #[test]
        fn one_shot_default_parses_without_numeric_knobs() {
            try_parse(&["vl", "dictation", "start"])
                .expect("default one-shot mode must parse with no numeric flags");
        }

        #[test]
        fn fixed_mode_requires_segment_secs() {
            let error = try_parse(&["vl", "dictation", "start", "--mode", "fixed"])
                .expect_err("--mode fixed without --segment-secs must error at parse time");
            assert!(
                error.to_string().contains("--segment-secs"),
                "error should name the missing flag; got {error}",
            );
        }

        #[test]
        fn vad_gated_mode_requires_probe_secs() {
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "vad-gated",
                "--max-segment-secs",
                "30",
            ])
            .expect_err("--mode vad-gated without --probe-secs must error");
        }

        #[test]
        fn vad_gated_mode_requires_max_segment_secs() {
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "vad-gated",
                "--probe-secs",
                "2",
            ])
            .expect_err("--mode vad-gated without --max-segment-secs must error");
        }

        #[test]
        fn vad_gated_mode_accepts_full_arg_set() {
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
            .expect("complete vad-gated arg set must parse cleanly");
        }

        #[test]
        fn fixed_mode_rejects_zero_segment_secs() {
            // The clap range parser must catch 0 before the request
            // reaches the daemon. Without `value_parser = ..range(1..)`
            // this would round-trip to the daemon and return a Failed
            // session — the user-facing error would be far less direct.
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "fixed",
                "--segment-secs",
                "0",
            ])
            .expect_err("--segment-secs 0 must be rejected at parse time");
        }

        #[test]
        fn vad_gated_mode_rejects_zero_probe_secs() {
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "vad-gated",
                "--probe-secs",
                "0",
                "--max-segment-secs",
                "30",
            ])
            .expect_err("--probe-secs 0 must be rejected at parse time");
        }

        #[test]
        fn vad_gated_mode_rejects_zero_max_segment_secs() {
            try_parse(&[
                "vl",
                "dictation",
                "start",
                "--mode",
                "vad-gated",
                "--probe-secs",
                "2",
                "--max-segment-secs",
                "0",
            ])
            .expect_err("--max-segment-secs 0 must be rejected at parse time");
        }

        #[test]
        fn duration_seconds_accepts_a_positive_integer() {
            try_parse(&["vl", "dictation", "start", "--duration-seconds", "10"])
                .expect("--duration-seconds 10 must parse cleanly on the default one-shot path");
        }

        #[test]
        fn duration_seconds_rejects_zero() {
            // Symmetric with the other clap range-parsed knobs: 0 is
            // rejected at parse time so the operator gets a precise
            // "0 is not in 1..=4294967295" error rather than a noop
            // round-trip to the daemon.
            try_parse(&["vl", "dictation", "start", "--duration-seconds", "0"])
                .expect_err("--duration-seconds 0 must be rejected at parse time");
        }

        #[test]
        fn duration_seconds_composes_with_vad_gated_mode() {
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
                "--duration-seconds",
                "8",
            ])
            .expect("vad-gated + duration-seconds must parse together");
        }

        #[test]
        fn list_parses_with_no_args() {
            try_parse(&["vl", "dictation", "list"])
                .expect("dictation list must parse with no positional args or flags");
        }

        /// The `--provider-id` flag is the user-facing knob for picking a
        /// non-default ASR provider (currently `mimo_v2_5_asr`) for an
        /// entire dictation session. Pin parse-time acceptance so a
        /// future refactor that drops the flag from the `Start`
        /// subcommand surfaces at `cargo test` rather than at runtime
        /// when an operator's PTT loop silently falls back to whisper.
        #[test]
        fn start_accepts_provider_id_flag() {
            let parsed = try_parse(&["vl", "dictation", "start", "--provider-id", "mimo_v2_5_asr"])
                .expect("dictation start must accept --provider-id mimo_v2_5_asr");
            let provider_id = match parsed.command {
                super::super::Command::Dictation {
                    command: super::super::DictationCommand::Start { provider_id, .. },
                } => provider_id,
                other => panic!("expected dictation start, got {other:?}"),
            };
            assert_eq!(
                provider_id.as_deref(),
                Some("mimo_v2_5_asr"),
                "--provider-id mimo_v2_5_asr must round-trip into the parsed Start variant",
            );
        }

        #[test]
        fn record_transcribe_accepts_provider_id_flag() {
            let parsed = try_parse(&[
                "vl",
                "record-transcribe",
                "--duration-seconds",
                "5",
                "--provider-id",
                "mimo_v2_5_asr",
            ])
            .expect("record-transcribe must accept --provider-id mimo_v2_5_asr");
            let provider_id = match parsed.command {
                super::super::Command::RecordTranscribe { provider_id, .. } => provider_id,
                other => panic!("expected record-transcribe, got {other:?}"),
            };
            assert_eq!(
                provider_id.as_deref(),
                Some("mimo_v2_5_asr"),
                "--provider-id mimo_v2_5_asr must round-trip into RecordTranscribe",
            );
        }

        #[test]
        fn foreground_ptt_accepts_provider_id_flag() {
            let parsed = try_parse(&[
                "vl",
                "dictation",
                "foreground-ptt",
                "--provider-id",
                "mimo_v2_5_asr",
            ])
            .expect("foreground-ptt must accept --provider-id mimo_v2_5_asr");
            let provider_id = match parsed.command {
                super::super::Command::Dictation {
                    command: super::super::DictationCommand::ForegroundPtt { provider_id, .. },
                } => provider_id,
                other => panic!("expected dictation foreground-ptt, got {other:?}"),
            };
            assert_eq!(
                provider_id.as_deref(),
                Some("mimo_v2_5_asr"),
                "--provider-id mimo_v2_5_asr must round-trip into ForegroundPtt",
            );
        }

        /// `vl --version` is the binary's "what release am I running?"
        /// affordance. Combined with the openapi `info.version`
        /// pin from #55, an operator who reports a bug can match the
        /// CLI version string against the contract version they
        /// were targeting. Pin the wiring so a future refactor that
        /// dropped the `#[command(version)]` attribute would surface
        /// at `cargo test`, not at first user complaint.
        #[test]
        fn top_level_version_flag_emits_cargo_pkg_version() {
            let error = try_parse(&["vl", "--version"])
                .expect_err("clap exits with DisplayVersion when --version is passed");
            assert_eq!(error.kind(), clap::error::ErrorKind::DisplayVersion);
            let rendered = error.to_string();
            assert!(
                rendered.contains(env!("CARGO_PKG_VERSION")),
                "version output should include CARGO_PKG_VERSION ({}); got {rendered}",
                env!("CARGO_PKG_VERSION"),
            );
        }

        #[test]
        fn stop_requires_a_session_id_positional() {
            let error = try_parse(&["vl", "dictation", "stop"])
                .expect_err("dictation stop without a session_id must error at parse time");
            // clap's missing-required-arg message names the argument.
            let message = error.to_string();
            assert!(
                message.contains("session_id") || message.contains("SESSION_ID"),
                "missing-arg error should name session_id; got {message}",
            );
        }

        #[test]
        fn stop_rejects_malformed_session_id() {
            // The CLI uses `uuid::Uuid` as the value type, so clap
            // rejects non-UUID strings at parse time. Pin that the
            // operator gets a clean error message instead of the
            // request reaching the daemon and surfacing as a 500.
            try_parse(&["vl", "dictation", "stop", "not-a-uuid"])
                .expect_err("dictation stop must reject a non-UUID session_id at parse time");
        }

        /// `vl --help` is the binary's discovery surface. Pin the
        /// top-level about string so a future refactor that strips or
        /// rewrites the `#[command(about = ...)]` attribute surfaces
        /// at `cargo test` rather than after operators report a stale
        /// header.
        #[test]
        fn top_level_help_renders_voicelayer_operator_cli_about() {
            let error = try_parse(&["vl", "--help"])
                .expect_err("clap exits with DisplayHelp when --help is passed");
            assert_eq!(error.kind(), clap::error::ErrorKind::DisplayHelp);
            let rendered = error.to_string();
            assert!(
                rendered.contains("VoiceLayer operator CLI"),
                "top-level help should advertise the binary; got {rendered}",
            );
        }

        /// Duplicate of `fixed_mode_requires_segment_secs` from a
        /// caller-shape angle: drive the parser through the same
        /// `Args::try_parse_from` entry point operators hit via
        /// `main()`'s `Args::parse()`, and pin that the missing-flag
        /// error message names `--segment-secs` (the user-facing
        /// remediation) so the `required_if_eq` wiring cannot be
        /// quietly dropped.
        #[test]
        fn start_fixed_mode_error_message_names_segment_secs_flag() {
            let result: Result<Args, clap::Error> =
                Args::try_parse_from(["vl", "dictation", "start", "--mode", "fixed"]);
            let error = result.expect_err("fixed mode without --segment-secs must error");
            assert!(
                error.to_string().contains("--segment-secs"),
                "missing-required error must name --segment-secs; got {error}",
            );
        }
    }

    mod readme_subcommand_alignment {
        //! Cross-check that every `vl <cmd>` mention in `README.md`
        //! resolves to a real top-level subcommand declared in this
        //! file's `enum Command`. Forward direction only: the CLI
        //! includes `preview` (an internal subcommand) which the
        //! README intentionally does not advertise; reverse-direction
        //! enforcement would require exposing it.
        //!
        //! The drift mode is silent: the README invites a contributor
        //! to run `vl scribe-file` (typo) or `vl old-name` (after a
        //! rename), the user copy-pastes the line, clap returns
        //! `error: unrecognized subcommand`, and the only fix is to
        //! reverse-engineer the real name from `--help`.
        use std::collections::BTreeSet;

        /// Walk `README.md` and pull every `vl <subcommand>` token,
        /// covering both shapes the README uses today:
        /// - `cargo run -p vl -- <cmd>` (in fenced code blocks)
        /// - `` `vl <cmd>` `` (inline backticks)
        ///
        /// Only the *first* token after `vl ` (or `-p vl -- `) is
        /// captured — that is the top-level subcommand. Sub-subcommands
        /// like `dictation foreground-ptt` contribute only `dictation`
        /// to the set; the deeper alignment is left to clap's own
        /// parse-time errors.
        pub(super) fn extract_readme_vl_subcommand_mentions(contents: &str) -> BTreeSet<String> {
            let mut subs = BTreeSet::new();
            for prefix in ["-p vl -- ", "`vl "] {
                let mut search = contents;
                while let Some(idx) = search.find(prefix) {
                    let after = &search[idx + prefix.len()..];
                    let token: String = after
                        .chars()
                        .take_while(|c| c.is_ascii_lowercase() || *c == '-' || c.is_ascii_digit())
                        .collect();
                    if !token.is_empty() && !token.starts_with('-') {
                        subs.insert(token);
                    }
                    search = after;
                }
            }
            subs
        }

        /// Walk this file (`crates/vl/src/cli.rs`) and pull every
        /// top-level `enum Command` variant name, kebab-cased to
        /// match clap's auto-derived subcommand names. Variants are
        /// recognised by their 4-space indent and PascalCase shape;
        /// inner sub-fields (deeper indent) and the closing brace
        /// are filtered out.
        pub(super) fn extract_clap_top_level_command_names(source: &str) -> BTreeSet<String> {
            let mut names = BTreeSet::new();
            let mut in_enum = false;
            for line in source.lines() {
                if line.trim() == "enum Command {" {
                    in_enum = true;
                    continue;
                }
                if !in_enum {
                    continue;
                }
                if line.trim() == "}" {
                    break;
                }
                let Some(rest) = line.strip_prefix("    ") else {
                    continue;
                };
                if rest.starts_with(' ') {
                    continue;
                }
                let token: String = rest
                    .chars()
                    .take_while(|c| c.is_ascii_alphanumeric() || *c == '_')
                    .collect();
                if !token.is_empty() && token.starts_with(|c: char| c.is_ascii_uppercase()) {
                    names.insert(pascal_to_kebab(&token));
                }
            }
            names
        }

        fn pascal_to_kebab(s: &str) -> String {
            let mut out = String::new();
            for (i, c) in s.chars().enumerate() {
                if c.is_ascii_uppercase() && i > 0 {
                    out.push('-');
                }
                out.push(c.to_ascii_lowercase());
            }
            out
        }

        #[test]
        fn pascal_to_kebab_handles_acronym_and_multiword_variants() {
            assert_eq!(pascal_to_kebab("Doctor"), "doctor");
            assert_eq!(
                pascal_to_kebab("PrintBracketedPaste"),
                "print-bracketed-paste"
            );
            assert_eq!(pascal_to_kebab("RecordTranscribe"), "record-transcribe");
        }

        #[test]
        fn extract_readme_vl_subcommand_mentions_handles_both_shapes() {
            let md = "\
The `vl doctor` invocation is a quick environment probe.

```bash
cargo run -p vl -- providers
```

`vl --help` prints clap's auto-generated reference.
";
            let mentions = extract_readme_vl_subcommand_mentions(md);
            assert_eq!(
                mentions,
                ["doctor", "providers"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                "`vl --help` must NOT contribute `--help` to the set",
            );
        }

        #[test]
        fn extract_clap_top_level_command_names_collects_unit_and_struct_variants() {
            let source = "\
enum Command {
    Doctor,
    PrintBracketedPaste {
        text: String,
    },
}
";
            let names = extract_clap_top_level_command_names(source);
            assert_eq!(
                names,
                ["doctor", "print-bracketed-paste"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
            );
        }

        #[test]
        fn every_readme_vl_subcommand_mention_resolves_to_a_clap_command_variant() {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let cli_rs = std::fs::read_to_string(format!("{manifest}/src/cli.rs"))
                .expect("read crates/vl/src/cli.rs");
            let readme = std::fs::read_to_string(format!("{manifest}/../../README.md"))
                .expect("read README.md");

            let mentions = extract_readme_vl_subcommand_mentions(&readme);
            let variants = extract_clap_top_level_command_names(&cli_rs);
            assert!(
                !mentions.is_empty(),
                "expected at least one `vl <cmd>` reference in README — \
                 the scanner may have lost its anchors",
            );
            assert!(
                !variants.is_empty(),
                "expected at least one variant in `enum Command` — \
                 the parser may have lost its anchor",
            );

            let invalid: Vec<&String> = mentions.difference(&variants).collect();
            assert!(
                invalid.is_empty(),
                "README references vl subcommands that do not exist in the \
                 CLI: {invalid:?}\n\nFix the typo, drop the mention, or add \
                 the variant to `enum Command` in crates/vl/src/cli.rs.",
            );
        }
    }

    mod readme_flag_alignment {
        //! Cross-check that every `--<flag>` token appearing in a
        //! `cargo run -p vl -- ...` invocation in `README.md`
        //! corresponds to a real `#[arg(long, ...)]` declaration on
        //! some Command variant in this file.
        //!
        //! Flags from non-vl tools (`kitten @ send-text --match ... --stdin`,
        //! `cargo build --all-features`) are not in scope: the
        //! extractor anchors at `cargo run -p vl --` and only collects
        //! flags from the tail of that line (and its `\` continuations),
        //! so external-tool flag mentions elsewhere in the README do
        //! not pollute the captured set.
        //!
        //! Reverse direction (every CLI flag is mentioned in README) is
        //! intentionally not enforced — the README documents
        //! operator-facing usage paths, not every internal knob.
        use std::collections::BTreeSet;

        /// Walk `README.md` and pull every `--<flag>` token that
        /// appears after `cargo run -p vl -- ` on the same logical
        /// line. `\<newline>` continuations are collapsed into a
        /// space first so multi-line invocations parse as a single
        /// command. Captured flag names are kebab-case and exclude
        /// the leading `--`.
        pub(super) fn extract_readme_vl_invocation_flags(contents: &str) -> BTreeSet<String> {
            let mut collapsed = String::new();
            for line in contents.lines() {
                let trimmed_end = line.trim_end_matches([' ', '\t']);
                if let Some(stripped) = trimmed_end.strip_suffix('\\') {
                    collapsed.push_str(stripped);
                    collapsed.push(' ');
                } else {
                    collapsed.push_str(trimmed_end);
                    collapsed.push('\n');
                }
            }

            let mut flags = BTreeSet::new();
            for line in collapsed.lines() {
                let Some(idx) = line.find("cargo run -p vl --") else {
                    continue;
                };
                let tail = &line[idx + "cargo run -p vl --".len()..];
                let mut search = tail;
                while let Some(idx) = search.find(" --") {
                    let after = &search[idx + 3..];
                    let token: String = after
                        .chars()
                        .take_while(|c| c.is_ascii_lowercase() || *c == '-' || c.is_ascii_digit())
                        .collect();
                    if !token.is_empty() && !token.starts_with('-') {
                        flags.insert(token);
                    }
                    search = after;
                }
            }
            flags
        }

        /// Walk this file (`crates/vl/src/cli.rs`) and pull every
        /// `--<flag>` clap exposes, derived from `#[arg(long, ...)]`
        /// (and `#[arg(long = "name", ...)]` for explicit overrides)
        /// followed by a field declaration. Multi-line attributes
        /// like
        ///
        /// ```ignore
        /// #[arg(
        ///     long,
        ///     value_parser = clap::value_parser!(u32).range(1..)
        /// )]
        /// segment_secs: Option<u32>,
        /// ```
        ///
        /// are handled by tracking the `#[arg(...)]` block boundaries
        /// (entered on a line starting with `#[arg(`, exited when a
        /// line ends with `)]`). Field names auto-convert from
        /// snake_case to kebab-case to match clap's derive behaviour.
        pub(super) fn extract_clap_long_flag_names(source: &str) -> BTreeSet<String> {
            let mut flags = BTreeSet::new();
            let mut in_arg_attr = false;
            let mut long_seen = false;
            let mut explicit_name: Option<String> = None;

            for line in source.lines() {
                let trimmed = line.trim();
                if trimmed.starts_with("#[arg(") {
                    in_arg_attr = true;
                    long_seen = false;
                    explicit_name = None;
                }
                if in_arg_attr {
                    if has_word_long(trimmed) {
                        long_seen = true;
                        if let Some(name) = parse_explicit_long_name(trimmed) {
                            explicit_name = Some(name);
                        }
                    }
                    if trimmed.ends_with(")]") {
                        in_arg_attr = false;
                    }
                    continue;
                }
                if !long_seen {
                    continue;
                }
                if trimmed.starts_with('#') || trimmed.is_empty() {
                    continue;
                }
                if let Some(colon) = trimmed.find(':') {
                    let name = trimmed[..colon].trim();
                    if !name.is_empty()
                        && name.chars().all(|c| c.is_ascii_alphanumeric() || c == '_')
                    {
                        let final_name = explicit_name
                            .take()
                            .unwrap_or_else(|| name.replace('_', "-"));
                        flags.insert(final_name);
                    }
                }
                long_seen = false;
                explicit_name = None;
            }
            flags
        }

        fn has_word_long(text: &str) -> bool {
            text.split(|c: char| !c.is_ascii_alphanumeric() && c != '_')
                .any(|w| w == "long")
        }

        fn parse_explicit_long_name(text: &str) -> Option<String> {
            let idx = text.find("long = \"")?;
            let after = &text[idx + "long = \"".len()..];
            let end = after.find('"')?;
            Some(after[..end].to_owned())
        }

        #[test]
        fn extract_readme_vl_invocation_flags_picks_up_continuation_and_skips_external_tools() {
            let md = "\
The kitten line uses an external tool:

```
kitten @ send-text --match title:Foo --stdin --bracketed-paste auto
cargo run -p vl -- dictation foreground-ptt \\
  --backend pipewire --copy-on-stop \\
  --tmux-target-pane %2
```

Inline `vl --help` is not in scope (no cargo run prefix).
";
            let flags = extract_readme_vl_invocation_flags(md);
            assert_eq!(
                flags,
                ["backend", "copy-on-stop", "tmux-target-pane"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                "external-tool flags (`--match`, `--stdin`, `--bracketed-paste`) \
                 must not leak into the vl-flag set",
            );
        }

        #[test]
        fn extract_clap_long_flag_names_handles_multiline_attributes_and_explicit_overrides() {
            let source = "\
struct Foo {
    #[arg(long)]
    plain_field: bool,

    #[arg(long = \"renamed\")]
    underlying_name: String,

    #[arg(
        long,
        value_parser = clap::value_parser!(u32).range(1..),
    )]
    multi_line_field: u32,

    #[arg(short)]
    no_long: bool,
}
";
            let flags = extract_clap_long_flag_names(source);
            assert_eq!(
                flags,
                ["multi-line-field", "plain-field", "renamed"]
                    .iter()
                    .map(|s| (*s).to_owned())
                    .collect(),
                "fields without `long` must be excluded; explicit-name fields \
                 must use the override; multi-line attribute must still attach \
                 to its field",
            );
        }

        #[test]
        fn extract_clap_long_flag_names_does_not_capture_long_lookalike_identifiers() {
            // A struct field literally named `long_term` and an
            // attribute that doesn't include the `long` directive
            // must not contribute a flag. The word-boundary check
            // in `has_word_long` is what prevents this.
            let source = "\
struct Foo {
    #[arg(short)]
    long_term: bool,
    #[arg(env = \"belongs\")]
    plain: bool,
}
";
            let flags = extract_clap_long_flag_names(source);
            assert!(flags.is_empty(), "got {flags:?}");
        }

        #[test]
        fn every_readme_vl_invocation_flag_resolves_to_a_clap_long_flag() {
            let manifest = env!("CARGO_MANIFEST_DIR");
            let cli_rs = std::fs::read_to_string(format!("{manifest}/src/cli.rs"))
                .expect("read crates/vl/src/cli.rs");
            let readme = std::fs::read_to_string(format!("{manifest}/../../README.md"))
                .expect("read README.md");

            let mentions = extract_readme_vl_invocation_flags(&readme);
            let flags = extract_clap_long_flag_names(&cli_rs);
            assert!(
                !mentions.is_empty(),
                "expected at least one `--<flag>` after `cargo run -p vl -- ` in README",
            );
            assert!(
                !flags.is_empty(),
                "expected at least one `#[arg(long, ...)]` declaration in cli.rs",
            );

            let invalid: Vec<&String> = mentions.difference(&flags).collect();
            assert!(
                invalid.is_empty(),
                "README references vl flags that do not exist in any \
                 `#[arg(long, ...)]` declaration: {invalid:?}\n\n\
                 Fix the typo, drop the mention, or wire up the flag in \
                 the matching Command variant in crates/vl/src/cli.rs.",
            );
        }
    }
}
