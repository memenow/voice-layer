//! Execution handlers for `vl` subcommands.

use voicelayer_core::client::DaemonClient;
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionReceipt, DictationCaptureRequest,
    DictationCaptureResult, HealthResponse, InjectRequest, InjectTarget, InjectionPlan,
    LanguageProfile, LanguageStrategy, RewriteRequest, StartDictationRequest, StopAction,
    StopDictationRequest, TranscribeRequest, TranscriptionResult, TranslateRequest, TriggerKind,
    VoiceLayerConfig,
};

use crate::config::{
    CliPttKey, build_segmentation_mode, load_vl_config, set_config_value, vl_config_path,
    write_vl_config,
};
use crate::ptt::{PttOptions, run_foreground_ptt};

use super::{Command, ConfigCommand, DictationCommand, HotkeysCommand, PreviewCommand};

type CliResult = Result<(), Box<dyn std::error::Error>>;

pub(super) async fn dispatch(command: Command) -> CliResult {
    let client = DaemonClient::from_env();
    match command {
        Command::Config { command } => handle_config(command),
        Command::Dictation { command } => handle_dictation(&client, command).await,
        Command::Hotkeys { command } => handle_hotkeys(&client, command).await,
        Command::Doctor => handle_doctor(&client).await,
        Command::Providers => {
            let providers: serde_json::Value = client.get_json("/v1/providers").await?;
            print_json(&providers)
        }
        Command::PrintBracketedPaste { text, auto_submit } => {
            let plan = InjectionPlan::from_request(&InjectRequest {
                target: InjectTarget::TerminalBracketedPaste,
                text,
                auto_submit,
            });
            print!("{}", plan.payload);
            Ok(())
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
                language_profile: locked_profile(language),
                duration_seconds,
                translate_to_english,
                keep_audio,
                provider_id,
            };
            let result: DictationCaptureResult =
                client.post_json("/v1/dictation/capture", &request).await?;
            print_json(&result)
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
            let result: TranscriptionResult =
                client.post_json("/v1/transcriptions", &request).await?;
            print_json(&result)
        }
        Command::Preview { command } => handle_preview(&client, command).await,
    }
}

fn handle_config(command: ConfigCommand) -> CliResult {
    match command {
        ConfigCommand::Path => {
            println!("{}", vl_config_path()?.display());
            Ok(())
        }
        ConfigCommand::Show => {
            let config = load_vl_config()?;
            println!("{}", toml::to_string_pretty(&config)?);
            Ok(())
        }
        ConfigCommand::InitDefaults => {
            let path = write_vl_config(&VoiceLayerConfig::default())?;
            print_json(&serde_json::json!({
                "status": "ok",
                "config_path": path,
            }))
        }
        ConfigCommand::Set { key, value } => {
            let mut config = load_vl_config()?;
            set_config_value(&mut config, &key, &value)?;
            let path = write_vl_config(&config)?;
            print_json(&serde_json::json!({
                "status": "ok",
                "config_path": path,
                "updated_key": key,
                "updated_value": value,
            }))
        }
    }
}

async fn handle_dictation(client: &DaemonClient, command: DictationCommand) -> CliResult {
    match command {
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
            let default_key = CliPttKey::parse_str(&defaults.key).unwrap_or(CliPttKey::Space);
            let cli_action: StopAction = default_stop_action.into();
            let options = PttOptions {
                language: language.or(defaults.language),
                translate_to_english: translate_to_english || defaults.translate_to_english,
                keep_audio: keep_audio || defaults.keep_audio,
                key: if key == CliPttKey::Space {
                    default_key
                } else {
                    key
                },
                tmux_target_pane: tmux_target_pane.or(defaults.tmux_target_pane),
                wezterm_target_pane_id: wezterm_target_pane_id.or(defaults.wezterm_target_pane_id),
                kitty_match: kitty_match.or(defaults.kitty_match),
                copy_on_stop: copy_on_stop || defaults.copy_on_stop,
                default_stop_action: if cli_action == StopAction::None {
                    defaults.default_stop_action
                } else {
                    cli_action
                },
                restore_clipboard_on_exit: restore_clipboard_on_exit
                    || defaults.restore_clipboard_on_exit,
                save_dir: save_dir.or_else(|| defaults.save_dir.map(Into::into)),
                provider_id: provider_id.or(defaults.provider_id),
            };
            run_foreground_ptt(client, options).await
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
                language_profile: locked_profile(language),
                translate_to_english,
                keep_audio,
                segmentation,
                provider_id,
            };
            let session: CaptureSession =
                client.post_json("/v1/sessions/dictation", &request).await?;

            match duration_seconds {
                None => print_json(&session),
                Some(seconds) => {
                    eprintln!("dictation listening; auto-stopping after {seconds}s");
                    tokio::time::sleep(std::time::Duration::from_secs(u64::from(seconds))).await;
                    let result: DictationCaptureResult = client
                        .post_json(
                            "/v1/sessions/dictation/stop",
                            &StopDictationRequest {
                                session_id: session.session_id,
                            },
                        )
                        .await?;
                    print_json(&result)
                }
            }
        }
        DictationCommand::Stop { session_id } => {
            let result: DictationCaptureResult = client
                .post_json(
                    "/v1/sessions/dictation/stop",
                    &StopDictationRequest { session_id },
                )
                .await?;
            print_json(&result)
        }
        DictationCommand::List => {
            let sessions: Vec<CaptureSession> = client.get_json("/v1/sessions").await?;
            print_json(&sessions)
        }
    }
}

async fn handle_hotkeys(client: &DaemonClient, command: HotkeysCommand) -> CliResult {
    match command {
        HotkeysCommand::PortalStatus => {
            let health: HealthResponse = client.get_json("/v1/health").await?;
            print_json(&serde_json::json!({
                "available": health.worker.global_hotkeys_available,
                "backend": health.worker.global_hotkeys_backend,
                "detail": health.worker.global_hotkeys_detail,
            }))
        }
    }
}

async fn handle_doctor(client: &DaemonClient) -> CliResult {
    let socket = client.socket_path().display().to_string();
    match client.get_json::<HealthResponse>("/v1/health").await {
        Ok(health) => print_json(&serde_json::json!({
            "daemon_reachable": true,
            "socket_path": socket,
            "os": std::env::consts::OS,
            "arch": std::env::consts::ARCH,
            "health": health,
        })),
        Err(error) => {
            let hint = if cfg!(target_os = "macos") {
                "start the daemon with `launchctl bootstrap gui/$(id -u) \
                 ~/Library/LaunchAgents/com.memenow.voicelayerd.plist` \
                 or run `voicelayerd` directly"
            } else {
                "start the daemon with `systemctl --user start voicelayerd` \
                 or run `voicelayerd` directly"
            };
            print_json(&serde_json::json!({
                "daemon_reachable": false,
                "socket_path": socket,
                "os": std::env::consts::OS,
                "arch": std::env::consts::ARCH,
                "error": error.to_string(),
                "hint": hint,
            }))
        }
    }
}

async fn handle_preview(client: &DaemonClient, command: PreviewCommand) -> CliResult {
    let receipt: CompositionReceipt = match command {
        PreviewCommand::Compose {
            spoken_prompt,
            archetype,
            output_language,
        } => {
            client
                .post_json(
                    "/v1/sessions/compose",
                    &ComposeRequest {
                        spoken_prompt,
                        archetype: archetype.map(Into::into),
                        output_language,
                    },
                )
                .await?
        }
        PreviewCommand::Rewrite {
            source_text,
            style,
            output_language,
        } => {
            client
                .post_json(
                    "/v1/rewrites",
                    &RewriteRequest {
                        source_text,
                        style: style.into(),
                        output_language,
                    },
                )
                .await?
        }
        PreviewCommand::Translate {
            source_text,
            target_language,
        } => {
            client
                .post_json(
                    "/v1/translations",
                    &TranslateRequest {
                        source_text,
                        target_language,
                    },
                )
                .await?
        }
    };
    print_json(&receipt)
}

fn locked_profile(language: Option<String>) -> Option<LanguageProfile> {
    language.map(|lang| LanguageProfile {
        strategy: LanguageStrategy::Locked,
        input_languages: vec![lang],
        output_language: None,
    })
}

fn print_json(value: &impl serde::Serialize) -> CliResult {
    println!("{}", serde_json::to_string_pretty(value)?);
    Ok(())
}
