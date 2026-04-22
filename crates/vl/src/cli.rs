use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use voicelayer_core::{
    ComposeRequest, CompositionArchetype, DictationCaptureRequest, InjectRequest, InjectTarget,
    InjectionPlan, LanguageProfile, LanguageStrategy, RecorderBackend, RewriteRequest,
    RewriteStyle, SegmentationMode, SessionMode, StartDictationRequest, StopDictationRequest,
    TranscribeRequest, TranslateRequest, TriggerKind,
};
use voicelayerd::{
    DaemonConfig, RecorderDiagnostics, WorkerHealthResult, capture_dictation_once,
    default_project_root, default_socket_path, probe_global_shortcuts_portal, recorder_diagnostics,
    run_daemon,
};

use crate::config::{
    CliPttKey, CliRecorderBackend, StopAction, VlConfig, load_vl_config, set_config_value,
    vl_config_path, write_vl_config,
};
use crate::foreground_ptt::run_foreground_ptt;
use crate::preview::{ready_receipt, worker_error_receipt};
use crate::uds::{cli_socket_path, uds_post_json};

#[derive(Debug, Parser)]
#[command(name = "vl")]
#[command(about = "VoiceLayer operator CLI")]
struct Args {
    #[command(subcommand)]
    command: Command,
}

#[derive(Debug, Subcommand)]
enum Command {
    Daemon {
        #[command(subcommand)]
        command: DaemonCommand,
    },
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
        #[arg(long, value_enum, default_value_t = CliRecorderBackend::Auto)]
        backend: CliRecorderBackend,
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
enum DaemonCommand {
    Run {
        #[arg(long, env = "VOICELAYER_SOCKET_PATH")]
        socket_path: Option<PathBuf>,
        #[arg(long, env = "VOICELAYER_PROJECT_ROOT")]
        project_root: Option<PathBuf>,
    },
}

#[derive(Debug, Subcommand)]
enum DictationCommand {
    ForegroundPtt {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, value_enum, default_value_t = CliRecorderBackend::Auto)]
        backend: CliRecorderBackend,
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
    },
    Start {
        #[arg(long)]
        language: Option<String>,
        #[arg(long, value_enum, default_value_t = CliRecorderBackend::Auto)]
        backend: CliRecorderBackend,
        #[arg(long, default_value_t = false)]
        translate_to_english: bool,
        #[arg(long, default_value_t = false)]
        keep_audio: bool,
    },
    Stop {
        session_id: uuid::Uuid,
    },
}

#[derive(Debug, Subcommand)]
enum HotkeysCommand {
    PortalStatus,
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

#[derive(Debug, Serialize)]
struct DoctorReport {
    socket_path: String,
    project_root: String,
    worker_command: String,
    worker_status: String,
    worker_error: Option<String>,
    asr_configured: bool,
    asr_binary: Option<String>,
    asr_model_path: Option<String>,
    asr_error: Option<String>,
    whisper_mode: &'static str,
    llm_configured: bool,
    llm_model: Option<String>,
    llm_endpoint: Option<String>,
    llm_reachable: bool,
    llm_error: Option<String>,
    llama_autostart_active: bool,
    global_shortcuts_portal_available: bool,
    global_shortcuts_portal_version: Option<u32>,
    global_shortcuts_portal_error: Option<String>,
    recorder: RecorderDiagnostics,
    systemd_unit_installed: bool,
    systemd_unit_active: bool,
    wayland_display: Option<String>,
    x11_display: Option<String>,
    dbus_session_bus_address: Option<String>,
    xdg_runtime_dir: Option<String>,
}

fn env_is_set(key: &str) -> bool {
    std::env::var_os(key)
        .map(|value| !value.is_empty())
        .unwrap_or(false)
}

fn env_is_truthy(key: &str) -> bool {
    std::env::var(key)
        .ok()
        .map(|value| value.trim().to_ascii_lowercase())
        .is_some_and(|value| matches!(value.as_str(), "1" | "true" | "yes" | "on"))
}

fn detect_whisper_mode() -> &'static str {
    let server_configured = [
        "VOICELAYER_WHISPER_SERVER_HOST",
        "VOICELAYER_WHISPER_SERVER_PORT",
        "VOICELAYER_WHISPER_SERVER_BIN",
    ]
    .iter()
    .any(|key| env_is_set(key))
        || env_is_truthy("VOICELAYER_WHISPER_SERVER_AUTO_START");

    if server_configured {
        return "server";
    }
    if env_is_set("VOICELAYER_WHISPER_MODEL_PATH") {
        "cli"
    } else {
        "unconfigured"
    }
}

fn provider_state_dir() -> PathBuf {
    let base = std::env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(std::env::temp_dir);
    base.join("voicelayer").join("providers")
}

fn pid_matches_command(pid: i32, expected_executable: &str) -> bool {
    if pid <= 0 {
        return false;
    }
    let cmdline_path = format!("/proc/{pid}/cmdline");
    // `/proc/<pid>/cmdline` is NUL-separated argv. Reading it also serves
    // as the liveness probe: the file disappears when the process exits.
    let Ok(bytes) = fs::read(&cmdline_path) else {
        return false;
    };
    let argv0 = bytes.split(|b| *b == 0).next().unwrap_or(&[]);
    let Ok(argv0) = std::str::from_utf8(argv0) else {
        return false;
    };
    // `expected_executable` may be an absolute path ("/usr/bin/llama-server")
    // or a bare name ("llama-server"). Match by file name so `$PATH`
    // resolution and explicit paths both work.
    let expected_name = Path::new(expected_executable)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(expected_executable);
    let argv0_name = Path::new(argv0)
        .file_name()
        .and_then(|n| n.to_str())
        .unwrap_or(argv0);
    argv0_name == expected_name
}

fn llama_autostart_active() -> bool {
    let Ok(entries) = fs::read_dir(provider_state_dir()) else {
        return false;
    };
    for entry in entries.flatten() {
        let path = entry.path();
        let Some(name) = path.file_name().and_then(|n| n.to_str()) else {
            continue;
        };
        // Skip whisper-server state; it has its own `whisper-server-*.json`
        // prefix. Everything else under providers/ is a llama-server
        // autostart record keyed by `{host}-{port}.json`.
        if !name.ends_with(".json") || name.starts_with("whisper-server-") {
            continue;
        }
        let Ok(body) = fs::read_to_string(&path) else {
            continue;
        };
        let Ok(value) = serde_json::from_str::<serde_json::Value>(&body) else {
            continue;
        };
        let Some(pid) = value.get("pid").and_then(serde_json::Value::as_i64) else {
            continue;
        };
        // Cross-check argv[0] against the command recorded at launch so a
        // recycled PID can't falsely report "autostart active".
        let expected = value
            .get("command")
            .and_then(|c| c.get(0))
            .and_then(serde_json::Value::as_str)
            .unwrap_or("llama-server");
        if pid_matches_command(pid as i32, expected) {
            return true;
        }
    }
    false
}

fn systemd_user_unit_path() -> Option<PathBuf> {
    let config_dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config_dir.join("systemd/user/voicelayerd.service"))
}

fn systemd_unit_installed() -> bool {
    systemd_user_unit_path()
        .map(|path| path.is_file())
        .unwrap_or(false)
}

fn systemd_unit_active() -> bool {
    ProcessCommand::new("systemctl")
        .args(["--user", "is-active", "voicelayerd"])
        .output()
        .map(|output| output.status.success())
        .unwrap_or(false)
}

#[allow(clippy::too_many_lines)]
pub(crate) async fn run() -> Result<(), Box<dyn std::error::Error>> {
    let args = Args::parse();

    match args.command {
        Command::Daemon { command } => match command {
            DaemonCommand::Run {
                socket_path,
                project_root,
            } => {
                let socket_path = socket_path.unwrap_or_else(default_socket_path);
                let project_root = project_root.unwrap_or_else(default_project_root);
                run_daemon(DaemonConfig::with_project_root(socket_path, project_root)).await?;
            }
        },
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
                backend,
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
                run_foreground_ptt(
                    language.or(defaults.language),
                    if backend == CliRecorderBackend::Auto {
                        defaults.backend
                    } else {
                        backend
                    },
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
                )
                .await?;
            }
            DictationCommand::Start {
                language,
                backend,
                translate_to_english,
                keep_audio,
            } => {
                let request = StartDictationRequest {
                    trigger: TriggerKind::Cli,
                    language_profile: language.map(|language| LanguageProfile {
                        strategy: LanguageStrategy::Locked,
                        input_languages: vec![language],
                        output_language: None,
                    }),
                    recorder_backend: Some(backend.into()),
                    translate_to_english,
                    keep_audio,
                    segmentation: SegmentationMode::default(),
                };
                let session: voicelayer_core::CaptureSession =
                    uds_post_json(&cli_socket_path(), "/v1/sessions/dictation", &request).await?;
                println!("{}", serde_json::to_string_pretty(&session)?);
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
        },
        Command::Hotkeys { command } => match command {
            HotkeysCommand::PortalStatus => {
                let status = probe_global_shortcuts_portal().await;
                println!("{}", serde_json::to_string_pretty(&status)?);
            }
        },
        Command::Doctor => {
            let socket_path = default_socket_path();
            let project_root = default_project_root();
            let config = DaemonConfig::with_project_root(socket_path.clone(), project_root.clone());
            let portal = probe_global_shortcuts_portal().await;
            let (
                worker_status,
                worker_error,
                asr_configured,
                asr_binary,
                asr_model_path,
                asr_error,
                llm_configured,
                llm_model,
                llm_endpoint,
                llm_reachable,
                llm_error,
            ) = match config.worker_command.health().await {
                Ok(WorkerHealthResult {
                    status,
                    asr_configured,
                    asr_binary,
                    asr_model_path,
                    asr_error,
                    llm_configured,
                    llm_model,
                    llm_endpoint,
                    llm_reachable,
                    llm_error,
                    ..
                }) => (
                    status,
                    None,
                    asr_configured,
                    asr_binary,
                    asr_model_path,
                    asr_error,
                    llm_configured,
                    llm_model,
                    llm_endpoint,
                    llm_reachable,
                    llm_error,
                ),
                Err(error) => (
                    "unavailable".to_owned(),
                    Some(error.to_string()),
                    false,
                    None,
                    None,
                    None,
                    false,
                    None,
                    None,
                    false,
                    None,
                ),
            };

            let report = DoctorReport {
                socket_path: socket_path.display().to_string(),
                project_root: project_root.display().to_string(),
                worker_command: config.worker_command.display(),
                worker_status,
                worker_error,
                asr_configured,
                asr_binary,
                asr_model_path,
                asr_error,
                whisper_mode: detect_whisper_mode(),
                llm_configured,
                llm_model,
                llm_endpoint,
                llm_reachable,
                llm_error,
                llama_autostart_active: llama_autostart_active(),
                global_shortcuts_portal_available: portal.available,
                global_shortcuts_portal_version: portal.version,
                global_shortcuts_portal_error: portal.error,
                recorder: recorder_diagnostics(RecorderBackend::Auto),
                systemd_unit_installed: systemd_unit_installed(),
                systemd_unit_active: systemd_unit_active(),
                wayland_display: std::env::var("WAYLAND_DISPLAY").ok(),
                x11_display: std::env::var("DISPLAY").ok(),
                dbus_session_bus_address: std::env::var("DBUS_SESSION_BUS_ADDRESS").ok(),
                xdg_runtime_dir: std::env::var("XDG_RUNTIME_DIR").ok(),
            };
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Providers => {
            let config =
                DaemonConfig::with_project_root(default_socket_path(), default_project_root());
            let mut providers = voicelayer_core::default_host_adapter_catalog();
            if let Ok(worker_providers) = config.worker_command.list_providers().await {
                providers.extend(worker_providers.providers);
            }
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
            backend,
            keep_audio,
        } => {
            let config =
                DaemonConfig::with_project_root(default_socket_path(), default_project_root());
            let result = capture_dictation_once(
                &config,
                DictationCaptureRequest {
                    trigger: TriggerKind::Cli,
                    language_profile: language.map(|language| LanguageProfile {
                        strategy: LanguageStrategy::Locked,
                        input_languages: vec![language],
                        output_language: None,
                    }),
                    duration_seconds,
                    recorder_backend: Some(backend.into()),
                    translate_to_english,
                    keep_audio,
                },
            )
            .await;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::TranscribeFile {
            audio_file,
            language,
            translate_to_english,
        } => {
            let config =
                DaemonConfig::with_project_root(default_socket_path(), default_project_root());
            let result = config
                .worker_command
                .transcribe(&TranscribeRequest {
                    audio_file,
                    language,
                    translate_to_english,
                })
                .await?;
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::Preview { command } => match command {
            PreviewCommand::Compose {
                spoken_prompt,
                archetype,
                output_language,
            } => {
                let config =
                    DaemonConfig::with_project_root(default_socket_path(), default_project_root());
                let receipt = match config
                    .worker_command
                    .compose(&ComposeRequest {
                        spoken_prompt,
                        archetype: archetype.map(Into::into),
                        output_language,
                    })
                    .await
                {
                    Ok(preview) => ready_receipt(preview),
                    Err(error) => worker_error_receipt(
                        SessionMode::Compose,
                        "Composition preview is waiting for a configured provider.",
                        error.to_string(),
                    ),
                };
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            PreviewCommand::Rewrite {
                source_text,
                style,
                output_language,
            } => {
                let config =
                    DaemonConfig::with_project_root(default_socket_path(), default_project_root());
                let receipt = match config
                    .worker_command
                    .rewrite(&RewriteRequest {
                        source_text,
                        style: style.into(),
                        output_language,
                    })
                    .await
                {
                    Ok(preview) => ready_receipt(preview),
                    Err(error) => worker_error_receipt(
                        SessionMode::Rewrite,
                        "Rewrite preview is waiting for a configured provider.",
                        error.to_string(),
                    ),
                };
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
            PreviewCommand::Translate {
                source_text,
                target_language,
            } => {
                let config =
                    DaemonConfig::with_project_root(default_socket_path(), default_project_root());
                let receipt = match config
                    .worker_command
                    .translate(&TranslateRequest {
                        source_text,
                        target_language: target_language.clone(),
                    })
                    .await
                {
                    Ok(preview) => ready_receipt(preview),
                    Err(error) => worker_error_receipt(
                        SessionMode::Translate,
                        &format!(
                            "Translation preview is waiting for a provider for {target_language}."
                        ),
                        error.to_string(),
                    ),
                };
                println!("{}", serde_json::to_string_pretty(&receipt)?);
            }
        },
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{detect_whisper_mode, env_is_set, env_is_truthy, pid_matches_command};

    /// Serializes process-env mutation across every test in this module.
    /// Cargo runs unit tests concurrently by default, and Rust 2024 flagged
    /// `std::env::set_var`/`remove_var` as `unsafe` precisely because a
    /// concurrent reader in another thread is UB. Any future test here
    /// that touches a `VOICELAYER_*` variable must take this lock.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    fn reset_var(key: &str) {
        // SAFETY: ENV_LOCK is held for the duration of the mutation.
        unsafe {
            std::env::remove_var(key);
        }
    }

    fn set_var(key: &str, value: &str) {
        // SAFETY: ENV_LOCK is held for the duration of the mutation.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    #[test]
    fn env_is_set_rejects_unset_and_empty() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        reset_var("VL_TEST_ENV_SET");
        assert!(!env_is_set("VL_TEST_ENV_SET"));
        set_var("VL_TEST_ENV_SET", "");
        assert!(!env_is_set("VL_TEST_ENV_SET"));
        set_var("VL_TEST_ENV_SET", "/abs/path");
        assert!(env_is_set("VL_TEST_ENV_SET"));
        reset_var("VL_TEST_ENV_SET");
    }

    #[test]
    fn env_is_truthy_accepts_canonical_forms() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for value in ["1", "true", "TRUE", " Yes ", "on", "YES"] {
            set_var("VL_TEST_ENV_TRUTHY", value);
            assert!(
                env_is_truthy("VL_TEST_ENV_TRUTHY"),
                "{value:?} should be truthy",
            );
        }
        for value in ["0", "false", "no", "off", "", "anything-else"] {
            set_var("VL_TEST_ENV_TRUTHY", value);
            assert!(
                !env_is_truthy("VL_TEST_ENV_TRUTHY"),
                "{value:?} should be falsy",
            );
        }
        reset_var("VL_TEST_ENV_TRUTHY");
    }

    #[test]
    fn detect_whisper_mode_prefers_server_over_cli() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        for key in [
            "VOICELAYER_WHISPER_SERVER_HOST",
            "VOICELAYER_WHISPER_SERVER_PORT",
            "VOICELAYER_WHISPER_SERVER_BIN",
            "VOICELAYER_WHISPER_SERVER_AUTO_START",
            "VOICELAYER_WHISPER_MODEL_PATH",
        ] {
            reset_var(key);
        }

        assert_eq!(detect_whisper_mode(), "unconfigured");

        set_var("VOICELAYER_WHISPER_MODEL_PATH", "/m.bin");
        assert_eq!(detect_whisper_mode(), "cli");

        set_var("VOICELAYER_WHISPER_SERVER_HOST", "127.0.0.1");
        assert_eq!(detect_whisper_mode(), "server");

        reset_var("VOICELAYER_WHISPER_SERVER_HOST");
        set_var("VOICELAYER_WHISPER_SERVER_AUTO_START", "true");
        assert_eq!(detect_whisper_mode(), "server");

        for key in [
            "VOICELAYER_WHISPER_SERVER_HOST",
            "VOICELAYER_WHISPER_SERVER_PORT",
            "VOICELAYER_WHISPER_SERVER_BIN",
            "VOICELAYER_WHISPER_SERVER_AUTO_START",
            "VOICELAYER_WHISPER_MODEL_PATH",
        ] {
            reset_var(key);
        }
    }

    #[test]
    fn pid_matches_command_recognizes_self_argv() {
        // argv[0] of the running test binary should always match /proc/self
        // regardless of absolute-path-vs-basename form.
        let my_pid = std::process::id() as i32;
        let my_exe = std::env::current_exe().expect("current_exe should resolve");
        let exe_name = my_exe.file_name().and_then(|n| n.to_str()).unwrap();
        assert!(pid_matches_command(my_pid, exe_name));
        assert!(pid_matches_command(my_pid, my_exe.to_str().unwrap()));
    }

    #[test]
    fn pid_matches_command_rejects_dead_and_zero_pids() {
        assert!(!pid_matches_command(0, "anything"));
        assert!(!pid_matches_command(-1, "anything"));
        // PID 1 (init/systemd) should not match a name we never ran.
        assert!(!pid_matches_command(1, "definitely-not-a-real-binary"));
    }
}
