use std::fs;
use std::path::{Path, PathBuf};
use std::process::Command as ProcessCommand;

use clap::{Parser, Subcommand, ValueEnum};
use serde::Serialize;
use voicelayer_core::{
    ComposeRequest, CompositionArchetype, DictationCaptureRequest, HealthResponse, InjectRequest,
    InjectTarget, InjectionPlan, LanguageProfile, LanguageStrategy, RecorderBackend,
    RewriteRequest, RewriteStyle, SessionMode, StartDictationRequest, StopDictationRequest,
    TranscribeRequest, TranslateRequest, TriggerKind,
};
use voicelayerd::{
    DaemonConfig, RecorderDiagnostics, WorkerHealthResult, capture_dictation_once,
    default_project_root, default_socket_path, probe_global_shortcuts_portal, recorder_diagnostics,
    run_daemon,
};

use crate::config::{
    CliPttKey, CliRecorderBackend, CliSegmentationMode, StopAction, VlConfig,
    build_segmentation_mode, load_vl_config, set_config_value, vl_config_path, write_vl_config,
};
use crate::foreground_ptt::run_foreground_ptt;
use crate::preview::{ready_receipt, worker_error_receipt};
use crate::uds::{DaemonOutcome, cli_socket_path, try_daemon_post, uds_get_json, uds_post_json};

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
        #[arg(long, default_value_t = 0)]
        overlap_secs: u32,
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

#[derive(Debug, Clone, Copy, Serialize)]
#[serde(rename_all = "snake_case")]
enum DaemonHealthSource {
    Daemon,
    LocalWorker,
}

#[derive(Debug, Serialize)]
struct DoctorReport {
    socket_path: String,
    project_root: String,
    worker_command: String,
    daemon_reachable: bool,
    daemon_health_source: DaemonHealthSource,
    worker_status: String,
    worker_error: Option<String>,
    asr_configured: bool,
    asr_binary: Option<String>,
    asr_model_path: Option<String>,
    asr_error: Option<String>,
    whisper_mode: String,
    whisper_server_url: Option<String>,
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
    whisper_server_unit_installed: bool,
    whisper_server_unit_active: bool,
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

/// Build the canonical path the install script lays a user unit at.
/// `unit_file_name` is the full filename (e.g. `"voicelayerd.service"`),
/// not the unit name passed to systemctl. Returns `None` when the
/// environment lacks both `XDG_CONFIG_HOME` and `HOME`, which would
/// only happen in a degenerate sandbox where no install location can
/// be inferred.
fn systemd_user_unit_path(unit_file_name: &str) -> Option<PathBuf> {
    let config_dir = std::env::var_os("XDG_CONFIG_HOME")
        .map(PathBuf::from)
        .or_else(|| std::env::var_os("HOME").map(|home| PathBuf::from(home).join(".config")))?;
    Some(config_dir.join("systemd/user").join(unit_file_name))
}

fn systemd_unit_installed(unit_file_name: &str) -> bool {
    systemd_user_unit_path(unit_file_name)
        .map(|path| path.is_file())
        .unwrap_or(false)
}

/// Probe whether a user-level systemd unit is currently active by
/// shelling out to `systemctl --user is-active`. `unit_name` is the
/// systemd unit name without the `.service` suffix
/// (`"voicelayerd"`, `"voicelayer-whisper-server"`).
fn systemd_unit_active(unit_name: &str) -> bool {
    ProcessCommand::new("systemctl")
        .args(["--user", "is-active", unit_name])
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
                mode,
                segment_secs,
                overlap_secs,
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
                duration_seconds,
            } => {
                let segmentation = build_segmentation_mode(
                    mode,
                    segment_secs,
                    overlap_secs,
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
                    recorder_backend: Some(backend.into()),
                    translate_to_english,
                    keep_audio,
                    segmentation,
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

            // Prefer the live daemon's `/v1/health` so doctor sees the same
            // env the daemon was started with (e.g. a systemd
            // EnvironmentFile). Fall back to spawning a one-off worker only
            // when the daemon socket is unreachable — doctor should still
            // work for pre-enable dry runs.
            let daemon_health: Option<HealthResponse> =
                uds_get_json(&socket_path, "/v1/health").await.ok();
            let (
                daemon_reachable,
                daemon_health_source,
                worker_status,
                worker_error,
                asr_configured,
                asr_binary,
                asr_model_path,
                asr_error,
                health_whisper_mode,
                health_whisper_server_url,
                llm_configured,
                llm_model,
                llm_endpoint,
                llm_reachable,
                llm_error,
            ) = match daemon_health {
                Some(response) => {
                    let worker = response.worker;
                    (
                        true,
                        DaemonHealthSource::Daemon,
                        worker.status,
                        // The daemon writes the underlying `worker_command.health()`
                        // failure into `worker.message` when it marks the worker
                        // "unavailable". Forward it so `vl doctor` surfaces the
                        // actionable reason (Python import error, missing uv,
                        // crashed subprocess, ...) instead of reporting an
                        // unavailable worker with no explanation.
                        worker.message,
                        worker.asr_configured,
                        worker.asr_binary,
                        worker.asr_model_path,
                        worker.asr_error,
                        worker.whisper_mode,
                        worker.whisper_server_url,
                        worker.llm_configured,
                        worker.llm_model,
                        worker.llm_endpoint,
                        worker.llm_reachable,
                        worker.llm_error,
                    )
                }
                None => match config.worker_command.health().await {
                    Ok(WorkerHealthResult {
                        status,
                        asr_configured,
                        asr_binary,
                        asr_model_path,
                        asr_error,
                        whisper_mode,
                        whisper_server_url,
                        llm_configured,
                        llm_model,
                        llm_endpoint,
                        llm_reachable,
                        llm_error,
                        ..
                    }) => (
                        false,
                        DaemonHealthSource::LocalWorker,
                        status,
                        None,
                        asr_configured,
                        asr_binary,
                        asr_model_path,
                        asr_error,
                        whisper_mode,
                        whisper_server_url,
                        llm_configured,
                        llm_model,
                        llm_endpoint,
                        llm_reachable,
                        llm_error,
                    ),
                    Err(error) => (
                        false,
                        DaemonHealthSource::LocalWorker,
                        "unavailable".to_owned(),
                        Some(error.to_string()),
                        false,
                        None,
                        None,
                        None,
                        None,
                        None,
                        false,
                        None,
                        None,
                        false,
                        None,
                    ),
                },
            };

            // Prefer the daemon's (or local worker's) reported whisper_mode
            // over sniffing the CLI's own environment — the daemon sees the
            // systemd EnvironmentFile and knows which path `transcribe` will
            // actually dispatch to. Fall back to local env detection only
            // when the worker returned no mode (older daemon / worker).
            let whisper_mode =
                health_whisper_mode.unwrap_or_else(|| detect_whisper_mode().to_owned());

            let report = DoctorReport {
                socket_path: socket_path.display().to_string(),
                project_root: project_root.display().to_string(),
                worker_command: config.worker_command.display(),
                daemon_reachable,
                daemon_health_source,
                worker_status,
                worker_error,
                asr_configured,
                asr_binary,
                asr_model_path,
                asr_error,
                whisper_mode,
                whisper_server_url: health_whisper_server_url,
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
                systemd_unit_installed: systemd_unit_installed("voicelayerd.service"),
                systemd_unit_active: systemd_unit_active("voicelayerd"),
                whisper_server_unit_installed: systemd_unit_installed(
                    "voicelayer-whisper-server.service",
                ),
                whisper_server_unit_active: systemd_unit_active("voicelayer-whisper-server"),
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
            let request = DictationCaptureRequest {
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
            };
            // Prefer the running daemon so recording + transcription
            // inherit its environment (systemd EnvironmentFile) and the
            // capture shows up on /v1/events/stream alongside anything
            // else an operator is watching. Falling back to the
            // in-process `capture_dictation_once` when no daemon is
            // listening keeps this command usable for scripting before
            // the unit is enabled. Missing socket → silent fallback;
            // existing socket that rejects the request → stderr warning,
            // same split as TranscribeFile.
            let socket_path = cli_socket_path();
            let outcome: DaemonOutcome<voicelayer_core::DictationCaptureResult> =
                try_daemon_post(&socket_path, "/v1/dictation/capture", &request).await;
            if let DaemonOutcome::Rejected(error) = &outcome {
                eprintln!(
                    "warning: daemon at {} rejected /v1/dictation/capture ({error}); \
                     falling back to in-process capture",
                    socket_path.display()
                );
            }
            let result = match outcome {
                DaemonOutcome::Ok(result) => result,
                _ => {
                    let config =
                        DaemonConfig::with_project_root(socket_path, default_project_root());
                    capture_dictation_once(&config, request).await
                }
            };
            println!("{}", serde_json::to_string_pretty(&result)?);
        }
        Command::TranscribeFile {
            audio_file,
            language,
            translate_to_english,
        } => {
            let request = TranscribeRequest {
                audio_file,
                language,
                translate_to_english,
            };
            // Prefer the running daemon so the request inherits its (e.g.
            // systemd-provided) environment. Fall back to spawning a
            // one-off local worker when no daemon is listening — useful
            // for pre-enable scripting, but the caller's shell must then
            // export VOICELAYER_WHISPER_* itself.
            //
            // Missing socket is a silent fallback; an existing socket
            // that rejects the request surfaces a warning. Without the
            // split, a broken daemon would silently route to the local
            // worker and the underlying error would be hidden.
            let socket_path = cli_socket_path();
            let outcome: DaemonOutcome<voicelayer_core::TranscriptionResult> =
                try_daemon_post(&socket_path, "/v1/transcriptions", &request).await;
            if let DaemonOutcome::Rejected(error) = &outcome {
                eprintln!(
                    "warning: daemon at {} rejected /v1/transcriptions ({error}); \
                     falling back to a local worker",
                    socket_path.display()
                );
            }
            let result: voicelayer_core::TranscriptionResult = match outcome {
                DaemonOutcome::Ok(result) => result,
                _ => {
                    let config =
                        DaemonConfig::with_project_root(socket_path, default_project_root());
                    config.worker_command.transcribe(&request).await?
                }
            };
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

    use super::{
        detect_whisper_mode, env_is_set, env_is_truthy, pid_matches_command,
        systemd_unit_installed, systemd_user_unit_path,
    };

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

    /// Pin clap's `required_if_eq` and range value-parser wiring on the
    /// `vl dictation start` command. These tests catch the failure modes
    /// that the runtime `build_segmentation_mode` `#[should_panic]`
    /// checks would otherwise only catch *after* the binary has parsed
    /// args — here we stop the regression at parse time, where the user
    /// sees the error.
    /// Helper that snapshots-and-clears two env vars under `ENV_LOCK`,
    /// runs `body`, and restores the originals. Avoids the boilerplate
    /// of preserving / restoring `XDG_CONFIG_HOME` and `HOME` in every
    /// test that needs to drive `systemd_user_unit_path` through one of
    /// its branches.
    fn with_clean_xdg_and_home<R>(body: impl FnOnce() -> R) -> R {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let prev_xdg = std::env::var_os("XDG_CONFIG_HOME");
        let prev_home = std::env::var_os("HOME");
        reset_var("XDG_CONFIG_HOME");
        reset_var("HOME");
        let result = body();
        match prev_xdg {
            Some(value) => set_var("XDG_CONFIG_HOME", value.to_str().unwrap_or_default()),
            None => reset_var("XDG_CONFIG_HOME"),
        }
        match prev_home {
            Some(value) => set_var("HOME", value.to_str().unwrap_or_default()),
            None => reset_var("HOME"),
        }
        result
    }

    /// Pins the XDG branch. Doctor's `systemd_unit_installed` field
    /// drives operator-facing diagnostics ("did the install script
    /// land the unit?"); a regression that ignored `XDG_CONFIG_HOME`
    /// would silently misreport on hosts that override it.
    #[test]
    fn systemd_user_unit_path_uses_xdg_config_home_when_set() {
        with_clean_xdg_and_home(|| {
            set_var("XDG_CONFIG_HOME", "/srv/test-xdg");
            let path =
                systemd_user_unit_path("voicelayerd.service").expect("XDG set must yield a path");
            assert_eq!(
                path,
                std::path::PathBuf::from("/srv/test-xdg/systemd/user/voicelayerd.service"),
            );
        });
    }

    /// Pins the HOME-fallback branch. The install script targets
    /// `~/.config/systemd/user/` regardless of XDG; this asserts the
    /// helper agrees so doctor cannot disagree with the installer.
    #[test]
    fn systemd_user_unit_path_falls_back_to_home_dot_config_when_xdg_unset() {
        with_clean_xdg_and_home(|| {
            set_var("HOME", "/home/test-user");
            let path = systemd_user_unit_path("voicelayer-whisper-server.service")
                .expect("HOME set must yield a path");
            assert_eq!(
                path,
                std::path::PathBuf::from(
                    "/home/test-user/.config/systemd/user/voicelayer-whisper-server.service",
                ),
            );
        });
    }

    /// Pins the no-anchor case. A degenerate sandbox (no XDG, no HOME)
    /// must yield `None` so doctor reports the unit as not installed
    /// instead of crashing or guessing a path that does not exist.
    #[test]
    fn systemd_user_unit_path_returns_none_when_xdg_and_home_both_unset() {
        with_clean_xdg_and_home(|| {
            assert!(systemd_user_unit_path("voicelayerd.service").is_none());
        });
    }

    /// Pins the affirmative branch of `systemd_unit_installed` and
    /// proves the unit-name parameter is honoured: the same XDG
    /// directory holds `voicelayerd.service` but not
    /// `voicelayer-whisper-server.service`, and the helper must
    /// report exactly one as installed.
    #[test]
    fn systemd_unit_installed_distinguishes_present_and_absent_unit_files() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let unit_dir = tempdir.path().join("systemd/user");
        std::fs::create_dir_all(&unit_dir).expect("create unit dir");
        std::fs::write(unit_dir.join("voicelayerd.service"), b"")
            .expect("touch voicelayerd.service");

        with_clean_xdg_and_home(|| {
            set_var("XDG_CONFIG_HOME", tempdir.path().to_str().expect("utf-8"));
            assert!(
                systemd_unit_installed("voicelayerd.service"),
                "voicelayerd.service exists in the XDG dir; helper must report installed",
            );
            assert!(
                !systemd_unit_installed("voicelayer-whisper-server.service"),
                "whisper-server unit absent; helper must report not installed even though \
                 the sibling unit is present in the same directory",
            );
        });
    }

    mod dictation_start_parsing {
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
    }
}
