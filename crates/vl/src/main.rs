use std::{io::Write, path::PathBuf};

use bytes::Bytes;
use clap::{Parser, Subcommand, ValueEnum};
use crossterm::{
    cursor::{Hide, MoveTo, Show},
    event::{
        Event, KeyCode, KeyEventKind, KeyboardEnhancementFlags, PopKeyboardEnhancementFlags,
        PushKeyboardEnhancementFlags, read,
    },
    execute,
    style::Print,
    terminal::{
        Clear, ClearType, EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode,
        enable_raw_mode, size,
    },
};
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use tokio::net::UnixStream;
use voicelayer_core::{
    ComposeRequest, CompositionArchetype, CompositionReceipt, DictationCaptureRequest,
    InjectRequest, InjectTarget, InjectionPlan, LanguageProfile, LanguageStrategy, PreviewArtifact,
    PreviewStatus, RecorderBackend, RewriteRequest, RewriteStyle, SessionMode, SessionState,
    StartDictationRequest, StopDictationRequest, TranscribeRequest, TranslateRequest, TriggerKind,
};
use voicelayerd::{
    DaemonConfig, RecorderDiagnostics, WorkerHealthResult, WorkerPreviewPayload,
    capture_dictation_once, default_project_root, default_socket_path,
    probe_global_shortcuts_portal, recorder_diagnostics, run_daemon,
};

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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, serde::Deserialize)]
enum CliRecorderBackend {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, serde::Deserialize)]
enum CliPttKey {
    Space,
    Enter,
    Tab,
    F8,
    F9,
    F10,
}

impl CliPttKey {
    fn as_key_code(self) -> KeyCode {
        match self {
            CliPttKey::Space => KeyCode::Char(' '),
            CliPttKey::Enter => KeyCode::Enter,
            CliPttKey::Tab => KeyCode::Tab,
            CliPttKey::F8 => KeyCode::F(8),
            CliPttKey::F9 => KeyCode::F(9),
            CliPttKey::F10 => KeyCode::F(10),
        }
    }

    fn label(self) -> &'static str {
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

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum, Serialize, serde::Deserialize)]
enum StopAction {
    None,
    Copy,
    Inject,
    Save,
}

#[derive(Debug, Clone, Serialize, serde::Deserialize)]
struct ForegroundPttConfig {
    language: Option<String>,
    backend: CliRecorderBackend,
    translate_to_english: bool,
    keep_audio: bool,
    key: CliPttKey,
    tmux_target_pane: Option<String>,
    wezterm_target_pane_id: Option<String>,
    kitty_match: Option<String>,
    copy_on_stop: bool,
    default_stop_action: StopAction,
    restore_clipboard_on_exit: bool,
    save_dir: Option<PathBuf>,
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

#[derive(Debug, Clone, Serialize, serde::Deserialize, Default)]
struct VlConfig {
    #[serde(default)]
    foreground_ptt: ForegroundPttConfig,
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
    llm_configured: bool,
    llm_model: Option<String>,
    llm_endpoint: Option<String>,
    llm_reachable: bool,
    llm_error: Option<String>,
    global_shortcuts_portal_available: bool,
    global_shortcuts_portal_version: Option<u32>,
    global_shortcuts_portal_error: Option<String>,
    recorder: RecorderDiagnostics,
    wayland_display: Option<String>,
    x11_display: Option<String>,
    dbus_session_bus_address: Option<String>,
    xdg_runtime_dir: Option<String>,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
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
                llm_configured,
                llm_model,
                llm_endpoint,
                llm_reachable,
                llm_error,
                global_shortcuts_portal_available: portal.available,
                global_shortcuts_portal_version: portal.version,
                global_shortcuts_portal_error: portal.error,
                recorder: recorder_diagnostics(RecorderBackend::Auto),
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

fn ready_receipt(preview: WorkerPreviewPayload) -> CompositionReceipt {
    CompositionReceipt {
        job_id: uuid::Uuid::new_v4(),
        preview: PreviewArtifact {
            artifact_id: uuid::Uuid::new_v4(),
            status: PreviewStatus::Ready,
            title: preview.title,
            generated_text: Some(preview.generated_text),
            notes: preview.notes,
        },
    }
}

fn worker_error_receipt(mode: SessionMode, title: &str, error: String) -> CompositionReceipt {
    let mut receipt = CompositionReceipt::needs_provider(title, mode);
    receipt
        .preview
        .notes
        .push(format!("Worker bridge detail: {error}"));
    receipt
}

async fn uds_post_json<TRequest, TResponse>(
    socket_path: &std::path::Path,
    path: &str,
    payload: &TRequest,
) -> Result<TResponse, Box<dyn std::error::Error>>
where
    TRequest: serde::Serialize,
    TResponse: serde::de::DeserializeOwned,
{
    let body = serde_json::to_vec(payload)?;
    let stream = UnixStream::connect(socket_path).await?;
    let io = TokioIo::new(stream);
    let (mut sender, connection) = http1::handshake(io).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::post(path)
        .header(HOST, "localhost")
        .header("content-type", "application/json")
        .body(Full::new(Bytes::from(body)))?;
    let response = sender.send_request(request).await?;
    let status = response.status();
    let response_bytes = response.into_body().collect().await?.to_bytes();
    if !status.is_success() {
        return Err(format!(
            "daemon returned non-success status {}: {}",
            status,
            String::from_utf8_lossy(&response_bytes)
        )
        .into());
    }
    Ok(serde_json::from_slice(&response_bytes)?)
}

fn cli_socket_path() -> std::path::PathBuf {
    std::env::var_os("VOICELAYER_SOCKET_PATH")
        .map(std::path::PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

fn vl_config_path() -> Result<PathBuf, Box<dyn std::error::Error>> {
    let project_dirs = directories::ProjectDirs::from("com", "memenow", "voicelayer")
        .ok_or("Unable to determine the platform config directory for VoiceLayer")?;
    Ok(project_dirs.config_dir().join("config.toml"))
}

fn load_vl_config() -> Result<VlConfig, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    if !path.is_file() {
        return Ok(VlConfig::default());
    }
    let contents = std::fs::read_to_string(path)?;
    Ok(toml::from_str(&contents)?)
}

fn write_vl_config(config: &VlConfig) -> Result<PathBuf, Box<dyn std::error::Error>> {
    let path = vl_config_path()?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    std::fs::write(&path, toml::to_string_pretty(config)?)?;
    Ok(path)
}

fn set_config_value(
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
            config.foreground_ptt.save_dir = if value.eq_ignore_ascii_case("none") {
                None
            } else {
                Some(PathBuf::from(value))
            };
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
        _ => {
            return Err(format!(
                "unsupported config key `{key}`; supported keys: foreground_ptt.default_stop_action, foreground_ptt.copy_on_stop, foreground_ptt.restore_clipboard_on_exit, foreground_ptt.save_dir, foreground_ptt.backend, foreground_ptt.key"
            )
            .into())
        }
    }
    Ok(())
}

fn parse_bool(value: &str) -> Result<bool, Box<dyn std::error::Error>> {
    match value.to_ascii_lowercase().as_str() {
        "true" | "1" | "yes" | "on" => Ok(true),
        "false" | "0" | "no" | "off" => Ok(false),
        _ => Err("expected a boolean value: true/false".into()),
    }
}

async fn run_foreground_ptt(
    language: Option<String>,
    backend: CliRecorderBackend,
    translate_to_english: bool,
    keep_audio: bool,
    key: CliPttKey,
    tmux_target_pane: Option<String>,
    wezterm_target_pane_id: Option<String>,
    kitty_match: Option<String>,
    copy_on_stop: bool,
    default_stop_action: StopAction,
    restore_clipboard_on_exit: bool,
    save_dir: Option<PathBuf>,
) -> Result<(), Box<dyn std::error::Error>> {
    let foreground_target = resolve_foreground_injection_target(
        tmux_target_pane.clone(),
        wezterm_target_pane_id,
        kitty_match,
    )?;
    let tmux_target_pane = resolve_tmux_target_pane(tmux_target_pane)?;
    let _raw_mode = RawTerminalMode::enter()?;
    let mut ui = ForegroundPttUiState::new(
        key.label().to_owned(),
        describe_foreground_target(&foreground_target, tmux_target_pane.as_deref()),
        default_stop_action,
        save_dir,
        restore_clipboard_on_exit,
    );
    ui.push_event(format!(
        "Press {} to start. Release to stop when supported; otherwise press {} again. Esc exits.",
        key.label(),
        key.label()
    ));
    render_foreground_ptt_ui(&ui)?;
    let mut active_session: Option<uuid::Uuid> = None;

    loop {
        let event = read()?;
        let Event::Key(key_event) = event else {
            continue;
        };

        if key_event.code == KeyCode::Esc {
            if let Some(session_id) = active_session.take() {
                let result: voicelayer_core::DictationCaptureResult = uds_post_json(
                    &cli_socket_path(),
                    "/v1/sessions/dictation/stop",
                    &StopDictationRequest { session_id },
                )
                .await?;
                apply_dictation_result_to_ui(
                    &mut ui,
                    &result,
                    &foreground_target,
                    tmux_target_pane.as_deref(),
                    copy_on_stop,
                );
                ui.push_event("Stopped active session before exit.".to_owned());
                render_foreground_ptt_ui(&ui)?;
            }
            if ui.restore_clipboard_on_exit {
                let _ = restore_clipboard_backup(&mut ui);
            }
            ui.push_event("Foreground PTT exited.".to_owned());
            break;
        }

        if key_event.code != key.as_key_code() {
            if key_event.kind == KeyEventKind::Press {
                match key_event.code {
                    KeyCode::Char('c') => {
                        if let Some(text) = ui.last_transcript_full.clone() {
                            if ui.clipboard_backup_text.is_none() {
                                ui.clipboard_backup_text = snapshot_clipboard_text();
                            }
                            match copy_transcript_to_clipboard(&text) {
                                Ok(()) => {
                                    ui.last_injection_status = Some(
                                        "Copied last transcript to system clipboard.".to_owned(),
                                    );
                                    ui.last_error = None;
                                    ui.push_event(
                                        "Copied last transcript to clipboard.".to_owned(),
                                    );
                                }
                                Err(error) => {
                                    ui.last_error = Some(format!("Clipboard copy failed: {error}"));
                                }
                            }
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('r') => {
                        match restore_clipboard_backup(&mut ui) {
                            Ok(()) => {
                                ui.push_event("Restored clipboard from saved backup.".to_owned());
                            }
                            Err(error) => {
                                ui.last_error = Some(error);
                            }
                        }
                        render_foreground_ptt_ui(&ui)?;
                    }
                    KeyCode::Char('i') => {
                        if let Some(result) = ui.last_result.clone() {
                            apply_dictation_result_to_ui(
                                &mut ui,
                                &result,
                                &foreground_target,
                                tmux_target_pane.as_deref(),
                                false,
                            );
                            ui.push_event("Re-applied last transcript injection.".to_owned());
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('s') => {
                        if let Some(text) = ui.last_transcript_full.clone() {
                            match save_transcript_to_file(&text) {
                                Ok(path) => {
                                    ui.last_injection_status =
                                        Some(format!("Saved transcript to {}.", path.display()));
                                    ui.last_error = None;
                                    ui.push_event(format!(
                                        "Saved last transcript to {}.",
                                        path.display()
                                    ));
                                }
                                Err(error) => {
                                    ui.last_error =
                                        Some(format!("Failed to save transcript: {error}"));
                                }
                            }
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('d') => {
                        ui.last_result = None;
                        ui.last_session_id = None;
                        ui.last_transcript_full = None;
                        ui.last_transcript_preview = None;
                        ui.transcript_scroll = 0;
                        ui.last_injection_status = None;
                        ui.last_error = None;
                        ui.push_event("Discarded last transcript from the panel.".to_owned());
                        render_foreground_ptt_ui(&ui)?;
                    }
                    KeyCode::Char('j') | KeyCode::Down => {
                        if scroll_transcript(&mut ui, 1) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::Char('k') | KeyCode::Up => {
                        if scroll_transcript(&mut ui, -1) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::PageDown => {
                        if scroll_transcript(&mut ui, 6) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    KeyCode::PageUp => {
                        if scroll_transcript(&mut ui, -6) {
                            render_foreground_ptt_ui(&ui)?;
                        }
                    }
                    _ => {}
                }
            }
            continue;
        }

        match key_event.kind {
            KeyEventKind::Press => {
                if active_session.is_none() {
                    let request = StartDictationRequest {
                        trigger: TriggerKind::Cli,
                        language_profile: language.clone().map(|language| LanguageProfile {
                            strategy: LanguageStrategy::Locked,
                            input_languages: vec![language],
                            output_language: None,
                        }),
                        recorder_backend: Some(backend.into()),
                        translate_to_english,
                        keep_audio,
                    };
                    let session: voicelayer_core::CaptureSession =
                        uds_post_json(&cli_socket_path(), "/v1/sessions/dictation", &request)
                            .await?;
                    active_session = Some(session.session_id);
                    ui.status_label = "Listening".to_owned();
                    ui.active_session_id = Some(session.session_id.to_string());
                    ui.last_error = None;
                    ui.push_event(format!("Dictation started: {}", session.session_id));
                    render_foreground_ptt_ui(&ui)?;
                } else if let Some(session_id) = active_session.take() {
                    ui.status_label = "Transcribing".to_owned();
                    ui.push_event(format!(
                        "Stopping session {} via key press toggle.",
                        session_id
                    ));
                    render_foreground_ptt_ui(&ui)?;
                    let result: voicelayer_core::DictationCaptureResult = uds_post_json(
                        &cli_socket_path(),
                        "/v1/sessions/dictation/stop",
                        &StopDictationRequest { session_id },
                    )
                    .await?;
                    apply_dictation_result_to_ui(
                        &mut ui,
                        &result,
                        &foreground_target,
                        tmux_target_pane.as_deref(),
                        copy_on_stop,
                    );
                    ui.push_event("Dictation stopped via key press toggle.".to_owned());
                    render_foreground_ptt_ui(&ui)?;
                }
            }
            KeyEventKind::Release => {
                if let Some(session_id) = active_session.take() {
                    ui.status_label = "Transcribing".to_owned();
                    ui.push_event(format!("Stopping session {} via key release.", session_id));
                    render_foreground_ptt_ui(&ui)?;
                    let result: voicelayer_core::DictationCaptureResult = uds_post_json(
                        &cli_socket_path(),
                        "/v1/sessions/dictation/stop",
                        &StopDictationRequest { session_id },
                    )
                    .await?;
                    apply_dictation_result_to_ui(
                        &mut ui,
                        &result,
                        &foreground_target,
                        tmux_target_pane.as_deref(),
                        copy_on_stop,
                    );
                    ui.push_event("Dictation stopped via key release.".to_owned());
                    render_foreground_ptt_ui(&ui)?;
                }
            }
            KeyEventKind::Repeat => {}
        }
    }

    Ok(())
}

struct RawTerminalMode;

impl RawTerminalMode {
    fn enter() -> Result<Self, Box<dyn std::error::Error>> {
        enable_raw_mode()?;
        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            EnterAlternateScreen,
            Hide,
            PushKeyboardEnhancementFlags(KeyboardEnhancementFlags::REPORT_EVENT_TYPES)
        );
        Ok(Self)
    }
}

impl Drop for RawTerminalMode {
    fn drop(&mut self) {
        let mut stdout = std::io::stdout();
        let _ = execute!(
            stdout,
            PopKeyboardEnhancementFlags,
            Show,
            LeaveAlternateScreen
        );
        let _ = disable_raw_mode();
    }
}

fn apply_dictation_result_to_ui(
    ui: &mut ForegroundPttUiState,
    result: &voicelayer_core::DictationCaptureResult,
    foreground_target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
    copy_on_stop: bool,
) {
    ui.active_session_id = None;
    ui.last_session_id = Some(result.session.session_id.to_string());
    ui.last_result = Some(result.clone());
    ui.status_label = match result.session.state {
        SessionState::Completed => "Completed".to_owned(),
        SessionState::Failed => "Failed".to_owned(),
        SessionState::Listening => "Listening".to_owned(),
        SessionState::Transcribing => "Transcribing".to_owned(),
        SessionState::Idle => "Idle".to_owned(),
        SessionState::Previewing => "Previewing".to_owned(),
        SessionState::AwaitingConfirmation => "AwaitingConfirmation".to_owned(),
    };
    ui.last_transcript_full = Some(result.transcription.text.clone());
    ui.last_transcript_preview = Some(truncate_for_panel(&result.transcription.text, 220));
    ui.transcript_scroll = 0;
    ui.last_error = None;
    ui.last_injection_status = None;

    let should_copy = copy_on_stop || matches!(ui.default_stop_action, StopAction::Copy);
    if should_copy {
        copy_result_to_clipboard(ui, &result.transcription.text);
    }

    if matches!(ui.default_stop_action, StopAction::Save) {
        if let Some(text) = ui.last_transcript_full.clone() {
            match save_transcript_to_target_dir(&text, ui.save_dir.as_deref()) {
                Ok(path) => {
                    ui.last_injection_status =
                        Some(format!("Saved transcript to {}.", path.display()));
                }
                Err(error) => {
                    ui.last_error = Some(format!("Failed to save transcript: {error}"));
                }
            }
        }
    }

    match foreground_target {
        ForegroundInjectionTarget::Tmux => {
            if let Some(target_pane) =
                tmux_target_pane.filter(|_| matches!(ui.default_stop_action, StopAction::Inject))
            {
                match paste_into_tmux_pane(target_pane, &result.transcription.text) {
                    Ok(()) => {
                        ui.last_injection_status = Some(format!(
                            "{} Pasted into tmux pane {target_pane}.",
                            ui.last_injection_status
                                .clone()
                                .unwrap_or_else(|| "Done.".to_owned())
                        ))
                    }
                    Err(error) => {
                        ui.last_error =
                            Some(format!("tmux pane {target_pane} injection failed: {error}"))
                    }
                }
            }
        }
        ForegroundInjectionTarget::Wezterm { pane_id } => {
            if matches!(ui.default_stop_action, StopAction::Inject) {
                match paste_into_wezterm_pane(pane_id, &result.transcription.text) {
                    Ok(()) => {
                        ui.last_injection_status = Some(format!(
                            "{} Sent to WezTerm pane {pane_id}.",
                            ui.last_injection_status
                                .clone()
                                .unwrap_or_else(|| "Done.".to_owned())
                        ))
                    }
                    Err(error) => {
                        ui.last_error =
                            Some(format!("WezTerm pane {pane_id} injection failed: {error}"))
                    }
                }
            }
        }
        ForegroundInjectionTarget::Kitty { r#match } => {
            if matches!(ui.default_stop_action, StopAction::Inject) {
                match paste_into_kitty_target(r#match, &result.transcription.text) {
                    Ok(()) => {
                        ui.last_injection_status = Some(format!(
                            "{} Sent to Kitty target {}.",
                            ui.last_injection_status
                                .clone()
                                .unwrap_or_else(|| "Done.".to_owned()),
                            r#match
                        ))
                    }
                    Err(error) => {
                        ui.last_error = Some(format!(
                            "Kitty target {} injection failed: {error}",
                            r#match
                        ))
                    }
                }
            }
        }
        ForegroundInjectionTarget::None => {
            if ui.last_injection_status.is_none() {
                ui.last_injection_status =
                    Some(if matches!(ui.default_stop_action, StopAction::Inject) {
                        "No foreground injection target configured.".to_owned()
                    } else {
                        "No default injection target configured.".to_owned()
                    })
            }
        }
    }
    if result.session.state == SessionState::Failed && ui.last_error.is_none() {
        ui.last_error = result.transcription.notes.first().cloned();
    }
}

#[derive(Debug, Clone)]
struct ForegroundPttUiState {
    key_label: String,
    target_label: String,
    status_label: String,
    default_stop_action: StopAction,
    save_dir: Option<PathBuf>,
    restore_clipboard_on_exit: bool,
    active_session_id: Option<String>,
    last_session_id: Option<String>,
    last_result: Option<voicelayer_core::DictationCaptureResult>,
    last_transcript_full: Option<String>,
    last_transcript_preview: Option<String>,
    transcript_scroll: usize,
    clipboard_backup_text: Option<String>,
    last_injection_status: Option<String>,
    last_error: Option<String>,
    recent_events: Vec<String>,
}

impl ForegroundPttUiState {
    fn new(
        key_label: String,
        target_label: String,
        default_stop_action: StopAction,
        save_dir: Option<PathBuf>,
        restore_clipboard_on_exit: bool,
    ) -> Self {
        Self {
            key_label,
            target_label,
            status_label: "Idle".to_owned(),
            default_stop_action,
            save_dir,
            restore_clipboard_on_exit,
            active_session_id: None,
            last_session_id: None,
            last_result: None,
            last_transcript_full: None,
            last_transcript_preview: None,
            transcript_scroll: 0,
            clipboard_backup_text: None,
            last_injection_status: None,
            last_error: None,
            recent_events: Vec::new(),
        }
    }

    fn push_event(&mut self, event: String) {
        self.recent_events.push(event);
        if self.recent_events.len() > 6 {
            let _ = self.recent_events.remove(0);
        }
    }
}

fn describe_foreground_target(
    target: &ForegroundInjectionTarget,
    tmux_target_pane: Option<&str>,
) -> String {
    match target {
        ForegroundInjectionTarget::Tmux => tmux_target_pane
            .map(|pane| format!("tmux pane {pane}"))
            .unwrap_or_else(|| "tmux disabled or no target pane".to_owned()),
        ForegroundInjectionTarget::Wezterm { pane_id } => format!("wezterm pane {pane_id}"),
        ForegroundInjectionTarget::Kitty { r#match } => {
            format!("kitty target {match}", match = r#match)
        }
        ForegroundInjectionTarget::None => "stdout panel only".to_owned(),
    }
}

fn truncate_for_panel(text: &str, max_chars: usize) -> String {
    let chars: Vec<char> = text.chars().collect();
    if chars.len() <= max_chars {
        return text.to_owned();
    }
    let mut truncated: String = chars.into_iter().take(max_chars).collect();
    truncated.push_str("...");
    truncated
}

fn render_foreground_ptt_ui(ui: &ForegroundPttUiState) -> Result<(), Box<dyn std::error::Error>> {
    let (width, height) = size().unwrap_or((100, 30));
    let transcript_width = width.saturating_sub(4).max(20) as usize;
    let transcript_lines = wrap_for_panel(
        ui.last_transcript_full.as_deref().unwrap_or("-"),
        transcript_width,
    );
    let transcript_height = height.saturating_sub(18).max(6) as usize;
    let max_scroll = transcript_lines.len().saturating_sub(transcript_height);
    let scroll = ui.transcript_scroll.min(max_scroll);
    let visible_lines = transcript_lines
        .iter()
        .skip(scroll)
        .take(transcript_height)
        .cloned()
        .collect::<Vec<_>>();

    let mut stdout = std::io::stdout();
    execute!(
        stdout,
        MoveTo(0, 0),
        Clear(ClearType::All),
        Print("VoiceLayer Foreground PTT\n"),
        Print("==========================\n"),
        Print(format!("Status: {}\n", ui.status_label)),
        Print(format!("Key: {}\n", ui.key_label)),
        Print(format!("Target: {}\n", ui.target_label)),
        Print(format!(
            "Active session: {}\n",
            ui.active_session_id.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last session: {}\n",
            ui.last_session_id.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last injection: {}\n",
            ui.last_injection_status.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "Last error: {}\n",
            ui.last_error.as_deref().unwrap_or("-")
        )),
        Print(format!(
            "\nTranscript view (line {} of {}):\n",
            scroll + 1,
            transcript_lines.len().max(1)
        )),
    )?;

    for line in visible_lines {
        execute!(stdout, Print(format!("  {line}\n")))?;
    }

    execute!(stdout, Print("\nRecent events:\n"),)?;

    for event in &ui.recent_events {
        execute!(stdout, Print(format!("  - {event}\n")))?;
    }

    execute!(
        stdout,
        Print("\nControls:\n"),
        Print("  press selected key to start\n"),
        Print("  release key to stop when supported\n"),
        Print("  otherwise press selected key again to stop\n"),
        Print("  j/k or Up/Down scroll transcript, PageUp/PageDown jump\n"),
        Print("  c copies the last transcript to the clipboard\n"),
        Print("  r restores the saved clipboard text backup\n"),
        Print("  i re-applies the last injection\n"),
        Print("  s saves the last transcript to a file\n"),
        Print("  d discards the last transcript from the panel\n"),
        Print("  Esc exits\n"),
    )?;
    Ok(())
}

fn wrap_for_panel(text: &str, width: usize) -> Vec<String> {
    let width = width.max(1);
    let mut lines = Vec::new();
    for raw_line in text.lines() {
        let chars: Vec<char> = raw_line.chars().collect();
        if chars.is_empty() {
            lines.push(String::new());
            continue;
        }
        for chunk in chars.chunks(width) {
            lines.push(chunk.iter().collect());
        }
    }
    if lines.is_empty() {
        lines.push(String::new());
    }
    lines
}

fn scroll_transcript(ui: &mut ForegroundPttUiState, delta: isize) -> bool {
    let text = ui.last_transcript_full.as_deref().unwrap_or("-");
    let lines = wrap_for_panel(text, 80);
    let max_scroll = lines.len().saturating_sub(6) as isize;
    let current = ui.transcript_scroll as isize;
    let next = (current + delta).clamp(0, max_scroll);
    let changed = next != current;
    ui.transcript_scroll = next as usize;
    changed
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum ForegroundInjectionTarget {
    None,
    Tmux,
    Wezterm { pane_id: String },
    Kitty { r#match: String },
}

fn paste_into_tmux_pane(target_pane: &str, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }

    if tmux_target_conflicts(target_pane, std::env::var("TMUX_PANE").ok().as_deref()) {
        return Err(
            "refusing to paste into the same tmux pane that is currently running the controller"
                .into(),
        );
    }

    let set_status = std::process::Command::new("tmux")
        .arg("set-buffer")
        .arg("--")
        .arg(text)
        .status()?;
    if !set_status.success() {
        return Err(format!("tmux set-buffer failed with status {set_status}").into());
    }

    let paste_status = std::process::Command::new("tmux")
        .arg("paste-buffer")
        .arg("-dpr")
        .arg("-t")
        .arg(target_pane)
        .status()?;
    if !paste_status.success() {
        return Err(format!("tmux paste-buffer failed with status {paste_status}").into());
    }

    Ok(())
}

fn tmux_target_conflicts(target_pane: &str, current_pane: Option<&str>) -> bool {
    current_pane == Some(target_pane)
}

fn wezterm_target_conflicts(target_pane: &str, current_pane: Option<&str>) -> bool {
    current_pane == Some(target_pane)
}

fn resolve_foreground_injection_target(
    tmux_target_pane: Option<String>,
    wezterm_target_pane_id: Option<String>,
    kitty_match: Option<String>,
) -> Result<ForegroundInjectionTarget, Box<dyn std::error::Error>> {
    let mut selected = Vec::new();
    if tmux_target_pane.is_some() {
        selected.push("tmux");
    }
    if wezterm_target_pane_id.is_some() {
        selected.push("wezterm");
    }
    if kitty_match.is_some() {
        selected.push("kitty");
    }

    if selected.len() > 1 {
        return Err(format!(
            "choose only one foreground injection target, got: {}",
            selected.join(", ")
        )
        .into());
    }

    if let Some(pane_id) = wezterm_target_pane_id {
        return Ok(ForegroundInjectionTarget::Wezterm { pane_id });
    }
    if let Some(r#match) = kitty_match {
        return Ok(ForegroundInjectionTarget::Kitty { r#match });
    }
    if tmux_target_pane.is_some() || std::env::var("TMUX").is_ok() {
        return Ok(ForegroundInjectionTarget::Tmux);
    }
    Ok(ForegroundInjectionTarget::None)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct TmuxPaneSummary {
    pane_id: String,
    location: String,
    window_name: String,
    current_command: String,
    active: bool,
}

fn resolve_tmux_target_pane(
    explicit_target: Option<String>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if explicit_target.is_some() {
        return Ok(explicit_target);
    }

    if std::env::var("TMUX").is_err() {
        return Ok(None);
    }

    let current_pane = std::env::var("TMUX_PANE").ok();
    let panes = discover_tmux_panes()?;
    let candidates: Vec<TmuxPaneSummary> = panes
        .into_iter()
        .filter(|pane| Some(pane.pane_id.as_str()) != current_pane.as_deref())
        .collect();

    match candidates.len() {
        0 => Ok(None),
        1 => {
            let pane = &candidates[0];
            eprintln!(
                "auto-selected tmux pane {} ({}, cmd: {})",
                pane.pane_id, pane.location, pane.current_command
            );
            Ok(Some(pane.pane_id.clone()))
        }
        _ => prompt_for_tmux_pane(&candidates),
    }
}

fn discover_tmux_panes() -> Result<Vec<TmuxPaneSummary>, Box<dyn std::error::Error>> {
    let output = std::process::Command::new("tmux")
        .arg("list-panes")
        .arg("-a")
        .arg("-F")
        .arg("#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{window_name}\t#{pane_current_command}\t#{pane_active}")
        .output()?;

    if !output.status.success() {
        return Err(format!("tmux list-panes failed with status {}", output.status).into());
    }

    Ok(parse_tmux_panes_output(&String::from_utf8(output.stdout)?))
}

fn parse_tmux_panes_output(output: &str) -> Vec<TmuxPaneSummary> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(5, '\t');
            let pane_id = parts.next()?.to_owned();
            let location = parts.next()?.to_owned();
            let window_name = parts.next()?.to_owned();
            let current_command = parts.next()?.to_owned();
            let active = matches!(parts.next(), Some("1"));
            Some(TmuxPaneSummary {
                pane_id,
                location,
                window_name,
                current_command,
                active,
            })
        })
        .collect()
}

fn paste_into_wezterm_pane(pane_id: &str, text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    if wezterm_target_conflicts(pane_id, std::env::var("WEZTERM_PANE").ok().as_deref()) {
        return Err(
            "refusing to send text into the same wezterm pane that is running the controller"
                .into(),
        );
    }

    let status = std::process::Command::new("wezterm")
        .arg("cli")
        .arg("send-text")
        .arg("--pane-id")
        .arg(pane_id)
        .arg(text)
        .status()?;
    if !status.success() {
        return Err(format!("wezterm cli send-text failed with status {status}").into());
    }
    Ok(())
}

fn paste_into_kitty_target(
    target_match: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    let mut child = std::process::Command::new("kitten")
        .arg("@")
        .arg("send-text")
        .arg("--match")
        .arg(target_match)
        .arg("--stdin")
        .arg("--bracketed-paste")
        .arg("auto")
        .stdin(std::process::Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        use std::io::Write;
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(format!("kitten @ send-text failed with status {status}").into());
    }
    Ok(())
}

fn copy_transcript_to_clipboard(text: &str) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    let mut clipboard = arboard::Clipboard::new()?;
    clipboard.set_text(text.to_owned())?;
    Ok(())
}

fn snapshot_clipboard_text() -> Option<String> {
    let mut clipboard = arboard::Clipboard::new().ok()?;
    clipboard.get_text().ok()
}

fn restore_clipboard_backup(ui: &mut ForegroundPttUiState) -> Result<(), String> {
    let Some(text) = ui.clipboard_backup_text.clone() else {
        return Err("No text clipboard backup is available to restore.".to_owned());
    };
    copy_transcript_to_clipboard(&text).map_err(|error| error.to_string())?;
    ui.last_injection_status = Some("Restored the saved clipboard text.".to_owned());
    ui.last_error = None;
    Ok(())
}

fn save_transcript_to_file(text: &str) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    save_transcript_to_target_dir(text, None)
}

fn save_transcript_to_target_dir(
    text: &str,
    save_dir: Option<&std::path::Path>,
) -> Result<std::path::PathBuf, Box<dyn std::error::Error>> {
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)?
        .as_secs();
    let base_dir = match save_dir {
        Some(dir) => dir.to_path_buf(),
        None => std::env::current_dir()?,
    };
    std::fs::create_dir_all(&base_dir)?;
    let path = base_dir.join(format!("voicelayer-transcript-{timestamp}.txt"));
    std::fs::write(&path, text)?;
    Ok(path)
}

fn copy_result_to_clipboard(ui: &mut ForegroundPttUiState, text: &str) {
    if ui.clipboard_backup_text.is_none() {
        ui.clipboard_backup_text = snapshot_clipboard_text();
    }
    match copy_transcript_to_clipboard(text) {
        Ok(()) => {
            ui.last_injection_status = Some("Copied transcript to system clipboard.".to_owned())
        }
        Err(error) => {
            ui.last_error = Some(format!("Clipboard copy failed: {error}"));
        }
    }
}

fn prompt_for_tmux_pane(
    candidates: &[TmuxPaneSummary],
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    eprintln!("select a tmux target pane for transcript injection:");
    for (index, pane) in candidates.iter().enumerate() {
        eprintln!(
            "  [{}] {}  {}  window={}  cmd={}",
            index + 1,
            pane.pane_id,
            pane.location,
            pane.window_name,
            pane.current_command
        );
    }
    eprint!("selection (number, pane id, Enter to skip): ");
    std::io::stderr().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();
    if input.is_empty() || input.eq_ignore_ascii_case("q") {
        return Ok(None);
    }

    if let Ok(index) = input.parse::<usize>() {
        if (1..=candidates.len()).contains(&index) {
            return Ok(Some(candidates[index - 1].pane_id.clone()));
        }
    }

    if let Some(pane) = candidates.iter().find(|pane| pane.pane_id == input) {
        return Ok(Some(pane.pane_id.clone()));
    }

    Err("invalid tmux pane selection".into())
}

#[cfg(test)]
mod tests {
    use super::{
        CliPttKey, CliRecorderBackend, ForegroundInjectionTarget, StopAction, VlConfig,
        parse_tmux_panes_output, resolve_foreground_injection_target, set_config_value,
        tmux_target_conflicts, wezterm_target_conflicts,
    };
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
    fn tmux_target_conflict_detects_same_pane() {
        assert!(tmux_target_conflicts("%1", Some("%1")));
        assert!(!tmux_target_conflicts("%1", Some("%2")));
        assert!(!tmux_target_conflicts("%1", None));
    }

    #[test]
    fn wezterm_target_conflict_detects_same_pane() {
        assert!(wezterm_target_conflicts("12", Some("12")));
        assert!(!wezterm_target_conflicts("12", Some("13")));
        assert!(!wezterm_target_conflicts("12", None));
    }

    #[test]
    fn parse_tmux_panes_output_reads_expected_fields() {
        let panes =
            parse_tmux_panes_output("%1\tdev:1.0\teditor\tbash\t1\n%2\tdev:1.1\ttests\tnvim\t0\n");
        assert_eq!(panes.len(), 2);
        assert_eq!(panes[0].pane_id, "%1");
        assert_eq!(panes[1].location, "dev:1.1");
        assert_eq!(panes[1].current_command, "nvim");
        assert!(!panes[1].active);
    }

    #[test]
    fn resolve_foreground_target_rejects_multiple_explicit_targets() {
        let error =
            resolve_foreground_injection_target(Some("%1".to_owned()), Some("12".to_owned()), None)
                .expect_err("multiple explicit targets should fail");
        assert!(
            error
                .to_string()
                .contains("choose only one foreground injection target")
        );
    }

    #[test]
    fn resolve_foreground_target_prefers_explicit_wezterm_target() {
        let target =
            resolve_foreground_injection_target(None, Some("12".to_owned()), None).unwrap();
        assert_eq!(
            target,
            ForegroundInjectionTarget::Wezterm {
                pane_id: "12".to_owned()
            }
        );
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
}
