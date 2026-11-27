//! Long-running VoiceLayer daemon.
//!
//! Serves the `/v1` HTTP control API over a Unix domain socket, supervises one
//! persistent Python worker for ASR and LLM calls, manages in-process audio
//! capture, and streams lifecycle events over Server-Sent Events.
//!
//! Modules: `api/` (router, handlers, RFC 9457 errors), `audio` (`cpal`
//! capture), `dictation/` (one-shot, fixed, and VAD-gated pipelines), `events`
//! (broadcast bus), `platform/` (OS capability probes), `session` (bounded
//! in-memory store), and `worker/` (persistent stdio JSON-RPC client).

pub mod api;
pub mod audio;
pub mod dictation;
pub mod events;
pub mod platform;
pub mod session;
pub mod worker;

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use tokio::net::UnixListener;
use tokio::sync::{Mutex, RwLock};
use tracing::{info, warn};
use voicelayer_core::{VoiceLayerConfig, default_runtime_dir};

use crate::{
    api::{AppState, refresh_health_state},
    events::EventBus,
    session::SessionStore,
    worker::WorkerCommand,
};

pub use worker::{WorkerCallError, WorkerHealthResult, WorkerPreviewPayload};

/// Runtime configuration for [`run_daemon`].
#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub project_root: PathBuf,
    pub worker_command: WorkerCommand,
    pub settings: VoiceLayerConfig,
    pub version: String,
}

impl DaemonConfig {
    /// Load settings from `config.toml` (with `VOICELAYER_*` overrides) using `socket_path`.
    pub fn new(socket_path: PathBuf) -> Self {
        let settings = VoiceLayerConfig::load().unwrap_or_else(|error| {
            warn!(%error, "failed to load config file; falling back to defaults + env");
            VoiceLayerConfig::default_with_env()
        });
        Self::with_settings(Some(socket_path), None, settings)
    }

    /// Load settings from `config.toml` with explicit `socket_path` and `project_root`.
    pub fn with_project_root(socket_path: PathBuf, project_root: PathBuf) -> Self {
        let settings = VoiceLayerConfig::load().unwrap_or_else(|error| {
            warn!(%error, "failed to load config file; falling back to defaults + env");
            VoiceLayerConfig::default_with_env()
        });
        Self::with_settings(Some(socket_path), Some(project_root), settings)
    }

    /// Build daemon config from explicit overrides and a loaded [`VoiceLayerConfig`].
    pub fn with_settings(
        socket_path: Option<PathBuf>,
        project_root: Option<PathBuf>,
        settings: VoiceLayerConfig,
    ) -> Self {
        let socket_path = socket_path.unwrap_or_else(|| settings.daemon.socket_path());
        let project_root = project_root.unwrap_or_else(|| settings.daemon.project_root());
        let inference_timeout = Duration::from_secs(settings.daemon.worker_timeout_seconds.max(1));
        let worker_command = WorkerCommand::discover(project_root.clone())
            .with_init_payload(settings.worker_payload())
            .with_inference_timeout(inference_timeout);
        Self {
            socket_path,
            project_root,
            worker_command,
            settings,
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

/// Bind the `/v1` API on the configured Unix socket and serve until Ctrl+C or SIGTERM.
pub async fn run_daemon(config: DaemonConfig) -> std::io::Result<()> {
    let socket_dir = config
        .socket_path
        .parent()
        .map(std::path::Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&socket_dir).await?;
    set_owner_only(&socket_dir).await;

    let recordings_dir = default_runtime_dir().join("dictation");
    tokio::fs::create_dir_all(&recordings_dir).await?;
    set_owner_only(&recordings_dir).await;

    if tokio::fs::try_exists(&config.socket_path).await? {
        tokio::fs::remove_file(&config.socket_path).await?;
    }
    let listener = UnixListener::bind(&config.socket_path)?;
    set_owner_only(&config.socket_path).await;

    let state = AppState {
        sessions: SessionStore::new(),
        active: Arc::new(Mutex::new(HashMap::new())),
        events: EventBus::new(),
        health: Arc::new(RwLock::new(None)),
        config: Arc::new(config),
        test_audio_silence: None,
    };

    // Keep cached health fresh without eagerly spawning an idle worker.
    let refresher_state = state.clone();
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(api::HEALTH_REFRESH_INTERVAL);
        loop {
            ticker.tick().await;
            let warm = refresher_state.health.read().await.is_some();
            if warm || refresher_state.config.worker_command.is_running().await {
                refresh_health_state(&refresher_state).await;
            }
        }
    });

    let socket_path = state.config.socket_path.clone();
    let worker = state.config.worker_command.clone();
    let app = api::router(state);

    info!(socket_path = %socket_path.display(), "starting VoiceLayer daemon");
    let serve_result = axum::serve(listener, app)
        .with_graceful_shutdown(shutdown_signal())
        .await;

    worker.shutdown().await;
    let _ = tokio::fs::remove_file(&socket_path).await;
    serve_result
}

async fn set_owner_only(path: &std::path::Path) {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        let permissions =
            std::fs::Permissions::from_mode(if path.is_dir() { 0o700 } else { 0o600 });
        if let Err(error) = tokio::fs::set_permissions(path, permissions).await {
            warn!(%error, path = %path.display(), "failed to tighten permissions");
        }
    }
}

async fn shutdown_signal() {
    let ctrl_c = async {
        if let Err(error) = tokio::signal::ctrl_c().await {
            warn!(%error, "failed to listen for Ctrl+C");
        }
    };

    #[cfg(unix)]
    let terminate = async {
        match tokio::signal::unix::signal(tokio::signal::unix::SignalKind::terminate()) {
            Ok(mut signal) => {
                signal.recv().await;
            }
            Err(error) => warn!(%error, "failed to listen for SIGTERM"),
        }
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
    info!("shutdown signal received; stopping daemon");
}
