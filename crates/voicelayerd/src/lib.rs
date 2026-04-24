use std::{
    collections::{HashMap, HashSet},
    convert::Infallible,
    env,
    path::{Path, PathBuf},
    sync::Arc,
    time::Duration,
};

use axum::{
    Json, Router,
    extract::State,
    response::{
        IntoResponse, Sse,
        sse::{Event, KeepAlive},
    },
    routing::{get, post},
};
use futures_util::stream::StreamExt;
use tokio::{
    net::UnixListener,
    sync::{Mutex, RwLock, broadcast, oneshot},
    task::JoinSet,
    time::{MissedTickBehavior, interval},
};
use tokio_stream::wrappers::BroadcastStream;
use tracing::info;
use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionReceipt, DictationCaptureRequest,
    DictationCaptureResult, DictationFailureKind, EventEnvelope, HealthResponse, InjectRequest,
    InjectionPlan, PreviewArtifact, PreviewStatus, RecorderBackend, RewriteRequest,
    SegmentationMode, SessionMode, SessionState, StartDictationRequest, StopDictationRequest,
    TranscribeRequest, TranscriptionResult, TranslateRequest, WorkerHealthSummary,
    default_host_adapter_catalog,
};

pub mod hotkeys;
pub mod recording;
pub mod segmented_recording;
pub mod worker;

pub use hotkeys::{GlobalShortcutsPortalStatus, probe_global_shortcuts_portal};
pub use recording::{
    ActiveRecording, RecorderDiagnostics, RecordingError, maybe_cleanup_audio_file,
    record_audio_file, recorder_diagnostics, start_recording_process, stop_recording_process,
    temp_audio_path,
};
pub use worker::{WorkerCallError, WorkerCommand, WorkerHealthResult, WorkerPreviewPayload};

#[derive(Debug, Clone)]
pub struct DaemonConfig {
    pub socket_path: PathBuf,
    pub project_root: PathBuf,
    pub worker_command: WorkerCommand,
    pub version: String,
}

impl DaemonConfig {
    pub fn new(socket_path: PathBuf) -> Self {
        let project_root = default_project_root();
        Self {
            socket_path,
            worker_command: WorkerCommand::discover(project_root.clone()),
            project_root,
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }

    pub fn with_project_root(socket_path: PathBuf, project_root: PathBuf) -> Self {
        Self {
            socket_path,
            worker_command: WorkerCommand::discover(project_root.clone()),
            project_root,
            version: env!("CARGO_PKG_VERSION").to_owned(),
        }
    }
}

pub fn default_project_root() -> PathBuf {
    env::var_os("VOICELAYER_PROJECT_ROOT")
        .map(PathBuf::from)
        .or_else(|| env::current_dir().ok())
        .unwrap_or_else(|| PathBuf::from("."))
}

pub fn default_socket_path() -> PathBuf {
    let base = env::var_os("XDG_RUNTIME_DIR")
        .map(PathBuf::from)
        .unwrap_or_else(env::temp_dir);
    base.join("voicelayer").join("daemon.sock")
}

#[derive(Clone)]
struct AppState {
    sessions: Arc<RwLock<HashMap<String, CaptureSession>>>,
    active_dictations: Arc<Mutex<HashMap<String, ActiveDictation>>>,
    events: broadcast::Sender<EventEnvelope>,
    config: DaemonConfig,
    // Spawns the recorder subprocess for a single segment (or one-shot
    // capture). Production wires this to `start_recording_process`,
    // which shells out to `pw-record` / `arecord`. HTTP-level tests
    // substitute a fake that writes a canned WAV and spawns a short-
    // lived child, so they can exercise handler wiring without a real
    // audio backend.
    recorder_spawner: segmented_recording::RecorderSpawner,
}

#[derive(Debug)]
#[allow(clippy::large_enum_variant)]
enum ActiveDictation {
    OneShot(OneShotActive),
    Segmented(SegmentedActive),
}

#[derive(Debug)]
struct OneShotActive {
    recording: ActiveRecording,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
}

#[derive(Debug)]
struct SegmentedActive {
    stop_tx: oneshot::Sender<()>,
    result_rx: oneshot::Receiver<DictationCaptureResult>,
}

#[derive(Debug)]
struct SegmentRecord {
    id: usize,
    audio_path: PathBuf,
    transcript: Option<String>,
    detected_language: Option<String>,
}

struct SegmentTranscribeOutcome {
    segment_id: usize,
    result: Result<TranscriptionResult, crate::worker::WorkerCallError>,
}

#[derive(serde::Serialize)]
struct ProviderListResponse {
    providers: Vec<voicelayer_core::ProviderDescriptor>,
}

pub async fn run_daemon(config: DaemonConfig) -> std::io::Result<()> {
    let socket_dir = config
        .socket_path
        .parent()
        .map(Path::to_path_buf)
        .unwrap_or_else(|| PathBuf::from("."));
    tokio::fs::create_dir_all(&socket_dir).await?;

    if tokio::fs::try_exists(&config.socket_path).await? {
        tokio::fs::remove_file(&config.socket_path).await?;
    }

    let listener = UnixListener::bind(&config.socket_path)?;
    let state = build_app_state(config.clone(), start_recording_process);
    let app = build_app_router(state);

    info!(socket_path = %config.socket_path.display(), "starting VoiceLayer daemon");
    axum::serve(listener, app).await
}

/// Construct the shared `AppState` with an explicit recorder spawner.
/// Production passes `start_recording_process`; tests inject a fake
/// that skips the real audio backend.
fn build_app_state(
    config: DaemonConfig,
    recorder_spawner: segmented_recording::RecorderSpawner,
) -> Arc<AppState> {
    let (events, _) = broadcast::channel(128);
    Arc::new(AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        active_dictations: Arc::new(Mutex::new(HashMap::new())),
        events,
        config,
        recorder_spawner,
    })
}

/// Build the axum router for the `/v1` control API. Shared by
/// `run_daemon` (which binds it to a Unix listener) and the HTTP-level
/// tests (which drive it in-process through `tower::ServiceExt`).
fn build_app_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(get_health))
        .route("/v1/providers", get(list_providers))
        .route("/v1/events/stream", get(stream_events))
        .route("/v1/sessions/dictation", post(create_dictation_session))
        .route("/v1/sessions/dictation/stop", post(stop_dictation_session))
        .route("/v1/sessions/compose", post(create_composition_job))
        .route("/v1/rewrites", post(create_rewrite_job))
        .route("/v1/translations", post(create_translation_job))
        .route("/v1/transcriptions", post(create_transcription))
        .route("/v1/dictation/capture", post(capture_dictation))
        .route("/v1/inject", post(plan_injection))
        .with_state(state)
}

async fn get_health(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let portal = probe_global_shortcuts_portal().await;
    let worker = match state.config.worker_command.health().await {
        Ok(health) => WorkerHealthSummary {
            status: if (health.llm_configured && !health.llm_reachable)
                || (health.asr_configured && health.asr_error.is_some())
            {
                "degraded".to_owned()
            } else {
                "ok".to_owned()
            },
            command: state.config.worker_command.display(),
            asr_configured: health.asr_configured,
            asr_binary: health.asr_binary,
            asr_model_path: health.asr_model_path,
            asr_error: health.asr_error,
            whisper_mode: health.whisper_mode,
            whisper_server_url: health.whisper_server_url,
            llm_configured: health.llm_configured,
            llm_model: health.llm_model,
            llm_endpoint: health.llm_endpoint,
            llm_reachable: health.llm_reachable,
            llm_error: health.llm_error,
            global_shortcuts_portal_available: portal.available,
            global_shortcuts_portal_version: portal.version,
            global_shortcuts_portal_error: portal.error,
            message: None,
        },
        Err(error) => WorkerHealthSummary {
            status: "unavailable".to_owned(),
            command: state.config.worker_command.display(),
            llm_configured: false,
            llm_model: None,
            llm_endpoint: None,
            llm_reachable: false,
            llm_error: None,
            global_shortcuts_portal_available: portal.available,
            global_shortcuts_portal_version: portal.version,
            global_shortcuts_portal_error: portal.error,
            asr_configured: false,
            asr_binary: None,
            asr_model_path: None,
            asr_error: None,
            whisper_mode: None,
            whisper_server_url: None,
            message: Some(error.to_string()),
        },
    };

    Json(HealthResponse {
        status: if worker.status == "ok" {
            "ok".to_owned()
        } else {
            "degraded".to_owned()
        },
        socket_path: state.config.socket_path.display().to_string(),
        version: state.config.version.clone(),
        worker,
    })
}

async fn list_providers(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let mut providers = default_host_adapter_catalog();
    match state.config.worker_command.list_providers().await {
        Ok(worker_catalog) => providers.extend(worker_catalog.providers),
        Err(error) => {
            let _ = state.events.send(EventEnvelope::new(
                "worker.providers_unavailable",
                None,
                format!("Worker provider listing failed: {error}"),
            ));
        }
    }

    Json(ProviderListResponse { providers })
}

async fn stream_events(
    State(state): State<Arc<AppState>>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.events.subscribe()).filter_map(|event| async move {
        match event {
            Ok(envelope) => match serde_json::to_string(&envelope) {
                Ok(payload) => Some(Ok(Event::default()
                    .event(envelope.event_type)
                    .data(payload))),
                Err(_) => None,
            },
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(std::time::Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn create_dictation_session(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StartDictationRequest>,
) -> impl IntoResponse {
    let detected_language = request.language_profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    });
    let recorder_backend = request.recorder_backend.unwrap_or(RecorderBackend::Auto);
    let keep_audio = request.keep_audio;
    let translate_to_english = request.translate_to_english;
    let base_session = CaptureSession::new(
        SessionMode::Dictation,
        request.trigger,
        request.language_profile.clone().unwrap_or_default(),
    );

    let recorder_spawner = state.recorder_spawner;
    let session = match request.segmentation {
        SegmentationMode::OneShot => {
            let session = transition_session_state(base_session, SessionState::Listening);
            let audio_file = temp_audio_path();
            match recorder_spawner(&audio_file, recorder_backend) {
                Ok(recording) => {
                    state.active_dictations.lock().await.insert(
                        session.session_id.to_string(),
                        ActiveDictation::OneShot(OneShotActive {
                            recording,
                            keep_audio,
                            translate_to_english,
                            language: detected_language,
                        }),
                    );
                    session
                }
                Err(_error) => transition_session_state(session, SessionState::Failed),
            }
        }
        SegmentationMode::Fixed {
            segment_secs,
            overlap_secs,
        } => {
            if segment_secs == 0 {
                transition_session_state(base_session, SessionState::Failed)
            } else {
                let session = transition_session_state(base_session, SessionState::Listening);
                let session_id = session.session_id;
                let (stop_tx, stop_rx) = oneshot::channel();
                let (result_tx, result_rx) = oneshot::channel();
                state.active_dictations.lock().await.insert(
                    session_id.to_string(),
                    ActiveDictation::Segmented(SegmentedActive { stop_tx, result_rx }),
                );

                let worker_command = state.config.worker_command.clone();
                let events = state.events.clone();
                let sessions = Arc::clone(&state.sessions);
                let language = detected_language.clone();
                let session_for_task = session.clone();
                tokio::spawn(async move {
                    let outcome = run_segmented_dictation_session(
                        session_for_task,
                        recorder_backend,
                        segment_secs,
                        overlap_secs,
                        keep_audio,
                        translate_to_english,
                        language,
                        worker_command,
                        events,
                        sessions,
                        stop_rx,
                        recorder_spawner,
                    )
                    .await;
                    let _ = result_tx.send(outcome);
                });
                let _ = state.events.send(EventEnvelope::new(
                    "dictation.segmented_started",
                    Some(session_id),
                    format!(
                        "Segmented dictation started with segment_secs={segment_secs}, overlap_secs={overlap_secs}."
                    ),
                ));
                session
            }
        }
    };

    store_session(&state, session.clone()).await;
    let _ = state.events.send(EventEnvelope::new(
        "dictation.session_created",
        Some(session.session_id),
        if session.state == SessionState::Failed {
            "Dictation session failed to start recording."
        } else {
            "Dictation session created and recording."
        },
    ));

    Json(session)
}

async fn stop_dictation_session(
    State(state): State<Arc<AppState>>,
    Json(request): Json<StopDictationRequest>,
) -> impl IntoResponse {
    Json(stop_live_dictation_with_state(state, request.session_id).await)
}

async fn create_composition_job(
    State(state): State<Arc<AppState>>,
    Json(request): Json<ComposeRequest>,
) -> impl IntoResponse {
    let receipt = match state.config.worker_command.compose(&request).await {
        Ok(preview) => ready_receipt(preview),
        Err(error) => worker_error_receipt(
            SessionMode::Compose,
            "Composition preview is waiting for a configured provider.",
            error,
        ),
    };
    let _ = state.events.send(EventEnvelope::new(
        "compose.job_created",
        None,
        preview_event_message(&receipt.preview),
    ));

    Json(receipt)
}

async fn create_rewrite_job(
    State(state): State<Arc<AppState>>,
    Json(request): Json<RewriteRequest>,
) -> impl IntoResponse {
    let receipt = match state.config.worker_command.rewrite(&request).await {
        Ok(preview) => ready_receipt(preview),
        Err(error) => worker_error_receipt(
            SessionMode::Rewrite,
            "Rewrite preview is waiting for a configured provider.",
            error,
        ),
    };
    let _ = state.events.send(EventEnvelope::new(
        "rewrite.job_created",
        None,
        preview_event_message(&receipt.preview),
    ));

    Json(receipt)
}

async fn create_translation_job(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TranslateRequest>,
) -> impl IntoResponse {
    let receipt = match state.config.worker_command.translate(&request).await {
        Ok(preview) => ready_receipt(preview),
        Err(error) => worker_error_receipt(
            SessionMode::Translate,
            "Translation preview is waiting for a configured provider.",
            error,
        ),
    };
    let _ = state.events.send(EventEnvelope::new(
        "translate.job_created",
        None,
        preview_event_message(&receipt.preview),
    ));

    Json(receipt)
}

async fn create_transcription(
    State(state): State<Arc<AppState>>,
    Json(request): Json<TranscribeRequest>,
) -> impl IntoResponse {
    let result = match state.config.worker_command.transcribe(&request).await {
        Ok(result) => result,
        Err(error) => voicelayer_core::TranscriptionResult {
            text: String::new(),
            detected_language: None,
            notes: vec![format!("Worker bridge detail: {error}")],
        },
    };
    let _ = state.events.send(EventEnvelope::new(
        "transcription.completed",
        None,
        if result.text.is_empty() {
            "Transcription request failed."
        } else {
            "Transcription request completed."
        },
    ));

    Json(result)
}

async fn capture_dictation(
    State(state): State<Arc<AppState>>,
    Json(request): Json<DictationCaptureRequest>,
) -> impl IntoResponse {
    Json(capture_dictation_with_state(state, request).await)
}

async fn plan_injection(Json(request): Json<InjectRequest>) -> impl IntoResponse {
    Json(InjectionPlan::from_request(&request))
}

pub async fn capture_dictation_once(
    config: &DaemonConfig,
    request: DictationCaptureRequest,
) -> DictationCaptureResult {
    let detected_language = request.language_profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    });
    let session = CaptureSession::new(
        SessionMode::Dictation,
        request.trigger.clone(),
        request.language_profile.clone().unwrap_or_default(),
    );

    let mut session = transition_session_state(session, SessionState::Listening);
    let audio_file = temp_audio_path();
    let mut failure_kind: Option<DictationFailureKind> = None;
    let transcription = match record_audio_file(
        &audio_file,
        request.duration_seconds,
        request.recorder_backend.unwrap_or(RecorderBackend::Auto),
    ) {
        Ok(()) => {
            session = transition_session_state(session, SessionState::Transcribing);
            match config
                .worker_command
                .transcribe(&TranscribeRequest {
                    audio_file: audio_file.display().to_string(),
                    language: detected_language,
                    translate_to_english: request.translate_to_english,
                })
                .await
            {
                Ok(result) => {
                    session = transition_session_state(session, SessionState::Completed);
                    result
                }
                Err(error) => {
                    session = transition_session_state(session, SessionState::Failed);
                    failure_kind = Some(DictationFailureKind::AsrFailed);
                    TranscriptionResult {
                        text: String::new(),
                        detected_language: None,
                        notes: vec![format!("Worker bridge detail: {error}")],
                    }
                }
            }
        }
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            failure_kind = Some(DictationFailureKind::RecordingFailed);
            TranscriptionResult {
                text: String::new(),
                detected_language: None,
                notes: vec![format!("Recording detail: {error}")],
            }
        }
    };

    let audio_file = maybe_cleanup_audio_file(&audio_file, request.keep_audio);
    DictationCaptureResult {
        session,
        transcription,
        audio_file,
        failure_kind,
    }
}

async fn stop_live_dictation_with_state(
    state: Arc<AppState>,
    session_id: uuid::Uuid,
) -> DictationCaptureResult {
    let session_key = session_id.to_string();
    let maybe_active = state.active_dictations.lock().await.remove(&session_key);
    let mut session = state
        .sessions
        .read()
        .await
        .get(&session_key)
        .cloned()
        .unwrap_or_else(|| {
            transition_session_state(
                CaptureSession::new(
                    SessionMode::Dictation,
                    voicelayer_core::TriggerKind::Cli,
                    voicelayer_core::LanguageProfile::default(),
                ),
                SessionState::Failed,
            )
        });

    let Some(active) = maybe_active else {
        session = transition_session_state(session, SessionState::Failed);
        return DictationCaptureResult {
            session,
            transcription: TranscriptionResult {
                text: String::new(),
                detected_language: None,
                notes: vec![
                    "No active dictation session exists for the requested session_id.".to_owned(),
                ],
            },
            audio_file: None,
            failure_kind: Some(DictationFailureKind::RecordingFailed),
        };
    };

    match active {
        ActiveDictation::OneShot(active) => stop_oneshot_dictation(state, session, active).await,
        ActiveDictation::Segmented(active) => stop_segmented_dictation(session, active).await,
    }
}

async fn stop_oneshot_dictation(
    state: Arc<AppState>,
    mut session: CaptureSession,
    active: OneShotActive,
) -> DictationCaptureResult {
    session = transition_session_state(session, SessionState::Transcribing);
    store_session(&state, session.clone()).await;
    let _ = state.events.send(EventEnvelope::new(
        "dictation.transcribing",
        Some(session.session_id),
        "Live dictation session is transcribing audio.",
    ));

    let audio_file = match stop_recording_process(active.recording).await {
        Ok(audio_file) => audio_file,
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            store_session(&state, session.clone()).await;
            let _ = state.events.send(EventEnvelope::new(
                "dictation.failed",
                Some(session.session_id),
                format!("Failed to stop dictation recording: {error}"),
            ));
            return DictationCaptureResult {
                session,
                transcription: TranscriptionResult {
                    text: String::new(),
                    detected_language: None,
                    notes: vec![format!("Recording detail: {error}")],
                },
                audio_file: None,
                failure_kind: Some(DictationFailureKind::RecordingFailed),
            };
        }
    };

    let mut failure_kind: Option<DictationFailureKind> = None;
    let transcription = match state
        .config
        .worker_command
        .transcribe(&TranscribeRequest {
            audio_file: audio_file.display().to_string(),
            language: active.language.clone(),
            translate_to_english: active.translate_to_english,
        })
        .await
    {
        Ok(result) => {
            session = transition_session_state(session, SessionState::Completed);
            result
        }
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            failure_kind = Some(DictationFailureKind::AsrFailed);
            TranscriptionResult {
                text: String::new(),
                detected_language: None,
                notes: vec![format!("Worker bridge detail: {error}")],
            }
        }
    };

    store_session(&state, session.clone()).await;
    let _ = state.events.send(EventEnvelope::new(
        if session.state == SessionState::Completed {
            "dictation.completed"
        } else {
            "dictation.failed"
        },
        Some(session.session_id),
        if session.state == SessionState::Completed {
            "Live dictation session completed successfully."
        } else {
            "Live dictation session failed during transcription."
        },
    ));

    DictationCaptureResult {
        session,
        transcription,
        audio_file: maybe_cleanup_audio_file(&audio_file, active.keep_audio),
        failure_kind,
    }
}

async fn stop_segmented_dictation(
    mut session: CaptureSession,
    active: SegmentedActive,
) -> DictationCaptureResult {
    let _ = active.stop_tx.send(());
    match active.result_rx.await {
        Ok(result) => result,
        Err(_) => {
            session = transition_session_state(session, SessionState::Failed);
            DictationCaptureResult {
                session,
                transcription: TranscriptionResult {
                    text: String::new(),
                    detected_language: None,
                    notes: vec![
                        "Segmented dictation task ended without returning a result.".to_owned(),
                    ],
                },
                audio_file: None,
                failure_kind: Some(DictationFailureKind::RecordingFailed),
            }
        }
    }
}

async fn capture_dictation_with_state(
    state: Arc<AppState>,
    request: DictationCaptureRequest,
) -> DictationCaptureResult {
    let detected_language = request.language_profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    });
    let session = CaptureSession::new(
        SessionMode::Dictation,
        request.trigger.clone(),
        request.language_profile.clone().unwrap_or_default(),
    );
    store_session(&state, session.clone()).await;
    let _ = state.events.send(EventEnvelope::new(
        "dictation.session_created",
        Some(session.session_id),
        "Dictation capture session created.",
    ));

    let mut session = transition_session_state(session, SessionState::Listening);
    store_session(&state, session.clone()).await;
    let _ = state.events.send(EventEnvelope::new(
        "dictation.listening",
        Some(session.session_id),
        "Dictation capture is recording audio.",
    ));

    let audio_file = temp_audio_path();
    let mut failure_kind: Option<DictationFailureKind> = None;
    let transcription = match record_audio_file(
        &audio_file,
        request.duration_seconds,
        request.recorder_backend.unwrap_or(RecorderBackend::Auto),
    ) {
        Ok(()) => {
            session = transition_session_state(session, SessionState::Transcribing);
            store_session(&state, session.clone()).await;
            let _ = state.events.send(EventEnvelope::new(
                "dictation.transcribing",
                Some(session.session_id),
                "Dictation capture is transcribing audio.",
            ));

            match state
                .config
                .worker_command
                .transcribe(&TranscribeRequest {
                    audio_file: audio_file.display().to_string(),
                    language: detected_language.clone(),
                    translate_to_english: request.translate_to_english,
                })
                .await
            {
                Ok(result) => {
                    session = transition_session_state(session, SessionState::Completed);
                    store_session(&state, session.clone()).await;
                    let _ = state.events.send(EventEnvelope::new(
                        "dictation.completed",
                        Some(session.session_id),
                        "Dictation capture completed successfully.",
                    ));
                    result
                }
                Err(error) => {
                    session = transition_session_state(session, SessionState::Failed);
                    failure_kind = Some(DictationFailureKind::AsrFailed);
                    store_session(&state, session.clone()).await;
                    let _ = state.events.send(EventEnvelope::new(
                        "dictation.failed",
                        Some(session.session_id),
                        format!("Dictation transcription failed: {error}"),
                    ));
                    TranscriptionResult {
                        text: String::new(),
                        detected_language: None,
                        notes: vec![format!("Worker bridge detail: {error}")],
                    }
                }
            }
        }
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            failure_kind = Some(DictationFailureKind::RecordingFailed);
            store_session(&state, session.clone()).await;
            let _ = state.events.send(EventEnvelope::new(
                "dictation.failed",
                Some(session.session_id),
                format!("Dictation recording failed: {error}"),
            ));
            TranscriptionResult {
                text: String::new(),
                detected_language: None,
                notes: vec![format!("Recording detail: {error}")],
            }
        }
    };

    DictationCaptureResult {
        session,
        transcription,
        audio_file: maybe_cleanup_audio_file(&audio_file, request.keep_audio),
        failure_kind,
    }
}

async fn store_session(state: &Arc<AppState>, session: CaptureSession) {
    state
        .sessions
        .write()
        .await
        .insert(session.session_id.to_string(), session);
}

fn transition_session_state(mut session: CaptureSession, state: SessionState) -> CaptureSession {
    session.state = state;
    session
}

#[allow(clippy::too_many_arguments)]
async fn run_segmented_dictation_session(
    mut session: CaptureSession,
    recorder_backend: RecorderBackend,
    segment_secs: u32,
    overlap_secs: u32,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
    worker_command: worker::WorkerCommand,
    events: broadcast::Sender<EventEnvelope>,
    sessions: Arc<RwLock<HashMap<String, CaptureSession>>>,
    mut stop_rx: oneshot::Receiver<()>,
    recorder_spawner: segmented_recording::RecorderSpawner,
) -> DictationCaptureResult {
    let session_id = session.session_id;
    let segment_dir = temp_audio_path().with_extension("").join("segments");

    // The production `recorder_spawner` is `start_recording_process`, which
    // internally calls `resolve_recorder_backend` — so passing `Auto` here
    // still resolves to pipewire/alsa at the point of subprocess spawn.
    // Test spawners receive the same `recorder_backend` but ignore it, which
    // is why the resolve step lives inside the spawner and not here.
    let mut recorder = match segmented_recording::SegmentedRecording::start_with_spawner(
        &segment_dir,
        recorder_backend,
        segment_secs,
        overlap_secs,
        recorder_spawner,
    ) {
        Ok(recorder) => recorder,
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            update_session_map(&sessions, session.clone()).await;
            let _ = events.send(EventEnvelope::new(
                "dictation.failed",
                Some(session_id),
                format!("Segmented recorder failed to start: {error}"),
            ));
            return DictationCaptureResult {
                session,
                transcription: TranscriptionResult {
                    text: String::new(),
                    detected_language: None,
                    notes: vec![format!("Recording detail: {error}")],
                },
                audio_file: None,
                failure_kind: Some(DictationFailureKind::RecordingFailed),
            };
        }
    };

    let mut segments: Vec<SegmentRecord> = Vec::new();
    let mut transcribe_tasks: JoinSet<SegmentTranscribeOutcome> = JoinSet::new();
    let mut ticker = interval(Duration::from_secs(u64::from(segment_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // The first tick fires immediately; skip it so the first rotation happens
    // after `segment_secs` seconds of actual recording.
    ticker.tick().await;

    let mut recording_failure: Option<String> = None;

    loop {
        tokio::select! {
            biased;
            _ = &mut stop_rx => {
                break;
            }
            _ = ticker.tick() => {
                match recorder.finalize_current().await {
                    Ok(Some((id, path))) => {
                        emit_segment_recorded(&events, session_id, id, &path);
                        segments.push(SegmentRecord {
                            id,
                            audio_path: path.clone(),
                            transcript: None,
                            detected_language: None,
                        });
                        spawn_segment_transcribe(
                            &mut transcribe_tasks,
                            worker_command.clone(),
                            events.clone(),
                            session_id,
                            id,
                            path,
                            language.clone(),
                            translate_to_english,
                        );
                    }
                    Ok(None) => {}
                    Err(error) => {
                        recording_failure = Some(error.to_string());
                        break;
                    }
                }
            }
        }
    }

    let finalized_paths = match recorder.stop().await {
        Ok(paths) => paths,
        Err(error) => {
            if recording_failure.is_none() {
                recording_failure = Some(error.to_string());
            }
            Vec::new()
        }
    };

    let known_ids: HashSet<usize> = segments.iter().map(|segment| segment.id).collect();
    for (id, path) in finalized_paths {
        if known_ids.contains(&id) {
            continue;
        }
        emit_segment_recorded(&events, session_id, id, &path);
        segments.push(SegmentRecord {
            id,
            audio_path: path.clone(),
            transcript: None,
            detected_language: None,
        });
        spawn_segment_transcribe(
            &mut transcribe_tasks,
            worker_command.clone(),
            events.clone(),
            session_id,
            id,
            path,
            language.clone(),
            translate_to_english,
        );
    }

    session = transition_session_state(session, SessionState::Transcribing);
    update_session_map(&sessions, session.clone()).await;

    let mut transcription_failure: Option<String> = None;
    while let Some(outcome) = transcribe_tasks.join_next().await {
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => continue,
        };
        let segment = segments
            .iter_mut()
            .find(|segment| segment.id == outcome.segment_id);
        match outcome.result {
            Ok(result) => {
                if let Some(segment) = segment {
                    segment.transcript = Some(result.text.clone());
                    segment.detected_language = result.detected_language.clone();
                }
            }
            Err(error) => {
                let detail = error.to_string();
                if transcription_failure.is_none() {
                    transcription_failure = Some(detail);
                }
                if let Some(segment) = segment {
                    segment.transcript = Some(String::new());
                }
            }
        }
    }

    segments.sort_by_key(|segment| segment.id);
    let (combined_text, detected_language) = stitch_segment_transcripts(&segments);

    // When `keep_audio` is set the operator usually wants the whole
    // segment set, not just the first file. Return the directory path so
    // `ls "$path"` lists every segment in order; clean it up otherwise.
    let audio_file_summary = if keep_audio {
        Some(segment_dir.display().to_string())
    } else {
        for segment in &segments {
            let _ = std::fs::remove_file(&segment.audio_path);
        }
        let _ = std::fs::remove_dir(&segment_dir);
        None
    };

    let (final_state, failure_kind, failure_detail) =
        classify_segmented_outcome(recording_failure, transcription_failure);
    session = transition_session_state(session, final_state);
    update_session_map(&sessions, session.clone()).await;

    let event_type = if final_state == SessionState::Completed {
        "dictation.completed"
    } else {
        "dictation.failed"
    };
    let event_message = if final_state == SessionState::Completed {
        format!(
            "Segmented dictation session completed with {} segments.",
            segments.len()
        )
    } else {
        format!(
            "Segmented dictation session failed: {}",
            failure_detail.as_deref().unwrap_or("unspecified failure")
        )
    };
    let _ = events.send(EventEnvelope::new(
        event_type,
        Some(session_id),
        event_message,
    ));

    let mut notes = vec![format!("{} segments stitched", segments.len())];
    if let Some(detail) = failure_detail {
        notes.push(detail);
    }

    DictationCaptureResult {
        session,
        transcription: TranscriptionResult {
            text: combined_text,
            detected_language,
            notes,
        },
        audio_file: audio_file_summary,
        failure_kind,
    }
}

fn emit_segment_recorded(
    events: &broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    segment_id: usize,
    path: &Path,
) {
    let _ = events.send(EventEnvelope::new(
        "dictation.segment_recorded",
        Some(session_id),
        format!("Segment {segment_id} recorded at {}.", path.display()),
    ));
}

/// Concatenate every non-empty segment transcript and pick the first detected
/// language. Whisper emits a leading space on Latin transcripts (which
/// naturally joins adjacent words) and no surrounding whitespace on CJK, so a
/// raw `collect::<String>()` handles both scripts; the final `trim` drops any
/// leading/trailing whitespace whisper left on the first or last segment.
///
/// The caller is expected to sort `segments` by `id` before calling this;
/// this helper only reads order, never reorders.
fn stitch_segment_transcripts(segments: &[SegmentRecord]) -> (String, Option<String>) {
    let combined_text: String = segments
        .iter()
        .filter_map(|segment| segment.transcript.as_deref())
        .filter(|chunk| !chunk.is_empty())
        .collect();
    let detected_language = segments
        .iter()
        .find_map(|segment| segment.detected_language.clone());
    (combined_text.trim().to_owned(), detected_language)
}

/// Map the two independent failure channels onto the final session state,
/// failure kind, and user-facing detail string. Recording errors dominate
/// transcription errors: when the recorder subprocess fails, every segment's
/// ASR attempt will also fail for reasons downstream of the original fault,
/// and surfacing `RecordingFailed` to the operator points at the root cause.
fn classify_segmented_outcome(
    recording_failure: Option<String>,
    transcription_failure: Option<String>,
) -> (SessionState, Option<DictationFailureKind>, Option<String>) {
    if let Some(detail) = recording_failure {
        (
            SessionState::Failed,
            Some(DictationFailureKind::RecordingFailed),
            Some(detail),
        )
    } else if let Some(detail) = transcription_failure {
        (
            SessionState::Failed,
            Some(DictationFailureKind::AsrFailed),
            Some(detail),
        )
    } else {
        (SessionState::Completed, None, None)
    }
}

#[allow(clippy::too_many_arguments)]
fn spawn_segment_transcribe(
    transcribe_tasks: &mut JoinSet<SegmentTranscribeOutcome>,
    worker_command: worker::WorkerCommand,
    events: broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    segment_id: usize,
    audio_path: PathBuf,
    language: Option<String>,
    translate_to_english: bool,
) {
    transcribe_tasks.spawn(async move {
        let request = TranscribeRequest {
            audio_file: audio_path.display().to_string(),
            language,
            translate_to_english,
        };
        let result = worker_command.transcribe(&request).await;
        let message = match &result {
            Ok(result) => {
                if result.text.trim().is_empty() {
                    format!("Segment {segment_id} transcribed with empty text.")
                } else {
                    format!(
                        "Segment {segment_id} transcribed ({} chars).",
                        result.text.len()
                    )
                }
            }
            Err(error) => format!("Segment {segment_id} transcription failed: {error}"),
        };
        let _ = events.send(EventEnvelope::new(
            "dictation.segment_transcribed",
            Some(session_id),
            message,
        ));
        SegmentTranscribeOutcome { segment_id, result }
    });
}

async fn update_session_map(
    sessions: &Arc<RwLock<HashMap<String, CaptureSession>>>,
    session: CaptureSession,
) {
    sessions
        .write()
        .await
        .insert(session.session_id.to_string(), session);
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

fn worker_error_receipt(
    mode: SessionMode,
    title: &str,
    error: WorkerCallError,
) -> CompositionReceipt {
    let mut receipt = CompositionReceipt::needs_provider(title, mode);
    receipt
        .preview
        .notes
        .push(format!("Worker bridge detail: {error}"));
    receipt
}

fn preview_event_message(preview: &PreviewArtifact) -> String {
    match &preview.status {
        PreviewStatus::Ready => "Preview artifact is ready.".to_owned(),
        PreviewStatus::NeedsProvider => {
            "Preview artifact is waiting for a configured provider.".to_owned()
        }
        PreviewStatus::Rejected => "Preview artifact was rejected.".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::{
        DaemonConfig, DictationFailureKind, SegmentRecord, SessionState,
        classify_segmented_outcome, default_project_root, default_socket_path,
        stitch_segment_transcripts,
    };
    use std::path::PathBuf;

    fn make_segment(
        id: usize,
        transcript: Option<&str>,
        detected_language: Option<&str>,
    ) -> SegmentRecord {
        SegmentRecord {
            id,
            audio_path: PathBuf::from(format!("/tmp/segment-{id:05}.wav")),
            transcript: transcript.map(str::to_owned),
            detected_language: detected_language.map(str::to_owned),
        }
    }

    #[test]
    fn daemon_config_tracks_version() {
        let config = DaemonConfig::new(PathBuf::from("/tmp/voicelayer.sock"));
        assert_eq!(config.version, env!("CARGO_PKG_VERSION"));
    }

    #[test]
    fn default_paths_are_resolved() {
        assert!(!default_socket_path().as_os_str().is_empty());
        assert!(!default_project_root().as_os_str().is_empty());
    }

    #[test]
    fn stitch_segment_transcripts_on_empty_input_returns_empty_string_and_no_language() {
        let (text, language) = stitch_segment_transcripts(&[]);
        assert!(text.is_empty());
        assert_eq!(language, None);
    }

    #[test]
    fn stitch_segment_transcripts_trims_whispers_leading_space_on_latin_output() {
        // whisper-cli emits a leading space on Latin transcripts; the stitcher
        // should trim the overall string so the first segment doesn't start
        // with whitespace but interior word spacing survives.
        let segments = [
            make_segment(0, Some(" Hello"), Some("en")),
            make_segment(1, Some(" world"), None),
        ];
        let (text, language) = stitch_segment_transcripts(&segments);
        assert_eq!(text, "Hello world");
        assert_eq!(language.as_deref(), Some("en"));
    }

    #[test]
    fn stitch_segment_transcripts_concatenates_cjk_without_adding_spaces() {
        // whisper-cli emits CJK transcripts without leading whitespace; a
        // naive `.join(" ")` would corrupt them, so the stitcher must plain-
        // concat.
        let segments = [
            make_segment(0, Some("你好"), Some("zh")),
            make_segment(1, Some("世界"), None),
        ];
        let (text, language) = stitch_segment_transcripts(&segments);
        assert_eq!(text, "你好世界");
        assert_eq!(language.as_deref(), Some("zh"));
    }

    #[test]
    fn stitch_segment_transcripts_skips_none_and_empty_segments() {
        let segments = [
            make_segment(0, Some(" alpha"), None),
            make_segment(1, None, None),
            make_segment(2, Some(""), None),
            make_segment(3, Some(" beta"), None),
        ];
        let (text, language) = stitch_segment_transcripts(&segments);
        assert_eq!(text, "alpha beta");
        assert_eq!(language, None);
    }

    #[test]
    fn stitch_segment_transcripts_respects_caller_supplied_order() {
        // The caller is expected to sort by segment id before invoking the
        // stitcher. This test documents that invariant — pre-sorted input
        // produces the expected order; the stitcher itself does not reorder.
        let segments = [
            make_segment(0, Some(" first"), None),
            make_segment(1, Some(" second"), None),
            make_segment(2, Some(" third"), None),
        ];
        let (text, _) = stitch_segment_transcripts(&segments);
        assert_eq!(text, "first second third");
    }

    #[test]
    fn stitch_segment_transcripts_picks_first_detected_language_it_finds() {
        // Whisper may report different languages for consecutive segments if
        // the speaker switches mid-session, but the aggregate result needs a
        // single label. The first non-None wins.
        let segments = [
            make_segment(0, Some(" hola"), None),
            make_segment(1, Some(" mundo"), Some("es")),
            make_segment(2, Some(" amigo"), Some("en")),
        ];
        let (_, language) = stitch_segment_transcripts(&segments);
        assert_eq!(language.as_deref(), Some("es"));
    }

    #[test]
    fn classify_segmented_outcome_without_failures_yields_completed_state() {
        let (state, kind, detail) = classify_segmented_outcome(None, None);
        assert_eq!(state, SessionState::Completed);
        assert_eq!(kind, None);
        assert_eq!(detail, None);
    }

    #[test]
    fn classify_segmented_outcome_maps_recording_failure_to_recording_failed() {
        let (state, kind, detail) =
            classify_segmented_outcome(Some("recorder exited".to_owned()), None);
        assert_eq!(state, SessionState::Failed);
        assert_eq!(kind, Some(DictationFailureKind::RecordingFailed));
        assert_eq!(detail.as_deref(), Some("recorder exited"));
    }

    #[test]
    fn classify_segmented_outcome_maps_transcription_failure_to_asr_failed() {
        let (state, kind, detail) =
            classify_segmented_outcome(None, Some("worker RPC timed out".to_owned()));
        assert_eq!(state, SessionState::Failed);
        assert_eq!(kind, Some(DictationFailureKind::AsrFailed));
        assert_eq!(detail.as_deref(), Some("worker RPC timed out"));
    }

    #[test]
    fn classify_segmented_outcome_lets_recording_failure_dominate_transcription_failure() {
        // A failed recorder usually causes every segment's ASR call to fail
        // too; surfacing the ASR error would mislead the operator away from
        // the root cause. Recording wins when both are Some.
        let (state, kind, detail) = classify_segmented_outcome(
            Some("pw-record crashed".to_owned()),
            Some("no audio on disk".to_owned()),
        );
        assert_eq!(state, SessionState::Failed);
        assert_eq!(kind, Some(DictationFailureKind::RecordingFailed));
        assert_eq!(detail.as_deref(), Some("pw-record crashed"));
    }
}

#[cfg(test)]
mod test_support {
    //! Shared fixtures for the lib.rs integration-style tests.
    //!
    //! The `segmented_orchestration_tests` and `http_api_tests` modules
    //! below both need (a) a recorder spawner that sidesteps the real
    //! audio backend and (b) a `WorkerCommand` that points at the
    //! mock-worker Python stub. Factoring these out here keeps each
    //! test file focused on its assertions and keeps the two fixtures
    //! in one place — so a future change to the fake spawner or the
    //! mock-worker contract touches a single location.
    use std::path::{Path, PathBuf};

    use tokio::process::Command;
    use voicelayer_core::{
        CaptureSession, LanguageProfile, RecorderBackend, SessionMode, TriggerKind,
    };

    use super::WorkerCommand;
    use super::recording::{ActiveRecording, RecordingError};

    /// Mirrors the fake spawner in `segmented_recording.rs::tests`. It
    /// writes a few bytes into the target WAV path so
    /// `stop_recording_process`'s non-empty-file check passes, then
    /// spawns a short `sh -c 'sleep 0.2'` child that exits naturally
    /// within the stop-path timeout. The backend argument is ignored —
    /// tests pass any variant of `RecorderBackend`.
    pub(super) fn fake_successful_spawner(
        path: &Path,
        backend: RecorderBackend,
    ) -> Result<ActiveRecording, RecordingError> {
        std::fs::write(path, b"fake-audio-bytes")?;
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.2")
            .spawn()
            .expect("spawn fake recorder child");
        Ok(ActiveRecording {
            backend,
            audio_file: path.to_path_buf(),
            child,
        })
    }

    /// Write `config` to a JSON file inside `dir` and return a
    /// `WorkerCommand` that spawns `mock_worker.py` against it. Using
    /// a per-test config file instead of an environment variable keeps
    /// the tests safe under parallel `cargo test` execution.
    pub(super) fn mock_worker_command(dir: &Path, config: serde_json::Value) -> WorkerCommand {
        let config_path = dir.join("mock_worker_config.json");
        std::fs::write(
            &config_path,
            serde_json::to_vec(&config).expect("encode config"),
        )
        .expect("write mock worker config");

        let mock_script = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("tests")
            .join("fixtures")
            .join("mock_worker.py");
        assert!(
            mock_script.is_file(),
            "mock worker script must exist at {}",
            mock_script.display(),
        );

        WorkerCommand {
            executable: "python3".to_owned(),
            args: vec![
                mock_script.display().to_string(),
                config_path.display().to_string(),
            ],
            project_root: dir.to_path_buf(),
        }
    }

    pub(super) fn fresh_dictation_session() -> CaptureSession {
        CaptureSession::new(
            SessionMode::Dictation,
            TriggerKind::Cli,
            LanguageProfile::default(),
        )
    }
}

#[cfg(test)]
mod segmented_orchestration_tests {
    //! Integration tests for `run_segmented_dictation_session`.
    //!
    //! These tests drive the orchestrator directly (not through the HTTP
    //! layer) so they can inject a fake recorder spawner and a mock
    //! JSON-RPC worker. Together those two doubles cover every moving
    //! piece of the segmented dictation flow — ticker, JoinSet, stop
    //! signal, event emission, and result stitching — without requiring
    //! a real audio backend (pw-record / arecord) or whisper on the
    //! host. The fake recorder follows the same
    //! `sh -c 'sleep 0.2'` pattern as `segmented_recording.rs` (the
    //! child exits naturally within the 5s stop-path timeout so no real
    //! SIGINT handling is needed); the mock worker is a Python stub at
    //! `tests/fixtures/mock_worker.py` configured via a per-test JSON
    //! file so concurrent tests never share state.
    use std::{collections::HashMap, path::Path, sync::Arc, time::Duration};

    use tokio::{
        sync::{RwLock, broadcast, oneshot},
        time::{sleep, timeout},
    };
    use voicelayer_core::{DictationFailureKind, EventEnvelope, RecorderBackend, SessionState};

    use super::run_segmented_dictation_session;
    use super::test_support::{
        fake_successful_spawner, fresh_dictation_session, mock_worker_command,
    };

    /// Drain every event currently sitting in `rx` into a `Vec`. Used
    /// after the orchestrator task has joined so nothing new can be
    /// pushed while we collect.
    fn drain_events(rx: &mut broadcast::Receiver<EventEnvelope>) -> Vec<EventEnvelope> {
        let mut collected = Vec::new();
        while let Ok(event) = rx.try_recv() {
            collected.push(event);
        }
        collected
    }

    fn event_types(events: &[EventEnvelope]) -> Vec<&str> {
        events.iter().map(|e| e.event_type.as_str()).collect()
    }

    #[tokio::test]
    async fn segmented_orchestrator_stops_immediately_and_stitches_tail_segment() {
        // segment_secs is large enough that the ticker never fires
        // before we signal stop; the orchestrator should still reap the
        // in-flight segment through the stop path, transcribe it once,
        // and emit the full event sequence.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " hello"},
                "fail_stems": [],
            }),
        );
        let session = fresh_dictation_session();
        let session_id = session.session_id;
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_segmented_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,
            0,
            false,
            false,
            None,
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        // Give the task a moment to spawn the first segment child so
        // stop reaps a real in-flight recording instead of running
        // before initialization. 50ms is comfortable on both dev and
        // CI hardware; the fake recorder child lives 200ms so this
        // window never overshoots.
        sleep(Duration::from_millis(50)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let result = timeout(Duration::from_secs(10), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let events = drain_events(&mut events_rx);
        let types = event_types(&events);

        assert!(
            types.contains(&"dictation.segment_recorded"),
            "segment_recorded event missing; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.segment_transcribed"),
            "segment_transcribed event missing; actual types: {types:?}",
        );
        assert_eq!(
            types.last().copied(),
            Some("dictation.completed"),
            "the last event must be completed; actual types: {types:?}",
        );

        assert_eq!(
            result.session.session_id, session_id,
            "the same session must come back",
        );
        assert_eq!(result.session.state, SessionState::Completed);
        assert_eq!(result.transcription.text, "hello");
        assert_eq!(
            result.transcription.detected_language.as_deref(),
            Some("en")
        );
        assert_eq!(result.failure_kind, None);
        assert_eq!(
            result.audio_file, None,
            "keep_audio=false must remove the segment_dir from the result",
        );
    }

    #[tokio::test]
    async fn segmented_orchestrator_reports_asr_failed_when_worker_errors_on_every_segment() {
        // Mock worker fails on the tail segment's stem. The orchestrator
        // should still complete (no recorder failure) but classify the
        // session as ASR-failed, since the transcription channel saw an
        // error and the recording channel did not.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": ["segment-00000"],
            }),
        );
        let session = fresh_dictation_session();
        let session_id = session.session_id;
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_segmented_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,
            0,
            false,
            false,
            None,
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        sleep(Duration::from_millis(50)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let result = timeout(Duration::from_secs(10), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let events = drain_events(&mut events_rx);
        let types = event_types(&events);
        assert_eq!(
            types.last().copied(),
            Some("dictation.failed"),
            "an ASR failure must end with dictation.failed; actual types: {types:?}",
        );

        assert_eq!(result.session.session_id, session_id);
        assert_eq!(result.session.state, SessionState::Failed);
        assert_eq!(result.failure_kind, Some(DictationFailureKind::AsrFailed));
        assert!(
            result.transcription.text.is_empty(),
            "a failing worker must not produce transcript text; got {:?}",
            result.transcription.text,
        );
    }

    #[tokio::test]
    async fn segmented_orchestrator_preserves_segment_dir_when_keep_audio_is_set() {
        // keep_audio=true must publish the segment directory path in
        // the result rather than deleting the segment WAVs. This is
        // the handoff contract with the operator who passed
        // `--keep-audio` expecting to find the captured audio on disk.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " kept"},
                "fail_stems": [],
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, _events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_segmented_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,
            0,
            true,
            false,
            None,
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        sleep(Duration::from_millis(50)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let result = timeout(Duration::from_secs(10), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let segment_dir = result
            .audio_file
            .as_deref()
            .expect("keep_audio=true must publish the segment dir");
        let segment_dir = Path::new(segment_dir);
        assert!(
            segment_dir.is_dir(),
            "segment_dir {} must still exist when keep_audio=true",
            segment_dir.display(),
        );
        let wav_path = segment_dir.join("segment-00000.wav");
        assert!(
            wav_path.is_file(),
            "the tail segment wav {} must be preserved",
            wav_path.display(),
        );

        // Clean up by hand so the tempdir drop doesn't race with the
        // session-directory contents the fake spawner wrote.
        let _ = std::fs::remove_dir_all(segment_dir);
    }
}

#[cfg(test)]
mod http_api_tests {
    //! Integration tests for the `/v1` axum router.
    //!
    //! These drive the same handler stack `run_daemon` mounts onto the
    //! Unix listener, but bypass the socket layer via
    //! `tower::ServiceExt::oneshot`. The goal is to pin the
    //! handler-level wiring — request deserialization, session
    //! lifecycle through `active_dictations`, and response shape —
    //! without re-testing the orchestration loop that
    //! `segmented_orchestration_tests` already covers. Pieces that
    //! need a recorder subprocess or a worker RPC reuse the shared
    //! fakes from `test_support`.
    use std::time::Duration;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use http_body_util::BodyExt;
    use tokio::time::sleep;
    use tower::ServiceExt;
    use uuid::Uuid;
    use voicelayer_core::{
        CaptureSession, CompositionArchetype, CompositionReceipt, DictationCaptureResult,
        DictationFailureKind, HealthResponse, InjectRequest, InjectTarget, InjectionPlan,
        PreviewStatus, RecorderBackend, RewriteStyle, SegmentationMode, SessionState,
        StartDictationRequest, StopDictationRequest, TranscriptionResult, TriggerKind,
        WorkerHealthSummary,
    };

    use super::test_support::{fake_successful_spawner, mock_worker_command};
    use super::worker::WorkerCommand;
    use super::{DaemonConfig, build_app_router, build_app_state};

    pub(super) fn test_config(worker_command: WorkerCommand) -> DaemonConfig {
        DaemonConfig {
            socket_path: std::path::PathBuf::from("/unused-in-tests.sock"),
            project_root: worker_command.project_root.clone(),
            worker_command,
            version: "test".to_owned(),
        }
    }

    pub(super) async fn post_json<R>(
        router: axum::Router,
        uri: &str,
        body: serde_json::Value,
    ) -> (StatusCode, R)
    where
        R: serde::de::DeserializeOwned,
    {
        let response = router
            .oneshot(
                Request::builder()
                    .method("POST")
                    .uri(uri)
                    .header("content-type", "application/json")
                    .body(Body::from(serde_json::to_vec(&body).expect("encode body")))
                    .expect("request build"),
            )
            .await
            .expect("router oneshot");
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        let parsed =
            serde_json::from_slice(&bytes).expect("response json should deserialize as expected");
        (status, parsed)
    }

    async fn get_json<R>(router: axum::Router, uri: &str) -> (StatusCode, R)
    where
        R: serde::de::DeserializeOwned,
    {
        let response = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri(uri)
                    .body(Body::empty())
                    .expect("request build"),
            )
            .await
            .expect("router oneshot");
        let status = response.status();
        let bytes = response
            .into_body()
            .collect()
            .await
            .expect("collect body")
            .to_bytes();
        let parsed =
            serde_json::from_slice(&bytes).expect("response json should deserialize as expected");
        (status, parsed)
    }

    #[tokio::test]
    async fn health_endpoint_propagates_mock_worker_summary() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let (status, body): (StatusCode, HealthResponse) = get_json(router, "/v1/health").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(body.version, "test");
        assert_eq!(
            body.worker.asr_binary.as_deref(),
            Some("/usr/bin/mock-whisper")
        );
        assert_eq!(body.worker.whisper_mode.as_deref(), Some("mock"));
        assert!(body.worker.asr_configured);
    }

    // Wire-level companion to the compile-time drift guard
    // `openapi_worker_health_summary_documents_every_field` in
    // `voicelayer-core::domain`. The drift guard proves every Rust field is
    // declared in the openapi contract; this test proves every Rust field
    // actually reaches the HTTP body of `GET /v1/health`. Decoding as
    // `serde_json::Value` (instead of the typed `HealthResponse`) is
    // deliberate — serde would silently fill a missing `Option<_>` with
    // `None`, hiding a regression where a field is dropped on the wire.
    // Key presence is the contract: values vary by host (no portal / no
    // llm on CI), so we assert shape, not content.
    #[tokio::test]
    async fn health_endpoint_serializes_every_worker_summary_field() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        // Mirror the drift guard's sentinel construction verbatim so a
        // future contributor who adds a pub field to `WorkerHealthSummary`
        // breaks both tests together. Every `Option<_>` is `Some(...)` so
        // the expected key set includes nullable fields too.
        let sentinel = WorkerHealthSummary {
            status: String::new(),
            command: String::new(),
            asr_configured: false,
            asr_binary: Some(String::new()),
            asr_model_path: Some(String::new()),
            asr_error: Some(String::new()),
            whisper_mode: Some(String::new()),
            whisper_server_url: Some(String::new()),
            llm_configured: false,
            llm_model: Some(String::new()),
            llm_endpoint: Some(String::new()),
            llm_reachable: false,
            llm_error: Some(String::new()),
            global_shortcuts_portal_available: false,
            global_shortcuts_portal_version: Some(0),
            global_shortcuts_portal_error: Some(String::new()),
            message: Some(String::new()),
        };
        let sentinel_value = serde_json::to_value(&sentinel)
            .expect("WorkerHealthSummary serializes to JSON without error");
        let expected_keys: Vec<String> = sentinel_value
            .as_object()
            .expect("WorkerHealthSummary serializes to a JSON object")
            .keys()
            .cloned()
            .collect();

        let (status, body): (StatusCode, serde_json::Value) = get_json(router, "/v1/health").await;
        assert_eq!(status, StatusCode::OK);

        let worker_body = body
            .get("worker")
            .expect("response body should contain a `worker` object");
        let worker_object = worker_body
            .as_object()
            .expect("`worker` field should serialize as a JSON object");

        for field in &expected_keys {
            assert!(
                worker_object.contains_key(field),
                "GET /v1/health is missing `worker.{field}` on the wire; \
                 expected keys derived from WorkerHealthSummary: {expected_keys:?}",
            );
        }
    }

    #[tokio::test]
    async fn providers_endpoint_includes_host_adapter_catalog() {
        // The providers endpoint merges `default_host_adapter_catalog`
        // with whatever the worker reports. Mock worker reports an
        // empty list; the response must still include the three host
        // adapters the daemon ships with.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let (status, body): (StatusCode, serde_json::Value) =
            get_json(router, "/v1/providers").await;
        assert_eq!(status, StatusCode::OK);
        let ids: Vec<&str> = body["providers"]
            .as_array()
            .expect("providers is a JSON array")
            .iter()
            .filter_map(|entry| entry["id"].as_str())
            .collect();
        assert!(
            ids.contains(&"atspi_accessible_text"),
            "response must include host adapter atspi_accessible_text; got {ids:?}",
        );
        assert!(
            ids.contains(&"global_shortcuts_portal"),
            "response must include host adapter global_shortcuts_portal; got {ids:?}",
        );
        assert!(
            ids.contains(&"terminal_bracketed_paste"),
            "response must include host adapter terminal_bracketed_paste; got {ids:?}",
        );
    }

    #[tokio::test]
    async fn segmented_dictation_http_roundtrip_returns_stitched_transcription() {
        // The goal here is the wiring between the two handlers, not
        // the orchestrator body (which `segmented_orchestration_tests`
        // already covers). Start a session, let the orchestrator
        // initialize the first segment, stop it, then verify the
        // response shape: the session id round-trips, the final state
        // is Completed, the stitched text matches what the mock worker
        // returned, and no failure_kind is set.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " via http"},
                "fail_stems": [],
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let start_req = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: Some(RecorderBackend::Pipewire),
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::Fixed {
                segment_secs: 60,
                overlap_secs: 0,
            },
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(session.state, SessionState::Listening);
        let session_id = session.session_id;

        // Let the tokio::spawn'd orchestrator reach `start_with_spawner`
        // and register the in-flight segment before we signal stop.
        // The fake recorder child lives 200ms so this window is safely
        // bounded.
        sleep(Duration::from_millis(50)).await;

        let stop_req = StopDictationRequest { session_id };
        let (status, result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&stop_req).expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result.session.session_id, session_id);
        assert_eq!(result.session.state, SessionState::Completed);
        assert_eq!(result.transcription.text, "via http");
        assert_eq!(result.failure_kind, None);
    }

    #[tokio::test]
    async fn segmented_dictation_start_with_zero_segment_secs_short_circuits_to_failed() {
        // segment_secs=0 is rejected by the handler before the
        // orchestrator is ever spawned. The response session should
        // come back already in Failed state, and the follow-up stop
        // call must classify the absent active dictation as a
        // recording failure (not an ASR one).
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let start_req = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: Some(RecorderBackend::Pipewire),
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::Fixed {
                segment_secs: 0,
                overlap_secs: 0,
            },
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            session.state,
            SessionState::Failed,
            "segment_secs=0 must land the session in Failed before any task spawns",
        );

        let (status, result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest {
                session_id: session.session_id,
            })
            .expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            result.failure_kind,
            Some(DictationFailureKind::RecordingFailed),
            "a never-started session has nothing to stop and must surface as a recording failure",
        );
    }

    #[tokio::test]
    async fn stop_dictation_with_unknown_session_id_returns_recording_failed() {
        // Defensive contract: posting a stop for a session the daemon
        // never saw must return a deterministic failure payload
        // instead of 5xx-ing or hanging.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let unknown_id = Uuid::new_v4();
        let (status, result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest {
                session_id: unknown_id,
            })
            .expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result.session.state, SessionState::Failed);
        assert_eq!(
            result.failure_kind,
            Some(DictationFailureKind::RecordingFailed),
        );
        assert!(
            result.transcription.text.is_empty(),
            "no transcription text should appear on a stop-without-start",
        );
        // Current behavior: when the requested session_id is unknown,
        // `stop_live_dictation_with_state` fabricates a fresh default
        // `CaptureSession` rather than echoing the caller's id. Pin that
        // so a future change (either direction — echo the requested id
        // or keep minting a new one) forces an explicit decision instead
        // of silently flipping the response shape.
        assert_ne!(
            result.session.session_id, unknown_id,
            "response session_id currently does not echo the request; update this assertion if the contract changes",
        );
    }

    #[tokio::test]
    async fn oneshot_dictation_http_roundtrip_returns_completed_session() {
        // The one-shot branch of `create_dictation_session` takes a
        // different path than the segmented one — it holds an
        // `ActiveRecording` directly in `active_dictations`, and stop
        // synchronously calls `stop_recording_process` +
        // `worker.transcribe`. Pin the handler-level roundtrip so a
        // regression on either side surfaces the same way the
        // segmented test catches regressions on the segmented side.
        //
        // The mock worker's `default_transcribe_text` knob handles the
        // UUID-named `temp_audio_path()` output that one-shot produces:
        // the stem is unknown-to-map, so the default reply fills in.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "default_transcribe_text": " oneshot reply",
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let start_req = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: Some(RecorderBackend::Pipewire),
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::OneShot,
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            session.state,
            SessionState::Listening,
            "one-shot start must ready the active recording, not short-circuit",
        );
        let session_id = session.session_id;

        // Let the fake recorder child live for a handful of milliseconds
        // before stop so `stop_recording_process`'s wait returns after
        // a natural child exit (~200ms) rather than racing the SIGINT
        // timeout path.
        sleep(Duration::from_millis(50)).await;

        let (status, result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest { session_id }).expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(result.session.session_id, session_id);
        assert_eq!(result.session.state, SessionState::Completed);
        assert_eq!(
            result.transcription.text, " oneshot reply",
            "one-shot returns the worker transcript verbatim — unlike the segmented path, no trim is applied because there is only one transcript and no stitching seam where leading whitespace could matter",
        );
        assert_eq!(result.failure_kind, None);
    }

    #[tokio::test]
    async fn compose_http_roundtrip_returns_ready_receipt_from_worker_preview() {
        // `create_composition_job` is a thin bridge onto
        // `WorkerCommand::compose`. The handler wraps the worker's
        // `{title, generated_text, notes}` payload into a
        // `CompositionReceipt` with `PreviewStatus::Ready`, so this test
        // pins that the mock worker's reply round-trips into the response
        // body verbatim — title, generated_text, and notes all present —
        // and that the preview status is Ready (not NeedsProvider).
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "compose_payload": {
                    "title": "Compose preview for email",
                    "generated_text": "Subject: status update\n\nAll green on my end.",
                    "notes": ["Generated by mock compose provider."],
                },
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request_body = serde_json::json!({
            "spoken_prompt": "draft a short status update for the team email",
            "archetype": CompositionArchetype::Email,
            "output_language": "en",
        });
        let (status, receipt): (StatusCode, CompositionReceipt) =
            post_json(router, "/v1/sessions/compose", request_body).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(receipt.preview.status, PreviewStatus::Ready);
        assert_eq!(receipt.preview.title, "Compose preview for email");
        assert_eq!(
            receipt.preview.generated_text.as_deref(),
            Some("Subject: status update\n\nAll green on my end."),
            "the worker's generated_text must flow through to the receipt preview verbatim",
        );
        assert_eq!(
            receipt.preview.notes,
            vec!["Generated by mock compose provider.".to_owned()],
            "worker-side notes must be preserved in the receipt preview",
        );
    }

    #[tokio::test]
    async fn rewrite_http_roundtrip_returns_ready_receipt_from_worker_preview() {
        // Mirrors the compose test for `create_rewrite_job`. The mock
        // worker returns a configured preview for the `rewrite` method
        // and the handler wraps it as a `CompositionReceipt` with
        // `PreviewStatus::Ready`.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "rewrite_payload": {
                    "title": "Rewrite preview for more formal",
                    "generated_text": "I would like to follow up on the previous message.",
                    "notes": ["Generated by mock rewrite provider."],
                },
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request_body = serde_json::json!({
            "source_text": "hey, any update on this?",
            "style": RewriteStyle::MoreFormal,
            "output_language": "en",
        });
        let (status, receipt): (StatusCode, CompositionReceipt) =
            post_json(router, "/v1/rewrites", request_body).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(receipt.preview.status, PreviewStatus::Ready);
        assert_eq!(receipt.preview.title, "Rewrite preview for more formal");
        assert_eq!(
            receipt.preview.generated_text.as_deref(),
            Some("I would like to follow up on the previous message."),
        );
        assert_eq!(
            receipt.preview.notes,
            vec!["Generated by mock rewrite provider.".to_owned()],
        );
    }

    #[tokio::test]
    async fn translate_http_roundtrip_returns_ready_receipt_from_worker_preview() {
        // Mirrors the compose/rewrite tests for `create_translation_job`.
        // The mock worker's `translate` response flows into a
        // `CompositionReceipt` with `PreviewStatus::Ready`.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "translate_payload": {
                    "title": "Translate preview for ja",
                    "generated_text": "こんにちは、世界。",
                    "notes": ["Generated by mock translate provider."],
                },
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request_body = serde_json::json!({
            "source_text": "Hello, world.",
            "target_language": "ja",
        });
        let (status, receipt): (StatusCode, CompositionReceipt) =
            post_json(router, "/v1/translations", request_body).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(receipt.preview.status, PreviewStatus::Ready);
        assert_eq!(receipt.preview.title, "Translate preview for ja");
        assert_eq!(
            receipt.preview.generated_text.as_deref(),
            Some("こんにちは、世界。"),
            "CJK generated_text must round-trip through JSON without mangling",
        );
        assert_eq!(
            receipt.preview.notes,
            vec!["Generated by mock translate provider.".to_owned()],
        );
    }

    #[tokio::test]
    async fn transcribe_http_roundtrip_returns_worker_transcription_result() {
        // `create_transcription` calls `WorkerCommand::transcribe` and
        // returns the `TranscriptionResult` verbatim on success. Use the
        // mock worker's `transcribe_map` keyed by the audio file stem
        // (here "fixture-audio") to pin that the mapped text reaches the
        // HTTP response intact and the detected language is populated.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("fixture-audio.wav");
        std::fs::write(&audio_file, b"fake-audio-bytes").expect("write audio fixture");

        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"fixture-audio": "hello from transcribe handler"},
                "fail_stems": [],
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request_body = serde_json::json!({
            "audio_file": audio_file.display().to_string(),
            "language": "en",
            "translate_to_english": false,
        });
        let (status, result): (StatusCode, TranscriptionResult) =
            post_json(router, "/v1/transcriptions", request_body).await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(result.text, "hello from transcribe handler");
        assert_eq!(
            result.detected_language.as_deref(),
            Some("en"),
            "mock worker tags non-empty transcripts with `en`",
        );
        assert!(
            result.notes.is_empty(),
            "a successful transcription carries no bridge-error notes; got {:?}",
            result.notes,
        );
    }

    #[tokio::test]
    async fn inject_http_roundtrip_returns_bracketed_paste_plan_for_terminal_target() {
        // `plan_injection` is pure Rust — it constructs an `InjectionPlan`
        // directly from the request without calling the worker. Pin the
        // two terminal-target cases: the bracketed-paste target wraps
        // text in the SGR escapes, and `auto_submit=false` (the safe
        // default) omits the trailing newline.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = InjectRequest {
            target: InjectTarget::TerminalBracketedPaste,
            text: "ls -la".to_owned(),
            auto_submit: false,
        };
        let (status, plan): (StatusCode, InjectionPlan) = post_json(
            router,
            "/v1/inject",
            serde_json::to_value(&request).expect("encode inject req"),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(plan.target, InjectTarget::TerminalBracketedPaste);
        assert!(!plan.auto_submit);
        assert_eq!(
            plan.payload, "\u{1b}[200~ls -la\u{1b}[201~",
            "terminal_bracketed_paste must wrap the text in the SGR start/end escapes and omit the trailing newline when auto_submit is false",
        );
    }

    #[tokio::test]
    async fn inject_http_roundtrip_returns_gui_accessible_plan_verbatim() {
        // The GUI path echoes the request text as the payload without
        // wrapping. Pinning this complements the terminal-target test so
        // both branches of `InjectionPlan::from_request` are covered at
        // the HTTP layer.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = InjectRequest {
            target: InjectTarget::GuiAccessible,
            text: "Inject into the focused editable surface.".to_owned(),
            auto_submit: false,
        };
        let (status, plan): (StatusCode, InjectionPlan) = post_json(
            router,
            "/v1/inject",
            serde_json::to_value(&request).expect("encode inject req"),
        )
        .await;

        assert_eq!(status, StatusCode::OK);
        assert_eq!(plan.target, InjectTarget::GuiAccessible);
        assert!(!plan.auto_submit);
        assert_eq!(
            plan.payload, "Inject into the focused editable surface.",
            "gui_accessible must pass the text through unchanged; no escape wrapping applies",
        );
    }
}

#[cfg(test)]
mod sse_stream_tests {
    //! Integration tests for `GET /v1/events/stream`.
    //!
    //! The SSE handler wraps the daemon's broadcast channel in
    //! `BroadcastStream` and frames each `EventEnvelope` into an SSE
    //! `event: <type>\ndata: <json>\n\n` block. The broadcast
    //! plumbing is already covered by the orchestrator tests that
    //! subscribe directly to the channel; what lives only in the HTTP
    //! layer is:
    //!   1. the per-request subscription creation,
    //!   2. the envelope → `Event::default().event(type).data(json)`
    //!      framing,
    //!   3. the keep-alive + stream filter shape.
    //!
    //! These tests drive the handler in-process via
    //! `tower::ServiceExt::oneshot`, which returns the response as
    //! soon as headers are ready; the streaming body is then pulled
    //! frame-by-frame. Both tests subscribe before emitting so there
    //! is no event-ordering race.
    use std::time::Duration;

    use axum::{
        body::Body,
        http::{Request, StatusCode},
    };
    use futures_util::StreamExt;
    use tokio::time::{sleep, timeout};
    use tower::ServiceExt;
    use uuid::Uuid;
    use voicelayer_core::{
        CaptureSession, DictationCaptureResult, EventEnvelope, RecorderBackend, SegmentationMode,
        StartDictationRequest, StopDictationRequest, TriggerKind,
    };

    use super::http_api_tests::{post_json, test_config};
    use super::test_support::{fake_successful_spawner, mock_worker_command};
    use super::{build_app_router, build_app_state};

    /// Read body chunks until `accumulated` contains `expected_frames`
    /// SSE frame terminators (`\n\n`) or the `overall_timeout`
    /// elapses, whichever comes first. SSE frames may straddle chunk
    /// boundaries, so the caller is expected to split the result on
    /// `\n\n` afterwards.
    async fn read_sse_frames_until(
        body: Body,
        expected_frames: usize,
        overall_timeout: Duration,
    ) -> String {
        let mut stream = body.into_data_stream();
        let mut buffer = String::new();
        let deadline = tokio::time::Instant::now() + overall_timeout;
        while buffer.matches("\n\n").count() < expected_frames {
            let remaining = deadline.saturating_duration_since(tokio::time::Instant::now());
            if remaining.is_zero() {
                break;
            }
            match timeout(remaining, stream.next()).await {
                Ok(Some(Ok(bytes))) => {
                    buffer.push_str(std::str::from_utf8(&bytes).expect("utf-8 SSE frame"));
                }
                Ok(Some(Err(_))) | Ok(None) | Err(_) => break,
            }
        }
        buffer
    }

    /// Extract the `event: <type>` header from each SSE frame in
    /// `raw`, in order. Ignores keep-alive frames (they do not carry
    /// an `event:` line).
    fn event_types(raw: &str) -> Vec<&str> {
        raw.split("\n\n")
            .filter_map(|frame| frame.lines().find_map(|line| line.strip_prefix("event: ")))
            .collect()
    }

    #[tokio::test]
    async fn events_stream_frames_a_single_broadcast_envelope() {
        // Minimum viable SSE test: subscribe, emit one envelope, read
        // one frame, verify the event type and payload round-trip
        // through the axum `Sse` framing. Nothing else runs concurrently
        // so the frame ordering is deterministic.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let events_tx = state.events.clone();
        let router = build_app_router(state);

        let response = router
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/events/stream")
                    .body(Body::empty())
                    .expect("request build"),
            )
            .await
            .expect("router oneshot");
        assert_eq!(response.status(), StatusCode::OK);
        assert_eq!(
            response
                .headers()
                .get("content-type")
                .and_then(|value| value.to_str().ok()),
            Some("text/event-stream"),
            "SSE handler must advertise the event-stream MIME type",
        );

        // The handler subscribed to the broadcast channel before
        // returning the response; the send below will reach that
        // subscription. Any envelope sent *before* `.oneshot().await`
        // returned would be lost.
        let session_id = Uuid::new_v4();
        events_tx
            .send(EventEnvelope::new(
                "test.single_event",
                Some(session_id),
                "single frame",
            ))
            .expect("broadcast send");

        let raw = read_sse_frames_until(response.into_body(), 1, Duration::from_secs(2)).await;
        let types = event_types(&raw);
        assert_eq!(
            types,
            vec!["test.single_event"],
            "the broadcast envelope's event_type must land on the SSE `event:` line; got frames: {raw:?}",
        );
        assert!(
            raw.contains(&session_id.to_string()),
            "SSE data payload must include the envelope's session_id; got: {raw:?}",
        );
        assert!(
            raw.contains(r#""message":"single frame""#),
            "SSE data payload must include the envelope's message; got: {raw:?}",
        );
    }

    #[tokio::test]
    async fn events_stream_surfaces_segmented_dictation_lifecycle_events() {
        // End-to-end: the SSE subscriber must see `segmented_started`
        // (emitted by `create_dictation_session`), `segment_recorded`
        // and `segment_transcribed` (emitted from the orchestrator task
        // during stop-path drain), and a terminal `completed` event.
        // This is the closest an HTTP-level test can get to what a
        // real vl-desktop overlay or foreground-ptt panel observes
        // during a dictation session.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " streamed"},
                "fail_stems": [],
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        // Open the SSE stream first so the broadcast subscription
        // exists before any event is emitted. The handler's
        // `state.events.subscribe()` runs inside the GET handler body,
        // so by the time `oneshot().await` returns, the subscription
        // is live.
        let sse_response = router
            .clone()
            .oneshot(
                Request::builder()
                    .method("GET")
                    .uri("/v1/events/stream")
                    .body(Body::empty())
                    .expect("request build"),
            )
            .await
            .expect("sse oneshot");
        assert_eq!(sse_response.status(), StatusCode::OK);

        // Spawn a concurrent reader that drains frames until the
        // full lifecycle (>= 5 frames: segmented_started,
        // session_created, segment_recorded, segment_transcribed,
        // completed) has arrived or the timeout lapses.
        let reader = tokio::spawn(async move {
            read_sse_frames_until(sse_response.into_body(), 5, Duration::from_secs(5)).await
        });

        // Kick off a segmented session and stop it immediately so the
        // full event sequence flows through the subscription.
        let start_req = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: Some(RecorderBackend::Pipewire),
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::Fixed {
                segment_secs: 60,
                overlap_secs: 0,
            },
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        sleep(Duration::from_millis(50)).await;

        let (status, _result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest {
                session_id: session.session_id,
            })
            .expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let raw = reader.await.expect("SSE reader joined");
        let types = event_types(&raw);
        assert!(
            types.contains(&"dictation.segmented_started"),
            "segmented_started must appear; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.session_created"),
            "session_created must appear; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.segment_recorded"),
            "segment_recorded must appear; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.segment_transcribed"),
            "segment_transcribed must appear; actual types: {types:?}",
        );
        assert_eq!(
            types.last().copied(),
            Some("dictation.completed"),
            "the terminal frame must be `dictation.completed`; actual types: {types:?}",
        );
        assert!(
            raw.contains(&session.session_id.to_string()),
            "every lifecycle frame should carry the session_id; got raw: {raw:?}",
        );
    }
}
