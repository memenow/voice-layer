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
    let (events, _) = broadcast::channel(128);
    let state = Arc::new(AppState {
        sessions: Arc::new(RwLock::new(HashMap::new())),
        active_dictations: Arc::new(Mutex::new(HashMap::new())),
        events,
        config: config.clone(),
    });

    let app = Router::new()
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
        .with_state(state);

    info!(socket_path = %config.socket_path.display(), "starting VoiceLayer daemon");
    axum::serve(listener, app).await
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

    let session = match request.segmentation {
        SegmentationMode::OneShot => {
            let session = transition_session_state(base_session, SessionState::Listening);
            let audio_file = temp_audio_path();
            match start_recording_process(&audio_file, recorder_backend) {
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
) -> DictationCaptureResult {
    let session_id = session.session_id;
    let segment_dir = temp_audio_path().with_extension("").join("segments");

    let mut recorder = match segmented_recording::SegmentedRecording::start(
        &segment_dir,
        recorder_backend,
        segment_secs,
        overlap_secs,
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
    // Preserve whisper's native leading whitespace on Latin transcripts
    // (it already separates words) and concatenate CJK transcripts
    // without adding artificial spaces. A final `trim` strips any
    // outer whitespace whisper may have emitted at the edges.
    let combined_text: String = segments
        .iter()
        .filter_map(|segment| segment.transcript.as_deref())
        .filter(|chunk| !chunk.is_empty())
        .collect();
    let combined_text = combined_text.trim().to_owned();
    let detected_language = segments
        .iter()
        .find_map(|segment| segment.detected_language.clone());

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
        if let Some(detail) = recording_failure.clone() {
            (
                SessionState::Failed,
                Some(DictationFailureKind::RecordingFailed),
                Some(detail),
            )
        } else if let Some(detail) = transcription_failure.clone() {
            (
                SessionState::Failed,
                Some(DictationFailureKind::AsrFailed),
                Some(detail),
            )
        } else {
            (SessionState::Completed, None, None)
        };
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
    use super::{DaemonConfig, default_project_root, default_socket_path};
    use std::path::PathBuf;

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
}
