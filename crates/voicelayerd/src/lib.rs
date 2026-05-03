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
    http::StatusCode,
    response::{
        IntoResponse, Response, Sse,
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
    SUPPORTED_TRANSCRIBE_PROVIDER_IDS, SegmentProbeRequest, SegmentProbeResult, SegmentationMode,
    SessionMode, SessionState, StartDictationRequest, StitchWavSegmentsRequest,
    StopDictationRequest, TranscribeRequest, TranscriptionResult, TranslateRequest,
    WorkerHealthSummary, default_host_adapter_catalog, is_supported_transcribe_provider_id,
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
    provider_id: Option<String>,
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

/// A single physical probe captured by the VAD-gated orchestrator.
/// Retained after classification so probe audio files can be dedup'd
/// across the live-loop and `recorder.stop()` paths and cleaned up on
/// session end. Verdicts themselves flow through
/// `dictation.probe_analyzed` events — no need to store them again here.
#[derive(Debug)]
struct ProbeRecord {
    id: usize,
    audio_path: PathBuf,
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

/// Validate the ASR `provider_id` and the request flag combination
/// before any audio capture starts. Returns the operator-facing
/// rejection reason as `Some(detail)` when the id is unknown to the
/// daemon (typo, advertised-only catalog entry, etc.) or when the
/// requested combination is documented to fail at the worker
/// (currently `mimo_v2_5_asr` + `translate_to_english`).
///
/// This is the moral equivalent of the worker-side check in
/// `python/voicelayer_orchestrator/worker.py`'s `transcribe` dispatch,
/// lifted to the dictation entry points so an operator who fat-fingers
/// `--provider-id mimo_v2_5` does not waste a full recording before the
/// stop call surfaces the error from the worker. The supported set
/// itself lives on `voicelayer_core::SUPPORTED_TRANSCRIBE_PROVIDER_IDS`
/// so daemon, worker, and OpenAPI all read from the same source.
fn pre_capture_provider_rejection_detail(
    provider_id: Option<&str>,
    translate_to_english: bool,
) -> Option<String> {
    if !is_supported_transcribe_provider_id(provider_id) {
        let id = provider_id.unwrap_or("");
        return Some(format!(
            "Unknown ASR provider id `{id}`. Supported ids: {}.",
            SUPPORTED_TRANSCRIBE_PROVIDER_IDS.join(", "),
        ));
    }
    if matches!(provider_id, Some("mimo_v2_5_asr")) && translate_to_english {
        return Some(
            "MiMo-V2.5-ASR does not support translate_to_english today; \
             run translation through the LLM workflow or pick the default \
             whisper.cpp provider."
                .to_owned(),
        );
    }
    None
}

/// HTTP wrapper over `pre_capture_provider_rejection_detail`. Returns
/// `Some(Response)` when the request must be rejected with a 400 so
/// the operator sees the same error class regardless of which
/// dictation entry point they hit; returns `None` when validation
/// passes. Modelled as `Option<Response>` rather than
/// `Result<(), Response>` because clippy's `result_large_err` lint
/// flags the latter (an axum `Response` is ~128 bytes).
fn dictation_provider_request_rejection(
    provider_id: Option<&str>,
    translate_to_english: bool,
) -> Option<Response> {
    let message = pre_capture_provider_rejection_detail(provider_id, translate_to_english)?;
    let error_code = if matches!(provider_id, Some("mimo_v2_5_asr")) && translate_to_english {
        "translate_to_english_unsupported"
    } else {
        "invalid_provider_id"
    };
    let body = serde_json::json!({
        "error": error_code,
        "message": message,
    });
    Some((StatusCode::BAD_REQUEST, Json(body)).into_response())
}

/// Build the axum router for the `/v1` control API. Shared by
/// `run_daemon` (which binds it to a Unix listener) and the HTTP-level
/// tests (which drive it in-process through `tower::ServiceExt`).
fn build_app_router(state: Arc<AppState>) -> Router {
    Router::new()
        .route("/v1/health", get(get_health))
        .route("/v1/providers", get(list_providers))
        .route("/v1/events/stream", get(stream_events))
        .route("/v1/sessions", get(list_sessions))
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
            mimo_configured: health.mimo_configured,
            mimo_model_path: health.mimo_model_path,
            mimo_error: health.mimo_error,
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
            mimo_configured: false,
            mimo_model_path: None,
            mimo_error: None,
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

/// List every `CaptureSession` the daemon currently tracks.
///
/// Sessions live in `state.sessions` from `create` until they reach a
/// terminal state (`Completed` / `Failed`); the map is not pruned on
/// stop so a recently-finished session is still discoverable here.
/// Operators reach this endpoint when they have lost the
/// `session_id` returned by an earlier `POST /v1/sessions/dictation`
/// (e.g. a CLI window scrolled off-screen) and need to recover it
/// before issuing the matching stop call.
///
/// The return value is a JSON array of `CaptureSession`. There is
/// intentionally no enclosing `{"sessions": [...]}` wrapper — the
/// list itself is the resource and `vl dictation list` consumes it
/// directly via `Vec<CaptureSession>`.
async fn list_sessions(State(state): State<Arc<AppState>>) -> impl IntoResponse {
    let sessions = state.sessions.read().await;
    let mut entries: Vec<CaptureSession> = sessions.values().cloned().collect();
    // Stable order so pretty-printed output and shell-script parsing
    // can rely on the listing — the underlying HashMap iteration order
    // is otherwise nondeterministic.
    entries.sort_by_key(|session| session.created_at_millis);
    Json(entries)
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
) -> Response {
    if let Some(rejection) = dictation_provider_request_rejection(
        request.provider_id.as_deref(),
        request.translate_to_english,
    ) {
        return rejection;
    }
    let detected_language = request.language_profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    });
    let recorder_backend = request.recorder_backend.unwrap_or(RecorderBackend::Auto);
    let keep_audio = request.keep_audio;
    let translate_to_english = request.translate_to_english;
    let provider_id = request.provider_id.clone();
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
                            provider_id,
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
                let provider_id_for_task = provider_id.clone();
                tokio::spawn(async move {
                    let outcome = run_segmented_dictation_session(
                        session_for_task,
                        recorder_backend,
                        segment_secs,
                        overlap_secs,
                        keep_audio,
                        translate_to_english,
                        language,
                        provider_id_for_task,
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
        SegmentationMode::VadGated {
            probe_secs,
            max_segment_secs,
            silence_gap_probes,
        } => {
            // Reject obviously degenerate configs at dispatch so the
            // session transitions cleanly to Failed without tying up the
            // recorder. `probe_secs == 0` would deadlock the ticker;
            // `max_segment_secs == 0` would force-flush every probe as a
            // degenerate 1-probe unit and defeat the point of gating.
            if probe_secs == 0 || max_segment_secs == 0 {
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
                let provider_id_for_task = provider_id.clone();
                tokio::spawn(async move {
                    let outcome = run_vad_gated_dictation_session(
                        session_for_task,
                        recorder_backend,
                        probe_secs,
                        max_segment_secs,
                        silence_gap_probes,
                        keep_audio,
                        translate_to_english,
                        language,
                        provider_id_for_task,
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
                    "dictation.vad_gated_started",
                    Some(session_id),
                    format!(
                        "VAD-gated dictation started with probe_secs={probe_secs}, \
                         max_segment_secs={max_segment_secs}, \
                         silence_gap_probes={silence_gap_probes}."
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

    Json(session).into_response()
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
) -> Response {
    if let Some(rejection) = dictation_provider_request_rejection(
        request.provider_id.as_deref(),
        request.translate_to_english,
    ) {
        return rejection;
    }
    Json(capture_dictation_with_state(state, request).await).into_response()
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

    // In-process fallback used by `vl record-transcribe` when no daemon
    // socket is reachable. The HTTP path returns 4xx for invalid
    // provider requests via `validate_dictation_provider_request`; this
    // call site has no HTTP envelope, so surface the same rejection by
    // short-circuiting before the recorder spawns. Without this, a
    // typo would still record a full duration and fail at the worker.
    if let Some(detail) = pre_capture_provider_rejection_detail(
        request.provider_id.as_deref(),
        request.translate_to_english,
    ) {
        let session = transition_session_state(session, SessionState::Failed);
        return DictationCaptureResult {
            session,
            transcription: TranscriptionResult {
                text: String::new(),
                detected_language: None,
                notes: vec![detail],
            },
            audio_file: None,
            failure_kind: Some(DictationFailureKind::AsrFailed),
        };
    }

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
                    provider_id: request.provider_id.clone(),
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
            provider_id: active.provider_id.clone(),
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
                    provider_id: request.provider_id.clone(),
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
    provider_id: Option<String>,
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
                            provider_id.clone(),
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
            provider_id.clone(),
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

/// Orchestrate a VAD-gated dictation session: the recorder rolls short
/// probes at `probe_secs` cadence; each finalized probe is classified by
/// the Python worker's `segment_probe` RPC; speech probes accumulate into
/// a pending buffer; the buffer flushes to `transcribe` on either a
/// silence run of `silence_gap_probes` probes or when the buffer's
/// duration would exceed `max_segment_secs`. Every flushed speech unit is
/// transcribed in the background so stop latency stays close to the stop
/// request's arrival.
#[allow(clippy::too_many_arguments, clippy::too_many_lines)]
async fn run_vad_gated_dictation_session(
    mut session: CaptureSession,
    recorder_backend: RecorderBackend,
    probe_secs: u32,
    max_segment_secs: u32,
    silence_gap_probes: u32,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
    provider_id: Option<String>,
    worker_command: worker::WorkerCommand,
    events: broadcast::Sender<EventEnvelope>,
    sessions: Arc<RwLock<HashMap<String, CaptureSession>>>,
    mut stop_rx: oneshot::Receiver<()>,
    recorder_spawner: segmented_recording::RecorderSpawner,
) -> DictationCaptureResult {
    let session_id = session.session_id;
    let probe_dir = temp_audio_path().with_extension("").join("probes");

    let mut recorder = match segmented_recording::SegmentedRecording::start_with_spawner(
        &probe_dir,
        recorder_backend,
        probe_secs,
        0,
        recorder_spawner,
    ) {
        Ok(recorder) => recorder,
        Err(error) => {
            session = transition_session_state(session, SessionState::Failed);
            update_session_map(&sessions, session.clone()).await;
            let _ = events.send(EventEnvelope::new(
                "dictation.failed",
                Some(session_id),
                format!("VAD-gated recorder failed to start: {error}"),
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

    let mut probes: Vec<ProbeRecord> = Vec::new();
    let mut pending_paths: Vec<PathBuf> = Vec::new();
    let mut pending_ids: Vec<usize> = Vec::new();
    let mut silent_streak: u32 = 0;
    let mut next_unit_id: usize = 0;
    let mut units: Vec<SegmentRecord> = Vec::new();
    let mut transcribe_tasks: JoinSet<SegmentTranscribeOutcome> = JoinSet::new();

    let mut ticker = interval(Duration::from_secs(u64::from(probe_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // The first tick fires immediately; skip it so the first rotation
    // happens after `probe_secs` of actual recording.
    ticker.tick().await;

    let mut recording_failure: Option<String> = None;
    let mut transcription_failure: Option<String> = None;

    loop {
        tokio::select! {
            biased;
            _ = &mut stop_rx => {
                break;
            }
            _ = ticker.tick() => {
                match recorder.finalize_current().await {
                    Ok(Some((id, path))) => {
                        let verdict = classify_probe(&worker_command, &path).await;
                        emit_probe_analyzed(&events, session_id, id, &path, &verdict);
                        probes.push(ProbeRecord {
                            id,
                            audio_path: path.clone(),
                        });

                        if verdict.has_speech {
                            pending_paths.push(path);
                            pending_ids.push(id);
                            silent_streak = 0;
                            let pending_secs = (pending_paths.len() as u32)
                                .saturating_mul(probe_secs);
                            if pending_secs >= max_segment_secs {
                                if let Err(detail) = flush_pending_speech_probes(
                                    &mut pending_paths,
                                    &mut pending_ids,
                                    &mut next_unit_id,
                                    &mut units,
                                    &mut transcribe_tasks,
                                    &probe_dir,
                                    &worker_command,
                                    &events,
                                    session_id,
                                    &language,
                                    translate_to_english,
                                    &provider_id,
                                )
                                .await
                                {
                                    transcription_failure.get_or_insert(detail);
                                }
                            }
                        } else if !pending_paths.is_empty() {
                            silent_streak = silent_streak.saturating_add(1);
                            if silent_streak >= silence_gap_probes {
                                if let Err(detail) = flush_pending_speech_probes(
                                    &mut pending_paths,
                                    &mut pending_ids,
                                    &mut next_unit_id,
                                    &mut units,
                                    &mut transcribe_tasks,
                                    &probe_dir,
                                    &worker_command,
                                    &events,
                                    session_id,
                                    &language,
                                    translate_to_english,
                                    &provider_id,
                                )
                                .await
                                {
                                    transcription_failure.get_or_insert(detail);
                                }
                                silent_streak = 0;
                            }
                        }
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

    // Drain probes produced by the recorder's graceful stop.
    let finalized_paths = match recorder.stop().await {
        Ok(paths) => paths,
        Err(error) => {
            if recording_failure.is_none() {
                recording_failure = Some(error.to_string());
            }
            Vec::new()
        }
    };

    let known_ids: HashSet<usize> = probes.iter().map(|probe| probe.id).collect();
    for (id, path) in finalized_paths {
        if known_ids.contains(&id) {
            continue;
        }
        let verdict = classify_probe(&worker_command, &path).await;
        emit_probe_analyzed(&events, session_id, id, &path, &verdict);
        probes.push(ProbeRecord {
            id,
            audio_path: path.clone(),
        });
        if verdict.has_speech {
            pending_paths.push(path);
            pending_ids.push(id);
        }
    }

    // Final flush: any pending speech that stop arrived mid-utterance.
    if !pending_paths.is_empty() {
        if let Err(detail) = flush_pending_speech_probes(
            &mut pending_paths,
            &mut pending_ids,
            &mut next_unit_id,
            &mut units,
            &mut transcribe_tasks,
            &probe_dir,
            &worker_command,
            &events,
            session_id,
            &language,
            translate_to_english,
            &provider_id,
        )
        .await
        {
            transcription_failure.get_or_insert(detail);
        }
    }

    session = transition_session_state(session, SessionState::Transcribing);
    update_session_map(&sessions, session.clone()).await;

    while let Some(outcome) = transcribe_tasks.join_next().await {
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => continue,
        };
        let unit = units.iter_mut().find(|unit| unit.id == outcome.segment_id);
        match outcome.result {
            Ok(result) => {
                if let Some(unit) = unit {
                    unit.transcript = Some(result.text.clone());
                    unit.detected_language = result.detected_language.clone();
                }
            }
            Err(error) => {
                let detail = error.to_string();
                if transcription_failure.is_none() {
                    transcription_failure = Some(detail);
                }
                if let Some(unit) = unit {
                    unit.transcript = Some(String::new());
                }
            }
        }
    }

    units.sort_by_key(|unit| unit.id);
    let (combined_text, detected_language) = stitch_segment_transcripts(&units);

    // `keep_audio=true` keeps every probe and every stitched unit so the
    // operator can inspect them; otherwise clean up both sets before
    // removing the probe directory.
    let audio_file_summary = if keep_audio {
        Some(probe_dir.display().to_string())
    } else {
        for probe in &probes {
            let _ = std::fs::remove_file(&probe.audio_path);
        }
        for unit in &units {
            // Only stitched units live at a path that isn't one of the
            // physical probes; single-probe units share storage with the
            // probe file and were already cleaned up above.
            if !probes
                .iter()
                .any(|probe| probe.audio_path == unit.audio_path)
            {
                let _ = std::fs::remove_file(&unit.audio_path);
            }
        }
        let _ = std::fs::remove_dir(&probe_dir);
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
            "VAD-gated dictation completed with {} speech unit(s) from {} probe(s).",
            units.len(),
            probes.len(),
        )
    } else {
        format!(
            "VAD-gated dictation failed: {}",
            failure_detail.as_deref().unwrap_or("unspecified failure"),
        )
    };
    let _ = events.send(EventEnvelope::new(
        event_type,
        Some(session_id),
        event_message,
    ));

    let mut notes = vec![format!(
        "{} speech unit(s) stitched from {} probe(s)",
        units.len(),
        probes.len(),
    )];
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
    provider_id: Option<String>,
) {
    transcribe_tasks.spawn(async move {
        let request = TranscribeRequest {
            audio_file: audio_path.display().to_string(),
            language,
            translate_to_english,
            provider_id,
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

fn emit_probe_analyzed(
    events: &broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    probe_id: usize,
    path: &Path,
    verdict: &SegmentProbeResult,
) {
    let classification = if verdict.has_speech {
        "speech"
    } else {
        "silence"
    };
    let _ = events.send(EventEnvelope::new(
        "dictation.probe_analyzed",
        Some(session_id),
        format!(
            "Probe {probe_id} at {} classified as {classification} (ratio={:.2}).",
            path.display(),
            verdict.speech_ratio,
        ),
    ));
}

/// Call the Python worker's `segment_probe` RPC. When the RPC itself
/// fails (VAD unconfigured, ONNX crash, etc.), fall back to a synthetic
/// "speech" verdict so the live dictation never silently drops audio
/// that might be speech. The synthetic note surfaces the reason so
/// operators can see why gating stopped behaving.
async fn classify_probe(worker_command: &worker::WorkerCommand, path: &Path) -> SegmentProbeResult {
    let request = SegmentProbeRequest {
        audio_file: path.display().to_string(),
    };
    match worker_command.segment_probe(&request).await {
        Ok(result) => result,
        Err(error) => SegmentProbeResult {
            has_speech: true,
            speech_ratio: 1.0,
            regions: Vec::new(),
            notes: vec![format!("segment_probe failed, treating as speech: {error}",)],
        },
    }
}

/// Stitch the accumulated speech probes into a single WAV (via the
/// Python worker's `stitch_wav_segments` RPC, skipped when only one
/// probe is pending), record the resulting speech unit, and spawn a
/// background transcription task. Clears `pending_paths` and
/// `pending_ids` on success and on stitch failure so the orchestrator
/// never re-emits a flushed probe.
#[allow(clippy::too_many_arguments)]
async fn flush_pending_speech_probes(
    pending_paths: &mut Vec<PathBuf>,
    pending_ids: &mut Vec<usize>,
    next_unit_id: &mut usize,
    units: &mut Vec<SegmentRecord>,
    transcribe_tasks: &mut JoinSet<SegmentTranscribeOutcome>,
    probe_dir: &Path,
    worker_command: &worker::WorkerCommand,
    events: &broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    language: &Option<String>,
    translate_to_english: bool,
    provider_id: &Option<String>,
) -> Result<(), String> {
    if pending_paths.is_empty() {
        return Ok(());
    }

    let unit_id = *next_unit_id;
    *next_unit_id += 1;

    let probe_count = pending_paths.len();
    let unit_path = if probe_count == 1 {
        pending_paths[0].clone()
    } else {
        let out_file = probe_dir.join(format!("unit-{unit_id:05}.wav"));
        let request = StitchWavSegmentsRequest {
            audio_files: pending_paths
                .iter()
                .map(|path| path.display().to_string())
                .collect(),
            out_file: out_file.display().to_string(),
        };
        match worker_command.stitch_wav_segments(&request).await {
            Ok(result) => PathBuf::from(result.audio_file),
            Err(error) => {
                let detail = format!("speech unit {unit_id} stitch failed: {error}");
                let _ = events.send(EventEnvelope::new(
                    "dictation.speech_unit_failed",
                    Some(session_id),
                    detail.clone(),
                ));
                pending_paths.clear();
                pending_ids.clear();
                return Err(detail);
            }
        }
    };

    pending_paths.clear();
    pending_ids.clear();

    units.push(SegmentRecord {
        id: unit_id,
        audio_path: unit_path.clone(),
        transcript: None,
        detected_language: None,
    });

    let _ = events.send(EventEnvelope::new(
        "dictation.speech_unit_flushed",
        Some(session_id),
        format!(
            "Speech unit {unit_id} flushed ({probe_count} probe{}).",
            if probe_count == 1 { "" } else { "s" },
        ),
    ));

    spawn_speech_unit_transcribe(
        transcribe_tasks,
        worker_command.clone(),
        events.clone(),
        session_id,
        unit_id,
        unit_path,
        language.clone(),
        translate_to_english,
        provider_id.clone(),
    );

    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spawn_speech_unit_transcribe(
    transcribe_tasks: &mut JoinSet<SegmentTranscribeOutcome>,
    worker_command: worker::WorkerCommand,
    events: broadcast::Sender<EventEnvelope>,
    session_id: Uuid,
    unit_id: usize,
    audio_path: PathBuf,
    language: Option<String>,
    translate_to_english: bool,
    provider_id: Option<String>,
) {
    transcribe_tasks.spawn(async move {
        let request = TranscribeRequest {
            audio_file: audio_path.display().to_string(),
            language,
            translate_to_english,
            provider_id,
        };
        let result = worker_command.transcribe(&request).await;
        let message = match &result {
            Ok(result) => {
                if result.text.trim().is_empty() {
                    format!("Speech unit {unit_id} transcribed with empty text.")
                } else {
                    format!(
                        "Speech unit {unit_id} transcribed ({} chars).",
                        result.text.len(),
                    )
                }
            }
            Err(error) => format!("Speech unit {unit_id} transcription failed: {error}"),
        };
        let _ = events.send(EventEnvelope::new(
            "dictation.speech_unit_transcribed",
            Some(session_id),
            message,
        ));
        SegmentTranscribeOutcome {
            segment_id: unit_id,
            result,
        }
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
    use std::sync::Mutex;

    /// Serialises any test that mutates process-wide env vars in this
    /// module. Cargo runs unit tests concurrently and Rust 2024 made
    /// `env::set_var` `unsafe` precisely because a concurrent reader
    /// in another thread is UB. Take this lock for the duration of
    /// every `XDG_RUNTIME_DIR` (or future env-var) mutation.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

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

    /// Pins the XDG-driven branch of `default_socket_path`. The
    /// systemd-managed install path goes through `XDG_RUNTIME_DIR`,
    /// so a regression that always returned `temp_dir()` would land
    /// the daemon socket in `/tmp/voicelayer/...` even on a properly
    /// configured user session.
    #[test]
    fn default_socket_path_uses_xdg_runtime_dir_when_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::set_var("XDG_RUNTIME_DIR", "/run/user/9001");
        }
        let path = default_socket_path();
        assert_eq!(path, PathBuf::from("/run/user/9001/voicelayer/daemon.sock"),);
        match previous {
            Some(value) => unsafe {
                std::env::set_var("XDG_RUNTIME_DIR", value);
            },
            None => unsafe {
                std::env::remove_var("XDG_RUNTIME_DIR");
            },
        }
    }

    /// Pins the fallback branch: with no `XDG_RUNTIME_DIR` configured
    /// (e.g. a headless `cargo test` running outside a logind session),
    /// the daemon must still resolve a socket path so the fallback
    /// path lands under `env::temp_dir()`. A regression that returned
    /// an empty `PathBuf` here would crash the daemon at bind time
    /// with no actionable error.
    #[test]
    fn default_socket_path_falls_back_to_temp_dir_when_xdg_runtime_dir_unset() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::remove_var("XDG_RUNTIME_DIR");
        }
        let path = default_socket_path();
        let expected_base = std::env::temp_dir();
        assert!(
            path.starts_with(&expected_base),
            "fallback path must live under env::temp_dir() (`{}`); got `{}`",
            expected_base.display(),
            path.display(),
        );
        assert!(
            path.ends_with("voicelayer/daemon.sock"),
            "fallback path must keep the `voicelayer/daemon.sock` suffix; got `{}`",
            path.display(),
        );
        if let Some(value) = previous {
            unsafe {
                std::env::set_var("XDG_RUNTIME_DIR", value);
            }
        }
    }

    /// Pins the env-set branch of `default_project_root`. The
    /// daemon resolves the Python worker's project root from this
    /// variable; a regression that always fell through to
    /// `current_dir()` would silently spawn the worker in the
    /// caller's cwd (e.g. `~`) where `uv run` cannot find the
    /// project, breaking JSON-RPC startup with a confusing error.
    #[test]
    fn default_project_root_uses_voicelayer_project_root_when_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_PROJECT_ROOT");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::set_var("VOICELAYER_PROJECT_ROOT", "/srv/voicelayer-from-systemd");
        }
        let path = default_project_root();
        assert_eq!(path, PathBuf::from("/srv/voicelayer-from-systemd"));
        match previous {
            Some(value) => unsafe {
                std::env::set_var("VOICELAYER_PROJECT_ROOT", value);
            },
            None => unsafe {
                std::env::remove_var("VOICELAYER_PROJECT_ROOT");
            },
        }
    }

    /// Pins the unset branch: with no override, the resolver falls
    /// through to `current_dir()`. The chained `unwrap_or_else(|| ".")`
    /// is the last-resort fallback for the unrealistic case where
    /// `current_dir()` itself fails (deleted cwd) — exercising it
    /// requires unsafe filesystem manipulation, so this test pins
    /// the realistic happy path: env unset, cwd available, result
    /// equals the cwd.
    #[test]
    fn default_project_root_falls_back_to_current_dir_when_env_unset() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_PROJECT_ROOT");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::remove_var("VOICELAYER_PROJECT_ROOT");
        }
        let path = default_project_root();
        let cwd = std::env::current_dir().expect("current_dir should be available in tests");
        assert_eq!(path, cwd);
        if let Some(value) = previous {
            unsafe {
                std::env::set_var("VOICELAYER_PROJECT_ROOT", value);
            }
        }
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
            timeout_override: None,
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
    use super::run_vad_gated_dictation_session;
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

    #[tokio::test]
    async fn vad_gated_orchestrator_stops_immediately_and_flushes_tail_probe_as_speech() {
        // Mirrors the segmented stop-immediately test but for VAD-gated
        // mode. `probe_secs` is large enough that the tick loop never
        // fires; the stop path runs `recorder.stop()` which returns the
        // in-flight probe (`segment-00000`), and the orchestrator
        // classifies + flushes it as a one-probe speech unit.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " tail probe text"},
                "segment_probe_default": {
                    "has_speech": true,
                    "speech_ratio": 0.9,
                    "regions": [],
                    "notes": [],
                },
            }),
        );
        let session = fresh_dictation_session();
        let session_id = session.session_id;
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,  // probe_secs — ticker never fires before stop
            120, // max_segment_secs
            1,   // silence_gap_probes
            false,
            false,
            None,
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

        assert!(
            types.contains(&"dictation.probe_analyzed"),
            "probe_analyzed event missing; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.speech_unit_flushed"),
            "speech_unit_flushed event missing; actual types: {types:?}",
        );
        assert!(
            types.contains(&"dictation.speech_unit_transcribed"),
            "speech_unit_transcribed event missing; actual types: {types:?}",
        );
        assert_eq!(
            types.last().copied(),
            Some("dictation.completed"),
            "the last event must be completed; actual types: {types:?}",
        );

        assert_eq!(result.session.session_id, session_id);
        assert_eq!(result.session.state, SessionState::Completed);
        assert_eq!(result.transcription.text, "tail probe text");
        assert_eq!(
            result.transcription.detected_language.as_deref(),
            Some("en")
        );
        assert!(result.failure_kind.is_none());
    }

    #[tokio::test]
    async fn vad_gated_orchestrator_treats_silent_tail_probe_as_no_flush() {
        // The same shape as above but the tail probe is classified as
        // silence. Nothing enters the pending buffer, no speech unit is
        // flushed, and the session still completes with an empty
        // transcript (zero flushed units is a legitimate outcome, not a
        // failure).
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " never read"},
                "segment_probe_default": {
                    "has_speech": false,
                    "speech_ratio": 0.0,
                    "regions": [],
                    "notes": [],
                },
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,
            120,
            1,
            false,
            false,
            None,
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

        assert!(
            types.contains(&"dictation.probe_analyzed"),
            "probe_analyzed event missing; actual types: {types:?}",
        );
        assert!(
            !types.contains(&"dictation.speech_unit_flushed"),
            "silent probe must not trigger a flush; actual types: {types:?}",
        );
        assert!(
            !types.contains(&"dictation.speech_unit_transcribed"),
            "no flush means no transcribe event; actual types: {types:?}",
        );
        assert_eq!(
            types.last().copied(),
            Some("dictation.completed"),
            "silence-only dictation still completes cleanly; actual types: {types:?}",
        );

        assert_eq!(result.session.state, SessionState::Completed);
        assert_eq!(result.transcription.text, "");
        assert!(result.transcription.detected_language.is_none());
        assert!(result.failure_kind.is_none());
    }

    #[tokio::test]
    async fn vad_gated_orchestrator_reports_asr_failed_when_transcribe_fails() {
        // Pessimistic fallback: classify the probe as speech, then force
        // the transcribe worker call to fail. The orchestrator must
        // record `AsrFailed` and surface the failure detail without
        // hanging.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "fail_stems": ["segment-00000"],
                "segment_probe_default": {
                    "has_speech": true,
                    "speech_ratio": 1.0,
                    "regions": [],
                    "notes": [],
                },
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, _events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            60,
            120,
            1,
            false,
            false,
            None,
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

        assert_eq!(result.session.state, SessionState::Failed);
        assert_eq!(result.failure_kind, Some(DictationFailureKind::AsrFailed));
        assert!(
            result
                .transcription
                .notes
                .iter()
                .any(|note| note.contains("segment-00000") || note.contains("mock worker")),
            "failure detail must mention the failing stem; got {:?}",
            result.transcription.notes,
        );
    }

    /// Pins the `silence_gap_probes > 1` branch of the orchestrator's
    /// flush logic. The earlier tail-probe tests use `probe_secs = 60`
    /// so the ticker never fires, exercising only the stop-path drain;
    /// none of them probe the multi-probe accumulation rule.
    ///
    /// Scenario: speech probe → silence probe → silence probe. With
    /// `silence_gap_probes = 2` the first silent probe must not flush
    /// (silent_streak < gap), and the second must (silent_streak == gap).
    /// A regression that compared `==` instead of `>=` would still pass
    /// the existing default-1 tests, so this scenario is the only
    /// place the threshold logic is genuinely exercised.
    #[tokio::test]
    async fn vad_gated_orchestrator_holds_speech_until_silence_gap_threshold_met() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            // The single-probe flush takes the shortcut: unit_path is the
            // probe path itself, so transcribe is keyed on the probe
            // stem rather than a synthesized `unit-00000`.
            serde_json::json!({
                "transcribe_map": {"segment-00000": " kept until silence ran"},
                "segment_probe_map": {
                    "segment-00000": {
                        "has_speech": true,
                        "speech_ratio": 0.9,
                        "regions": [],
                        "notes": [],
                    },
                    "segment-00001": {
                        "has_speech": false,
                        "speech_ratio": 0.0,
                        "regions": [],
                        "notes": [],
                    },
                    "segment-00002": {
                        "has_speech": false,
                        "speech_ratio": 0.0,
                        "regions": [],
                        "notes": [],
                    },
                },
                "segment_probe_default": {
                    "has_speech": false,
                    "speech_ratio": 0.0,
                    "regions": [],
                    "notes": [],
                },
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            1,   // probe_secs
            120, // max_segment_secs (large enough not to interfere)
            2,   // silence_gap_probes — the contract under test
            false,
            false,
            None,
            None,
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        // Real-time wait for ~3 ticks (probe_secs=1) plus margin for
        // the worker subprocess spawns. Tests on the existing CI
        // runner finish within ~1m20s for similar shapes; this slice
        // is well under that budget.
        sleep(Duration::from_millis(3500)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let result = timeout(Duration::from_secs(30), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let events = drain_events(&mut events_rx);
        let types = event_types(&events);

        // The first silent probe alone must not flush; the test passes
        // only if a flush event eventually appears (after the streak
        // crosses the threshold) and exactly one transcribed speech
        // unit lands.
        let flush_count = events
            .iter()
            .filter(|e| e.event_type == "dictation.speech_unit_flushed")
            .count();
        assert_eq!(
            flush_count, 1,
            "silence_gap=2 with one speech probe followed by silence must \
             flush exactly once after the streak crosses the threshold; \
             event types: {types:?}",
        );

        assert_eq!(result.session.state, SessionState::Completed);
        assert!(result.failure_kind.is_none());
        assert!(
            result.transcription.text.contains("kept until silence ran"),
            "the buffered speech probe must transcribe through to the result; got {:?}",
            result.transcription.text,
        );
    }

    /// Pins the `pending_secs >= max_segment_secs` forced-flush branch.
    /// The orchestrator must not let a continuous speech run accumulate
    /// past `max_segment_secs` even when no silence appears — otherwise
    /// a never-pausing speaker would block the entire session in the
    /// pending buffer until stop.
    ///
    /// Scenario: speech, speech with `max_segment_secs = 2`,
    /// `silence_gap_probes = 99` (large enough that silence cannot
    /// trigger a flush). After the second probe, `pending_secs` reaches
    /// 2 and the forced flush fires; the two-probe flush goes through
    /// `stitch_wav_segments` (probe_count > 1 branch) into a synthesized
    /// `unit-00000.wav` whose transcribe key the mock worker honours.
    #[tokio::test]
    async fn vad_gated_orchestrator_force_flushes_when_max_segment_secs_reached() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"unit-00000": " forced flush text"},
                "segment_probe_default": {
                    "has_speech": true,
                    "speech_ratio": 1.0,
                    "regions": [],
                    "notes": [],
                },
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, mut events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            1,  // probe_secs
            2,  // max_segment_secs — flushes after two speech probes
            99, // silence_gap_probes — large enough not to trigger
            false,
            false,
            None,
            None,
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        // Two probe ticks (~2 s) plus margin for subprocess spawns.
        sleep(Duration::from_millis(2800)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let result = timeout(Duration::from_secs(30), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let events = drain_events(&mut events_rx);
        let types = event_types(&events);

        let flush_count = events
            .iter()
            .filter(|e| e.event_type == "dictation.speech_unit_flushed")
            .count();
        assert!(
            flush_count >= 1,
            "max_segment_secs=2 with continuous speech must force at least \
             one flush; event types: {types:?}",
        );

        assert_eq!(result.session.state, SessionState::Completed);
        assert!(result.failure_kind.is_none());
        assert!(
            result.transcription.text.contains("forced flush text"),
            "the stitched speech unit must transcribe through to the result; got {:?}",
            result.transcription.text,
        );
    }

    /// Pin the dictation pipeline's session-level `provider_id` plumbing.
    /// When the orchestrator is started with a non-`None` provider, every
    /// fresh transcribe call it spawns — segments captured during the
    /// session and segments reaped from the recorder's graceful stop —
    /// must carry the same value into the worker. Without this guarantee
    /// a session whose operator selected `mimo_v2_5_asr` could silently
    /// fall back to whisper for any subset of its segments, which is
    /// exactly the behavior the new flag is meant to rule out.
    #[tokio::test]
    async fn segmented_orchestrator_threads_session_provider_id_to_every_transcribe() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let log_path = tempdir.path().join("transcribe_calls.jsonl");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " hello"},
                "fail_stems": [],
                "transcribe_log_path": log_path.display().to_string(),
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
            false,
            false,
            None,
            Some("mimo_v2_5_asr".to_owned()),
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        sleep(Duration::from_millis(50)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let _result = timeout(Duration::from_secs(10), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let log_contents =
            std::fs::read_to_string(&log_path).expect("mock worker writes the transcribe log");
        let provider_ids: Vec<Option<String>> = log_contents
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let value: serde_json::Value =
                    serde_json::from_str(line).expect("each log line is valid json");
                value
                    .get("provider_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert!(
            !provider_ids.is_empty(),
            "the orchestrator must drive at least one transcribe call (segment-00000); got log: {log_contents:?}",
        );
        for provider_id in &provider_ids {
            assert_eq!(
                provider_id.as_deref(),
                Some("mimo_v2_5_asr"),
                "session-level provider_id must be threaded into every transcribe call; \
                 saw {provider_id:?} in log: {log_contents:?}",
            );
        }
    }

    /// Mirror the segmented pin for the VAD-gated path: the speech-unit
    /// transcribe spawned after a forced flush must carry the session's
    /// `provider_id` exactly the same as a fixed-segment transcribe
    /// would. Without this, an operator who picks `mimo_v2_5_asr` would
    /// have only the OneShot/Fixed paths actually route to MiMo while
    /// the VAD-gated path stayed on whisper.
    #[tokio::test]
    async fn vad_gated_orchestrator_threads_session_provider_id_to_every_transcribe() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let log_path = tempdir.path().join("transcribe_calls.jsonl");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"unit-00000": " forced flush text"},
                "fail_stems": [],
                "transcribe_log_path": log_path.display().to_string(),
            }),
        );
        let session = fresh_dictation_session();
        let (events_tx, _events_rx) = broadcast::channel(64);
        let sessions = Arc::new(RwLock::new(HashMap::new()));
        let (stop_tx, stop_rx) = oneshot::channel();

        let orchestrator = tokio::spawn(run_vad_gated_dictation_session(
            session,
            RecorderBackend::Pipewire,
            1,  // probe_secs
            2,  // max_segment_secs — flushes after two speech probes
            99, // silence_gap_probes — large enough not to trigger
            false,
            false,
            None,
            Some("mimo_v2_5_asr".to_owned()),
            worker,
            events_tx,
            Arc::clone(&sessions),
            stop_rx,
            fake_successful_spawner,
        ));

        // Let the orchestrator drive a couple of probe-tick cycles so a
        // forced flush actually fires and the unit transcribe spawns.
        sleep(Duration::from_millis(2_500)).await;
        stop_tx.send(()).expect("orchestrator listening on stop_rx");

        let _result = timeout(Duration::from_secs(30), orchestrator)
            .await
            .expect("orchestrator finished in time")
            .expect("orchestrator task joined cleanly");

        let log_contents =
            std::fs::read_to_string(&log_path).expect("mock worker writes the transcribe log");
        let provider_ids: Vec<Option<String>> = log_contents
            .lines()
            .filter(|line| !line.is_empty())
            .map(|line| {
                let value: serde_json::Value =
                    serde_json::from_str(line).expect("each log line is valid json");
                value
                    .get("provider_id")
                    .and_then(serde_json::Value::as_str)
                    .map(str::to_owned)
            })
            .collect();
        assert!(
            !provider_ids.is_empty(),
            "the vad-gated orchestrator must drive at least one transcribe call after a flush; \
             got log: {log_contents:?}",
        );
        for provider_id in &provider_ids {
            assert_eq!(
                provider_id.as_deref(),
                Some("mimo_v2_5_asr"),
                "session-level provider_id must be threaded into every speech-unit transcribe; \
                 saw {provider_id:?} in log: {log_contents:?}",
            );
        }
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
            mimo_configured: false,
            mimo_model_path: Some(String::new()),
            mimo_error: Some(String::new()),
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
            provider_id: None,
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
            provider_id: None,
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
    async fn vad_gated_dictation_http_roundtrip_returns_speech_unit_transcription() {
        // Mirrors `segmented_dictation_http_roundtrip_returns_stitched_transcription`
        // but for the VAD-gated mode: start a session with `probe_secs`
        // large enough that the tick loop never fires, let the
        // orchestrator boot the first probe, stop it, and verify the
        // tail probe is classified as speech, flushed as a single
        // speech unit, and transcribed through to the HTTP response.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {"segment-00000": " vad gated http"},
                "segment_probe_default": {
                    "has_speech": true,
                    "speech_ratio": 0.85,
                    "regions": [],
                    "notes": [],
                },
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
            segmentation: SegmentationMode::VadGated {
                probe_secs: 60,
                max_segment_secs: 120,
                silence_gap_probes: 1,
            },
            provider_id: None,
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
        assert_eq!(result.transcription.text, "vad gated http");
        assert_eq!(result.failure_kind, None);
    }

    #[tokio::test]
    async fn vad_gated_dictation_start_with_zero_probe_secs_short_circuits_to_failed() {
        // `probe_secs == 0` is rejected at dispatch: the ticker would
        // spin forever. The session returns already in Failed state and
        // the follow-up stop classifies the missing active dictation as
        // a recording failure — mirroring the `segment_secs == 0` pin
        // for Fixed mode.
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
            segmentation: SegmentationMode::VadGated {
                probe_secs: 0,
                max_segment_secs: 60,
                silence_gap_probes: 1,
            },
            provider_id: None,
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(session.state, SessionState::Failed);

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
        );
    }

    #[tokio::test]
    async fn list_sessions_returns_empty_array_when_no_sessions_have_been_created() {
        // The endpoint is unconditionally available; on a fresh daemon
        // it must return `[]`, not `{"sessions": []}` or 404. Pin both
        // the status and the empty-array shape so a future wrapper
        // type would require an explicit migration.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let (status, sessions): (StatusCode, Vec<CaptureSession>) =
            get_json(router, "/v1/sessions").await;
        assert_eq!(status, StatusCode::OK);
        assert!(sessions.is_empty());
    }

    #[tokio::test]
    async fn list_sessions_retains_completed_session_after_stop() {
        // Pin the contract `vl dictation list`'s docstring promises:
        // sessions stay in the daemon's map across the listening →
        // completed transition, so an operator can recover a
        // `session_id` even after the dictation has already finished.
        // `update_session_map` is `insert`-only — never `remove` — so
        // a future change that pruned terminal-state sessions to cap
        // memory growth would break this contract silently. This test
        // catches that.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "default_transcribe_text": " retained",
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
            provider_id: None,
        };
        let (_, started): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;

        // Same 50 ms beat as `oneshot_dictation_http_roundtrip_*`:
        // lets the fake recorder child exit naturally rather than
        // racing the SIGINT timeout path.
        sleep(Duration::from_millis(50)).await;

        let (_, _): (StatusCode, DictationCaptureResult) = post_json(
            router.clone(),
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest {
                session_id: started.session_id,
            })
            .expect("encode stop req"),
        )
        .await;

        let (status, sessions): (StatusCode, Vec<CaptureSession>) =
            get_json(router, "/v1/sessions").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(
            sessions.len(),
            1,
            "the stopped session must still be enumerable; pruning would silently break \
             `vl dictation list` for terminal-state recovery",
        );
        assert_eq!(sessions[0].session_id, started.session_id);
        assert_eq!(
            sessions[0].state,
            SessionState::Completed,
            "the listing must reflect the post-stop state, not the original Listening",
        );
    }

    #[tokio::test]
    async fn list_sessions_includes_a_running_session_after_dictation_start() {
        // Start a one-shot live session via the public API and confirm
        // `GET /v1/sessions` surfaces it. Pinning the round-trip
        // protects against a future change that registered sessions
        // somewhere other than `state.sessions` and silently broke
        // the listing.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = StartDictationRequest {
            trigger: TriggerKind::Cli,
            language_profile: None,
            recorder_backend: None,
            translate_to_english: false,
            keep_audio: false,
            segmentation: SegmentationMode::OneShot,
            provider_id: None,
        };
        let (post_status, started): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&request).expect("encode start req"),
        )
        .await;
        assert_eq!(post_status, StatusCode::OK);
        assert_eq!(started.state, SessionState::Listening);

        let (status, sessions): (StatusCode, Vec<CaptureSession>) =
            get_json(router, "/v1/sessions").await;
        assert_eq!(status, StatusCode::OK);
        assert_eq!(sessions.len(), 1);
        assert_eq!(sessions[0].session_id, started.session_id);
        assert_eq!(sessions[0].state, SessionState::Listening);
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
            provider_id: None,
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

    /// Pin the HTTP-level wiring for session-level provider selection on
    /// the one-shot path. POST `/v1/sessions/dictation` with a non-`None`
    /// `provider_id`, drive the recorder briefly, then POST stop and
    /// confirm the worker observed the provider id on the resulting
    /// transcribe call. Without this the operator could set the flag in
    /// the request and the daemon could silently drop it on the way to
    /// `stop_oneshot_dictation`'s `TranscribeRequest`, sending whisper
    /// instead of MiMo.
    #[tokio::test]
    async fn oneshot_dictation_http_threads_provider_id_into_worker_transcribe() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let log_path = tempdir.path().join("transcribe_calls.jsonl");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "default_transcribe_text": " oneshot reply",
                "transcribe_log_path": log_path.display().to_string(),
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
            provider_id: Some("mimo_v2_5_asr".to_owned()),
        };
        let (status, session): (StatusCode, CaptureSession) = post_json(
            router.clone(),
            "/v1/sessions/dictation",
            serde_json::to_value(&start_req).expect("encode start req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);
        let session_id = session.session_id;

        sleep(Duration::from_millis(50)).await;

        let (status, _result): (StatusCode, DictationCaptureResult) = post_json(
            router,
            "/v1/sessions/dictation/stop",
            serde_json::to_value(&StopDictationRequest { session_id }).expect("encode stop req"),
        )
        .await;
        assert_eq!(status, StatusCode::OK);

        let log_contents =
            std::fs::read_to_string(&log_path).expect("mock worker writes the transcribe log");
        let lines: Vec<&str> = log_contents
            .lines()
            .filter(|line| !line.is_empty())
            .collect();
        assert_eq!(
            lines.len(),
            1,
            "one-shot stop must drive exactly one transcribe call; got log: {log_contents:?}",
        );
        let entry: serde_json::Value =
            serde_json::from_str(lines[0]).expect("the single log line is valid json");
        assert_eq!(
            entry.get("provider_id").and_then(serde_json::Value::as_str),
            Some("mimo_v2_5_asr"),
            "session provider_id must reach the worker; entry: {entry}",
        );
    }

    // The ad-hoc capture happy path (`POST /v1/dictation/capture` with
    // a non-`None` `provider_id`) is intentionally not pinned with a
    // mock-worker log here. `capture_dictation_with_state` reaches a
    // real `pw-record` / `arecord` subprocess via
    // `recording::record_audio_file`, which is not present on the CI
    // runner. The provider_id transport itself is a one-line
    // `request.provider_id.clone()` into `TranscribeRequest` (see
    // crates/voicelayerd/src/lib.rs around line 985), structurally
    // identical to the OneShot path that
    // `oneshot_dictation_http_threads_provider_id_into_worker_transcribe`
    // already pins end-to-end. The negative pin
    // `capture_dictation_rejects_unknown_provider_id_before_recording`
    // covers this endpoint's validation contract without spawning a
    // real recorder.

    /// Pre-flight provider validation: an unknown `provider_id` on
    /// `POST /v1/sessions/dictation` must be rejected with 400 before
    /// the recorder spawns. Without this guard, an operator who
    /// mistypes `--provider-id mimo_v2_5` (missing `_asr`) would
    /// successfully start a listening session, record audio for the
    /// full duration, and only see the failure when stop reaches the
    /// worker — wasting the entire capture.
    #[tokio::test]
    async fn create_dictation_session_rejects_unknown_provider_id_before_recording() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let log_path = tempdir.path().join("transcribe_calls.jsonl");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "transcribe_log_path": log_path.display().to_string(),
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = serde_json::json!({
            "trigger": "cli",
            "segmentation": {"mode": "one_shot"},
            "provider_id": "mimo_v2_5",
        });
        let (status, body): (StatusCode, serde_json::Value) =
            post_json(router, "/v1/sessions/dictation", request).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "unknown provider_id must surface as 400 before recording starts; got body: {body}",
        );
        assert_eq!(
            body.get("error").and_then(serde_json::Value::as_str),
            Some("invalid_provider_id"),
            "rejection error code must classify the failure; got body: {body}",
        );
        // No transcribe call should have reached the worker — the
        // request was rejected before any audio was captured.
        assert!(
            !log_path.exists()
                || std::fs::read_to_string(&log_path)
                    .unwrap_or_default()
                    .is_empty(),
            "no transcribe call should fire when the request is rejected at the entry point",
        );
    }

    /// Pre-flight provider validation: `mimo_v2_5_asr` does not
    /// support `translate_to_english`. The worker would surface the
    /// limitation, but only after recording — fail at request time
    /// instead so the operator never wastes capture on a flag combo
    /// that is documented to fail.
    #[tokio::test]
    async fn create_dictation_session_rejects_mimo_with_translate_to_english() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({"transcribe_map": {}, "fail_stems": []}),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = serde_json::json!({
            "trigger": "cli",
            "segmentation": {"mode": "one_shot"},
            "provider_id": "mimo_v2_5_asr",
            "translate_to_english": true,
        });
        let (status, body): (StatusCode, serde_json::Value) =
            post_json(router, "/v1/sessions/dictation", request).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "mimo_v2_5_asr + translate_to_english must surface as 400; got body: {body}",
        );
        assert_eq!(
            body.get("error").and_then(serde_json::Value::as_str),
            Some("translate_to_english_unsupported"),
            "rejection error code must classify the failure; got body: {body}",
        );
    }

    /// Same pre-flight on the ad-hoc capture path: `POST
    /// /v1/dictation/capture` rejects an unknown provider_id before
    /// the recorder spawns. Mirrors the
    /// `create_dictation_session_rejects_unknown_provider_id_before_recording`
    /// pin one layer up, since the desktop overlay and `vl
    /// record-transcribe` reach this endpoint when no live session is
    /// in flight.
    #[tokio::test]
    async fn capture_dictation_rejects_unknown_provider_id_before_recording() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let log_path = tempdir.path().join("transcribe_calls.jsonl");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "transcribe_log_path": log_path.display().to_string(),
            }),
        );
        let state = build_app_state(test_config(worker), fake_successful_spawner);
        let router = build_app_router(state);

        let request = serde_json::json!({
            "trigger": "cli",
            "duration_seconds": 1,
            "recorder_backend": "pipewire",
            "translate_to_english": false,
            "keep_audio": false,
            "provider_id": "voxtral_realtime",
        });
        let (status, body): (StatusCode, serde_json::Value) =
            post_json(router, "/v1/dictation/capture", request).await;
        assert_eq!(
            status,
            StatusCode::BAD_REQUEST,
            "advertised-only catalog id (voxtral_realtime) must be rejected at the entry point; \
             got body: {body}",
        );
        assert_eq!(
            body.get("error").and_then(serde_json::Value::as_str),
            Some("invalid_provider_id"),
        );
        assert!(
            !log_path.exists()
                || std::fs::read_to_string(&log_path)
                    .unwrap_or_default()
                    .is_empty(),
            "no transcribe call should fire on a rejected capture request",
        );
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
    async fn compose_http_roundtrip_surfaces_worker_error_as_needs_provider_receipt() {
        // Complements the happy-path compose test by driving the
        // `Err(error)` arm of `create_composition_job` through the
        // mock worker's `fail_methods` knob. The handler wraps the
        // JSON-RPC error as a `CompositionReceipt` with
        // `PreviewStatus::NeedsProvider`, returns HTTP 200 (not 5xx)
        // so the CLI/UI can render the preview state, and appends a
        // `"Worker bridge detail: ..."` note for diagnostics.
        let tempdir = tempfile::tempdir().expect("tempdir");
        let worker = mock_worker_command(
            tempdir.path(),
            serde_json::json!({
                "transcribe_map": {},
                "fail_stems": [],
                "fail_methods": ["compose"],
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
        assert_eq!(receipt.preview.status, PreviewStatus::NeedsProvider);
        assert_eq!(
            receipt.preview.title,
            "Composition preview is waiting for a configured provider.",
        );
        assert!(
            receipt.preview.generated_text.is_none(),
            "NeedsProvider receipts must not carry fabricated generated_text; got {:?}",
            receipt.preview.generated_text,
        );
        assert!(
            receipt
                .preview
                .notes
                .iter()
                .any(|note| note.starts_with("Worker bridge detail:")),
            "error path must append a `Worker bridge detail:` diagnostic note; got {:?}",
            receipt.preview.notes,
        );
        // Mode-specific pin: `PreviewArtifact::needs_provider` derives the
        // first note from the `SessionMode` passed by the handler. If
        // `create_composition_job` were wired to the wrong mode, the
        // title assertion above would still pass but this note would
        // identify the wrong workflow.
        assert!(
            receipt
                .preview
                .notes
                .iter()
                .any(|note| note.contains("composition workflow")),
            "first note must identify the composition workflow; got {:?}",
            receipt.preview.notes,
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
            provider_id: None,
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

#[cfg(test)]
mod openapi_route_alignment_tests {
    //! Cross-check that every axum route declared in
    //! `build_app_router` has a matching `paths:` entry in
    //! `openapi/voicelayerd.v1.yaml`, and that every openapi path has
    //! a matching axum route. The per-schema drift guards in
    //! `voicelayer-core::domain::tests` (#28 / #33 / #34 / #35 / #36 /
    //! #37) cover the *response shape* contract; this module covers
    //! the *endpoint surface* contract — a shipped-but-undocumented
    //! route or a documented-but-unimplemented path was previously
    //! invisible to every existing guard.

    use std::collections::{BTreeMap, BTreeSet};

    /// Allowlist of HTTP method tokens we recognise on either side of
    /// the contract. Keeps the parsers from mistaking other
    /// four-space-indent keys (e.g. `summary:`, `responses:`,
    /// `requestBody:`, `parameters:`) for methods just because they
    /// happen to be at the same depth, and excludes axum builder
    /// helpers like `with_state` that are never legitimate methods.
    const HTTP_METHODS: &[&str] = &[
        "get", "post", "put", "delete", "patch", "head", "options", "trace",
    ];

    /// Walk a source string and pull out every `(path, method)` pair
    /// declared by `.route("PATH", METHOD(...))` calls. Stops at the
    /// first `#[cfg(test)]` line so synthetic fixtures and helper
    /// strings inside test modules do not pollute the production
    /// route set. Production routes in `voicelayerd::lib.rs` all
    /// land in `build_app_router` ahead of every test module.
    fn collect_axum_route_methods(source: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut routes: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        for line in source.lines() {
            if line.trim() == "#[cfg(test)]" {
                break;
            }
            let trimmed = line.trim_start();
            let Some(after_path_open) = trimmed.strip_prefix(".route(\"") else {
                continue;
            };
            let Some(quote_end) = after_path_open.find('"') else {
                continue;
            };
            let path = after_path_open[..quote_end].to_owned();
            // After the closing `"` of the path, the next non-comma /
            // non-space token is the method builder.
            let tail = after_path_open[quote_end + 1..]
                .trim_start_matches(',')
                .trim_start();
            let Some(open_paren) = tail.find('(') else {
                continue;
            };
            let method = tail[..open_paren].trim().to_owned();
            if !HTTP_METHODS.contains(&method.as_str()) {
                // `.route(path, with_state(...))` and similar
                // non-method builder calls would otherwise pollute
                // the set. We only care about the eight HTTP verbs.
                continue;
            }
            routes.entry(path).or_default().insert(method);
        }
        routes
    }

    /// Pull a method token out of a four-space-indent line. Tolerates
    /// both the block form (`    get:` followed by nested keys on
    /// the next line) and the inline form (`    get: {}`,
    /// `    get: someValue`); both end up with the method name in the
    /// substring before the first `:`. Returns `None` when the line is
    /// not at four-space indent, has no colon, or names a key that is
    /// not on `HTTP_METHODS` (so `summary:`, `responses:`,
    /// `requestBody:`, etc. are filtered out).
    fn extract_method_at_method_indent(line: &str) -> Option<&str> {
        let rest = line.strip_prefix("    ")?;
        if rest.starts_with(' ') {
            return None;
        }
        let colon_pos = rest.find(':')?;
        let method = &rest[..colon_pos];
        if HTTP_METHODS.contains(&method) {
            Some(method)
        } else {
            None
        }
    }

    /// Walk an openapi YAML and pull out every `(path, method)` pair.
    /// Path keys are at two-space indent under the `paths:` block;
    /// each method is a four-space-indent key under its parent path
    /// whose name is one of the allowlisted HTTP verbs. Bigger keys
    /// at the same indent (`summary:`, `responses:`, `requestBody:`)
    /// are filtered out by the allowlist.
    fn collect_openapi_path_methods(contents: &str) -> BTreeMap<String, BTreeSet<String>> {
        let mut paths: BTreeMap<String, BTreeSet<String>> = BTreeMap::new();
        let mut in_paths = false;
        let mut current_path: Option<String> = None;
        for line in contents.lines() {
            if line == "paths:" {
                in_paths = true;
                continue;
            }
            if in_paths
                && !line.is_empty()
                && line.chars().next().is_some_and(|c| !c.is_whitespace())
            {
                in_paths = false;
                current_path = None;
                continue;
            }
            if !in_paths {
                continue;
            }
            if let Some(rest) = line.strip_prefix("  /")
                && let Some(name) = rest.strip_suffix(':')
                && !name.contains(' ')
                && !name.is_empty()
            {
                let path = format!("/{name}");
                paths.entry(path.clone()).or_default();
                current_path = Some(path);
                continue;
            }
            if let Some(path) = &current_path
                && let Some(method) = extract_method_at_method_indent(line)
            {
                paths
                    .get_mut(path)
                    .expect("current_path entry was inserted above")
                    .insert(method.to_owned());
            }
        }
        paths
    }

    #[test]
    fn collect_axum_route_methods_extracts_each_method_token() {
        let source = "\
            Router::new()\n\
                .route(\"/v1/health\", get(get_health))\n\
                .route(\"/v1/sessions\", get(list_sessions))\n\
                .route(\"/v1/sessions/dictation\", post(create_dictation_session))\n\
                .with_state(state)\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert_eq!(routes.len(), 3);
        assert!(routes.get("/v1/health").is_some_and(|m| m.contains("get")));
        assert!(
            routes
                .get("/v1/sessions")
                .is_some_and(|m| m.contains("get"))
        );
        assert!(
            routes
                .get("/v1/sessions/dictation")
                .is_some_and(|m| m.contains("post"))
        );
    }

    #[test]
    fn collect_axum_route_methods_ignores_non_method_builders() {
        // `.with_state(state)` looks like `.method(...)` to a naive
        // parser. The HTTP-method allowlist filters it out so a
        // builder chain entry never lands as a synthetic route.
        let source = "\
            Router::new()\n\
                .route(\"/v1/real\", get(real_handler))\n\
                .with_state(state)\n\
                .layer(some_layer())\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert_eq!(routes.len(), 1);
        assert!(routes.get("/v1/real").is_some_and(|m| m.contains("get")));
    }

    #[test]
    fn collect_axum_route_methods_stops_at_first_cfg_test_line() {
        let source = "\
            Router::new()\n\
                .route(\"/v1/real\", get(handler))\n\
            #[cfg(test)]\n\
            mod tests {\n\
                let _ = router.route(\"/v1/synthetic\", get(handler));\n\
            }\n\
        ";
        let routes = collect_axum_route_methods(source);
        assert!(routes.contains_key("/v1/real"));
        assert!(
            !routes.contains_key("/v1/synthetic"),
            "test-block routes must be ignored; got {routes:?}",
        );
    }

    #[test]
    fn collect_openapi_path_methods_extracts_methods_under_each_path() {
        let yaml = concat!(
            "openapi: 3.1.0\n",
            "info:\n",
            "  title: Local Control API\n",
            "paths:\n",
            "  /v1/health:\n",
            "    get:\n",
            "      operationId: getHealth\n",
            "  /v1/sessions/dictation:\n",
            "    post:\n",
            "      operationId: startLiveDictationSession\n",
            "      requestBody:\n",
            "        required: true\n",
            "components:\n",
            "  schemas:\n",
            "    HealthResponse:\n",
            "      type: object\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        assert_eq!(paths.len(), 2);
        assert!(paths.get("/v1/health").is_some_and(|m| m.contains("get")));
        assert!(
            paths
                .get("/v1/sessions/dictation")
                .is_some_and(|m| m.contains("post"))
        );
    }

    #[test]
    fn collect_openapi_path_methods_ignores_non_method_keys_at_method_indent() {
        // `summary:`, `description:`, `responses:`, `requestBody:`,
        // `parameters:` all live at four-space indent under a path
        // entry. Only the eight HTTP verbs are real method keys; the
        // allowlist must filter the rest out.
        let yaml = concat!(
            "paths:\n",
            "  /v1/foo:\n",
            "    summary: not a method\n",
            "    description: also not a method\n",
            "    get:\n",
            "      operationId: getFoo\n",
            "    requestBody:\n",
            "      content: {}\n",
            "components: {}\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        let methods = paths.get("/v1/foo").expect("/v1/foo declared");
        assert_eq!(methods.len(), 1);
        assert!(methods.contains("get"));
    }

    #[test]
    fn collect_openapi_path_methods_ignores_indented_keys_outside_paths_block() {
        let yaml = concat!(
            "paths:\n",
            "  /v1/real:\n",
            "    get: {}\n",
            "components:\n",
            "  schemas:\n",
            "    Foo:\n",
            "      description: |\n",
            "        See /v1/not-a-path: discussed in the changelog.\n",
        );
        let paths = collect_openapi_path_methods(yaml);
        assert_eq!(paths.len(), 1);
        let methods = paths.get("/v1/real").expect("real path");
        assert!(methods.contains("get"));
    }

    #[test]
    fn every_axum_route_method_matches_an_openapi_method_and_vice_versa() {
        let source_path = format!("{}/src/lib.rs", env!("CARGO_MANIFEST_DIR"));
        let openapi_path = format!(
            "{}/../../openapi/voicelayerd.v1.yaml",
            env!("CARGO_MANIFEST_DIR"),
        );

        let source = std::fs::read_to_string(&source_path).expect("read voicelayerd lib.rs");
        let openapi = std::fs::read_to_string(&openapi_path).expect("read openapi contract");

        let routes = collect_axum_route_methods(&source);
        let paths = collect_openapi_path_methods(&openapi);

        assert!(
            !routes.is_empty(),
            "expected at least one axum route in build_app_router; \
             collect_axum_route_methods may be misparsing",
        );
        assert!(
            !paths.is_empty(),
            "expected at least one path in openapi/voicelayerd.v1.yaml; \
             collect_openapi_path_methods may be misparsing",
        );

        for (path, methods) in &routes {
            let yaml_methods = paths
                .get(path)
                .unwrap_or_else(|| panic!("axum route `{path}` is missing from openapi `paths:`"));
            for method in methods {
                assert!(
                    yaml_methods.contains(method),
                    "axum route `{method} {path}` has no matching openapi method declaration. \
                     Add `{method}:` under `paths.{path}` (or drop the route).",
                );
            }
        }
        for (path, methods) in &paths {
            let axum_methods = routes
                .get(path)
                .unwrap_or_else(|| panic!("openapi path `{path}` has no matching axum route"));
            for method in methods {
                assert!(
                    axum_methods.contains(method),
                    "openapi `{method} {path}` has no matching `.route(...)` call. \
                     Add `.route(\"{path}\", {method}(handler))` (or drop the openapi entry).",
                );
            }
        }
    }
}

#[cfg(test)]
mod doc_v1_endpoint_alignment_tests {
    //! Cross-check that every `/v1/<path>` URL mentioned in any
    //! operator-facing markdown (the project README plus everything
    //! under `docs/`) resolves to either a real axum route in
    //! `build_app_router` or an external OpenAI-compatible LLM
    //! endpoint the daemon talks *to* (e.g.
    //! `/v1/chat/completions`, `/v1/models`).
    //!
    //! The drift mode is silent and operator-facing: a doc claims
    //! `POST /v1/sessions/dictation/start` is the right entrypoint
    //! after the route was renamed from `/start` to bare
    //! `/sessions/dictation`, the operator copy-pastes the URL into
    //! curl, the daemon returns 404, and the only fix is to read
    //! the actual router source.
    //!
    //! The LLM allowlist is no longer a hard-coded const inside
    //! this test. It is derived at test time from the Python worker
    //! module
    //! `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`,
    //! which is the single source of truth for the OpenAI-compatible
    //! suffixes the daemon dials. If that worker migrates to a new
    //! suffix (e.g. `/v1/responses`), the allowlist tracks
    //! automatically without a parallel edit here.
    //!
    //! Reverse direction (every axum route is mentioned in some
    //! doc) is intentionally not enforced — the openapi document
    //! is the canonical surface contract (see
    //! `openapi_route_alignment_tests`); markdown prose is merely
    //! a guide.
    use std::collections::BTreeSet;

    /// Walk a Markdown body and pull every `/v1/<path>` URL token.
    /// Anchors on the literal `/v1/` prefix and walks forward
    /// taking ASCII alphanumerics, hyphens, underscores, and
    /// internal `/` separators. Stops at any other character (a
    /// space, punctuation, backtick, etc.) so trailing prose like
    /// `/v1/health.` (sentence-final period) yields the bare path.
    ///
    /// Uppercase letters are intentionally captured (not folded to
    /// lowercase) so that operator-facing typos such as
    /// `/v1/Sessions/Dictation` survive extraction and surface as
    /// allowlist violations downstream, instead of being silently
    /// truncated at the first uppercase character.
    fn extract_doc_v1_endpoint_paths(contents: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find("/v1/") {
            let after = &search[idx + 1..];
            let token: String = after
                .chars()
                .take_while(|c| c.is_ascii_alphanumeric() || matches!(c, '-' | '_' | '/'))
                .collect();
            // Step past the full captured token so pathological
            // inputs like `/v1/v1/foo` are not double-counted by a
            // re-scan of the same suffix.
            search = &after[token.len()..];
            if token.len() <= "v1/".len() {
                continue;
            }
            // Trim a trailing `/` so `POST /v1/sessions/dictation/`
            // (markdown-prose form) collapses onto the bare path.
            let cleaned = token.trim_end_matches('/');
            if cleaned.is_empty() {
                continue;
            }
            paths.insert(format!("/{cleaned}"));
        }
        paths
    }

    /// Walk this file's source and pull the first-argument path of
    /// every `.route("PATH", ...)` call. Stops at the first
    /// `#[cfg(test)]` line so synthetic fixtures and helper strings
    /// inside test modules do not pollute the production route set
    /// (the same truncation pattern as
    /// `collect_axum_route_methods` in
    /// `openapi_route_alignment_tests`).
    ///
    /// Assumption: in this file, `#[cfg(test)]` is only used as the
    /// test-module boundary, never to gate individual production
    /// `.route(...)` calls (e.g. a debug-only handler). If that ever
    /// changes, this scanner needs a smarter delimiter, otherwise it
    /// will silently truncate the production route set at the first
    /// such guard.
    fn collect_axum_route_paths(source: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        for line in source.lines() {
            if line.trim() == "#[cfg(test)]" {
                break;
            }
            let trimmed = line.trim_start();
            let Some(after) = trimmed.strip_prefix(".route(\"") else {
                continue;
            };
            let Some(end) = after.find('"') else {
                continue;
            };
            paths.insert(after[..end].to_owned());
        }
        paths
    }

    /// Walk the Python OpenAI-compatible worker source and pull
    /// every `/v1/<segment>` literal it mentions. The worker is the
    /// single source of truth for the OpenAI-compatible suffixes the
    /// daemon dials, so this extractor lets the doc-alignment guard
    /// follow the worker without a parallel const edit here.
    ///
    /// Matching rule:
    ///
    /// - Anchor on the literal `/v1/` prefix.
    /// - Walk forward taking lowercase ASCII letters, digits,
    ///   hyphens, underscores, and internal `/` separators (the same
    ///   character set the worker uses for its f-string suffixes).
    /// - Trim a trailing `/` so a stray `/v1/foo/` collapses onto
    ///   the bare path.
    /// - Skip any line whose first non-whitespace character is `#`
    ///   so commented-out URLs (e.g. `# /v1/should-not-be-captured`)
    ///   never reach the allowlist.
    fn extract_llm_external_endpoints_from_python_worker(source: &str) -> BTreeSet<String> {
        let mut paths = BTreeSet::new();
        for line in source.lines() {
            if line.trim_start().starts_with('#') {
                continue;
            }
            let mut search = line;
            while let Some(idx) = search.find("/v1/") {
                // `/v1/` is 4 ASCII bytes, so slicing past the
                // matched prefix is safe.
                let after_v1_slash = &search[idx + 4..];
                let suffix: String = after_v1_slash
                    .chars()
                    .take_while(|c| {
                        c.is_ascii_lowercase() || c.is_ascii_digit() || matches!(c, '-' | '_' | '/')
                    })
                    .collect();
                // Step past the full captured token (`/v1/` plus
                // the suffix we just consumed) so pathological
                // inputs like `/v1/v1/foo` are not double-counted
                // by a re-scan of the same suffix. Mirrors the
                // advancement strategy in
                // `extract_doc_v1_endpoint_paths` above.
                search = &after_v1_slash[suffix.len()..];
                // Require a real first segment that starts with a
                // lowercase ASCII letter, mirroring how the worker
                // names its OpenAI-compatible paths
                // (`chat/completions`, `models`, ...).
                let Some(first_char) = suffix.chars().next() else {
                    continue;
                };
                if !first_char.is_ascii_lowercase() {
                    continue;
                }
                let cleaned = suffix.trim_end_matches('/');
                if cleaned.is_empty() {
                    continue;
                }
                paths.insert(format!("/v1/{cleaned}"));
            }
        }
        paths
    }

    #[test]
    fn extract_llm_external_endpoints_from_python_worker_captures_chat_completions_and_models() {
        let fixture = "\
\"\"\"OpenAI-compatible chat completion provider fixture.\"\"\"


def resolve_chat_completions_url(endpoint: str) -> str:
    normalized = endpoint.rstrip(\"/\")
    if normalized.endswith(\"/v1\"):
        return f\"{normalized}/chat/completions\"
    return f\"{normalized}/v1/chat/completions\"


def resolve_models_url(endpoint: str) -> str:
    normalized = endpoint.rstrip(\"/\")
    if normalized.endswith(\"/v1/chat/completions\"):
        return normalized.removesuffix(\"/chat/completions\") + \"/models\"
    return f\"{normalized}/v1/models\"


# /v1/should-not-be-captured  -- comment line must be filtered out
";
        let paths = extract_llm_external_endpoints_from_python_worker(fixture);
        assert_eq!(
            paths,
            ["/v1/chat/completions", "/v1/models"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "expected only the two real OpenAI-compatible URLs; the `#`-prefixed \
             comment line `/v1/should-not-be-captured` must be skipped, and the \
             `removesuffix(\"/chat/completions\") + \"/models\"` branch must not \
             produce a spurious capture",
        );
    }

    #[test]
    fn extract_llm_external_endpoints_from_python_worker_does_not_double_count_overlapping_v1_segments()
     {
        let fixture = "return f\"{normalized}/v1/v1/foo\"\n";
        let paths = extract_llm_external_endpoints_from_python_worker(fixture);
        assert_eq!(
            paths,
            ["/v1/v1/foo"].iter().map(|s| (*s).to_owned()).collect(),
            "the loop must step past the full captured token so a \
             pathological worker URL like `/v1/v1/foo` yields one entry, \
             not a second `/v1/foo` re-discovered inside the suffix",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_handles_method_prefix_and_trailing_punctuation() {
        let md = "\
- `POST /v1/sessions/dictation` starts recording.
- The `GET /v1/events/stream` channel emits SSE.
- See `/v1/health.` for liveness — note the trailing period must drop.
- `/v1/path/with/multiple/segments` should resolve as one entry.
- The string `/v1/` alone (no segment) must not be captured.
";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            [
                "/v1/events/stream",
                "/v1/health",
                "/v1/path/with/multiple/segments",
                "/v1/sessions/dictation",
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
            "the bare `/v1/` prefix without a path segment must NOT be captured",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_captures_uppercase_so_doc_typos_surface_as_violations() {
        let md = "- See `/v1/Sessions/Dictation` for the live entry.";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            ["/v1/Sessions/Dictation"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "uppercase characters must be captured verbatim so an \
             operator-facing typo like `/v1/Sessions/Dictation` \
             reaches the allowlist comparison and surfaces as a \
             violation, instead of being silently truncated to \
             `/v1/` and dropped",
        );
    }

    #[test]
    fn extract_doc_v1_endpoint_paths_does_not_double_count_overlapping_v1_segments() {
        let md = "`/v1/v1/foo` is a degenerate but legal substring.";
        let paths = extract_doc_v1_endpoint_paths(md);
        assert_eq!(
            paths,
            ["/v1/v1/foo"].iter().map(|s| (*s).to_owned()).collect(),
            "the loop must step past the full captured token so a \
             pathological input like `/v1/v1/foo` yields one entry, \
             not a second `/v1/foo` re-discovered inside the suffix",
        );
    }

    #[test]
    fn collect_axum_route_paths_strips_method_builder_and_skips_test_module() {
        let source = "\
fn build_app_router() -> Router {
    Router::new()
        .route(\"/v1/health\", get(get_health))
        .route(\"/v1/sessions\", get(list_sessions))
}

#[cfg(test)]
mod tests {
    let _ = router.route(\"/v1/synthetic\", get(handler));
}
";
        let paths = collect_axum_route_paths(source);
        assert_eq!(
            paths,
            ["/v1/health", "/v1/sessions"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "the test-module fixture path must be excluded by #[cfg(test)] truncation",
        );
    }

    /// Every `/v1/<path>` URL the README or any doc mentions must
    /// be either a real daemon route or a known external
    /// OpenAI-compatible endpoint. The LLM-side allowlist is
    /// derived at test time from the Python worker module
    /// `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`,
    /// which is the single source of truth for the suffixes the
    /// daemon dials on the configured `VOICELAYER_LLM_ENDPOINT`
    /// host. The same URLs are operator-documented in
    /// `docs/guides/local-llm-provider.md`.
    #[test]
    fn every_doc_v1_endpoint_mention_resolves_to_an_axum_route_or_llm_allowlist() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let lib_source = std::fs::read_to_string(format!("{manifest}/src/lib.rs"))
            .expect("read voicelayerd lib.rs");
        let routes = collect_axum_route_paths(&lib_source);
        assert!(
            !routes.is_empty(),
            "expected at least one .route() call in voicelayerd lib.rs",
        );

        let repo_root = std::path::PathBuf::from(format!("{manifest}/../.."));
        let python_worker_path =
            repo_root.join("python/voicelayer_orchestrator/providers/llm_openai_compatible.py");
        let python_source = std::fs::read_to_string(&python_worker_path)
            .unwrap_or_else(|err| panic!("read {}: {err}", python_worker_path.display()));
        let llm_external_endpoints =
            extract_llm_external_endpoints_from_python_worker(&python_source);
        assert!(
            !llm_external_endpoints.is_empty(),
            "expected at least one /v1/<path> literal in the python worker — \
             extract_llm_external_endpoints_from_python_worker may be misparsing, \
             or the worker may have moved to a different file path",
        );

        let mut docs = vec![repo_root.join("README.md")];
        voicelayer_doc_test_utils::collect_markdown_files(&repo_root.join("docs"), &mut docs)
            .expect("walk docs/");

        let mut allowed: BTreeSet<String> = routes;
        allowed.extend(llm_external_endpoints);

        let mut violations: Vec<String> = Vec::new();
        for doc_path in &docs {
            let contents = match std::fs::read_to_string(doc_path) {
                Ok(s) => s,
                Err(_) => continue,
            };
            for mention in extract_doc_v1_endpoint_paths(&contents) {
                if allowed.contains(&mention) {
                    continue;
                }
                violations.push(format!(
                    "{}: `{mention}`",
                    doc_path
                        .strip_prefix(&repo_root)
                        .unwrap_or(doc_path)
                        .display(),
                ));
            }
        }
        assert!(
            violations.is_empty(),
            "docs reference `/v1/...` endpoints that are neither real daemon \
             routes nor on the LLM external-endpoint allowlist:\n  - {}\n\n\
             Either fix the doc URL to match an actual `.route(...)` call in \
             crates/voicelayerd/src/lib.rs, drop the mention, or add the URL \
             as a `/v1/...` literal in \
             `python/voicelayer_orchestrator/providers/llm_openai_compatible.py`, \
             since that file is the single source of truth scanned by this guard.",
            violations.join("\n  - "),
        );
    }
}

#[cfg(test)]
mod sse_event_doc_alignment_tests {
    //! Cross-check that every backtick-quoted SSE event name in
    //! `docs/architecture/overview.md` corresponds to an
    //! `EventEnvelope::new("<name>", ...)` emission somewhere in
    //! `crates/voicelayerd/src/lib.rs`. Forward direction only: the
    //! doc enumerates the dictation-pipeline events that operators
    //! consume, but the daemon also emits non-dictation events
    //! (`compose.job_created`, `transcription.completed`, etc.) that
    //! the doc intentionally does not mention. Reverse-direction
    //! enforcement would require lifting the doc to an exhaustive
    //! event reference, which is out of scope for this guard.
    //!
    //! The drift mode the forward direction catches is doc rot: the
    //! doc claims `dictation.fancy_new_event` is emitted but the
    //! daemon never produces it. Operators wiring an SSE consumer
    //! against the doc see no event and have no way to discover the
    //! name was retired.
    use std::collections::BTreeSet;

    /// Walk a Rust source string and pull every event name from
    /// `EventEnvelope::new("<name>", ...)` calls. Tolerates the
    /// multi-line layout the daemon uses today
    /// (`EventEnvelope::new(\n    "<name>",\n ...)`): the scanner
    /// finds each `EventEnvelope::new(` occurrence, skips
    /// whitespace, and reads the next `"..."` literal. Stops at the
    /// first `#[cfg(test)]` line so synthetic test fixtures
    /// (`test.single_event`) do not leak into the production set.
    fn collect_event_envelope_names(source: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let mut search = source;
        // Truncate at the first `#[cfg(test)]` so test-only event
        // fixtures don't pollute the captured set.
        if let Some(idx) = search.find("#[cfg(test)]") {
            search = &search[..idx];
        }
        while let Some(idx) = search.find("EventEnvelope::new(") {
            let after = &search[idx + "EventEnvelope::new(".len()..];
            let trimmed = after.trim_start();
            search = after;
            let Some(stripped) = trimmed.strip_prefix('"') else {
                continue;
            };
            let Some(end) = stripped.find('"') else {
                continue;
            };
            let name = &stripped[..end];
            if !name.is_empty() {
                names.insert(name.to_owned());
            }
        }
        names
    }

    /// Walk a Markdown body and pull every backtick-quoted token of
    /// shape `dictation.<word>` (lowercase letters and underscores
    /// only after the dot). Designed for the SSE event mentions in
    /// `docs/architecture/overview.md`. Excludes the `worker.*`,
    /// `compose.*`, and `transcription.*` namespaces, which the
    /// doc does not enumerate today and the forward-direction
    /// guard does not require to be enumerated.
    fn collect_doc_dictation_event_names(contents: &str) -> BTreeSet<String> {
        let mut names = BTreeSet::new();
        let mut search = contents;
        while let Some(idx) = search.find('`') {
            let after = &search[idx + 1..];
            let Some(close) = after.find('`') else {
                break;
            };
            let inner = &after[..close];
            search = &after[close + 1..];
            let Some(rest) = inner.strip_prefix("dictation.") else {
                continue;
            };
            if rest.is_empty() {
                continue;
            }
            if !rest.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                continue;
            }
            names.insert(format!("dictation.{rest}"));
        }
        names
    }

    #[test]
    fn collect_event_envelope_names_handles_multiline_layout_and_skips_test_module() {
        let source = "
let _ = state.events.send(EventEnvelope::new(
    \"dictation.session_created\",
    payload,
));
let _ = state.events.send(EventEnvelope::new(\"dictation.completed\", payload));
#[cfg(test)]
mod tests {
    let _ = events.send(EventEnvelope::new(\"test.synthetic\", payload));
}
";
        let names = collect_event_envelope_names(source);
        assert_eq!(
            names,
            ["dictation.completed", "dictation.session_created"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "the test-module event must be excluded by the #[cfg(test)] truncation",
        );
    }

    #[test]
    fn collect_doc_dictation_event_names_filters_to_dictation_namespace_only() {
        let md = "\
- `dictation.session_created` — fired when the session is created.
- `dictation.completed` / `dictation.failed` — terminal pair.
- `compose.job_created` — out of scope for this guard.
- The `dictation.with space` token is not an event.
";
        let names = collect_doc_dictation_event_names(md);
        assert_eq!(
            names,
            [
                "dictation.completed",
                "dictation.failed",
                "dictation.session_created"
            ]
            .iter()
            .map(|s| (*s).to_owned())
            .collect(),
        );
    }

    /// Every dictation-namespace SSE event name a public-facing doc
    /// enumerates must correspond to a real `EventEnvelope::new(...)`
    /// emission in the daemon. Scans the architecture overview AND
    /// the project README, since both are operator-facing and either
    /// can drift independently — a contributor renaming an event
    /// might fix overview.md but forget the README example, or vice
    /// versa.
    ///
    /// Reverse direction (every emitted event is documented) is
    /// intentionally not enforced — the daemon emits several
    /// non-dictation events (`compose.job_created`,
    /// `transcription.completed`, `worker.providers_unavailable`)
    /// the docs do not promise to list.
    #[test]
    fn every_doc_dictation_event_resolves_to_an_event_envelope_emission() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let lib_source = std::fs::read_to_string(format!("{manifest}/src/lib.rs"))
            .expect("read voicelayerd lib.rs");
        let emitted = collect_event_envelope_names(&lib_source);
        assert!(
            !emitted.is_empty(),
            "expected at least one EventEnvelope::new emission in lib.rs",
        );

        let doc_paths: &[&str] = &["../../docs/architecture/overview.md", "../../README.md"];
        let mut total_doc_events = 0usize;
        let mut undocumented_in_code: Vec<String> = Vec::new();
        for rel in doc_paths {
            let abs = format!("{manifest}/{rel}");
            let contents =
                std::fs::read_to_string(&abs).unwrap_or_else(|err| panic!("read {abs}: {err}"));
            let doc_events = collect_doc_dictation_event_names(&contents);
            total_doc_events += doc_events.len();
            for event in doc_events.difference(&emitted) {
                undocumented_in_code.push(format!("{rel}: `{event}`"));
            }
        }

        assert!(
            total_doc_events > 0,
            "expected at least one dictation event reference across the scanned docs \
             (overview.md, README.md) — `collect_doc_dictation_event_names` may have \
             lost its anchor",
        );
        assert!(
            undocumented_in_code.is_empty(),
            "scanned docs mention dictation events the daemon does not emit:\n  - {}\n\n\
             Either rename the doc reference to match the actual event, drop the \
             mention, or wire up an `EventEnvelope::new(\"<name>\", ...)` emission \
             in crates/voicelayerd/src/lib.rs.",
            undocumented_in_code.join("\n  - "),
        );
    }
}
