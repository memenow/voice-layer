//! HTTP control plane: router, handlers, and health caching.

pub mod error;

use std::{convert::Infallible, sync::Arc, time::Duration};

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
use serde::Serialize;
use tokio::sync::RwLock;
use tokio_stream::wrappers::{BroadcastStream, errors::BroadcastStreamRecvError};
use voicelayer_core::{
    ComposeRequest, CompositionReceipt, DaemonEvent, DictationCaptureRequest,
    DictationCaptureResult, EventEnvelope, HealthResponse, PreviewArtifact, PreviewStatus,
    RewriteRequest, StartDictationRequest, StopDictationRequest, TranscribeRequest,
    TranscriptionResult, TranslateRequest, WorkerHealthSummary, default_host_adapter_catalog,
};

use crate::{
    DaemonConfig,
    dictation::{self, ActiveDictations},
    events::EventBus,
    platform,
    session::SessionStore,
    worker::{WorkerManager, WorkerPreviewPayload},
};

use error::ApiError;

pub(crate) const HEALTH_REFRESH_INTERVAL: Duration = Duration::from_secs(30);

#[derive(Clone)]
pub struct AppState {
    pub sessions: SessionStore,
    pub active: ActiveDictations,
    pub events: EventBus,
    pub worker: Arc<WorkerManager>,
    pub health: Arc<RwLock<Option<HealthResponse>>>,
    pub config: Arc<DaemonConfig>,
}

#[derive(Serialize)]
struct ProviderListResponse {
    providers: Vec<voicelayer_core::ProviderDescriptor>,
}

pub fn router(state: AppState) -> Router {
    Router::new()
        .route("/v1/health", get(get_health))
        .route("/v1/health/refresh", post(refresh_health))
        .route("/v1/providers", get(list_providers))
        .route("/v1/events/stream", get(stream_events))
        .route("/v1/sessions/dictation", post(create_dictation_session))
        .route("/v1/sessions/dictation/stop", post(stop_dictation_session))
        .route("/v1/sessions/compose", post(create_composition_job))
        .route("/v1/rewrites", post(create_rewrite_job))
        .route("/v1/translations", post(create_translation_job))
        .route("/v1/transcriptions", post(create_transcription))
        .route("/v1/dictation/capture", post(capture_dictation))
        .with_state(state)
}

/// Probe worker and platform state and refresh the cache. Invoked by the
/// background refresher and by `POST /v1/health/refresh`.
pub async fn refresh_health_state(state: &AppState) -> HealthResponse {
    let hotkeys = platform::probe_global_hotkeys().await;
    let worker = match state.worker.health().await {
        Ok(health) => WorkerHealthSummary {
            status: if (health.llm_configured && !health.llm_reachable)
                || (health.asr_configured && health.asr_error.is_some())
            {
                "degraded".to_owned()
            } else {
                "ok".to_owned()
            },
            command: state.worker.command_display(),
            asr_configured: health.asr_configured,
            asr_binary: health.asr_binary,
            asr_model_path: health.asr_model_path,
            asr_error: health.asr_error,
            llm_configured: health.llm_configured,
            llm_model: health.llm_model,
            llm_endpoint: health.llm_endpoint,
            llm_reachable: health.llm_reachable,
            llm_error: health.llm_error,
            global_hotkeys_available: hotkeys.available,
            global_hotkeys_backend: Some(hotkeys.backend),
            global_hotkeys_detail: hotkeys.detail,
            message: None,
        },
        Err(error) => WorkerHealthSummary {
            status: "unavailable".to_owned(),
            command: state.worker.command_display(),
            asr_configured: false,
            asr_binary: None,
            asr_model_path: None,
            asr_error: None,
            llm_configured: false,
            llm_model: None,
            llm_endpoint: None,
            llm_reachable: false,
            llm_error: None,
            global_hotkeys_available: hotkeys.available,
            global_hotkeys_backend: Some(hotkeys.backend),
            global_hotkeys_detail: hotkeys.detail,
            message: Some(error.to_string()),
        },
    };

    let response = HealthResponse {
        status: if worker.status == "ok" {
            "ok".to_owned()
        } else {
            "degraded".to_owned()
        },
        socket_path: state.config.socket_path.display().to_string(),
        version: state.config.version.clone(),
        worker,
    };
    *state.health.write().await = Some(response.clone());
    response
}

/// Cheap liveness read: returns the last cached snapshot, refreshing inline
/// only when no probe has completed yet.
async fn get_health(State(state): State<AppState>) -> Json<HealthResponse> {
    if let Some(cached) = state.health.read().await.clone() {
        return Json(cached);
    }
    Json(refresh_health_state(&state).await)
}

async fn refresh_health(State(state): State<AppState>) -> Json<HealthResponse> {
    Json(refresh_health_state(&state).await)
}

async fn list_providers(State(state): State<AppState>) -> Result<impl IntoResponse, ApiError> {
    let mut providers = default_host_adapter_catalog();
    let worker_catalog = state.worker.list_providers().await.map_err(|error| {
        state.events.emit(DaemonEvent::WorkerProvidersUnavailable {
            detail: error.to_string(),
        });
        ApiError::Worker(error)
    })?;
    providers.extend(worker_catalog.providers);
    Ok(Json(ProviderListResponse { providers }))
}

async fn stream_events(
    State(state): State<AppState>,
) -> Sse<impl futures_util::Stream<Item = Result<Event, Infallible>>> {
    let stream = BroadcastStream::new(state.events.subscribe()).filter_map(|event| async move {
        let envelope = match event {
            Ok(envelope) => envelope,
            Err(BroadcastStreamRecvError::Lagged(count)) => {
                EventEnvelope::new(DaemonEvent::EventsLost { count })
            }
        };
        match serde_json::to_string(&envelope) {
            Ok(payload) => Some(Ok(Event::default()
                .event(envelope.event.name())
                .data(payload))),
            Err(_) => None,
        }
    });

    Sse::new(stream).keep_alive(
        KeepAlive::new()
            .interval(Duration::from_secs(15))
            .text("keepalive"),
    )
}

async fn create_dictation_session(
    State(state): State<AppState>,
    Json(request): Json<StartDictationRequest>,
) -> Result<impl IntoResponse, ApiError> {
    Ok(Json(dictation::start_session(&state, request).await?))
}

async fn stop_dictation_session(
    State(state): State<AppState>,
    Json(request): Json<StopDictationRequest>,
) -> Result<Json<DictationCaptureResult>, ApiError> {
    Ok(Json(
        dictation::stop_session(&state, request.session_id).await?,
    ))
}

async fn create_composition_job(
    State(state): State<AppState>,
    Json(request): Json<ComposeRequest>,
) -> Result<Json<CompositionReceipt>, ApiError> {
    if request.spoken_prompt.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "spoken_prompt must not be empty".to_owned(),
        ));
    }
    let preview = state.worker.compose(&request).await?;
    let receipt = ready_receipt(preview);
    state.events.emit(DaemonEvent::ComposeJobCreated {
        title: receipt.preview.title.clone(),
    });
    Ok(Json(receipt))
}

async fn create_rewrite_job(
    State(state): State<AppState>,
    Json(request): Json<RewriteRequest>,
) -> Result<Json<CompositionReceipt>, ApiError> {
    if request.source_text.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "source_text must not be empty".to_owned(),
        ));
    }
    let preview = state.worker.rewrite(&request).await?;
    let receipt = ready_receipt(preview);
    state.events.emit(DaemonEvent::RewriteJobCreated {
        title: receipt.preview.title.clone(),
    });
    Ok(Json(receipt))
}

async fn create_translation_job(
    State(state): State<AppState>,
    Json(request): Json<TranslateRequest>,
) -> Result<Json<CompositionReceipt>, ApiError> {
    if request.source_text.trim().is_empty() || request.target_language.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "source_text and target_language must not be empty".to_owned(),
        ));
    }
    let preview = state.worker.translate(&request).await?;
    let receipt = ready_receipt(preview);
    state.events.emit(DaemonEvent::TranslateJobCreated {
        title: receipt.preview.title.clone(),
    });
    Ok(Json(receipt))
}

async fn create_transcription(
    State(state): State<AppState>,
    Json(request): Json<TranscribeRequest>,
) -> Result<Json<TranscriptionResult>, ApiError> {
    if request.audio_file.trim().is_empty() {
        return Err(ApiError::BadRequest(
            "audio_file must not be empty".to_owned(),
        ));
    }
    let result = state.worker.transcribe(&request).await?;
    state.events.emit(DaemonEvent::TranscriptionCompleted {
        transcript_chars: result.text.chars().count(),
    });
    Ok(Json(result))
}

async fn capture_dictation(
    State(state): State<AppState>,
    Json(request): Json<DictationCaptureRequest>,
) -> Result<Json<DictationCaptureResult>, ApiError> {
    Ok(Json(dictation::capture_once(&state, request).await?))
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
