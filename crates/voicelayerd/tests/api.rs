//! HTTP-layer integration tests for the control API.
//!
//! Exercises the axum router in-process via `tower::ServiceExt::oneshot` —
//! no socket is bound. The worker manager underneath spawns the real Python
//! worker from the repository `uv` environment.

use std::{collections::HashMap, sync::Arc, time::Duration};

use axum::body::Body;
use axum::http::{Request, StatusCode};
use http_body_util::BodyExt;
use tokio::sync::{Mutex, RwLock};
use tower::ServiceExt;
use voicelayer_core::{ProblemDetails, VoiceLayerConfig, default_project_root};
use voicelayerd::{
    DaemonConfig,
    api::{self, AppState},
    events::EventBus,
    session::SessionStore,
    worker::WorkerManager,
};

fn test_state() -> AppState {
    let config = DaemonConfig::with_settings(
        Some(std::env::temp_dir().join("voicelayer-test.sock")),
        Some(default_project_root()),
        VoiceLayerConfig::default(),
    );
    let worker = Arc::new(WorkerManager::new(
        config.project_root.clone(),
        config.settings.worker_payload(),
        Duration::from_secs(60),
    ));
    AppState {
        sessions: SessionStore::new(),
        active: Arc::new(Mutex::new(HashMap::new())),
        events: EventBus::new(),
        worker,
        health: Arc::new(RwLock::new(None)),
        config: Arc::new(config),
    }
}

async fn call(
    state: AppState,
    method: &str,
    path: &str,
    body: Option<serde_json::Value>,
) -> (StatusCode, String, serde_json::Value) {
    let app = api::router(state);
    let mut builder = Request::builder().method(method).uri(path);
    let body = match body {
        Some(value) => {
            builder = builder.header("content-type", "application/json");
            Body::from(serde_json::to_vec(&value).unwrap())
        }
        None => Body::empty(),
    };
    let response = app.oneshot(builder.body(body).unwrap()).await.unwrap();
    let status = response.status();
    let content_type = response
        .headers()
        .get("content-type")
        .and_then(|value| value.to_str().ok())
        .unwrap_or("")
        .to_owned();
    let bytes = response.into_body().collect().await.unwrap().to_bytes();
    let json = serde_json::from_slice(&bytes).unwrap_or(serde_json::Value::Null);
    (status, content_type, json)
}

#[tokio::test]
async fn compose_with_empty_prompt_is_a_400_problem() {
    let (status, content_type, body) = call(
        test_state(),
        "POST",
        "/v1/sessions/compose",
        Some(serde_json::json!({"spoken_prompt": "   "})),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(content_type, "application/problem+json");
    let problem: ProblemDetails = serde_json::from_value(body).unwrap();
    assert_eq!(problem.problem_type, "urn:voicelayer:problem:bad_request");
    assert_eq!(problem.status, 400);
}

#[tokio::test]
async fn stopping_an_unknown_session_is_a_404_problem() {
    let (status, content_type, body) = call(
        test_state(),
        "POST",
        "/v1/sessions/dictation/stop",
        Some(serde_json::json!({"session_id": uuid::Uuid::new_v4()})),
    )
    .await;

    assert_eq!(status, StatusCode::NOT_FOUND);
    assert_eq!(content_type, "application/problem+json");
    assert_eq!(
        body["type"],
        serde_json::json!("urn:voicelayer:problem:session_not_found")
    );
}

#[tokio::test]
async fn transcribe_with_empty_audio_file_is_a_400_problem() {
    let (status, _, body) = call(
        test_state(),
        "POST",
        "/v1/transcriptions",
        Some(serde_json::json!({"audio_file": ""})),
    )
    .await;

    assert_eq!(status, StatusCode::BAD_REQUEST);
    assert_eq!(body["type"], "urn:voicelayer:problem:bad_request");
}

#[tokio::test]
async fn compose_without_provider_is_a_503_problem() {
    let (status, _, body) = call(
        test_state(),
        "POST",
        "/v1/sessions/compose",
        Some(serde_json::json!({"spoken_prompt": "write a status update"})),
    )
    .await;

    assert_eq!(status, StatusCode::SERVICE_UNAVAILABLE);
    assert_eq!(
        body["type"],
        serde_json::json!("urn:voicelayer:problem:provider_unavailable")
    );
}

#[tokio::test]
async fn health_returns_a_cached_snapshot() {
    let state = test_state();
    let (status, _, body) = call(state.clone(), "GET", "/v1/health", None).await;
    assert_eq!(status, StatusCode::OK);
    assert!(body["worker"]["status"].is_string());

    // Second call must come from the cache (worker status identical, and
    // the health slot is populated).
    assert!(state.health.read().await.is_some());
    let (status, _, _) = call(state, "GET", "/v1/health", None).await;
    assert_eq!(status, StatusCode::OK);
}
