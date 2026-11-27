//! Typed `/v1` daemon client for the desktop shell, built on
//! [`voicelayer_core::client::DaemonClient`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use serde::Deserialize;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;
use voicelayer_core::client::{ClientError, DaemonClient, resolve_socket_path};
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionReceipt, DictationCaptureRequest,
    DictationCaptureResult, EventEnvelope, HealthResponse, InjectRequest, InjectionPlan,
    ProviderDescriptor, RewriteRequest, StartDictationRequest, StopDictationRequest,
    TranslateRequest,
};

use crate::state::SharedError;

/// Typed client for a daemon socket path.
#[derive(Debug, Clone)]
pub struct Client {
    inner: DaemonClient,
}

impl Client {
    pub fn new(socket_path: PathBuf) -> Self {
        Self {
            inner: DaemonClient::new(socket_path),
        }
    }

    pub fn socket_path(&self) -> &Path {
        self.inner.socket_path()
    }

    pub async fn health(&self) -> Result<HealthResponse, SharedError> {
        self.inner.get_json("/v1/health").await.map_err(shared)
    }

    pub async fn providers(&self) -> Result<Vec<ProviderDescriptor>, SharedError> {
        let response: ProviderListResponse =
            self.inner.get_json("/v1/providers").await.map_err(shared)?;
        Ok(response.providers)
    }

    pub async fn sessions(&self) -> Result<Vec<CaptureSession>, SharedError> {
        self.inner.get_json("/v1/sessions").await.map_err(shared)
    }

    pub async fn start_dictation(
        &self,
        request: StartDictationRequest,
    ) -> Result<CaptureSession, SharedError> {
        self.inner
            .post_json("/v1/sessions/dictation", &request)
            .await
            .map_err(shared)
    }

    pub async fn stop_dictation(
        &self,
        session_id: Uuid,
    ) -> Result<DictationCaptureResult, SharedError> {
        let request = StopDictationRequest { session_id };
        self.inner
            .post_json("/v1/sessions/dictation/stop", &request)
            .await
            .map_err(shared)
    }

    pub async fn capture(
        &self,
        request: DictationCaptureRequest,
    ) -> Result<DictationCaptureResult, SharedError> {
        self.inner
            .post_json("/v1/dictation/capture", &request)
            .await
            .map_err(shared)
    }

    pub async fn compose(
        &self,
        request: ComposeRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        self.inner
            .post_json("/v1/sessions/compose", &request)
            .await
            .map_err(shared)
    }

    pub async fn rewrite(
        &self,
        request: RewriteRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        self.inner
            .post_json("/v1/rewrites", &request)
            .await
            .map_err(shared)
    }

    pub async fn translate(
        &self,
        request: TranslateRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        self.inner
            .post_json("/v1/translations", &request)
            .await
            .map_err(shared)
    }

    pub async fn inject(&self, request: InjectRequest) -> Result<InjectionPlan, SharedError> {
        self.inner
            .post_json("/v1/inject", &request)
            .await
            .map_err(shared)
    }
}

#[derive(Debug, Deserialize)]
struct ProviderListResponse {
    providers: Vec<ProviderDescriptor>,
}

/// Resolve the daemon socket path from `VOICELAYER_SOCKET_PATH` or the shared default.
pub fn desktop_socket_path() -> PathBuf {
    resolve_socket_path()
}

fn shared(error: ClientError) -> SharedError {
    Arc::new(error.to_string())
}

/// Stream `GET /v1/events/stream` and forward each decoded [`EventEnvelope`] to
/// `sink` until the stream closes, errors, or the receiver is dropped.
pub async fn stream_events(
    socket_path: &Path,
    sink: UnboundedSender<EventEnvelope>,
) -> Result<(), ClientError> {
    DaemonClient::new(socket_path.to_path_buf())
        .stream_events(|event| sink.send(event).is_ok())
        .await
}
