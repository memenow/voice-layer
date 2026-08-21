//! Typed client for the VoiceLayer daemon's local `/v1` API over a Unix domain
//! socket, plus the Server-Sent Events reader for `/v1/events/stream`.
//!
//! NOTE: the low-level UDS helpers mirror `crates/vl/src/uds.rs` — the desktop
//! shell keeps an independent copy so we can iterate on the GUI without pulling
//! in `vl` as a library crate. When a third consumer appears we'll extract a
//! shared `voicelayer-client` crate and replace both copies.
//!
//! Every request and response type is reused verbatim from [`voicelayer_core`]
//! (all of them derive both `Serialize` and `Deserialize`), so the GUI and the
//! daemon share one wire-shape source of truth. The only daemon-local envelope
//! is `GET /v1/providers`, mirrored here as [`ProviderListResponse`].

use std::path::{Path, PathBuf};
use std::sync::Arc;

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use serde::Deserialize;
use tokio::net::UnixStream;
use tokio::sync::mpsc::UnboundedSender;
use uuid::Uuid;
use voicelayer_core::default_socket_path;
use voicelayer_core::{
    CaptureSession, ComposeRequest, CompositionReceipt, DictationCaptureRequest,
    DictationCaptureResult, EventEnvelope, HealthResponse, InjectRequest, InjectionPlan,
    ProviderDescriptor, RewriteRequest, StartDictationRequest, StopDictationRequest,
    TranscribeRequest, TranscriptionResult, TranslateRequest,
};

use crate::state::SharedError;

type BoxError = Box<dyn std::error::Error + Send + Sync>;

/// A typed handle to a daemon's `/v1` API at a fixed socket path. Cheap to
/// clone (just the path), so the app hands a clone to each async task.
#[derive(Debug, Clone)]
pub struct Client {
    socket_path: PathBuf,
}

impl Client {
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// The socket this client targets, for display in the connection panel.
    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// `GET /v1/health` — daemon liveness plus worker/provider diagnostics.
    pub async fn health(&self) -> Result<HealthResponse, SharedError> {
        uds_get_json(&self.socket_path, "/v1/health")
            .await
            .map_err(shared)
    }

    /// `GET /v1/providers` — the registered ASR / LLM / host adapters. Unwraps
    /// the daemon's `{ "providers": [...] }` envelope into the bare list.
    pub async fn providers(&self) -> Result<Vec<ProviderDescriptor>, SharedError> {
        let response: ProviderListResponse = uds_get_json(&self.socket_path, "/v1/providers")
            .await
            .map_err(shared)?;
        Ok(response.providers)
    }

    /// `GET /v1/sessions` — known capture sessions, oldest first. Drives the
    /// History panel.
    pub async fn sessions(&self) -> Result<Vec<CaptureSession>, SharedError> {
        uds_get_json(&self.socket_path, "/v1/sessions")
            .await
            .map_err(shared)
    }

    /// `POST /v1/sessions/dictation` — open a streaming dictation session.
    pub async fn start_dictation(
        &self,
        request: StartDictationRequest,
    ) -> Result<CaptureSession, SharedError> {
        uds_post_json(&self.socket_path, "/v1/sessions/dictation", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/sessions/dictation/stop` — stop a session and collect its
    /// final transcription.
    pub async fn stop_dictation(
        &self,
        session_id: Uuid,
    ) -> Result<DictationCaptureResult, SharedError> {
        let request = StopDictationRequest { session_id };
        uds_post_json(&self.socket_path, "/v1/sessions/dictation/stop", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/dictation/capture` — fixed-duration one-shot capture, driven by
    /// the dictation panel's one-shot button.
    pub async fn capture(
        &self,
        request: DictationCaptureRequest,
    ) -> Result<DictationCaptureResult, SharedError> {
        uds_post_json(&self.socket_path, "/v1/dictation/capture", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/sessions/compose` — draft longer text from a spoken prompt.
    pub async fn compose(
        &self,
        request: ComposeRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        uds_post_json(&self.socket_path, "/v1/sessions/compose", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/rewrites` — restyle existing text.
    pub async fn rewrite(
        &self,
        request: RewriteRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        uds_post_json(&self.socket_path, "/v1/rewrites", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/translations` — translate text to a target language.
    pub async fn translate(
        &self,
        request: TranslateRequest,
    ) -> Result<CompositionReceipt, SharedError> {
        uds_post_json(&self.socket_path, "/v1/translations", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/transcriptions` — transcribe an existing audio file. No view
    /// surfaces a file picker yet; the method keeps `api` the complete `/v1`
    /// surface so a future "transcribe a file" affordance only adds a view.
    #[allow(dead_code)]
    pub async fn transcribe(
        &self,
        request: TranscribeRequest,
    ) -> Result<TranscriptionResult, SharedError> {
        uds_post_json(&self.socket_path, "/v1/transcriptions", &request)
            .await
            .map_err(shared)
    }

    /// `POST /v1/inject` — plan an injection of text into the focused target.
    pub async fn inject(&self, request: InjectRequest) -> Result<InjectionPlan, SharedError> {
        uds_post_json(&self.socket_path, "/v1/inject", &request)
            .await
            .map_err(shared)
    }
}

/// The `GET /v1/providers` response envelope. Defined daemon-side as a thin
/// wrapper around the provider list; mirrored here for deserialization.
#[derive(Debug, Deserialize)]
struct ProviderListResponse {
    providers: Vec<ProviderDescriptor>,
}

/// Resolve the daemon socket: `VOICELAYER_SOCKET_PATH` when set, else the same
/// default the CLI uses, so the shell and `vl` land on one socket by default.
pub fn desktop_socket_path() -> PathBuf {
    std::env::var_os("VOICELAYER_SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

fn shared(error: BoxError) -> SharedError {
    Arc::new(error.to_string())
}

async fn uds_post_json<TRequest, TResponse>(
    socket_path: &Path,
    path: &str,
    payload: &TRequest,
) -> Result<TResponse, BoxError>
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

async fn uds_get_json<TResponse>(socket_path: &Path, path: &str) -> Result<TResponse, BoxError>
where
    TResponse: serde::de::DeserializeOwned,
{
    let stream = UnixStream::connect(socket_path).await?;
    let io = TokioIo::new(stream);
    let (mut sender, connection) = http1::handshake(io).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::get(path)
        .header(HOST, "localhost")
        .body(Full::new(Bytes::new()))?;
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

/// Open `GET /v1/events/stream` and forward each decoded [`EventEnvelope`] to
/// `sink` until the stream ends, the socket errors, or the receiver is dropped.
/// The caller owns reconnection (the iced subscription loops on return).
pub async fn stream_events(
    socket_path: &Path,
    sink: UnboundedSender<EventEnvelope>,
) -> Result<(), BoxError> {
    let stream = UnixStream::connect(socket_path).await?;
    let io = TokioIo::new(stream);
    let (mut sender, connection) = http1::handshake(io).await?;
    tokio::spawn(async move {
        let _ = connection.await;
    });

    let request = Request::get("/v1/events/stream")
        .header(HOST, "localhost")
        .body(Full::new(Bytes::new()))?;
    let response = sender.send_request(request).await?;
    let status = response.status();
    if !status.is_success() {
        return Err(format!("event stream returned non-success status {status}").into());
    }

    let mut body = response.into_body();
    let mut decoder = SseDecoder::default();
    while let Some(frame) = body.frame().await {
        let frame = frame?;
        if let Some(chunk) = frame.data_ref() {
            for event in decoder.push(chunk) {
                if sink.send(event).is_err() {
                    // Receiver dropped (window closed): stop quietly.
                    return Ok(());
                }
            }
        }
    }
    Ok(())
}

/// Incremental decoder for the daemon's Server-Sent Events framing. The daemon
/// emits one JSON [`EventEnvelope`] per `data:` field and terminates each event
/// with a blank line; `event:` names, `:`-comment keepalives, and other SSE
/// fields are ignored because the envelope already carries its `event_type`.
///
/// Bytes are buffered until a full `\n`-terminated line is available, so a UTF-8
/// sequence split across socket reads is never decoded mid-character.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    data: String,
}

impl SseDecoder {
    /// Feed a chunk of stream bytes; return every event completed by it.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<EventEnvelope> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            line.pop(); // drop the trailing '\n'
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line);
            if line.is_empty() {
                // Blank line dispatches the buffered event.
                if !self.data.is_empty() {
                    if let Ok(event) = serde_json::from_str::<EventEnvelope>(&self.data) {
                        events.push(event);
                    }
                    self.data.clear();
                }
            } else if let Some(rest) = line.strip_prefix("data:") {
                let rest = rest.strip_prefix(' ').unwrap_or(rest);
                if !self.data.is_empty() {
                    self.data.push('\n');
                }
                self.data.push_str(rest);
            }
            // `event:` / `id:` / `retry:` / `:comment` (keepalive) → ignored.
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use super::{SseDecoder, default_socket_path, desktop_socket_path};
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serializes any test that mutates `VOICELAYER_SOCKET_PATH`. Same rationale
    /// as the matching lock in `vl::uds::tests`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// The override branch: the desktop shell must honor an operator's
    /// `VOICELAYER_SOCKET_PATH` rather than silently using the default.
    #[test]
    fn desktop_socket_path_uses_voicelayer_socket_path_when_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serializes every mutation of this variable.
        unsafe {
            std::env::set_var("VOICELAYER_SOCKET_PATH", "/run/test-desktop/daemon.sock");
        }
        let path = desktop_socket_path();
        assert_eq!(path, PathBuf::from("/run/test-desktop/daemon.sock"));
        match previous {
            Some(value) => unsafe {
                std::env::set_var("VOICELAYER_SOCKET_PATH", value);
            },
            None => unsafe {
                std::env::remove_var("VOICELAYER_SOCKET_PATH");
            },
        }
    }

    /// The unset branch: with no override, delegate to the shared default so the
    /// CLI and desktop shell always agree on the socket.
    #[test]
    fn desktop_socket_path_falls_back_to_default_socket_path_when_env_unset() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serializes every mutation of this variable.
        unsafe {
            std::env::remove_var("VOICELAYER_SOCKET_PATH");
        }
        assert_eq!(desktop_socket_path(), default_socket_path());
        if let Some(value) = previous {
            unsafe {
                std::env::set_var("VOICELAYER_SOCKET_PATH", value);
            }
        }
    }

    const SAMPLE: &str = "{\"event_type\":\"dictation_listening\",\"session_id\":\"00000000-0000-0000-0000-000000000000\",\"created_at_millis\":7}";

    #[test]
    fn sse_decoder_emits_one_event_per_blank_line_terminated_frame() {
        let mut decoder = SseDecoder::default();
        let frame = format!("event:dictation.listening\ndata:{SAMPLE}\n\n");
        let events = decoder.push(frame.as_bytes());
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.name(), "dictation_listening");
        assert_eq!(events[0].created_at_millis, 7);
    }

    #[test]
    fn sse_decoder_waits_for_the_terminating_blank_line() {
        let mut decoder = SseDecoder::default();
        // A complete data line but no blank line yet → nothing dispatched.
        let partial = format!("data:{SAMPLE}\n");
        assert!(decoder.push(partial.as_bytes()).is_empty());
        // The blank line on the next read completes the event.
        let events = decoder.push(b"\n");
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].event.name(), "dictation_listening");
    }

    #[test]
    fn sse_decoder_reassembles_a_frame_split_across_reads() {
        let mut decoder = SseDecoder::default();
        let frame = format!("data:{SAMPLE}\n\n");
        let bytes = frame.as_bytes();
        let (head, tail) = bytes.split_at(bytes.len() / 2);
        assert!(decoder.push(head).is_empty());
        let events = decoder.push(tail);
        assert_eq!(events.len(), 1);
    }

    #[test]
    fn sse_decoder_ignores_keepalive_comments() {
        let mut decoder = SseDecoder::default();
        let events = decoder.push(b":keepalive\n");
        assert!(events.is_empty());
        // And a real frame still decodes after a keepalive.
        let frame = format!("data:{SAMPLE}\n\n");
        assert_eq!(decoder.push(frame.as_bytes()).len(), 1);
    }

    #[test]
    fn sse_decoder_handles_two_events_in_one_chunk() {
        let mut decoder = SseDecoder::default();
        let frame = format!("data:{SAMPLE}\n\ndata:{SAMPLE}\n\n");
        let events = decoder.push(frame.as_bytes());
        assert_eq!(events.len(), 2);
    }
}
