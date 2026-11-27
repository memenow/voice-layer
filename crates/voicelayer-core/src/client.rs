//! Shared Unix-domain-socket HTTP/1.1 client for `voicelayerd`.

use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, StatusCode, header::HOST};
use hyper_util::rt::TokioIo;
use serde::Serialize;
use serde::de::DeserializeOwned;
use tokio::net::UnixStream;

use crate::config::{VoiceLayerConfig, config_path, default_socket_path};
use crate::domain::EventEnvelope;

/// Errors returned by [`DaemonClient`].
#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("failed to connect to daemon socket: {0}")]
    Io(#[from] std::io::Error),
    #[error("HTTP transport error: {0}")]
    Hyper(#[from] hyper::Error),
    #[error("failed to construct HTTP request: {0}")]
    Http(#[from] hyper::http::Error),
    #[error("failed to serialize or parse JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("daemon returned non-success status {status}: {body}")]
    HttpStatus { status: StatusCode, body: String },
}

/// Resolve the daemon socket path: `VOICELAYER_SOCKET_PATH`, then `[daemon] socket_path`
/// in `config.toml`, then [`default_socket_path`].
pub fn resolve_socket_path() -> PathBuf {
    if let Some(override_path) = std::env::var_os("VOICELAYER_SOCKET_PATH")
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
    {
        return override_path;
    }
    if let Ok(path) = config_path()
        && let Ok(config) = VoiceLayerConfig::load_file_only(&path)
        && config.daemon.socket_path_raw.is_some()
    {
        return config.daemon.socket_path();
    }
    default_socket_path()
}

/// HTTP/1.1 JSON and SSE client talking to `voicelayerd` over a Unix domain socket.
#[derive(Debug, Clone)]
pub struct DaemonClient {
    socket_path: PathBuf,
}

impl DaemonClient {
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    pub fn from_env() -> Self {
        Self::new(resolve_socket_path())
    }

    pub fn socket_path(&self) -> &Path {
        &self.socket_path
    }

    /// Issue a `GET` request and deserialize the JSON response body.
    pub async fn get_json<TResponse>(&self, path: &str) -> Result<TResponse, ClientError>
    where
        TResponse: DeserializeOwned,
    {
        let request = Request::get(path)
            .header(HOST, "localhost")
            .body(Full::new(Bytes::new()))?;
        self.send_json(request).await
    }

    /// Issue a `POST` request with a JSON body and deserialize the JSON response body.
    pub async fn post_json<TRequest, TResponse>(
        &self,
        path: &str,
        payload: &TRequest,
    ) -> Result<TResponse, ClientError>
    where
        TRequest: Serialize,
        TResponse: DeserializeOwned,
    {
        let body = serde_json::to_vec(payload)?;
        let request = Request::post(path)
            .header(HOST, "localhost")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body)))?;
        self.send_json(request).await
    }

    /// Open `GET /v1/events/stream` and invoke `on_event` for each decoded [`EventEnvelope`].
    ///
    /// Returns `Ok(())` when the stream ends or `on_event` returns `false`.
    pub async fn stream_events<F>(&self, mut on_event: F) -> Result<(), ClientError>
    where
        F: FnMut(EventEnvelope) -> bool,
    {
        let mut sender = self.connect().await?;
        let request = Request::get("/v1/events/stream")
            .header(HOST, "localhost")
            .body(Full::new(Bytes::new()))?;
        let response = sender.send_request(request).await?;
        let status = response.status();
        if !status.is_success() {
            let body = response.into_body().collect().await?.to_bytes();
            return Err(ClientError::HttpStatus {
                status,
                body: String::from_utf8_lossy(&body).into_owned(),
            });
        }

        let mut body = response.into_body();
        let mut decoder = SseDecoder::default();
        while let Some(frame) = body.frame().await {
            let frame = frame?;
            if let Some(chunk) = frame.data_ref() {
                for event in decoder.push(chunk) {
                    if !on_event(event) {
                        return Ok(());
                    }
                }
            }
        }
        Ok(())
    }

    async fn connect(&self) -> Result<http1::SendRequest<Full<Bytes>>, ClientError> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let io = TokioIo::new(stream);
        let (sender, connection) = http1::handshake(io).await?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        Ok(sender)
    }

    async fn send_json<TResponse>(
        &self,
        request: Request<Full<Bytes>>,
    ) -> Result<TResponse, ClientError>
    where
        TResponse: DeserializeOwned,
    {
        let mut sender = self.connect().await?;
        let response = sender.send_request(request).await?;
        let status = response.status();
        let bytes = response.into_body().collect().await?.to_bytes();
        if !status.is_success() {
            return Err(ClientError::HttpStatus {
                status,
                body: String::from_utf8_lossy(&bytes).into_owned(),
            });
        }
        Ok(serde_json::from_slice(&bytes)?)
    }
}

/// Incremental SSE line decoder for `GET /v1/events/stream`.
///
/// Buffers raw bytes until `\n` so multi-byte UTF-8 sequences split across reads
/// decode cleanly. Ignores `event:`, `id:`, `retry:`, and comment keepalives
/// because each JSON payload already carries `event_type`.
#[derive(Debug, Default)]
pub struct SseDecoder {
    buf: Vec<u8>,
    data: String,
}

impl SseDecoder {
    /// Feed raw bytes and return any completed [`EventEnvelope`] items.
    pub fn push(&mut self, chunk: &[u8]) -> Vec<EventEnvelope> {
        self.buf.extend_from_slice(chunk);
        let mut events = Vec::new();
        while let Some(pos) = self.buf.iter().position(|&b| b == b'\n') {
            let mut line: Vec<u8> = self.buf.drain(..=pos).collect();
            line.pop();
            if line.last() == Some(&b'\r') {
                line.pop();
            }
            let line = String::from_utf8_lossy(&line);
            if line.is_empty() {
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
        }
        events
    }
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;
    use voicelayer_testkit::EnvGuard;

    use super::{ClientError, DaemonClient, SseDecoder, resolve_socket_path};
    use crate::config::default_socket_path;
    use crate::domain::{DaemonEvent, EventEnvelope};

    #[test]
    fn resolve_socket_path_honors_env_override() {
        let mut env = EnvGuard::lock();
        env.set("VOICELAYER_SOCKET_PATH", "/run/test-vl/daemon.sock");
        assert_eq!(
            resolve_socket_path(),
            PathBuf::from("/run/test-vl/daemon.sock")
        );
    }

    #[test]
    fn resolve_socket_path_falls_back_to_default_when_unset() {
        let mut env = EnvGuard::lock();
        env.remove("VOICELAYER_SOCKET_PATH");
        assert_eq!(resolve_socket_path(), default_socket_path());
    }

    #[test]
    fn sse_decoder_reassembles_split_utf8_chunks_and_ignores_keepalives() {
        let envelope = EventEnvelope::new(DaemonEvent::ComposeJobCreated {
            title: "你好世界".to_owned(),
        });
        let json = serde_json::to_string(&envelope).expect("encode envelope");
        let wire = format!(": keepalive\nevent: compose_job_created\ndata: {json}\n\n");
        let bytes = wire.as_bytes();
        let split = bytes.len() / 2;

        let mut decoder = SseDecoder::default();
        assert!(decoder.push(&bytes[..split]).is_empty());
        let events = decoder.push(&bytes[split..]);
        assert_eq!(events, vec![envelope]);
    }

    #[tokio::test]
    async fn daemon_client_returns_http_status_error_on_non_2xx() {
        let dir = tempfile::tempdir().expect("tempdir");
        let sock = dir.path().join("test.sock");
        let listener = UnixListener::bind(&sock).expect("bind uds");

        tokio::spawn(async move {
            let (mut stream, _) = listener.accept().await.expect("accept");
            let mut req_buf = [0u8; 512];
            let _ = stream.read(&mut req_buf).await;
            let reply = b"HTTP/1.1 503 Service Unavailable\r\nContent-Length: 4\r\n\r\nbusy";
            let _ = stream.write_all(reply).await;
        });

        let client = DaemonClient::new(sock);
        let err = client
            .get_json::<serde_json::Value>("/v1/health")
            .await
            .expect_err("expected HTTP error");
        assert!(matches!(
            err,
            ClientError::HttpStatus { status, body } if status.as_u16() == 503 && body == "busy"
        ));
    }
}
