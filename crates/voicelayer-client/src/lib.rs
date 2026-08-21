//! Typed client for the VoiceLayer daemon's local control API.
//!
//! Every consumer of the daemon (CLI, desktop shell, future tooling) talks
//! to `voicelayerd` exclusively through this crate over the Unix domain
//! socket. Non-2xx responses are surfaced as [`ClientError::Problem`]
//! carrying the RFC 9457 problem details body.

use std::path::PathBuf;

use bytes::Bytes;
use futures_util::{Stream, StreamExt as _};
use http_body_util::{BodyExt, Full};
use hyper::{Request, StatusCode, header::HOST};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use voicelayer_core::{EventEnvelope, ProblemDetails, default_socket_path};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("failed to reach the daemon socket: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode or decode a JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    #[error("HTTP transport error: {0}")]
    Transport(String),
    #[error("daemon returned {status}: {}", .problem.detail)]
    Problem {
        status: StatusCode,
        problem: ProblemDetails,
    },
    /// Non-2xx response without a parseable problem details body.
    #[error("daemon returned unexpected status {0}: {1}")]
    UnexpectedStatus(StatusCode, String),
}

#[derive(Debug, Clone)]
pub struct Client {
    socket_path: PathBuf,
}

impl Client {
    pub fn new(socket_path: PathBuf) -> Self {
        Self { socket_path }
    }

    /// `VOICELAYER_SOCKET_PATH` override, else the platform default.
    pub fn from_env() -> Self {
        let socket_path = std::env::var_os("VOICELAYER_SOCKET_PATH")
            .map(PathBuf::from)
            .unwrap_or_else(default_socket_path);
        Self { socket_path }
    }

    pub fn socket_path(&self) -> &std::path::Path {
        &self.socket_path
    }

    pub async fn get<R>(&self, path: &str) -> Result<R, ClientError>
    where
        R: serde::de::DeserializeOwned,
    {
        let request = Request::get(path)
            .header(HOST, "localhost")
            .body(Full::new(Bytes::new()))
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        self.round_trip(request).await
    }

    pub async fn post<P, R>(&self, path: &str, payload: &P) -> Result<R, ClientError>
    where
        P: serde::Serialize,
        R: serde::de::DeserializeOwned,
    {
        let body = serde_json::to_vec(payload)?;
        let request = Request::post(path)
            .header(HOST, "localhost")
            .header("content-type", "application/json")
            .body(Full::new(Bytes::from(body)))
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        self.round_trip(request).await
    }

    /// Subscribe to the daemon's Server-Sent Events stream.
    pub async fn events(
        &self,
    ) -> Result<impl Stream<Item = Result<EventEnvelope, ClientError>>, ClientError> {
        let mut sender = self.connect().await?;
        let request = Request::get("/v1/events/stream")
            .header(HOST, "localhost")
            .body(Full::new(Bytes::new()))
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        let response = sender
            .send_request(request)
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        let status = response.status();
        if !status.is_success() {
            let body = response
                .into_body()
                .collect()
                .await
                .map_err(|error| ClientError::Transport(error.to_string()))?
                .to_bytes();
            return Err(decode_error(status, &body));
        }

        let byte_stream = response.into_body().into_data_stream();
        Ok(futures_util::stream::try_unfold(
            (byte_stream, String::new()),
            |(mut byte_stream, mut pending)| async move {
                loop {
                    if let Some(frame_end) = pending.find("\n\n") {
                        let frame = pending[..frame_end].to_owned();
                        pending = pending[frame_end + 2..].to_owned();
                        if let Some(envelope) = parse_sse_frame(&frame) {
                            return Ok(Some((envelope, (byte_stream, pending))));
                        }
                        continue;
                    }
                    match byte_stream.next().await {
                        Some(Ok(chunk)) => {
                            pending.push_str(&String::from_utf8_lossy(&chunk));
                        }
                        Some(Err(error)) => {
                            return Err(ClientError::Transport(error.to_string()));
                        }
                        None => return Ok(None),
                    }
                }
            },
        ))
    }

    async fn round_trip<R>(&self, request: Request<Full<Bytes>>) -> Result<R, ClientError>
    where
        R: serde::de::DeserializeOwned,
    {
        let mut sender = self.connect().await?;
        let response = sender
            .send_request(request)
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        let status = response.status();
        let body = response
            .into_body()
            .collect()
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?
            .to_bytes();
        if !status.is_success() {
            return Err(decode_error(status, &body));
        }
        Ok(serde_json::from_slice(&body)?)
    }

    async fn connect(
        &self,
    ) -> Result<hyper::client::conn::http1::SendRequest<Full<Bytes>>, ClientError> {
        let stream = UnixStream::connect(&self.socket_path).await?;
        let io = TokioIo::new(stream);
        let (sender, connection) = hyper::client::conn::http1::handshake(io)
            .await
            .map_err(|error| ClientError::Transport(error.to_string()))?;
        tokio::spawn(async move {
            let _ = connection.await;
        });
        Ok(sender)
    }
}

fn decode_error(status: StatusCode, body: &[u8]) -> ClientError {
    match serde_json::from_slice::<ProblemDetails>(body) {
        Ok(problem) if problem.status == status.as_u16() => {
            ClientError::Problem { status, problem }
        }
        _ => ClientError::UnexpectedStatus(status, String::from_utf8_lossy(body).into_owned()),
    }
}

/// Extract the `data:` payload of one SSE frame as an [`EventEnvelope`].
fn parse_sse_frame(frame: &str) -> Option<EventEnvelope> {
    let data: String = frame
        .lines()
        .filter_map(|line| line.strip_prefix("data:").map(str::trim_start))
        .collect::<Vec<_>>()
        .join("\n");
    if data.is_empty() {
        return None;
    }
    serde_json::from_str(&data).ok()
}
