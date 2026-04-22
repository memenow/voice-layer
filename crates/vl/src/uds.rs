use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use voicelayerd::default_socket_path;

/// Tri-state outcome from a daemon-first dispatch attempt.
///
/// Command handlers use this to decide whether an in-process fallback
/// should happen silently or should first warn the operator:
///
/// - [`DaemonOutcome::Ok`] — the daemon responded successfully.
/// - [`DaemonOutcome::SilentMiss`] — the socket does not exist. Normal
///   pre-daemon / scripting scenario; fall back without comment.
/// - [`DaemonOutcome::Rejected`] — the socket exists but the request
///   failed (transport, non-2xx status, or deserialization). Operators
///   should see the error even if we fall back, since a misconfigured
///   daemon silently routing to a local worker would hide the cause.
pub(crate) enum DaemonOutcome<T> {
    Ok(T),
    SilentMiss,
    Rejected(Box<dyn std::error::Error>),
}

/// Probe the daemon socket, POST the payload when present, and return
/// a [`DaemonOutcome`] that preserves the silent-miss vs noisy-failure
/// distinction.
///
/// The caller supplies its own in-process fallback after matching on
/// the outcome; this helper only encapsulates the probe-and-dispatch
/// pattern so it can be unit-tested in isolation from the command
/// handler.
pub(crate) async fn try_daemon_post<TRequest, TResponse>(
    socket_path: &Path,
    path: &str,
    payload: &TRequest,
) -> DaemonOutcome<TResponse>
where
    TRequest: serde::Serialize,
    TResponse: serde::de::DeserializeOwned,
{
    if !socket_path.exists() {
        return DaemonOutcome::SilentMiss;
    }
    match uds_post_json(socket_path, path, payload).await {
        Ok(value) => DaemonOutcome::Ok(value),
        Err(error) => DaemonOutcome::Rejected(error),
    }
}

pub(crate) async fn uds_post_json<TRequest, TResponse>(
    socket_path: &Path,
    path: &str,
    payload: &TRequest,
) -> Result<TResponse, Box<dyn std::error::Error>>
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

pub(crate) async fn uds_get_json<TResponse>(
    socket_path: &Path,
    path: &str,
) -> Result<TResponse, Box<dyn std::error::Error>>
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

pub(crate) fn cli_socket_path() -> PathBuf {
    std::env::var_os("VOICELAYER_SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

#[cfg(test)]
mod tests {
    use super::{DaemonOutcome, try_daemon_post};
    use std::path::PathBuf;
    use tempfile::TempDir;
    use tokio::io::{AsyncReadExt, AsyncWriteExt};
    use tokio::net::UnixListener;

    /// Minimal HTTP/1.1 responder on a UDS path. Reads the client's
    /// request bytes (discarded — the test doesn't care what hyper
    /// sent, only that it sent *something*), writes the supplied raw
    /// bytes, and closes the connection. We bind synchronously before
    /// spawning the accept task so the caller can connect without
    /// racing on socket creation.
    fn spawn_fake_daemon(
        socket_path: PathBuf,
        raw_response: &'static str,
    ) -> tokio::task::JoinHandle<()> {
        let listener = UnixListener::bind(&socket_path).expect("bind test UDS");
        tokio::spawn(async move {
            let (mut stream, _) = match listener.accept().await {
                Ok(pair) => pair,
                Err(_) => return,
            };
            let mut buf = [0u8; 4096];
            let _ = stream.read(&mut buf).await;
            let _ = stream.write_all(raw_response.as_bytes()).await;
            let _ = stream.shutdown().await;
        })
    }

    const OK_JSON: &str = "HTTP/1.1 200 OK\r\n\
        Content-Type: application/json\r\n\
        Content-Length: 13\r\n\
        Connection: close\r\n\
        \r\n\
        {\"text\":\"hi\"}";

    const INTERNAL_ERROR: &str = "HTTP/1.1 500 Internal Server Error\r\n\
        Content-Type: text/plain\r\n\
        Content-Length: 15\r\n\
        Connection: close\r\n\
        \r\n\
        worker exploded";

    #[tokio::test]
    async fn try_daemon_post_returns_silent_miss_when_socket_absent() {
        let tmp = TempDir::new().expect("tempdir");
        let socket_path = tmp.path().join("no-daemon.sock");
        // Intentionally don't bind — the path must not exist.
        assert!(!socket_path.exists());

        let outcome: DaemonOutcome<serde_json::Value> =
            try_daemon_post(&socket_path, "/v1/any", &serde_json::json!({})).await;

        assert!(
            matches!(outcome, DaemonOutcome::SilentMiss),
            "absent socket should produce SilentMiss, not a probe attempt",
        );
    }

    #[tokio::test]
    async fn try_daemon_post_returns_ok_when_daemon_responds_with_2xx() {
        let tmp = TempDir::new().expect("tempdir");
        let socket_path = tmp.path().join("daemon.sock");
        let server = spawn_fake_daemon(socket_path.clone(), OK_JSON);

        let outcome: DaemonOutcome<serde_json::Value> =
            try_daemon_post(&socket_path, "/v1/any", &serde_json::json!({"x": 1})).await;

        match outcome {
            DaemonOutcome::Ok(value) => assert_eq!(value["text"], "hi"),
            DaemonOutcome::SilentMiss => panic!("socket existed; expected Ok not SilentMiss"),
            DaemonOutcome::Rejected(error) => panic!("expected Ok, got Rejected: {error}"),
        }
        let _ = server.await;
    }

    #[tokio::test]
    async fn try_daemon_post_returns_rejected_when_daemon_responds_with_5xx() {
        let tmp = TempDir::new().expect("tempdir");
        let socket_path = tmp.path().join("bad-daemon.sock");
        let server = spawn_fake_daemon(socket_path.clone(), INTERNAL_ERROR);

        let outcome: DaemonOutcome<serde_json::Value> =
            try_daemon_post(&socket_path, "/v1/any", &serde_json::json!({})).await;

        match outcome {
            DaemonOutcome::Rejected(error) => {
                let message = error.to_string();
                assert!(
                    message.contains("500"),
                    "error should mention the HTTP status, got: {message}",
                );
            }
            DaemonOutcome::Ok(_) => panic!("expected Rejected, got Ok"),
            DaemonOutcome::SilentMiss => panic!("expected Rejected, got SilentMiss"),
        }
        let _ = server.await;
    }

    #[tokio::test]
    async fn try_daemon_post_returns_rejected_when_daemon_sends_garbage() {
        // Server exists and accepts the connection, but returns bytes
        // hyper cannot parse as HTTP. This covers the transport-error
        // branch of Rejected (distinct from the 5xx-status branch).
        let tmp = TempDir::new().expect("tempdir");
        let socket_path = tmp.path().join("garbage-daemon.sock");
        let server = spawn_fake_daemon(socket_path.clone(), "this is not HTTP\r\n\r\n");

        let outcome: DaemonOutcome<serde_json::Value> =
            try_daemon_post(&socket_path, "/v1/any", &serde_json::json!({})).await;

        assert!(
            matches!(outcome, DaemonOutcome::Rejected(_)),
            "non-HTTP response must produce Rejected so the CLI warns the operator",
        );
        let _ = server.await;
    }
}
