use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use voicelayer_core::default_socket_path;

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
    use super::cli_socket_path;
    use std::path::PathBuf;
    use std::sync::Mutex;
    use voicelayer_core::default_socket_path;

    /// Serialises any test that mutates `VOICELAYER_SOCKET_PATH`.
    /// Cargo runs unit tests concurrently and Rust 2024 made
    /// `env::set_var` `unsafe` precisely because a concurrent reader
    /// in another thread is UB.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Minimal HTTP/1.1 responder on a UDS path. Reads the client's
    /// request bytes (discarded — the test doesn't care what hyper
    /// sent, only that it sent *something*), writes the supplied raw
    /// bytes, and closes the connection. We bind synchronously before
    /// spawning the accept task so the caller can connect without
    /// Pins the env-set branch of `cli_socket_path`. The CLI honours
    /// `VOICELAYER_SOCKET_PATH` so contributors can point `vl` at a
    /// non-default socket without rebuilding; a regression that
    /// always returned `default_socket_path()` would silently ignore
    /// the override and the operator would have no obvious failure
    /// to diagnose.
    #[test]
    fn cli_socket_path_uses_voicelayer_socket_path_when_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::set_var("VOICELAYER_SOCKET_PATH", "/run/test-vl/daemon.sock");
        }
        let path = cli_socket_path();
        assert_eq!(path, PathBuf::from("/run/test-vl/daemon.sock"));
        match previous {
            Some(value) => unsafe {
                std::env::set_var("VOICELAYER_SOCKET_PATH", value);
            },
            None => unsafe {
                std::env::remove_var("VOICELAYER_SOCKET_PATH");
            },
        }
    }

    /// Pins the unset branch: with no override, `cli_socket_path`
    /// must delegate to `default_socket_path()`. The exact path is
    /// XDG-driven (already pinned in voicelayerd #38), so this
    /// asserts equality with the live default rather than a literal.
    #[test]
    fn cli_socket_path_falls_back_to_default_socket_path_when_env_unset() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
        unsafe {
            std::env::remove_var("VOICELAYER_SOCKET_PATH");
        }
        assert_eq!(cli_socket_path(), default_socket_path());
        if let Some(value) = previous {
            unsafe {
                std::env::set_var("VOICELAYER_SOCKET_PATH", value);
            }
        }
    }
}
