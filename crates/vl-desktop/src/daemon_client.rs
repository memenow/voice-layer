// NOTE: keep in sync with crates/vl/src/uds.rs — the desktop shell needs an
// independent copy of the UDS POST helper so we can iterate on the GUI
// without pulling in `vl` as a library crate. When a third consumer shows up
// we'll extract a shared `voicelayer-client` crate and replace both copies.

use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use voicelayerd::default_socket_path;

pub async fn uds_post_json<TRequest, TResponse>(
    socket_path: &Path,
    path: &str,
    payload: &TRequest,
) -> Result<TResponse, Box<dyn std::error::Error + Send + Sync>>
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

pub async fn uds_get_json<TResponse>(
    socket_path: &Path,
    path: &str,
) -> Result<TResponse, Box<dyn std::error::Error + Send + Sync>>
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

pub fn desktop_socket_path() -> PathBuf {
    std::env::var_os("VOICELAYER_SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}

#[cfg(test)]
mod tests {
    use super::{default_socket_path, desktop_socket_path};
    use std::path::PathBuf;
    use std::sync::Mutex;

    /// Serialises any test that mutates `VOICELAYER_SOCKET_PATH`.
    /// Same rationale as the matching lock in `vl::uds::tests`.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Pins the env-set branch. The desktop shell shares the socket
    /// override convention with the CLI; a regression that always
    /// returned `default_socket_path()` would silently ignore the
    /// operator's `VOICELAYER_SOCKET_PATH` and connect to the wrong
    /// daemon.
    #[test]
    fn desktop_socket_path_uses_voicelayer_socket_path_when_set() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
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

    /// Pins the unset branch: with no override, the resolver must
    /// delegate to `voicelayerd::default_socket_path()` so the CLI
    /// and desktop shell always land on the same socket by default.
    #[test]
    fn desktop_socket_path_falls_back_to_default_socket_path_when_env_unset() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("VOICELAYER_SOCKET_PATH");
        // SAFETY: ENV_LOCK serialises every mutation of this variable.
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
}
