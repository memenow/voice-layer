use std::path::{Path, PathBuf};

use bytes::Bytes;
use http_body_util::{BodyExt, Full};
use hyper::client::conn::http1;
use hyper::{Request, header::HOST};
use hyper_util::rt::TokioIo;
use tokio::net::UnixStream;
use voicelayerd::default_socket_path;

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

pub(crate) fn cli_socket_path() -> PathBuf {
    std::env::var_os("VOICELAYER_SOCKET_PATH")
        .map(PathBuf::from)
        .unwrap_or_else(default_socket_path)
}
