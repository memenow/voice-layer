//! JSON-RPC 2.0 wire types, error definitions, and single-frame exchange logic.

use std::time::Duration;

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{io::AsyncWriteExt, time::timeout};
use voicelayer_core::{ProviderDescriptor, WorkerInitPayload};

use super::process::WorkerProcess;

pub(super) const JSONRPC_VERSION: &str = "2.0";

/// Brief pause before `try_wait` so a respond-then-exit child completes its exit syscall.
const POST_RESPONSE_EXIT_GRACE: Duration = Duration::from_millis(1);

/// Deserialized reply to the worker `health` RPC.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerHealthResult {
    pub status: String,
    pub worker: String,
    pub protocol: String,
    pub asr_configured: bool,
    pub asr_binary: Option<String>,
    pub asr_model_path: Option<String>,
    pub asr_error: Option<String>,
    #[serde(default)]
    pub whisper_mode: Option<String>,
    #[serde(default)]
    pub whisper_server_url: Option<String>,
    #[serde(default)]
    pub mimo_configured: bool,
    #[serde(default)]
    pub mimo_model_path: Option<String>,
    #[serde(default)]
    pub mimo_error: Option<String>,
    #[serde(default)]
    pub qwen3_asr_configured: bool,
    #[serde(default)]
    pub qwen3_asr_model_path: Option<String>,
    #[serde(default)]
    pub qwen3_asr_error: Option<String>,
    pub llm_configured: bool,
    pub llm_model: Option<String>,
    pub llm_endpoint: Option<String>,
    pub llm_reachable: bool,
    pub llm_error: Option<String>,
}

/// Reply to the worker `list_providers` RPC.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerProviderList {
    pub providers: Vec<ProviderDescriptor>,
}

/// Generated text returned by worker `compose`, `rewrite`, and `translate` RPCs.
#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerPreviewPayload {
    pub title: String,
    pub generated_text: String,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Serialize)]
pub(super) struct JsonRpcRequest<P> {
    pub jsonrpc: &'static str,
    pub id: String,
    pub method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub params: Option<P>,
}

impl<'a> JsonRpcRequest<&'a WorkerInitPayload> {
    pub(super) fn initialize(id: String, params: &'a WorkerInitPayload) -> Self {
        Self {
            jsonrpc: JSONRPC_VERSION,
            id,
            method: "initialize".to_owned(),
            params: Some(params),
        }
    }
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse<R> {
    jsonrpc: String,
    id: Option<String>,
    result: Option<R>,
    error: Option<JsonRpcError>,
}

/// JSON-RPC error object returned by the Python worker.
#[derive(Debug, Clone, Deserialize, Error)]
#[error("{message}")]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl JsonRpcError {
    /// Returns `true` for the worker's provider-unavailable error code (`-32004`).
    pub fn is_provider_unavailable(&self) -> bool {
        self.code == -32004
    }
}

/// Transport and protocol failures from communicating with the Python worker.
#[derive(Debug, Error)]
pub enum WorkerCallError {
    #[error("failed to spawn worker process: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode or decode JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    #[error("worker process did not provide a {0} pipe")]
    MissingPipe(&'static str),
    #[error("worker process returned no response")]
    EmptyResponse,
    #[error("worker process timed out while waiting for a response")]
    TimedOut,
    #[error("worker process exited with code {0:?}: {1}")]
    ProcessExited(Option<i32>, String),
    #[error("worker returned unsupported JSON-RPC version `{0}`")]
    InvalidProtocolVersion(String),
    #[error("worker response id mismatch: expected {expected}, got {got:?}")]
    IdMismatch {
        expected: String,
        got: Option<String>,
    },
    #[error("worker returned a malformed JSON-RPC response")]
    MalformedResponse,
    #[error("worker RPC error {0}")]
    Rpc(JsonRpcError),
}

pub(super) enum ExchangeFailure {
    Retryable(WorkerCallError),
    Fatal {
        error: WorkerCallError,
        drop_child: bool,
    },
}

/// Write a serialized JSON-RPC line and parse the matching response line within `budget`.
pub(super) async fn write_and_read<R: DeserializeOwned>(
    process: &mut WorkerProcess,
    payload: &str,
    expected_id: &str,
    budget: Duration,
) -> Result<R, WorkerCallError> {
    process.stdin.write_all(payload.as_bytes()).await?;
    process.stdin.flush().await?;

    let line = timeout(budget, process.stdout.next_line())
        .await
        .map_err(|_| WorkerCallError::TimedOut)?
        .map_err(WorkerCallError::Io)?
        .ok_or(WorkerCallError::EmptyResponse)?;

    decode_response(&line, expected_id, true)
}

/// Perform one JSON-RPC request/response round-trip and inspect post-response child status.
pub(super) async fn exchange_once<R: DeserializeOwned>(
    process: &mut WorkerProcess,
    payload: &str,
    expected_id: &str,
    budget: Duration,
    stderr_tail: impl FnOnce() -> String,
) -> Result<(R, bool), ExchangeFailure> {
    let write_res = async {
        process.stdin.write_all(payload.as_bytes()).await?;
        process.stdin.flush().await?;
        Ok::<(), std::io::Error>(())
    }
    .await;
    if write_res.is_err() {
        return Err(ExchangeFailure::Retryable(WorkerCallError::ProcessExited(
            None,
            stderr_tail(),
        )));
    }

    let line = match timeout(budget, process.stdout.next_line()).await {
        Ok(Ok(Some(line))) => line,
        Ok(Ok(None)) => return Err(ExchangeFailure::Retryable(WorkerCallError::EmptyResponse)),
        Ok(Err(io_err)) => {
            return Err(ExchangeFailure::Fatal {
                error: WorkerCallError::Io(io_err),
                drop_child: true,
            });
        }
        Err(_) => {
            return Err(ExchangeFailure::Fatal {
                error: WorkerCallError::TimedOut,
                drop_child: true,
            });
        }
    };

    let response = parse_envelope::<R>(&line, expected_id, false)
        .map_err(|(error, drop_child)| ExchangeFailure::Fatal { error, drop_child })?;

    tokio::time::sleep(POST_RESPONSE_EXIT_GRACE).await;
    let keep_alive = match process.child.try_wait() {
        Ok(Some(status)) if !status.success() => {
            return Err(ExchangeFailure::Fatal {
                error: WorkerCallError::ProcessExited(status.code(), stderr_tail()),
                drop_child: true,
            });
        }
        Ok(Some(_)) | Err(_) => false,
        Ok(None) => true,
    };

    match (response.result, response.error) {
        (Some(result), None) => Ok((result, keep_alive)),
        (None, Some(error)) => Err(ExchangeFailure::Fatal {
            error: WorkerCallError::Rpc(error),
            drop_child: !keep_alive,
        }),
        _ => Err(ExchangeFailure::Fatal {
            error: WorkerCallError::MalformedResponse,
            drop_child: !keep_alive,
        }),
    }
}

fn decode_response<R: DeserializeOwned>(
    line: &str,
    expected_id: &str,
    require_id: bool,
) -> Result<R, WorkerCallError> {
    let response = parse_envelope::<R>(line, expected_id, require_id).map_err(|(err, _)| err)?;
    match (response.result, response.error) {
        (Some(result), None) => Ok(result),
        (None, Some(error)) => Err(WorkerCallError::Rpc(error)),
        _ => Err(WorkerCallError::MalformedResponse),
    }
}

fn parse_envelope<R: DeserializeOwned>(
    line: &str,
    expected_id: &str,
    require_id: bool,
) -> Result<JsonRpcResponse<R>, (WorkerCallError, bool)> {
    let response: JsonRpcResponse<R> =
        serde_json::from_str(line).map_err(|err| (WorkerCallError::Json(err), false))?;
    if response.jsonrpc != JSONRPC_VERSION {
        return Err((
            WorkerCallError::InvalidProtocolVersion(response.jsonrpc),
            false,
        ));
    }
    let id_mismatch = if require_id {
        response.id.as_deref() != Some(expected_id)
    } else {
        response.id.as_deref().is_some_and(|got| got != expected_id)
    };
    if id_mismatch {
        return Err((
            WorkerCallError::IdMismatch {
                expected: expected_id.to_owned(),
                got: response.id,
            },
            true,
        ));
    }
    Ok(response)
}
