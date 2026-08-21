//! Persistent Python worker process manager.
//!
//! The daemon spawns `voicelayer_orchestrator.worker` once (lazily, on the
//! first inference call) and multiplexes every JSON-RPC method over the same
//! stdio pair. The first frame after spawn is always `initialize`, carrying
//! the provider configuration payload; the worker answers it before serving
//! anything else.
//!
//! Calls are serialized through a single mutex: the worker serves one
//! request at a time by design, so pipelining would buy nothing. A process
//! that dies mid-call fails the in-flight request and is respawned on the
//! next call.

use std::{path::PathBuf, process::Stdio, time::Duration};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout},
    sync::Mutex,
    time::timeout,
};
use uuid::Uuid;
use voicelayer_core::{ProviderDescriptor, TranscriptionResult, WorkerInitPayload};

const JSONRPC_VERSION: &str = "2.0";
const WORKER_MODULE: &str = "voicelayer_orchestrator.worker";

/// `health` and `list_providers` are cheap probes and stay short so CLI
/// commands that depend on them don't hang on a misconfigured worker. Every
/// other method invokes real inference and uses the configured budget.
const PROBE_TIMEOUT: Duration = Duration::from_secs(15);

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct WorkerCommand {
    pub executable: String,
    pub args: Vec<String>,
    pub project_root: PathBuf,
}

impl WorkerCommand {
    pub fn discover(project_root: PathBuf) -> Self {
        let uv_python = project_root.join(".venv").join("bin").join("python");
        if uv_python.is_file() {
            return Self {
                executable: uv_python.display().to_string(),
                args: vec!["-m".to_owned(), WORKER_MODULE.to_owned()],
                project_root,
            };
        }

        Self {
            executable: "uv".to_owned(),
            args: vec![
                "run".to_owned(),
                "--project".to_owned(),
                project_root.display().to_string(),
                "python".to_owned(),
                "-m".to_owned(),
                WORKER_MODULE.to_owned(),
            ],
            project_root,
        }
    }

    pub fn display(&self) -> String {
        std::iter::once(self.executable.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }
}

/// Owns the worker child process. Dropping the manager kills the child
/// (`kill_on_drop`).
pub struct WorkerManager {
    command: WorkerCommand,
    init_payload: WorkerInitPayload,
    inference_timeout: Duration,
    process: Mutex<Option<WorkerProcess>>,
}

struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
}

impl WorkerManager {
    pub fn new(
        project_root: PathBuf,
        init_payload: WorkerInitPayload,
        inference_timeout: Duration,
    ) -> Self {
        Self {
            command: WorkerCommand::discover(project_root),
            init_payload,
            inference_timeout,
            process: Mutex::new(None),
        }
    }

    pub fn command_display(&self) -> String {
        self.command.display()
    }

    /// Whether the worker child is currently running (used to keep health
    /// refreshes lazy: no refresh may spawn the worker by itself).
    pub async fn is_running(&self) -> bool {
        self.process.lock().await.is_some()
    }

    fn call_timeout(&self, method: &str) -> Duration {
        match method {
            "health" | "list_providers" => PROBE_TIMEOUT,
            _ => self.inference_timeout,
        }
    }

    pub async fn health(&self) -> Result<WorkerHealthResult, WorkerCallError> {
        self.call::<(), _>("health", None).await
    }

    pub async fn list_providers(&self) -> Result<WorkerProviderList, WorkerCallError> {
        self.call::<(), _>("list_providers", None).await
    }

    pub async fn compose(
        &self,
        request: &voicelayer_core::ComposeRequest,
    ) -> Result<WorkerPreviewPayload, WorkerCallError> {
        self.call("compose", Some(request)).await
    }

    pub async fn rewrite(
        &self,
        request: &voicelayer_core::RewriteRequest,
    ) -> Result<WorkerPreviewPayload, WorkerCallError> {
        self.call("rewrite", Some(request)).await
    }

    pub async fn translate(
        &self,
        request: &voicelayer_core::TranslateRequest,
    ) -> Result<WorkerPreviewPayload, WorkerCallError> {
        self.call("translate", Some(request)).await
    }

    pub async fn transcribe(
        &self,
        request: &voicelayer_core::TranscribeRequest,
    ) -> Result<TranscriptionResult, WorkerCallError> {
        self.call("transcribe", Some(request)).await
    }

    async fn call<P, R>(&self, method: &str, params: Option<P>) -> Result<R, WorkerCallError>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let mut guard = self.process.lock().await;
        if guard.is_none() {
            *guard = Some(self.spawn_and_initialize().await?);
        }
        let process = guard.as_mut().expect("worker process just spawned");

        match self.round_trip(process, method, params).await {
            Ok(result) => Ok(result),
            Err(error) => {
                if error.is_process_failure() {
                    // The pipe is gone or the process died; drop it so the
                    // next call respawns instead of writing into a void.
                    if let Some(mut dead) = guard.take() {
                        let _ = dead.child.kill().await;
                    }
                }
                Err(error)
            }
        }
    }

    async fn spawn_and_initialize(&self) -> Result<WorkerProcess, WorkerCallError> {
        let mut command = tokio::process::Command::new(&self.command.executable);
        command
            .args(&self.command.args)
            .current_dir(&self.command.project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::null())
            .kill_on_drop(true);

        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or(WorkerCallError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(WorkerCallError::MissingPipe("stdout"))?;

        let mut process = WorkerProcess {
            child,
            stdin,
            stdout: BufReader::new(stdout).lines(),
        };

        let id = Uuid::new_v4().to_string();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: id.clone(),
            method: "initialize".to_owned(),
            params: Some(&self.init_payload),
        };
        write_request(&mut process.stdin, &request).await?;
        let response: JsonRpcResponse<InitializeResult> =
            read_response(&mut process.stdout, PROBE_TIMEOUT).await?;
        match response.into_result(&id)? {
            Ok(result) if result.status == "ok" => Ok(process),
            Ok(result) => Err(WorkerCallError::InitializeFailed(format!(
                "worker reported status `{}`",
                result.status
            ))),
            Err(error) => Err(WorkerCallError::InitializeFailed(error.to_string())),
        }
    }

    async fn round_trip<P, R>(
        &self,
        process: &mut WorkerProcess,
        method: &str,
        params: Option<P>,
    ) -> Result<R, WorkerCallError>
    where
        P: Serialize,
        R: DeserializeOwned,
    {
        let id = Uuid::new_v4().to_string();
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: id.clone(),
            method: method.to_owned(),
            params,
        };
        write_request(&mut process.stdin, &request).await?;
        let response: JsonRpcResponse<R> =
            read_response(&mut process.stdout, self.call_timeout(method)).await?;
        response.into_result(&id)?.map_err(WorkerCallError::Rpc)
    }

    /// Kill the worker child. Called on daemon shutdown.
    pub async fn shutdown(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut process) = guard.take() {
            let _ = process.child.kill().await;
        }
    }
}

async fn write_request<P: Serialize>(
    stdin: &mut ChildStdin,
    request: &JsonRpcRequest<P>,
) -> Result<(), WorkerCallError> {
    let payload = serde_json::to_string(request)?;
    stdin
        .write_all(payload.as_bytes())
        .await
        .map_err(|_| WorkerCallError::ProcessDied)?;
    stdin
        .write_all(b"\n")
        .await
        .map_err(|_| WorkerCallError::ProcessDied)?;
    stdin
        .flush()
        .await
        .map_err(|_| WorkerCallError::ProcessDied)
}

async fn read_response<R: DeserializeOwned>(
    stdout: &mut Lines<BufReader<ChildStdout>>,
    budget: Duration,
) -> Result<JsonRpcResponse<R>, WorkerCallError> {
    let line = timeout(budget, stdout.next_line())
        .await
        .map_err(|_| WorkerCallError::TimedOut)?
        .map_err(|_| WorkerCallError::ProcessDied)?
        .ok_or(WorkerCallError::ProcessDied)?;
    Ok(serde_json::from_str(&line)?)
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerHealthResult {
    pub status: String,
    pub worker: String,
    pub protocol: String,
    pub asr_configured: bool,
    pub asr_binary: Option<String>,
    pub asr_model_path: Option<String>,
    pub asr_error: Option<String>,
    pub llm_configured: bool,
    pub llm_model: Option<String>,
    pub llm_endpoint: Option<String>,
    pub llm_reachable: bool,
    pub llm_error: Option<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerProviderList {
    pub providers: Vec<ProviderDescriptor>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct WorkerPreviewPayload {
    pub title: String,
    pub generated_text: String,
    #[serde(default)]
    pub notes: Vec<String>,
}

#[derive(Debug, Clone, Deserialize, PartialEq, Eq)]
pub struct InitializeResult {
    pub status: String,
}

#[derive(Debug, Serialize)]
struct JsonRpcRequest<P> {
    jsonrpc: &'static str,
    id: String,
    method: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    params: Option<P>,
}

#[derive(Debug, Deserialize)]
struct JsonRpcResponse<R> {
    jsonrpc: String,
    id: Option<String>,
    result: Option<R>,
    error: Option<JsonRpcError>,
}

impl<R> JsonRpcResponse<R> {
    /// Validate protocol version and response id, then unwrap result/error.
    fn into_result(self, expected_id: &str) -> Result<Result<R, JsonRpcError>, WorkerCallError> {
        if self.jsonrpc != JSONRPC_VERSION {
            return Err(WorkerCallError::InvalidProtocolVersion(self.jsonrpc));
        }
        if self.id.as_deref() != Some(expected_id) {
            return Err(WorkerCallError::IdMismatch {
                expected: expected_id.to_owned(),
                got: self.id,
            });
        }
        match (self.result, self.error) {
            (Some(result), None) => Ok(Ok(result)),
            (None, Some(error)) => Ok(Err(error)),
            _ => Err(WorkerCallError::MalformedResponse),
        }
    }
}

#[derive(Debug, Clone, Deserialize, Error)]
#[error("{message}")]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

impl JsonRpcError {
    pub fn is_provider_unavailable(&self) -> bool {
        self.code == -32004
    }
}

#[derive(Debug, Error)]
pub enum WorkerCallError {
    #[error("failed to spawn worker process: {0}")]
    Io(#[from] std::io::Error),
    #[error("failed to encode or decode JSON payload: {0}")]
    Json(#[from] serde_json::Error),
    #[error("worker process did not provide a {0} pipe")]
    MissingPipe(&'static str),
    #[error("worker process died or closed its pipes")]
    ProcessDied,
    #[error("worker process timed out while waiting for a response")]
    TimedOut,
    #[error("worker initialize handshake failed: {0}")]
    InitializeFailed(String),
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

impl WorkerCallError {
    /// Errors after which the process must be discarded and respawned.
    ///
    /// `TimedOut` is included: a late response from the timed-out request is
    /// undrainable, so the pipe state is unknowable and the next call would
    /// read the stale line and fail with an id mismatch.
    fn is_process_failure(&self) -> bool {
        matches!(
            self,
            Self::Io(_)
                | Self::MissingPipe(_)
                | Self::ProcessDied
                | Self::TimedOut
                | Self::InvalidProtocolVersion(_)
                | Self::IdMismatch { .. }
                | Self::MalformedResponse
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn response_id_is_validated() {
        let response: JsonRpcResponse<serde_json::Value> =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":"abc","result":{"ok":true}}"#).unwrap();
        assert!(response.into_result("abc").is_ok());

        let mismatched: JsonRpcResponse<serde_json::Value> =
            serde_json::from_str(r#"{"jsonrpc":"2.0","id":"other","result":{"ok":true}}"#).unwrap();
        assert!(matches!(
            mismatched.into_result("abc"),
            Err(WorkerCallError::IdMismatch { .. })
        ));
    }

    #[test]
    fn response_rejects_wrong_protocol_version() {
        let response: JsonRpcResponse<serde_json::Value> =
            serde_json::from_str(r#"{"jsonrpc":"1.0","id":"abc","result":null}"#).unwrap();
        assert!(matches!(
            response.into_result("abc"),
            Err(WorkerCallError::InvalidProtocolVersion(_))
        ));
    }

    #[test]
    fn timeout_is_a_process_failure() {
        // A late response to a timed-out request is undrainable; keeping the
        // process would poison the next call with an id mismatch.
        assert!(WorkerCallError::TimedOut.is_process_failure());
        assert!(WorkerCallError::ProcessDied.is_process_failure());
        assert!(
            !WorkerCallError::Rpc(JsonRpcError {
                code: -32004,
                message: "unavailable".to_owned(),
            })
            .is_process_failure()
        );
    }

    #[tokio::test]
    async fn worker_manager_serves_multiple_calls_on_one_process() {
        let manager = WorkerManager::new(
            std::env::current_dir().expect("current_dir should be available"),
            voicelayer_core::VoiceLayerConfig::default().worker_payload(),
            Duration::from_secs(60),
        );

        let health = manager
            .health()
            .await
            .expect("worker health should succeed");
        assert_eq!(health.status, "ok");
        assert_eq!(health.protocol, "2.0");

        let providers = manager
            .list_providers()
            .await
            .expect("worker provider listing should succeed");
        assert!(
            providers
                .providers
                .iter()
                .any(|provider| provider.id == "whisper_cpp")
        );

        manager.shutdown().await;
    }
}
