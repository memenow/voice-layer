use std::{path::PathBuf, process::Stdio, time::Duration};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncReadExt, AsyncWriteExt, BufReader},
    process::Command,
    time::timeout,
};
use uuid::Uuid;
use voicelayer_core::{ProviderDescriptor, TranscriptionResult};

const JSONRPC_VERSION: &str = "2.0";
const WORKER_MODULE: &str = "voicelayer_orchestrator.worker";

/// Upper bound on how long the daemon waits for a worker response.
///
/// `health` and `list_providers` are cheap probes and stay short so CLI
/// commands that depend on them (e.g., `vl doctor`, `vl providers`) don't
/// hang on a misconfigured worker. Every other method invokes real
/// inference (whisper-cli / whisper-server for transcribe, llama-server
/// for compose/rewrite/translate) and must outlast the provider's own
/// timeout. The inference budget is overridable through the
/// `VOICELAYER_WORKER_TIMEOUT_SECONDS` environment variable.
const DEFAULT_PROBE_TIMEOUT_SECS: u64 = 15;
const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 600;

fn worker_call_timeout(method: &str) -> Duration {
    let seconds = match method {
        "health" | "list_providers" => DEFAULT_PROBE_TIMEOUT_SECS,
        _ => std::env::var("VOICELAYER_WORKER_TIMEOUT_SECONDS")
            .ok()
            .and_then(|value| value.parse::<u64>().ok())
            .unwrap_or(DEFAULT_INFERENCE_TIMEOUT_SECS),
    };
    Duration::from_secs(seconds)
}

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
        let request = JsonRpcRequest {
            jsonrpc: JSONRPC_VERSION,
            id: Uuid::new_v4().to_string(),
            method: method.to_owned(),
            params,
        };
        let payload = serde_json::to_string(&request)?;

        let mut command = Command::new(&self.executable);
        command
            .args(&self.args)
            .current_dir(&self.project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn()?;
        let mut stdin = child
            .stdin
            .take()
            .ok_or(WorkerCallError::MissingPipe("stdin"))?;
        let stdout = child
            .stdout
            .take()
            .ok_or(WorkerCallError::MissingPipe("stdout"))?;
        let stderr = child
            .stderr
            .take()
            .ok_or(WorkerCallError::MissingPipe("stderr"))?;

        let stderr_task = tokio::spawn(async move {
            let mut buffer = String::new();
            let mut reader = BufReader::new(stderr);
            let _ = reader.read_to_string(&mut buffer).await;
            buffer
        });

        stdin.write_all(payload.as_bytes()).await?;
        stdin.write_all(b"\n").await?;
        stdin.shutdown().await?;
        drop(stdin);

        let mut reader = BufReader::new(stdout).lines();
        let line = timeout(worker_call_timeout(method), reader.next_line())
            .await
            .map_err(|_| WorkerCallError::TimedOut)??
            .ok_or(WorkerCallError::EmptyResponse)?;

        let response: JsonRpcResponse<R> = serde_json::from_str(&line)?;
        if response.jsonrpc != JSONRPC_VERSION {
            return Err(WorkerCallError::InvalidProtocolVersion(response.jsonrpc));
        }

        let status = child.wait().await?;
        let stderr_output = stderr_task.await.unwrap_or_default();
        if !status.success() {
            return Err(WorkerCallError::ProcessExited(status.code(), stderr_output));
        }

        match (response.result, response.error) {
            (Some(result), None) => Ok(result),
            (None, Some(error)) => Err(WorkerCallError::Rpc(error)),
            _ => Err(WorkerCallError::MalformedResponse),
        }
    }
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
    result: Option<R>,
    error: Option<JsonRpcError>,
}

#[derive(Debug, Clone, Deserialize, Error)]
#[error("{message}")]
pub struct JsonRpcError {
    pub code: i64,
    pub message: String,
}

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
    #[error("worker returned a malformed JSON-RPC response")]
    MalformedResponse,
    #[error("worker RPC error {0}")]
    Rpc(JsonRpcError),
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::{
        DEFAULT_INFERENCE_TIMEOUT_SECS, DEFAULT_PROBE_TIMEOUT_SECS, WorkerCommand,
        worker_call_timeout,
    };

    /// Serializes any test that mutates process-wide env vars. Cargo runs
    /// unit tests concurrently by default, and Rust 2024 flagged
    /// `std::env::set_var` as `unsafe` precisely because a concurrent
    /// reader in another thread is UB. Any future test that touches
    /// `VOICELAYER_WORKER_TIMEOUT_SECONDS` (or any other process env)
    /// must take this lock for the duration of the mutation.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    #[test]
    fn probe_methods_use_short_timeout() {
        assert_eq!(
            worker_call_timeout("health").as_secs(),
            DEFAULT_PROBE_TIMEOUT_SECS,
        );
        assert_eq!(
            worker_call_timeout("list_providers").as_secs(),
            DEFAULT_PROBE_TIMEOUT_SECS,
        );
    }

    #[test]
    fn inference_methods_use_env_overridden_budget() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("VOICELAYER_WORKER_TIMEOUT_SECONDS").ok();
        // SAFETY: ENV_LOCK serializes every mutation of this variable with
        // every other env-touching test in this module, so no other thread
        // can observe the mutation while we hold the lock.
        unsafe {
            std::env::remove_var("VOICELAYER_WORKER_TIMEOUT_SECONDS");
        }
        assert_eq!(
            worker_call_timeout("transcribe").as_secs(),
            DEFAULT_INFERENCE_TIMEOUT_SECS,
        );
        unsafe {
            std::env::set_var("VOICELAYER_WORKER_TIMEOUT_SECONDS", "42");
        }
        assert_eq!(worker_call_timeout("transcribe").as_secs(), 42);
        assert_eq!(worker_call_timeout("compose").as_secs(), 42);
        match previous {
            Some(value) => unsafe {
                std::env::set_var("VOICELAYER_WORKER_TIMEOUT_SECONDS", value);
            },
            None => unsafe {
                std::env::remove_var("VOICELAYER_WORKER_TIMEOUT_SECONDS");
            },
        }
    }

    #[tokio::test]
    async fn worker_command_can_call_health() {
        let project_root = std::env::current_dir().expect("current_dir should be available");
        let worker = WorkerCommand::discover(project_root);

        let health = worker.health().await.expect("worker health should succeed");
        assert_eq!(health.status, "ok");
        assert_eq!(health.protocol, "2.0");
    }

    #[tokio::test]
    async fn worker_command_can_list_providers() {
        let project_root = std::env::current_dir().expect("current_dir should be available");
        let worker = WorkerCommand::discover(project_root);

        let providers = worker
            .list_providers()
            .await
            .expect("worker provider listing should succeed");
        assert!(
            providers
                .providers
                .iter()
                .any(|provider| provider.id == "whisper_cpp")
        );
    }
}
