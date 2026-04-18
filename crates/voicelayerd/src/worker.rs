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
        let line = timeout(Duration::from_secs(10), reader.next_line())
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
    use super::WorkerCommand;

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
