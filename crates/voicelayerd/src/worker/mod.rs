//! Persistent stdio JSON-RPC client for the Python inference worker.

mod process;
mod rpc;

use std::{
    collections::VecDeque,
    path::PathBuf,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use serde::{Serialize, de::DeserializeOwned};
use tokio::sync::Mutex as TokioMutex;
use uuid::Uuid;
use voicelayer_core::{TranscriptionResult, WorkerInitPayload};

use self::process::{STDERR_TAIL_MAX_LINES, WorkerProcess, spawn_process};
use self::rpc::{JSONRPC_VERSION, JsonRpcRequest, exchange_once};

pub use self::rpc::{
    JsonRpcError, WorkerCallError, WorkerHealthResult, WorkerPreviewPayload, WorkerProviderList,
};

pub const WORKER_MODULE: &str = "voicelayer_worker.worker";

const DEFAULT_PROBE_TIMEOUT_SECS: u64 = 15;
const DEFAULT_INFERENCE_TIMEOUT_SECS: u64 = 600;

fn is_probe_method(method: &str) -> bool {
    matches!(
        method,
        "health" | "list_providers" | "segment_probe" | "stitch_wav_segments"
    )
}

fn worker_call_timeout(method: &str, configured_inference: Option<Duration>) -> Duration {
    if is_probe_method(method) {
        return Duration::from_secs(DEFAULT_PROBE_TIMEOUT_SECS);
    }
    if let Some(duration) = configured_inference {
        return duration;
    }
    let seconds = std::env::var("VOICELAYER_WORKER_TIMEOUT_SECONDS")
        .ok()
        .and_then(|value| value.parse::<u64>().ok())
        .unwrap_or(DEFAULT_INFERENCE_TIMEOUT_SECS);
    Duration::from_secs(seconds)
}

/// Handle supervising a single long-lived Python worker child process.
///
/// Clones share the same underlying process and serialize JSON-RPC frames over
/// stdin/stdout via an async mutex. Crashes and protocol errors clear the child
/// handle so the next call respawns lazily.
#[derive(Debug, Clone)]
pub struct WorkerCommand {
    pub executable: String,
    pub args: Vec<String>,
    pub project_root: PathBuf,
    inference_timeout: Option<Duration>,
    pub(crate) timeout_override: Option<Duration>,
    process: Arc<TokioMutex<Option<WorkerProcess>>>,
    stderr_tail: Arc<StdMutex<VecDeque<String>>>,
    init_payload: Option<Arc<WorkerInitPayload>>,
}

impl WorkerCommand {
    pub fn new(executable: String, args: Vec<String>, project_root: PathBuf) -> Self {
        Self {
            executable,
            args,
            project_root,
            inference_timeout: None,
            timeout_override: None,
            process: Arc::new(TokioMutex::new(None)),
            stderr_tail: Arc::new(StdMutex::new(VecDeque::with_capacity(
                STDERR_TAIL_MAX_LINES,
            ))),
            init_payload: None,
        }
    }

    /// Attach provider settings sent during the worker `initialize` handshake.
    pub fn with_init_payload(mut self, payload: WorkerInitPayload) -> Self {
        self.init_payload = Some(Arc::new(payload));
        self
    }

    /// Set the inference RPC timeout budget (applied to non-probe methods).
    pub fn with_inference_timeout(mut self, timeout: Duration) -> Self {
        self.inference_timeout = Some(timeout);
        self
    }

    /// Resolve `.venv/bin/python` under `project_root` when present, falling back
    /// to `uv run --project <root> python -m voicelayer_worker.worker`.
    pub fn discover(project_root: PathBuf) -> Self {
        let uv_python = project_root.join(".venv").join("bin").join("python");
        if uv_python.is_file() {
            return Self::new(
                uv_python.display().to_string(),
                vec!["-m".to_owned(), WORKER_MODULE.to_owned()],
                project_root,
            );
        }

        let project_display = project_root.display().to_string();
        Self::new(
            "uv".to_owned(),
            vec![
                "run".to_owned(),
                "--project".to_owned(),
                project_display,
                "python".to_owned(),
                "-m".to_owned(),
                WORKER_MODULE.to_owned(),
            ],
            project_root,
        )
    }

    #[cfg(test)]
    pub(crate) fn with_timeout_override(mut self, timeout: Duration) -> Self {
        self.timeout_override = Some(timeout);
        self
    }

    /// Format the launch command for logs and health responses.
    pub fn display(&self) -> String {
        std::iter::once(self.executable.as_str())
            .chain(self.args.iter().map(String::as_str))
            .collect::<Vec<_>>()
            .join(" ")
    }

    /// Returns `true` when a worker subprocess is currently alive.
    pub async fn is_running(&self) -> bool {
        self.process.lock().await.is_some()
    }

    /// Terminate the active worker subprocess, if running.
    pub async fn shutdown(&self) {
        let mut guard = self.process.lock().await;
        if let Some(mut process) = guard.take() {
            let _ = process.child.kill().await;
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

    pub async fn segment_probe(
        &self,
        request: &voicelayer_core::SegmentProbeRequest,
    ) -> Result<voicelayer_core::SegmentProbeResult, WorkerCallError> {
        self.call("segment_probe", Some(request)).await
    }

    pub async fn stitch_wav_segments(
        &self,
        request: &voicelayer_core::StitchWavSegmentsRequest,
    ) -> Result<voicelayer_core::StitchWavSegmentsResult, WorkerCallError> {
        self.call("stitch_wav_segments", Some(request)).await
    }

    fn timeout_for(&self, method: &str) -> Duration {
        self.timeout_override
            .unwrap_or_else(|| worker_call_timeout(method, self.inference_timeout))
    }

    fn snapshot_stderr_tail(&self) -> String {
        match self.stderr_tail.lock() {
            Ok(buffer) => buffer.iter().cloned().collect::<Vec<_>>().join("\n"),
            Err(_) => String::new(),
        }
    }

    async fn initialize(&self, process: &mut WorkerProcess) -> Result<(), WorkerCallError> {
        let Some(payload) = &self.init_payload else {
            return Ok(());
        };
        let id = Uuid::new_v4().to_string();
        let request = JsonRpcRequest::initialize(id.clone(), payload.as_ref());
        let mut line = serde_json::to_string(&request)?;
        line.push('\n');

        let budget = self.timeout_for("health");
        let _: serde_json::Value = rpc::write_and_read(process, &line, &id, budget).await?;
        Ok(())
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
        let mut payload = serde_json::to_string(&request)?;
        payload.push('\n');
        let timeout_duration = self.timeout_for(method);

        let mut guard = self.process.lock().await;
        // Retry once if the persistent child died between calls.
        for attempt in 0..2 {
            if guard.is_none() {
                let spawned = spawn_process(
                    &self.executable,
                    &self.args,
                    &self.project_root,
                    Arc::clone(&self.stderr_tail),
                )?;
                *guard = Some(spawned);
                if let Err(error) = self.initialize(guard.as_mut().expect("just spawned")).await {
                    *guard = None;
                    if attempt == 0 {
                        continue;
                    }
                    return Err(error);
                }
            }

            let process = guard.as_mut().expect("spawned or existing");
            match exchange_once::<R>(process, &payload, &request.id, timeout_duration, || {
                self.snapshot_stderr_tail()
            })
            .await
            {
                Ok((result, keep_alive)) => {
                    if !keep_alive {
                        *guard = None;
                    }
                    return Ok(result);
                }
                Err(rpc::ExchangeFailure::Retryable(error)) => {
                    *guard = None;
                    if attempt == 0 {
                        continue;
                    }
                    return Err(error);
                }
                Err(rpc::ExchangeFailure::Fatal { error, drop_child }) => {
                    if drop_child {
                        *guard = None;
                    }
                    return Err(error);
                }
            }
        }

        unreachable!("call retry loop exits within two attempts")
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use voicelayer_testkit::EnvGuard;

    use super::{
        DEFAULT_INFERENCE_TIMEOUT_SECS, DEFAULT_PROBE_TIMEOUT_SECS, WORKER_MODULE, WorkerCallError,
        WorkerCommand, worker_call_timeout,
    };

    #[test]
    fn probe_methods_use_short_timeout() {
        for method in [
            "health",
            "list_providers",
            "segment_probe",
            "stitch_wav_segments",
        ] {
            assert_eq!(
                worker_call_timeout(method, None).as_secs(),
                DEFAULT_PROBE_TIMEOUT_SECS,
            );
            assert_eq!(
                worker_call_timeout(method, Some(Duration::from_secs(900))).as_secs(),
                DEFAULT_PROBE_TIMEOUT_SECS,
            );
        }
    }

    #[test]
    fn inference_methods_honor_configured_and_env_timeouts() {
        let mut env = EnvGuard::lock();
        env.remove("VOICELAYER_WORKER_TIMEOUT_SECONDS");
        assert_eq!(
            worker_call_timeout("transcribe", None).as_secs(),
            DEFAULT_INFERENCE_TIMEOUT_SECS,
        );
        assert_eq!(
            worker_call_timeout("transcribe", Some(Duration::from_secs(120))).as_secs(),
            120,
        );
        env.set("VOICELAYER_WORKER_TIMEOUT_SECONDS", "42");
        assert_eq!(worker_call_timeout("transcribe", None).as_secs(), 42);
        assert_eq!(worker_call_timeout("compose", None).as_secs(), 42);
    }

    #[test]
    fn inference_timeout_falls_back_to_default_when_env_is_unparsable() {
        let mut env = EnvGuard::lock();
        env.set("VOICELAYER_WORKER_TIMEOUT_SECONDS", "not-a-number");
        assert_eq!(
            worker_call_timeout("transcribe", None).as_secs(),
            DEFAULT_INFERENCE_TIMEOUT_SECS,
        );
    }

    #[test]
    fn discover_picks_venv_python_when_present() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let bin_dir = tempdir.path().join(".venv").join("bin");
        std::fs::create_dir_all(&bin_dir).expect("create .venv/bin");
        let python_path = bin_dir.join("python");
        std::fs::write(&python_path, b"").expect("touch python");

        let worker = WorkerCommand::discover(tempdir.path().to_path_buf());
        assert_eq!(worker.executable, python_path.display().to_string());
        assert_eq!(worker.args, vec!["-m".to_owned(), WORKER_MODULE.to_owned()]);
        assert_eq!(worker.project_root, tempdir.path().to_path_buf());
        assert!(worker.timeout_override.is_none());
    }

    #[test]
    fn discover_falls_back_to_uv_run_when_venv_absent() {
        let tempdir = tempfile::tempdir().expect("tempdir");

        let worker = WorkerCommand::discover(tempdir.path().to_path_buf());
        assert_eq!(worker.executable, "uv");
        assert_eq!(
            worker.args,
            vec![
                "run".to_owned(),
                "--project".to_owned(),
                tempdir.path().display().to_string(),
                "python".to_owned(),
                "-m".to_owned(),
                WORKER_MODULE.to_owned(),
            ],
        );
    }

    fn shell_worker(response_script: &str) -> WorkerCommand {
        let combined = format!("head -n 1 > /dev/null; {response_script}");
        WorkerCommand::new(
            "sh".to_owned(),
            vec!["-c".to_owned(), combined],
            std::env::current_dir().expect("current_dir"),
        )
    }

    #[tokio::test]
    async fn worker_call_returns_timed_out_when_subprocess_exceeds_budget() {
        let worker = shell_worker("sleep 5").with_timeout_override(Duration::from_millis(50));
        assert!(matches!(
            worker.health().await,
            Err(WorkerCallError::TimedOut)
        ));
    }
}
