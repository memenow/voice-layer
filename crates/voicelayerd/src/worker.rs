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
        // `segment_probe` runs silero-vad on a short (1-2 s) probe clip,
        // and `stitch_wav_segments` is stdlib `wave` I/O over a handful
        // of short files — both are fast and share the probe budget with
        // health/list_providers rather than the inference budget.
        "health" | "list_providers" | "segment_probe" | "stitch_wav_segments" => {
            DEFAULT_PROBE_TIMEOUT_SECS
        }
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
    /// Optional per-instance timeout override that bypasses the
    /// env-driven default in `worker_call_timeout`. Only used by tests
    /// to drive the `TimedOut` error variant with a short budget —
    /// production constructions leave it `None` and the env-driven
    /// default applies. Using a per-instance field instead of mutating
    /// `VOICELAYER_WORKER_TIMEOUT_SECONDS` avoids the
    /// `clippy::await_holding_lock` problem that arises when an
    /// `ENV_LOCK` mutex is held across the subprocess `.await`.
    pub(crate) timeout_override: Option<Duration>,
}

impl WorkerCommand {
    pub fn discover(project_root: PathBuf) -> Self {
        let uv_python = project_root.join(".venv").join("bin").join("python");
        if uv_python.is_file() {
            return Self {
                executable: uv_python.display().to_string(),
                args: vec!["-m".to_owned(), WORKER_MODULE.to_owned()],
                project_root,
                timeout_override: None,
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
            timeout_override: None,
        }
    }

    /// Override the per-call timeout for this `WorkerCommand`. Used by
    /// tests; production code never sets this.
    #[cfg(test)]
    pub(crate) fn with_timeout_override(mut self, timeout: Duration) -> Self {
        self.timeout_override = Some(timeout);
        self
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

    /// Run a silero-vad pass on a short WAV file to classify it as speech
    /// or silence. Intended for the VAD-gated segmentation orchestrator;
    /// uses the probe timeout budget because each call is a fast in-proc
    /// ONNX inference over a 1-2 s clip.
    pub async fn segment_probe(
        &self,
        request: &voicelayer_core::SegmentProbeRequest,
    ) -> Result<voicelayer_core::SegmentProbeResult, WorkerCallError> {
        self.call("segment_probe", Some(request)).await
    }

    /// Concatenate N probe WAV files into a single output WAV. Shares the
    /// probe timeout — stdlib `wave` I/O over a few short clips is fast
    /// and does not warrant the inference budget.
    pub async fn stitch_wav_segments(
        &self,
        request: &voicelayer_core::StitchWavSegmentsRequest,
    ) -> Result<voicelayer_core::StitchWavSegmentsResult, WorkerCallError> {
        self.call("stitch_wav_segments", Some(request)).await
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
        let timeout_duration = self
            .timeout_override
            .unwrap_or_else(|| worker_call_timeout(method));
        let line = timeout(timeout_duration, reader.next_line())
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
    #[serde(default)]
    pub whisper_mode: Option<String>,
    #[serde(default)]
    pub whisper_server_url: Option<String>,
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
    use std::{sync::Mutex, time::Duration};

    use super::{
        DEFAULT_INFERENCE_TIMEOUT_SECS, DEFAULT_PROBE_TIMEOUT_SECS, WORKER_MODULE, WorkerCallError,
        WorkerCommand, worker_call_timeout,
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
        assert_eq!(
            worker_call_timeout("segment_probe").as_secs(),
            DEFAULT_PROBE_TIMEOUT_SECS,
        );
        assert_eq!(
            worker_call_timeout("stitch_wav_segments").as_secs(),
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

    /// Closes the third case in `worker_call_timeout`'s env branch:
    /// the variable is **set** but not parseable as `u64`. The first
    /// `.ok()` keeps the `Some("...")`, `.parse::<u64>().ok()` returns
    /// `None`, and `unwrap_or(DEFAULT_INFERENCE_TIMEOUT_SECS)` wins —
    /// the same default the unset case uses. The existing test only
    /// exercises "unset" and "set to a number", so a regression that
    /// swapped the parse for `expect` or an explicit `panic!` would
    /// slip through.
    #[test]
    fn inference_timeout_falls_back_to_default_when_env_value_is_unparsable() {
        let _guard = ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var("VOICELAYER_WORKER_TIMEOUT_SECONDS").ok();
        // SAFETY: ENV_LOCK serializes every mutation of this variable
        // with every other env-touching test in this module.
        unsafe {
            std::env::set_var("VOICELAYER_WORKER_TIMEOUT_SECONDS", "not-a-number");
        }
        assert_eq!(
            worker_call_timeout("transcribe").as_secs(),
            DEFAULT_INFERENCE_TIMEOUT_SECS,
        );
        match previous {
            Some(value) => unsafe {
                std::env::set_var("VOICELAYER_WORKER_TIMEOUT_SECONDS", value);
            },
            None => unsafe {
                std::env::remove_var("VOICELAYER_WORKER_TIMEOUT_SECONDS");
            },
        }
    }

    /// Pins the `.venv/bin/python` branch of `WorkerCommand::discover`.
    /// Live workspace runs already hit this path through the checked-in
    /// venv, but the field assertions catch a silent regression where
    /// a future refactor flips to `uv run` even when a usable interpreter
    /// is present — that would turn every JSON-RPC call into a `uv`
    /// process-resolve with measurable cold-start cost.
    #[test]
    fn discover_picks_venv_python_when_present() {
        let tempdir = tempfile::tempdir().expect("tempdir should be creatable");
        let bin_dir = tempdir.path().join(".venv").join("bin");
        std::fs::create_dir_all(&bin_dir).expect("create .venv/bin");
        let python_path = bin_dir.join("python");
        std::fs::write(&python_path, b"").expect("touch .venv/bin/python");

        let worker = WorkerCommand::discover(tempdir.path().to_path_buf());
        assert_eq!(worker.executable, python_path.display().to_string());
        assert_eq!(worker.args, vec!["-m".to_owned(), WORKER_MODULE.to_owned()]);
        assert_eq!(worker.project_root, tempdir.path().to_path_buf());
        assert!(worker.timeout_override.is_none());
    }

    /// Pins the fallback branch: a project root without a `.venv`
    /// resolves through `uv run --project <root> python -m <module>`.
    /// This is the clone-and-run path for contributors who haven't
    /// created a venv yet; a regression that dropped `--project` or
    /// reordered the args would break worker startup on fresh clones.
    #[test]
    fn discover_falls_back_to_uv_run_when_venv_absent() {
        let tempdir = tempfile::tempdir().expect("tempdir should be creatable");

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
        assert_eq!(worker.project_root, tempdir.path().to_path_buf());
        assert!(worker.timeout_override.is_none());
    }

    /// Cross-check the `WORKER_MODULE` constant against the Python
    /// source tree. The Rust daemon spawns the worker via
    /// `python -m voicelayer_orchestrator.worker`, so the constant
    /// must point at a real importable module *and* that module
    /// must run a JSON-RPC loop when invoked as `__main__`. A
    /// regression on either side — renaming the python file, or
    /// dropping the `if __name__ == "__main__":` guard — would
    /// silently break worker startup with no compile-time signal.
    /// Only `cargo test` reaches both sides.
    #[test]
    fn worker_module_constant_resolves_to_a_python_file_with_main_entrypoint() {
        let repo_root = format!("{}/../..", env!("CARGO_MANIFEST_DIR"));
        let module_relative = WORKER_MODULE.replace('.', "/");
        let module_path = format!("{repo_root}/python/{module_relative}.py");

        let contents = std::fs::read_to_string(&module_path).unwrap_or_else(|err| {
            panic!(
                "WORKER_MODULE = `{WORKER_MODULE}` does not resolve to a real \
                 .py file at `{module_path}`. Either rename the constant or \
                 restore the python module. Underlying error: {err}",
            );
        });

        assert!(
            contents.contains("if __name__ == \"__main__\":"),
            "python module at `{module_path}` lacks an `if __name__ == \"__main__\":` \
             entrypoint. The daemon spawns it via `python -m {WORKER_MODULE}`; \
             without the guard the module imports cleanly but never runs \
             `serve()`, so the JSON-RPC handshake hangs forever.",
        );
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

    #[tokio::test]
    async fn segment_probe_returns_provider_unavailable_when_vad_not_configured() {
        // End-to-end pin for the new JSON-RPC method: with no VAD env
        // configured, the Python worker's `load_whisper_vad_config`
        // returns None and the dispatch surfaces PROVIDER_UNAVAILABLE
        // (-32004).
        //
        // The test assumes the ambient environment does not set
        // `VOICELAYER_WHISPER_VAD_ENABLED` — if the developer exported
        // it for an integration run, skip the assertion rather than
        // fight it. We intentionally do not mutate env under a mutex
        // here: holding a sync mutex across the subprocess `.await`
        // trips clippy's `await_holding_lock`, and any release-then-
        // spawn pattern races with other env-touching tests anyway.
        if std::env::var("VOICELAYER_WHISPER_VAD_ENABLED").is_ok() {
            eprintln!(
                "skipping segment_probe unavailable-error pin because \
                 VOICELAYER_WHISPER_VAD_ENABLED is set in the ambient env",
            );
            return;
        }

        let project_root = std::env::current_dir().expect("current_dir should be available");
        let worker = WorkerCommand::discover(project_root);

        let request = voicelayer_core::SegmentProbeRequest {
            audio_file: "/tmp/voicelayer-probe-does-not-matter.wav".to_owned(),
        };
        let error = worker
            .segment_probe(&request)
            .await
            .expect_err("segment_probe without VAD config must error");

        match error {
            WorkerCallError::Rpc(rpc) => {
                assert_eq!(
                    rpc.code, -32004,
                    "unconfigured VAD must surface PROVIDER_UNAVAILABLE; got {rpc:?}",
                );
                assert!(
                    rpc.message.contains("VOICELAYER_WHISPER_VAD_ENABLED"),
                    "error must hint at the required env var; got {}",
                    rpc.message,
                );
            }
            other => panic!("expected Rpc error from unconfigured VAD; got {other:?}"),
        }
    }

    /// Build a `WorkerCommand` that runs an arbitrary shell script as
    /// the JSON-RPC worker. The helper prepends `cat > /dev/null;`
    /// so the subprocess always consumes the daemon's request from
    /// stdin before running the caller's response snippet — without
    /// that, `printf '...'; exit 0` style scripts close the stdin pipe
    /// before our `write_all` lands and the test fails with
    /// `BrokenPipe` instead of the variant we wanted to exercise. The
    /// script's stdout becomes the response line and its final exit
    /// status feeds the post-response classification (success /
    /// `ProcessExited`).
    fn shell_worker(response_script: &str) -> WorkerCommand {
        let combined = format!("cat > /dev/null; {response_script}");
        WorkerCommand {
            executable: "sh".to_owned(),
            args: vec!["-c".to_owned(), combined],
            project_root: std::env::current_dir().expect("current_dir should resolve"),
            timeout_override: None,
        }
    }

    #[tokio::test]
    async fn worker_call_returns_io_error_when_executable_does_not_exist() {
        let worker = WorkerCommand {
            executable: "/voicelayer-test/definitely-not-a-real-binary".to_owned(),
            args: vec![],
            project_root: std::env::current_dir().expect("current_dir should resolve"),
            timeout_override: None,
        };
        match worker.health().await {
            Err(WorkerCallError::Io(_)) => {}
            other => panic!("expected Io error from missing executable; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_empty_response_when_subprocess_writes_nothing() {
        // The shell exits cleanly without ever writing to stdout, so
        // `BufReader::next_line` resolves to `None` and the call
        // surface translates that to `EmptyResponse`.
        let worker = shell_worker("exit 0");
        match worker.health().await {
            Err(WorkerCallError::EmptyResponse) => {}
            other => panic!("expected EmptyResponse from silent subprocess; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_json_error_when_response_is_not_json() {
        // `printf` writes a single line that is not parseable as
        // JSON-RPC. `serde_json::from_str` fails and the helper
        // converts it to `WorkerCallError::Json` via `From`.
        let worker = shell_worker("printf '%s\\n' 'not-json-at-all'");
        match worker.health().await {
            Err(WorkerCallError::Json(_)) => {}
            other => panic!("expected Json error from garbage stdout; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_invalid_protocol_version_when_jsonrpc_field_disagrees() {
        // The response parses cleanly as JsonRpcResponse but the
        // `jsonrpc` field doesn't match the protocol constant — the
        // helper rejects it before reading the result.
        let worker =
            shell_worker(r#"printf '%s\n' '{"jsonrpc":"1.0","id":"x","result":{"providers":[]}}'"#);
        match worker.list_providers().await {
            Err(WorkerCallError::InvalidProtocolVersion(version)) => {
                assert_eq!(version, "1.0");
            }
            other => panic!("expected InvalidProtocolVersion; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_malformed_response_when_neither_result_nor_error_present() {
        // Valid JSON-RPC envelope but neither `result` nor `error` is
        // populated. The helper defends against this since servers
        // sometimes ship a half-formed response on internal failures.
        let worker = shell_worker(r#"printf '%s\n' '{"jsonrpc":"2.0","id":"x"}'"#);
        match worker.list_providers().await {
            Err(WorkerCallError::MalformedResponse) => {}
            other => panic!("expected MalformedResponse; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_process_exited_when_subprocess_returns_nonzero_after_response() {
        // Subprocess writes a valid response then exits with a
        // non-zero status. The helper surfaces the exit code so the
        // operator can tell the worker crashed mid-shutdown rather
        // than the response itself being broken.
        let worker = shell_worker(
            r#"printf '%s\n' '{"jsonrpc":"2.0","id":"x","result":{"providers":[]}}'; exit 7"#,
        );
        match worker.list_providers().await {
            Err(WorkerCallError::ProcessExited(code, _stderr)) => {
                assert_eq!(code, Some(7));
            }
            other => panic!("expected ProcessExited; got {other:?}"),
        }
    }

    #[tokio::test]
    async fn worker_call_returns_timed_out_when_subprocess_never_responds() {
        // Closes the gap left by #29: subprocess consumes stdin then
        // sleeps far longer than the override budget, so
        // `reader.next_line()` never resolves and `timeout()` produces
        // `WorkerCallError::TimedOut`. `kill_on_drop(true)` reaps the
        // shell when the helper future drops on the early return.
        //
        // Uses `with_timeout_override` instead of mutating
        // `VOICELAYER_WORKER_TIMEOUT_SECONDS` so no global mutex is
        // held across the await.
        let worker = shell_worker("sleep 30").with_timeout_override(Duration::from_millis(100));
        match worker.health().await {
            Err(WorkerCallError::TimedOut) => {}
            other => panic!("expected TimedOut from unresponsive subprocess; got {other:?}"),
        }
    }
}
