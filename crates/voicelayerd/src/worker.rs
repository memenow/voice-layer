use std::{
    collections::VecDeque,
    path::PathBuf,
    process::Stdio,
    sync::{Arc, Mutex as StdMutex},
    time::Duration,
};

use serde::{Deserialize, Serialize, de::DeserializeOwned};
use thiserror::Error;
use tokio::{
    io::{AsyncBufReadExt, AsyncWriteExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    sync::Mutex as TokioMutex,
    task::JoinHandle,
    time::timeout,
};
use uuid::Uuid;
use voicelayer_core::{ProviderDescriptor, TranscriptionResult};

const JSONRPC_VERSION: &str = "2.0";
const WORKER_MODULE: &str = "voicelayer_orchestrator.worker";

/// How many recent stderr lines to keep around for inclusion in error
/// reports. Most Python tracebacks fit comfortably; long log streams
/// rotate the oldest entries out.
const STDERR_TAIL_MAX_LINES: usize = 64;

/// Wall-clock grace before the post-response `try_wait` poll. The
/// daemon's pipe reactor can wake the parent before the kernel
/// finishes the child's `exit` transition, so a respond-then-exit
/// fixture occasionally still shows as alive on the first
/// `try_wait` even though it has already entered zombie state.
/// 1 ms is enough margin to make the detection deterministic across
/// every system the verification chain has been observed on, and
/// far below the 25 ms grace the previous busy-loop implementation
/// charged on every successful call (the production path used to
/// pay the full window because the loop did not break early on a
/// living child).
const POST_RESPONSE_EXIT_GRACE: Duration = Duration::from_millis(1);

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

/// Long-lived stdio JSON-RPC connection to a Python worker subprocess.
///
/// The daemon previously spawned a fresh `python -m
/// voicelayer_orchestrator.worker` per request and shut its stdin
/// after one line, which made every transcribe call pay the model
/// cold load — multi-GB GPU providers like MiMo-V2.5-ASR and
/// Qwen3-ASR-1.7B were re-loading on every dictation segment because
/// the in-process model cache was effectively dead under that
/// architecture.
///
/// The current implementation keeps a single child alive across
/// requests and serializes JSON-RPC line exchanges through a tokio
/// mutex; clones of the same `WorkerCommand` share the underlying
/// process via `Arc`. Crashes, EOFs, and protocol failures all drop
/// the child so the next call lazily respawns and the operator never
/// sees a stuck handle. Stderr is drained into a ring buffer
/// (`STDERR_TAIL_MAX_LINES` lines) so error reports include the
/// Python-side traceback even though stderr is no longer read to EOF
/// per call.
#[derive(Debug)]
struct WorkerProcess {
    child: Child,
    stdin: ChildStdin,
    stdout: Lines<BufReader<ChildStdout>>,
    stderr_drainer: JoinHandle<()>,
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        // Cancel the stderr drainer; the underlying child has
        // `kill_on_drop(true)` so the kernel reaps it as soon as the
        // `Child` itself drops. Aborting the drainer prevents the
        // task from outliving its pipe and warning about a closed
        // reader.
        self.stderr_drainer.abort();
    }
}

#[derive(Debug, Clone)]
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
    /// Lazy-spawned long-lived child process shared by every clone of
    /// this `WorkerCommand`. The tokio mutex serializes JSON-RPC line
    /// exchanges; clones share the same `Arc` so all request handlers
    /// cooperate on one worker process. Setting the inner Option to
    /// `None` (after a crash, EOF, or protocol failure) tells the
    /// next call to lazily respawn before retrying.
    process: Arc<TokioMutex<Option<WorkerProcess>>>,
    /// Last `STDERR_TAIL_MAX_LINES` lines of stderr, populated by the
    /// drainer task that owns the child's stderr pipe. Snapshotted on
    /// `ProcessExited` errors so the operator sees the Python-side
    /// traceback rather than an opaque EOF. Outlives any single
    /// `WorkerProcess`: when a respawn happens, the buffer accumulates
    /// across the boundary so a stable read of the most recent failure
    /// is always available.
    stderr_tail: Arc<StdMutex<VecDeque<String>>>,
}

impl WorkerCommand {
    /// Build a `WorkerCommand` from the spec fields. The lazy process
    /// handle and stderr tail buffer are initialised empty; the first
    /// `call` spawns the worker and seeds them.
    pub fn new(executable: String, args: Vec<String>, project_root: PathBuf) -> Self {
        Self {
            executable,
            args,
            project_root,
            timeout_override: None,
            process: Arc::new(TokioMutex::new(None)),
            stderr_tail: Arc::new(StdMutex::new(VecDeque::with_capacity(
                STDERR_TAIL_MAX_LINES,
            ))),
        }
    }

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

    /// Spawn a fresh worker subprocess and wire its pipes into a
    /// `WorkerProcess`. The stderr drainer task owns the stderr pipe
    /// and pushes lines into the shared `stderr_tail` ring buffer so
    /// every error reported by `call` carries the most recent
    /// Python-side log lines without blocking the JSON-RPC dance on a
    /// `read_to_string` to EOF (which would never return for a
    /// healthy long-lived worker).
    fn spawn_process(&self) -> Result<WorkerProcess, WorkerCallError> {
        let mut command = Command::new(&self.executable);
        command
            .args(&self.args)
            .current_dir(&self.project_root)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .kill_on_drop(true);

        let mut child = command.spawn()?;
        let stdin = child
            .stdin
            .take()
            .ok_or(WorkerCallError::MissingPipe("stdin"))?;
        let stdout = BufReader::new(
            child
                .stdout
                .take()
                .ok_or(WorkerCallError::MissingPipe("stdout"))?,
        )
        .lines();
        let stderr = child
            .stderr
            .take()
            .ok_or(WorkerCallError::MissingPipe("stderr"))?;

        let tail = Arc::clone(&self.stderr_tail);
        let stderr_drainer = tokio::spawn(async move {
            let mut lines = BufReader::new(stderr).lines();
            while let Ok(Some(line)) = lines.next_line().await {
                if let Ok(mut buffer) = tail.lock() {
                    if buffer.len() >= STDERR_TAIL_MAX_LINES {
                        buffer.pop_front();
                    }
                    buffer.push_back(line);
                }
            }
        });

        Ok(WorkerProcess {
            child,
            stdin,
            stdout,
            stderr_drainer,
        })
    }

    /// Snapshot the current contents of the stderr ring buffer as a
    /// single newline-joined `String`, suitable for inclusion in
    /// `WorkerCallError::ProcessExited`. Poisoned mutex degrades to
    /// the empty string rather than propagating panics out of the
    /// daemon's request path.
    fn snapshot_stderr_tail(&self) -> String {
        match self.stderr_tail.lock() {
            Ok(buffer) => buffer.iter().cloned().collect::<Vec<_>>().join("\n"),
            Err(_) => String::new(),
        }
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
        let timeout_duration = self
            .timeout_override
            .unwrap_or_else(|| worker_call_timeout(method));

        let mut guard = self.process.lock().await;

        // Two attempts at most: if the child died between calls, the
        // first write/read fails fast and we re-spawn before retrying.
        // After the second failure we surface the error so the operator
        // sees the underlying problem (typically a Python import failure
        // or a model load crash, captured in the stderr tail).
        for attempt in 0..2 {
            if guard.is_none() {
                match self.spawn_process() {
                    Ok(process) => *guard = Some(process),
                    Err(err) => return Err(err),
                }
            }

            let process = guard.as_mut().expect("just spawned or pre-existing");

            // Write the JSON-RPC line. `BrokenPipe` and similar I/O
            // errors mean the child died between calls; drop the handle
            // and retry once with a fresh spawn.
            let write_result = async {
                process.stdin.write_all(payload.as_bytes()).await?;
                process.stdin.flush().await?;
                Ok::<(), std::io::Error>(())
            }
            .await;
            if write_result.is_err() {
                *guard = None;
                if attempt == 0 {
                    continue;
                }
                let stderr_tail = self.snapshot_stderr_tail();
                return Err(WorkerCallError::ProcessExited(None, stderr_tail));
            }

            // Read one response line. `Ok(Ok(None))` is EOF (child
            // exited without writing); `Ok(Err(_))` is an I/O error
            // mid-stream; `Err(_)` is the timeout.
            let line_result = timeout(timeout_duration, process.stdout.next_line()).await;
            let line = match line_result {
                Ok(Ok(Some(line))) => line,
                Ok(Ok(None)) => {
                    *guard = None;
                    if attempt == 0 {
                        continue;
                    }
                    return Err(WorkerCallError::EmptyResponse);
                }
                Ok(Err(io_err)) => {
                    *guard = None;
                    return Err(WorkerCallError::Io(io_err));
                }
                Err(_) => {
                    // Timeout: we have no way to recover an in-flight
                    // worker call cleanly, so kill the child and let
                    // the next call respawn.
                    *guard = None;
                    return Err(WorkerCallError::TimedOut);
                }
            };

            let response: JsonRpcResponse<R> = serde_json::from_str(&line)?;
            if response.jsonrpc != JSONRPC_VERSION {
                return Err(WorkerCallError::InvalidProtocolVersion(response.jsonrpc));
            }

            // Detect the "respond-then-exit" pattern with a single
            // non-blocking poll after a 1 ms grace. Production workers
            // stay alive forever, so the cost is bounded at
            // `POST_RESPONSE_EXIT_GRACE` per call (down from the 25 ms
            // the previous busy-loop charged); test fixtures and rare
            // crash-after-response paths get the margin they need for
            // the kernel to complete the child's `exit` transition
            // before `try_wait` is asked to observe it. Without the
            // grace the tokio pipe reactor can race ahead of the
            // child's exit syscall and `try_wait` flakily returns
            // `Ok(None)` for a child that is, microseconds later,
            // observably zombie.
            tokio::time::sleep(POST_RESPONSE_EXIT_GRACE).await;
            match process.child.try_wait() {
                Ok(Some(status)) if !status.success() => {
                    *guard = None;
                    let stderr_tail = self.snapshot_stderr_tail();
                    return Err(WorkerCallError::ProcessExited(status.code(), stderr_tail));
                }
                Ok(Some(_)) => {
                    // Clean exit after a clean response; mark for
                    // respawn on the next call but pass the response
                    // through.
                    *guard = None;
                }
                Ok(None) => {
                    // Still alive — production path. Leave the worker
                    // alive for the next call.
                }
                Err(_) => {
                    // try_wait surfaced an I/O error; assume the child
                    // is gone and respawn next call. The current
                    // response is still valid.
                    *guard = None;
                }
            }

            return match (response.result, response.error) {
                (Some(result), None) => Ok(result),
                (None, Some(error)) => Err(WorkerCallError::Rpc(error)),
                _ => Err(WorkerCallError::MalformedResponse),
            };
        }

        // The retry loop always returns inside the body; reaching
        // here would mean a logic bug.
        unreachable!("call retry loop must return after at most two attempts")
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
    /// Whether MiMo-V2.5-ASR is configured and validated. Defaults to
    /// `false` so worker payloads predating the optional MiMo provider
    /// continue to deserialize.
    #[serde(default)]
    pub mimo_configured: bool,
    #[serde(default)]
    pub mimo_model_path: Option<String>,
    #[serde(default)]
    pub mimo_error: Option<String>,
    /// Whether Qwen3-ASR-1.7B is configured and validated. Defaults to
    /// `false` so worker payloads predating the optional Qwen3 provider
    /// continue to deserialize.
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

    /// Walk a Rust source string and collect every method name passed
    /// as the first quoted string after `.call(...)` or
    /// `.call::<...>(...)`. Stops at the first column-zero
    /// `#[cfg(test)]` line — the marker for a top-level test
    /// module — so synthetic fixtures inside `worker::tests` do not
    /// pollute the production set. Indented `#[cfg(test)]`
    /// attributes that gate inline test-only items (e.g.
    /// `with_timeout_override` on `WorkerCommand`) are *not* test
    /// fixtures; they sit alongside production code and must be
    /// included in the scan. The trailing-lowercase filter rejects
    /// mixed-case or symbol-bearing tokens so a quoted error
    /// message would not slip through.
    fn collect_rust_call_methods(source: &str) -> std::collections::BTreeSet<String> {
        let mut methods = std::collections::BTreeSet::new();
        for line in source.lines() {
            if line == "#[cfg(test)]" {
                break;
            }
            let Some(call_pos) = line.find(".call") else {
                continue;
            };
            let after_call = &line[call_pos + ".call".len()..];
            let Some(first_quote) = after_call.find('"') else {
                continue;
            };
            let inside = &after_call[first_quote + 1..];
            let Some(close) = inside.find('"') else {
                continue;
            };
            let method = &inside[..close];
            if !method.is_empty() && method.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                methods.insert(method.to_owned());
            }
        }
        methods
    }

    /// Walk `docs/architecture/python-worker-protocol.html` and pull
    /// the JSON-RPC method names listed under the Required Methods
    /// heading. Each entry is a code literal whose value is lowercase
    /// letters and underscores only. The section terminates at the
    /// next `<h2>` so prose mentions of method names later in the doc
    /// (e.g. under Current Behavior) are not re-captured.
    ///
    /// Markdown input is still accepted for the unit fixture so the
    /// section-boundary behavior stays easy to read.
    fn extract_protocol_doc_method_names(contents: &str) -> std::collections::BTreeSet<String> {
        let mut methods = std::collections::BTreeSet::new();

        if let Some(start) = contents.find("<h2 id=\"required-methods\"") {
            let after_heading = &contents[start..];
            let section_start = after_heading
                .find("</h2>")
                .map_or(0, |idx| idx + "</h2>".len());
            let section = &after_heading[section_start..];
            let section = section
                .find("<h2")
                .map_or(section, |next_heading| &section[..next_heading]);
            for literal in voicelayer_doc_test_utils::extract_doc_code_literals(section) {
                if !literal.is_empty()
                    && literal.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                {
                    methods.insert(literal);
                }
            }
            return methods;
        }

        let mut in_section = false;
        for line in contents.lines() {
            let trimmed = line.trim();
            if trimmed.starts_with("## ") {
                in_section = trimmed == "## Required Methods";
                continue;
            }
            if !in_section {
                continue;
            }
            let Some(rest) = trimmed.strip_prefix("- `") else {
                continue;
            };
            let Some(end) = rest.find('`') else {
                continue;
            };
            let name = &rest[..end];
            if !name.is_empty() && name.chars().all(|c| c.is_ascii_lowercase() || c == '_') {
                methods.insert(name.to_owned());
            }
        }
        methods
    }

    /// Walk a Python source string and collect every method name the
    /// dispatch tree compares against. Two patterns are recognised:
    /// `method == "X"` (comparison form, used at six call sites in
    /// the current worker) and `method in {"X", "Y", ...}` (set
    /// form, used once for the compose/rewrite/translate group).
    /// The trailing-lowercase filter rejects accidental matches.
    fn collect_python_dispatched_methods(source: &str) -> std::collections::BTreeSet<String> {
        let mut methods = std::collections::BTreeSet::new();
        for line in source.lines() {
            if let Some(idx) = line.find("method == \"") {
                let after = &line[idx + "method == \"".len()..];
                if let Some(close) = after.find('"') {
                    let token = &after[..close];
                    if !token.is_empty()
                        && token.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                    {
                        methods.insert(token.to_owned());
                    }
                }
            }
            if let Some(idx) = line.find("method in {") {
                let after = &line[idx + "method in {".len()..];
                if let Some(close) = after.find('}') {
                    let inner = &after[..close];
                    let mut search = inner;
                    while let Some(start) = search.find('"') {
                        let after_quote = &search[start + 1..];
                        let Some(end) = after_quote.find('"') else {
                            break;
                        };
                        let token = &after_quote[..end];
                        if !token.is_empty()
                            && token.chars().all(|c| c.is_ascii_lowercase() || c == '_')
                        {
                            methods.insert(token.to_owned());
                        }
                        search = &after_quote[end + 1..];
                    }
                }
            }
        }
        methods
    }

    #[test]
    fn collect_rust_call_methods_extracts_names_from_turbofish_and_plain_call() {
        let source = "\
            self.call::<(), _>(\"health\", None).await\n\
            self.call(\"compose\", Some(request)).await\n\
            self.call(\"transcribe\", Some(request)).await\n\
        ";
        let methods = collect_rust_call_methods(source);
        assert_eq!(methods.len(), 3);
        assert!(methods.contains("health"));
        assert!(methods.contains("compose"));
        assert!(methods.contains("transcribe"));
    }

    #[test]
    fn collect_rust_call_methods_stops_at_first_cfg_test_line() {
        let source = "\
            self.call(\"real\", None)\n\
            #[cfg(test)]\n\
            mod tests {\n\
                router.call(\"synthetic\", None)\n\
            }\n\
        ";
        let methods = collect_rust_call_methods(source);
        assert!(methods.contains("real"));
        assert!(!methods.contains("synthetic"));
    }

    #[test]
    fn collect_python_dispatched_methods_handles_equality_and_set_forms() {
        let source = "\
if method == \"health\":\n\
    return health_response()\n\
elif method == \"transcribe\":\n\
    return transcribe_response()\n\
if method in {\"compose\", \"rewrite\", \"translate\"}:\n\
    return generate_response()\n\
";
        let methods = collect_python_dispatched_methods(source);
        assert_eq!(methods.len(), 5);
        assert!(methods.contains("health"));
        assert!(methods.contains("transcribe"));
        assert!(methods.contains("compose"));
        assert!(methods.contains("rewrite"));
        assert!(methods.contains("translate"));
    }

    #[test]
    fn extract_protocol_doc_method_names_collects_only_required_methods_section() {
        let md = "\
# Worker Protocol

## Transport

prose about stdin/stdout

## Required Methods

- `health`
- `compose`
- `rewrite`

## Current Behavior

The `health` method also reports llm and asr probes.
The `transcribe` method (mentioned only in prose here) is not in
the bullet list above and must not be captured.
";
        let methods = extract_protocol_doc_method_names(md);
        assert_eq!(
            methods,
            ["compose", "health", "rewrite"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "later prose mentions of `transcribe` must not leak into the set",
        );

        let html = "\
<h1 id=\"python-worker-protocol\">Python Worker Protocol</h1>
<h2 id=\"transport\">Transport</h2>
<p><code>health</code> in prose before the list must not count.</p>
<h2 id=\"required-methods\">Required Methods</h2>
<ul>
<li><code>health</code></li>
<li><code>compose</code></li>
<li><code>rewrite</code></li>
</ul>
<h2 id=\"current-behavior\">Current Behavior</h2>
<p>The <code>transcribe</code> method is mentioned only in prose here.</p>
";
        let html_methods = extract_protocol_doc_method_names(html);
        assert_eq!(
            html_methods,
            ["compose", "health", "rewrite"]
                .iter()
                .map(|s| (*s).to_owned())
                .collect(),
            "HTML prose outside the Required Methods section must not leak into the set",
        );
    }

    /// Cross-check the protocol doc's enumerated method list against
    /// the Python worker's dispatch tree. By transitivity through
    /// `every_rust_call_method_has_a_python_dispatch_arm_and_vice_versa`,
    /// passing this test plus that one means the doc, the Rust
    /// caller, and the Python implementation are all in agreement
    /// about the set of supported JSON-RPC methods.
    ///
    /// The drift mode is documentation rot: a method gets added to
    /// the worker but the doc bullet list still enumerates the old
    /// set, or a method is renamed and the doc lags behind.
    /// Operators reading the doc in isolation see a stale picture of
    /// the protocol.
    #[test]
    fn every_protocol_doc_method_is_dispatched_in_python_worker() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let doc_source = std::fs::read_to_string(format!(
            "{manifest}/../../docs/architecture/python-worker-protocol.html"
        ))
        .expect("read python-worker-protocol.html");
        let python_source = std::fs::read_to_string(format!(
            "{manifest}/../../python/voicelayer_orchestrator/worker.py"
        ))
        .expect("read python worker.py");

        let doc_methods = extract_protocol_doc_method_names(&doc_source);
        assert!(
            !doc_methods.is_empty(),
            "expected at least one method in the Required Methods section — \
             extract_protocol_doc_method_names may be misparsing or \
             the heading may have moved",
        );
        let dispatched = collect_python_dispatched_methods(&python_source);

        let missing: Vec<&String> = doc_methods.difference(&dispatched).collect();
        assert!(
            missing.is_empty(),
            "protocol doc lists methods the Python worker does not dispatch: \
             {missing:?}\n\nEither add the dispatch arm in \
             python/voicelayer_orchestrator/worker.py or drop the entry from \
             the Required Methods list in \
             docs/architecture/python-worker-protocol.html.",
        );

        let undocumented: Vec<&String> = dispatched.difference(&doc_methods).collect();
        assert!(
            undocumented.is_empty(),
            "Python worker dispatches methods not listed in the protocol doc: \
             {undocumented:?}\n\nAdd a `<code>&lt;name&gt;</code>` entry under \
             the Required Methods list in docs/architecture/python-worker-protocol.html \
             so the doc reflects what the worker actually accepts.",
        );
    }

    /// Cross-check every Rust JSON-RPC method literal against the
    /// Python worker's dispatch tree, in both directions. A regression
    /// renaming the Rust constant or the Python dispatch arm would
    /// surface as `-32601 Method not found` at runtime; this test
    /// catches the drift before the code ships.
    #[test]
    fn every_rust_call_method_has_a_python_dispatch_arm_and_vice_versa() {
        let manifest = env!("CARGO_MANIFEST_DIR");
        let rust_source = std::fs::read_to_string(format!("{manifest}/src/worker.rs"))
            .expect("read voicelayerd worker.rs");
        let python_source = std::fs::read_to_string(format!(
            "{manifest}/../../python/voicelayer_orchestrator/worker.py"
        ))
        .expect("read python worker.py");

        let rust_methods = collect_rust_call_methods(&rust_source);
        let python_methods = collect_python_dispatched_methods(&python_source);

        assert!(
            !rust_methods.is_empty(),
            "expected at least one .call(\"...\") in worker.rs; \
             collect_rust_call_methods may be misparsing",
        );
        assert!(
            !python_methods.is_empty(),
            "expected at least one method dispatch in worker.py; \
             collect_python_dispatched_methods may be misparsing",
        );

        for method in &rust_methods {
            assert!(
                python_methods.contains(method),
                "rust .call(\"{method}\") has no matching Python dispatch arm. \
                 Add `if method == \"{method}\":` (or include `\"{method}\"` in \
                 the existing set form) to python/voicelayer_orchestrator/worker.py.",
            );
        }
        for method in &python_methods {
            assert!(
                rust_methods.contains(method),
                "Python dispatches `{method}` but the Rust daemon never calls it. \
                 Either add a `.call(\"{method}\", ...)` site in worker.rs or drop \
                 the dispatch arm.",
            );
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
        // The optional Xiaomi MiMo-V2.5-ASR backend must show up in
        // the catalog so `vl providers` can list it, but with
        // `default_enabled=false` and `experimental=true` so callers
        // know the whisper.cpp chain stays the production default.
        let mimo = providers
            .providers
            .iter()
            .find(|provider| provider.id == "mimo_v2_5_asr")
            .expect(
                "the Python worker must advertise the mimo_v2_5_asr ASR \
                 provider so operators can opt into it via \
                 TranscribeRequest.provider_id",
            );
        assert!(!mimo.default_enabled);
        assert!(mimo.experimental);

        // Same shape for the optional Qwen3-ASR-1.7B backend: advertised
        // in the catalog with `default_enabled=false` /
        // `experimental=true` so the whisper.cpp chain remains the
        // production default and operators opt in via
        // `TranscribeRequest.provider_id`.
        let qwen3 = providers
            .providers
            .iter()
            .find(|provider| provider.id == "qwen3_asr_1_7b")
            .expect(
                "the Python worker must advertise the qwen3_asr_1_7b ASR \
                 provider so operators can opt into it via \
                 TranscribeRequest.provider_id",
            );
        assert!(!qwen3.default_enabled);
        assert!(qwen3.experimental);
        assert_eq!(qwen3.license, "Apache-2.0");
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
    /// the JSON-RPC worker. The helper prepends `head -n 1 > /dev/null;`
    /// so the subprocess consumes exactly one line of the daemon's
    /// request before running the caller's response snippet — without
    /// it, `printf '...'; exit 0` style scripts close stdin before our
    /// `write_all` lands and the test fails with `BrokenPipe` instead
    /// of the variant we wanted to exercise. `head -n 1` (rather than
    /// `cat > /dev/null`) is required by the persistent-worker mode in
    /// `WorkerCommand::call`: the daemon now keeps stdin open across
    /// calls so `cat` would block forever waiting for EOF, while the
    /// JSON-RPC dance is one line in / one line out per call. The
    /// script's stdout becomes the response line and its final exit
    /// status feeds the post-response classification (success /
    /// `ProcessExited`) via the 25 ms try_wait grace window.
    fn shell_worker(response_script: &str) -> WorkerCommand {
        let combined = format!("head -n 1 > /dev/null; {response_script}");
        WorkerCommand::new(
            "sh".to_owned(),
            vec!["-c".to_owned(), combined],
            std::env::current_dir().expect("current_dir should resolve"),
        )
    }

    #[tokio::test]
    async fn worker_call_returns_io_error_when_executable_does_not_exist() {
        let worker = WorkerCommand::new(
            "/voicelayer-test/definitely-not-a-real-binary".to_owned(),
            vec![],
            std::env::current_dir().expect("current_dir should resolve"),
        );
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

    /// End-to-end pin for the persistent-worker mode added to retire
    /// the per-call spawn cost on GPU providers like MiMo-V2.5-ASR and
    /// Qwen3-ASR-1.7B. The shell records one line per spawn into a
    /// counter file on startup, then loops on stdin emitting a static
    /// JSON-RPC response per request line. After two `list_providers`
    /// calls through the same `WorkerCommand` clone, the counter must
    /// end at exactly one entry — proof that both calls flowed through
    /// the same child process and that the in-process model cache on
    /// the Python side now actually warms across requests. A
    /// regression that flipped back to spawn-per-call (or that broke
    /// `Arc` sharing across `WorkerCommand` clones) would land the
    /// counter at two.
    #[tokio::test]
    async fn persistent_worker_reuses_child_across_calls() {
        let tempdir = tempfile::tempdir().expect("tempdir should be creatable");
        let spawn_counter = tempdir.path().join("spawn_counter");
        let counter_path = spawn_counter.display().to_string();

        // `IFS= read -r _` reads one line per request without stripping
        // whitespace; the loop body emits a fixed valid JSON-RPC
        // response each time.
        let script = format!(
            r#"echo x >> {counter_path}; while IFS= read -r _; do printf '%s\n' '{{"jsonrpc":"2.0","id":"x","result":{{"providers":[]}}}}'; done"#,
        );
        let worker = WorkerCommand::new(
            "sh".to_owned(),
            vec!["-c".to_owned(), script],
            std::env::current_dir().expect("current_dir should resolve"),
        );

        worker
            .list_providers()
            .await
            .expect("first list_providers call must succeed");
        worker
            .list_providers()
            .await
            .expect("second list_providers call must succeed");

        let spawn_marks = std::fs::read_to_string(&spawn_counter)
            .expect("counter file should exist after spawn")
            .lines()
            .count();
        assert_eq!(
            spawn_marks, 1,
            "persistent-worker mode must spawn the child exactly once across \
             multiple calls; got {spawn_marks} spawn marks. A regression that \
             reverts to spawn-per-call would land at 2.",
        );
    }

    /// Crash recovery: when the shared child dies between calls, the
    /// next call must transparently respawn instead of surfacing the
    /// EOF as a permanent error. The script writes one response per
    /// request line and exits cleanly after each, so the persistent
    /// worker handle goes stale between calls and the next call has
    /// to rebuild it. Both calls must succeed and the spawn counter
    /// must show two distinct child processes.
    #[tokio::test]
    async fn persistent_worker_respawns_after_child_exits_cleanly() {
        let tempdir = tempfile::tempdir().expect("tempdir should be creatable");
        let spawn_counter = tempdir.path().join("spawn_counter");
        let counter_path = spawn_counter.display().to_string();

        let script = format!(
            r#"echo x >> {counter_path}; head -n 1 > /dev/null; printf '%s\n' '{{"jsonrpc":"2.0","id":"x","result":{{"providers":[]}}}}'"#,
        );
        let worker = WorkerCommand::new(
            "sh".to_owned(),
            vec!["-c".to_owned(), script],
            std::env::current_dir().expect("current_dir should resolve"),
        );

        worker
            .list_providers()
            .await
            .expect("first list_providers call must succeed");
        worker
            .list_providers()
            .await
            .expect("second list_providers call must respawn and succeed");

        let spawn_marks = std::fs::read_to_string(&spawn_counter)
            .expect("counter file should exist after spawn")
            .lines()
            .count();
        assert_eq!(
            spawn_marks, 2,
            "after a clean child exit, the second call must respawn the worker; \
             got {spawn_marks} spawns. A regression that left the dead handle \
             in place would surface an EOF error on the second call.",
        );
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
