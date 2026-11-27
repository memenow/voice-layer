//! Subprocess management and stderr ring-buffer draining for the Python worker.

use std::{
    collections::VecDeque,
    path::Path,
    process::Stdio,
    sync::{Arc, Mutex as StdMutex},
};

use tokio::{
    io::{AsyncBufReadExt, BufReader, Lines},
    process::{Child, ChildStdin, ChildStdout, Command},
    task::JoinHandle,
};

use super::rpc::WorkerCallError;

/// Maximum stderr lines retained for process-exit diagnostics.
pub(super) const STDERR_TAIL_MAX_LINES: usize = 64;

#[derive(Debug)]
pub(super) struct WorkerProcess {
    pub child: Child,
    pub stdin: ChildStdin,
    pub stdout: Lines<BufReader<ChildStdout>>,
    stderr_drainer: JoinHandle<()>,
}

impl Drop for WorkerProcess {
    fn drop(&mut self) {
        self.stderr_drainer.abort();
    }
}

pub(super) fn spawn_process(
    executable: &str,
    args: &[String],
    project_root: &Path,
    stderr_tail: Arc<StdMutex<VecDeque<String>>>,
) -> Result<WorkerProcess, WorkerCallError> {
    let mut command = Command::new(executable);
    command
        .args(args)
        .current_dir(project_root)
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

    let stderr_drainer = tokio::spawn(async move {
        let mut lines = BufReader::new(stderr).lines();
        while let Ok(Some(line)) = lines.next_line().await {
            if let Ok(mut buffer) = stderr_tail.lock() {
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
