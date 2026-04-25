use std::{
    fs,
    path::{Path, PathBuf},
    process::Command as StdCommand,
    time::Duration,
};

use serde::Serialize;
use thiserror::Error;
use tokio::{
    process::{Child, Command},
    time::timeout,
};
use voicelayer_core::RecorderBackend;

#[derive(Debug, Clone, Serialize)]
pub struct RecorderDiagnostics {
    pub selected_backend: Option<String>,
    pub pw_record_available: bool,
    pub arecord_available: bool,
    pub timeout_available: bool,
}

#[derive(Debug, Error)]
pub enum RecordingError {
    #[error("pipewire recording requires both `pw-record` and `timeout`")]
    PipewireUnavailable,
    #[error("alsa recording requires `arecord`")]
    AlsaUnavailable,
    #[error("no supported recorder backend is available; install `pw-record` or `arecord`")]
    NoSupportedBackend,
    #[error("failed to start recorder process: {0}")]
    Io(#[from] std::io::Error),
    #[error("pw-record failed with status {0}")]
    PipewireFailed(String),
    #[error("arecord failed with status {0}")]
    AlsaFailed(String),
    #[error("recorder did not create the expected file: {0}")]
    MissingOutput(String),
    #[error("recorder did not expose a process identifier")]
    MissingPid,
    #[error("failed to send interrupt to recorder: {0}")]
    InterruptFailed(String),
    #[error("recorder did not stop before timeout")]
    StopTimedOut,
}

#[derive(Debug)]
pub struct ActiveRecording {
    pub backend: RecorderBackend,
    pub audio_file: PathBuf,
    pub child: Child,
}

pub fn recorder_diagnostics(preferred: RecorderBackend) -> RecorderDiagnostics {
    let pw_record_available = resolve_executable("pw-record").is_some();
    let arecord_available = resolve_executable("arecord").is_some();
    let timeout_available = resolve_executable("timeout").is_some();
    let selected_backend = match resolve_recorder_backend(preferred) {
        Ok(RecorderBackend::Pipewire) => Some("pipewire".to_owned()),
        Ok(RecorderBackend::Alsa) => Some("alsa".to_owned()),
        Ok(RecorderBackend::Auto) => Some("auto".to_owned()),
        Err(_) => None,
    };

    RecorderDiagnostics {
        selected_backend,
        pw_record_available,
        arecord_available,
        timeout_available,
    }
}

pub fn temp_audio_path() -> PathBuf {
    std::env::temp_dir().join(format!("voicelayer-recording-{}.wav", uuid::Uuid::new_v4()))
}

pub fn record_audio_file(
    audio_file: &Path,
    duration_seconds: u32,
    preferred: RecorderBackend,
) -> Result<(), RecordingError> {
    match resolve_recorder_backend(preferred)? {
        RecorderBackend::Pipewire => {
            let status = StdCommand::new("timeout")
                .arg("--signal=INT")
                .arg(format!("{duration_seconds}s"))
                .arg("pw-record")
                .arg("--rate")
                .arg("16000")
                .arg("--channels")
                .arg("1")
                .arg("--format")
                .arg("s16")
                .arg(audio_file.as_os_str())
                .status()?;

            if !(status.success() || status.code() == Some(124)) {
                return Err(RecordingError::PipewireFailed(status.to_string()));
            }
        }
        RecorderBackend::Alsa => {
            let status = StdCommand::new("arecord")
                .arg("-q")
                .arg("-t")
                .arg("wav")
                .arg("-d")
                .arg(duration_seconds.to_string())
                .arg("-f")
                .arg("S16_LE")
                .arg("-r")
                .arg("16000")
                .arg("-c")
                .arg("1")
                .arg(audio_file.as_os_str())
                .status()?;

            if !status.success() {
                return Err(RecordingError::AlsaFailed(status.to_string()));
            }
        }
        RecorderBackend::Auto => return Err(RecordingError::NoSupportedBackend),
    }

    if !audio_file.is_file() {
        return Err(RecordingError::MissingOutput(
            audio_file.display().to_string(),
        ));
    }
    Ok(())
}

pub fn start_recording_process(
    audio_file: &Path,
    preferred: RecorderBackend,
) -> Result<ActiveRecording, RecordingError> {
    let backend = resolve_recorder_backend(preferred)?;
    let mut command = match backend {
        RecorderBackend::Pipewire => {
            let mut command = Command::new("pw-record");
            command
                .arg("--rate")
                .arg("16000")
                .arg("--channels")
                .arg("1")
                .arg("--format")
                .arg("s16")
                .arg(audio_file.as_os_str());
            command
        }
        RecorderBackend::Alsa => {
            let mut command = Command::new("arecord");
            command
                .arg("-q")
                .arg("-t")
                .arg("wav")
                .arg("-f")
                .arg("S16_LE")
                .arg("-r")
                .arg("16000")
                .arg("-c")
                .arg("1")
                .arg(audio_file.as_os_str());
            command
        }
        RecorderBackend::Auto => return Err(RecordingError::NoSupportedBackend),
    };

    let child = command.spawn()?;
    Ok(ActiveRecording {
        backend,
        audio_file: audio_file.to_path_buf(),
        child,
    })
}

pub async fn stop_recording_process(
    mut recording: ActiveRecording,
) -> Result<PathBuf, RecordingError> {
    let pid = recording.child.id().ok_or(RecordingError::MissingPid)?;
    let signal = match recording.backend {
        RecorderBackend::Pipewire | RecorderBackend::Alsa => "-INT",
        RecorderBackend::Auto => "-TERM",
    };

    let output = StdCommand::new("kill")
        .arg(signal)
        .arg(pid.to_string())
        .output()?;
    if !output.status.success() {
        let stderr = String::from_utf8_lossy(&output.stderr).trim().to_owned();
        let detail = if stderr.is_empty() {
            output.status.to_string()
        } else {
            format!("{} ({stderr})", output.status)
        };
        tracing::error!(
            pid = pid,
            signal = signal,
            exit = ?output.status.code(),
            stderr = %stderr,
            "kill failed while stopping recorder",
        );
        return Err(RecordingError::InterruptFailed(detail));
    }

    // After `kill -INT` succeeds the child has one of three outcomes:
    //   - terminated via SIGINT (status.signal() == Some(2))
    //   - cleaned up during its signal handler and exited with status 0 or
    //     a non-zero code (pw-record returns 1 here, which is its normal
    //     shutdown path)
    //   - ignored SIGINT and kept running, in which case we fall through
    //     to the 5s timeout branch below and SIGKILL it
    // So the exit status alone does not tell us whether the recording is
    // usable. Instead we verify the WAV file landed on disk with content.
    let wait_result = timeout(Duration::from_secs(5), recording.child.wait()).await;
    match wait_result {
        Ok(Ok(_status)) => {}
        Ok(Err(error)) => return Err(RecordingError::Io(error)),
        Err(_) => {
            recording.child.start_kill()?;
            let _ = recording.child.wait().await;
            return Err(RecordingError::StopTimedOut);
        }
    }

    match fs::metadata(&recording.audio_file) {
        Ok(meta) if meta.is_file() && meta.len() > 0 => Ok(recording.audio_file),
        _ => Err(RecordingError::MissingOutput(
            recording.audio_file.display().to_string(),
        )),
    }
}

pub fn maybe_cleanup_audio_file(audio_file: &Path, keep_audio: bool) -> Option<String> {
    if keep_audio {
        Some(audio_file.display().to_string())
    } else {
        let _ = fs::remove_file(audio_file);
        None
    }
}

pub fn resolve_recorder_backend(
    preferred: RecorderBackend,
) -> Result<RecorderBackend, RecordingError> {
    let pw_record_available = resolve_executable("pw-record").is_some();
    let arecord_available = resolve_executable("arecord").is_some();
    let timeout_available = resolve_executable("timeout").is_some();

    match preferred {
        RecorderBackend::Pipewire => {
            if !pw_record_available || !timeout_available {
                return Err(RecordingError::PipewireUnavailable);
            }
            Ok(RecorderBackend::Pipewire)
        }
        RecorderBackend::Alsa => {
            if !arecord_available {
                return Err(RecordingError::AlsaUnavailable);
            }
            Ok(RecorderBackend::Alsa)
        }
        RecorderBackend::Auto => {
            if pw_record_available && timeout_available {
                Ok(RecorderBackend::Pipewire)
            } else if arecord_available {
                Ok(RecorderBackend::Alsa)
            } else {
                Err(RecordingError::NoSupportedBackend)
            }
        }
    }
}

fn resolve_executable(name: &str) -> Option<PathBuf> {
    let candidate = PathBuf::from(name);
    if candidate.components().count() > 1 {
        return candidate.is_file().then_some(candidate);
    }

    let path = std::env::var_os("PATH")?;
    for directory in std::env::split_paths(&path) {
        let full_path = directory.join(name);
        if full_path.is_file() && is_executable(&full_path) {
            return Some(full_path);
        }
    }
    None
}

fn is_executable(path: &Path) -> bool {
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;

        path.metadata()
            .map(|metadata| metadata.permissions().mode() & 0o111 != 0)
            .unwrap_or(false)
    }

    #[cfg(not(unix))]
    {
        path.is_file()
    }
}

#[cfg(test)]
mod tests {
    use super::{
        ActiveRecording, RecordingError, maybe_cleanup_audio_file, recorder_diagnostics,
        resolve_recorder_backend, stop_recording_process,
    };
    use std::fs;
    use tokio::process::Command;
    use voicelayer_core::RecorderBackend;

    #[test]
    fn auto_backend_resolves_on_current_machine() {
        let _ = resolve_recorder_backend(RecorderBackend::Auto);
    }

    #[test]
    fn diagnostics_report_current_machine_state() {
        // Validates internal consistency rather than the presence of a
        // recorder: CI runners do not always install pipewire/alsa, so
        // asserting availability would hard-fail there. The invariant
        // worth pinning is that the reported `selected_backend` must
        // match the underlying `*_available` flags.
        let diagnostics = recorder_diagnostics(RecorderBackend::Auto);
        match diagnostics.selected_backend.as_deref() {
            Some("pipewire") => assert!(
                diagnostics.pw_record_available,
                "Auto selected pipewire but pw-record is not reported as available",
            ),
            Some("alsa") => assert!(
                diagnostics.arecord_available,
                "Auto selected alsa but arecord is not reported as available",
            ),
            Some(other) => panic!("Unexpected selected_backend label: {other}"),
            None => assert!(
                !diagnostics.pw_record_available && !diagnostics.arecord_available,
                "selected_backend is None while a recorder is reported as available",
            ),
        }
    }

    // Spawn a shell child that sleeps briefly and then exits with the
    // requested code. This models `pw-record`'s real shutdown path —
    // exits with code 1 after cleanup — without relying on SIGINT
    // delivery to dash, which tokio's process spawn makes unreliable:
    // dash inherits SIGINT with SIG_IGN from tokio's fork point, and
    // POSIX dash cannot re-enable a signal that was ignored at exec.
    // The production code path really does deliver SIGINT to
    // `pw-record` (which installs its own sigaction handler and honors
    // it), so what these tests pin is the post-wait invariant: once
    // the child has exited, stop_recording_process decides pass/fail
    // based on whether the WAV file landed, not on the exit status.
    fn spawn_exiting_child(code: u8) -> tokio::process::Child {
        let script = format!("sleep 0.2; exit {code}");
        Command::new("sh")
            .arg("-c")
            .arg(script)
            .spawn()
            .expect("spawn sh")
    }

    #[tokio::test]
    async fn stop_recording_process_accepts_nonzero_exit_when_audio_file_has_content() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("fake.wav");
        fs::write(&audio_file, b"fake-audio-bytes").expect("seed audio");

        let child = spawn_exiting_child(1);
        let recording = ActiveRecording {
            backend: RecorderBackend::Pipewire,
            audio_file: audio_file.clone(),
            child,
        };

        let resolved = stop_recording_process(recording)
            .await
            .expect("nonzero child exit with populated audio file must succeed");
        assert_eq!(resolved, audio_file);
    }

    #[tokio::test]
    async fn stop_recording_process_reports_missing_output_when_audio_file_absent() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("never_written.wav");
        // Deliberately do not create audio_file.

        let child = spawn_exiting_child(1);
        let recording = ActiveRecording {
            backend: RecorderBackend::Pipewire,
            audio_file: audio_file.clone(),
            child,
        };

        match stop_recording_process(recording).await {
            Err(RecordingError::MissingOutput(path)) => {
                assert!(path.contains("never_written.wav"), "got path: {path}")
            }
            other => panic!("expected MissingOutput, got {other:?}"),
        }
    }

    #[tokio::test]
    async fn stop_recording_process_reports_missing_output_when_audio_file_empty() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("empty.wav");
        fs::write(&audio_file, b"").expect("seed empty file");

        let child = spawn_exiting_child(0);
        let recording = ActiveRecording {
            backend: RecorderBackend::Pipewire,
            audio_file: audio_file.clone(),
            child,
        };

        // A zero-byte file means the recorder started but captured no
        // frames (or the user's audio path is broken). Do not pass that
        // off to whisper as if it were usable input.
        match stop_recording_process(recording).await {
            Err(RecordingError::MissingOutput(_)) => {}
            other => panic!("expected MissingOutput for empty file, got {other:?}"),
        }
    }

    /// Pins the `keep_audio = true` branch. The dictation lifecycle
    /// hands the cleanup decision to the operator via the
    /// `keep_audio` flag on `StartDictationRequest` /
    /// `DictationCaptureRequest`; with the flag set the recording
    /// must survive on disk and its path must be reported back so
    /// `DictationCaptureResult.audio_file` can carry it. A regression
    /// that always cleaned up regardless of the flag would silently
    /// break debugging workflows that rely on retaining captures.
    #[test]
    fn maybe_cleanup_audio_file_keeps_file_and_returns_path_when_keep_audio_set() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("seed.wav");
        fs::write(&audio_file, b"audio bytes").expect("seed audio");

        let result = maybe_cleanup_audio_file(&audio_file, true);

        assert_eq!(
            result.as_deref(),
            Some(audio_file.display().to_string()).as_deref()
        );
        assert!(
            audio_file.exists(),
            "keep_audio=true must leave the file in place; was deleted",
        );
    }

    /// Pins the `keep_audio = false` branch. Default flow: the
    /// recording is consumed by the worker for transcription and
    /// then removed from the runtime directory. A regression that
    /// returned the path even though the file was deleted would
    /// surface a phantom audio_file in `DictationCaptureResult` and
    /// the operator would dereference a nonexistent path.
    #[test]
    fn maybe_cleanup_audio_file_deletes_file_and_returns_none_when_keep_audio_false() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("seed.wav");
        fs::write(&audio_file, b"audio bytes").expect("seed audio");

        let result = maybe_cleanup_audio_file(&audio_file, false);

        assert!(
            result.is_none(),
            "keep_audio=false must return None to suppress audio_file in the result",
        );
        assert!(
            !audio_file.exists(),
            "keep_audio=false must delete the file from disk",
        );
    }

    /// Defensive: with `keep_audio = false` and the file already
    /// missing, the cleanup must not panic. The current
    /// implementation discards the `fs::remove_file` error, so this
    /// pin protects against a future refactor that would surface
    /// the absent-file IO error and crash the dictation path.
    #[test]
    fn maybe_cleanup_audio_file_tolerates_already_missing_file_when_keep_audio_false() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let audio_file = tempdir.path().join("never-existed.wav");
        assert!(
            !audio_file.exists(),
            "test precondition: the audio file must not exist",
        );

        let result = maybe_cleanup_audio_file(&audio_file, false);
        assert!(result.is_none());
    }
}
