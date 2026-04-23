//! Fixed-duration segmented recording for VoiceLayer dictation sessions.
//!
//! A [`SegmentedRecording`] rolls the underlying recorder subprocess every
//! `segment_secs` seconds so a Phase 3 transcription pipeline can hand the
//! finalized chunk to the worker while the next segment starts without a gap.
//!
//! The caller drives cadence: the daemon task ticks on an interval and invokes
//! [`SegmentedRecording::finalize_current`] to stop the running chunk, collect
//! its path, and immediately start the next one. [`SegmentedRecording::stop`]
//! consumes the recorder at the end of the session and returns the full
//! ordered list of segment paths, including the partial one in flight when
//! the stop was requested.
//!
//! The `overlap_secs` knob is reserved for a later stage that stitches
//! overlapping windows at segment boundaries; the MVP implementation keeps it
//! in the struct but defaults to zero overlap.

use std::fs;
use std::path::{Path, PathBuf};

use voicelayer_core::RecorderBackend;

use crate::recording::{ActiveRecording, RecordingError, resolve_recorder_backend};
use crate::recording::{start_recording_process, stop_recording_process};

/// Function pointer type for spawning a segment's recorder subprocess.
///
/// Production code passes [`start_recording_process`]; tests pass a fake
/// that pre-populates the WAV path and spawns a benign child so the state
/// machine can be exercised without requiring `pw-record` / `arecord` on
/// the host.
pub(crate) type RecorderSpawner =
    fn(&Path, RecorderBackend) -> Result<ActiveRecording, RecordingError>;

#[derive(Debug)]
pub struct SegmentedRecording {
    segment_dir: PathBuf,
    backend: RecorderBackend,
    segment_secs: u32,
    overlap_secs: u32,
    next_id: usize,
    current_id: usize,
    current: Option<ActiveRecording>,
    finalized: Vec<(usize, PathBuf)>,
    spawner: RecorderSpawner,
}

impl SegmentedRecording {
    /// Start a segmented recorder. The first segment's subprocess begins
    /// immediately so the caller can start the interval ticker as soon as
    /// this returns `Ok`.
    pub fn start(
        segment_dir: &Path,
        backend: RecorderBackend,
        segment_secs: u32,
        overlap_secs: u32,
    ) -> Result<Self, RecordingError> {
        let resolved = resolve_recorder_backend(backend)?;
        Self::start_with_spawner(
            segment_dir,
            resolved,
            segment_secs,
            overlap_secs,
            start_recording_process,
        )
    }

    /// Construct a [`SegmentedRecording`] with an explicit spawner.
    ///
    /// Used by the production [`Self::start`] after `resolve_recorder_backend`
    /// picks a real backend, and by the test module to inject a fake spawner
    /// that sidesteps the `pw-record` / `arecord` binaries. Kept `pub(crate)`
    /// — this is not a stability surface.
    pub(crate) fn start_with_spawner(
        segment_dir: &Path,
        backend: RecorderBackend,
        segment_secs: u32,
        overlap_secs: u32,
        spawner: RecorderSpawner,
    ) -> Result<Self, RecordingError> {
        if segment_secs == 0 {
            return Err(RecordingError::NoSupportedBackend);
        }
        fs::create_dir_all(segment_dir)?;
        let mut recorder = Self {
            segment_dir: segment_dir.to_path_buf(),
            backend,
            segment_secs,
            overlap_secs,
            next_id: 0,
            current_id: 0,
            current: None,
            finalized: Vec::new(),
            spawner,
        };
        recorder.begin_next_segment()?;
        Ok(recorder)
    }

    /// Number of seconds each segment should capture before being rolled.
    pub fn segment_secs(&self) -> u32 {
        self.segment_secs
    }

    /// Requested overlap between segments in seconds. Currently informational.
    pub fn overlap_secs(&self) -> u32 {
        self.overlap_secs
    }

    /// Directory that holds the recorded segment WAV files.
    pub fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    /// Stop the in-flight segment and start the next one.
    ///
    /// Returns the finalized segment's `(id, path)` on success, or `None`
    /// when no segment was in flight (for example, after [`Self::stop`] has
    /// already been called).
    pub async fn finalize_current(&mut self) -> Result<Option<(usize, PathBuf)>, RecordingError> {
        let Some(active) = self.current.take() else {
            return Ok(None);
        };
        let id = self.current_id;
        let path = stop_recording_process(active).await.inspect_err(|err| {
            tracing::error!(segment = id, %err, "failed to finalize segment");
        })?;
        self.finalized.push((id, path.clone()));
        self.begin_next_segment()?;
        Ok(Some((id, path)))
    }

    /// Consume the recorder, stop the in-flight segment (if any), and return
    /// every segment in recording order.
    pub async fn stop(mut self) -> Result<Vec<(usize, PathBuf)>, RecordingError> {
        if let Some(active) = self.current.take() {
            let id = self.current_id;
            let path = stop_recording_process(active).await?;
            self.finalized.push((id, path));
        }
        Ok(self.finalized)
    }

    fn begin_next_segment(&mut self) -> Result<(), RecordingError> {
        let id = self.next_id;
        let path = self.segment_dir.join(segment_file_name(id));
        let recording = (self.spawner)(&path, self.backend)?;
        self.current = Some(recording);
        self.current_id = id;
        self.next_id = id + 1;
        Ok(())
    }
}

pub(crate) fn segment_file_name(id: usize) -> String {
    format!("segment-{id:05}.wav")
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use tokio::process::Command;
    use voicelayer_core::RecorderBackend;

    use super::{ActiveRecording, RecordingError, SegmentedRecording, segment_file_name};

    /// Fake spawner used by the state-machine tests. Writes bytes into the
    /// target path so `stop_recording_process`'s size check passes, then
    /// spawns a short-lived `sh -c 'sleep 0.2'` child that exits on its
    /// own within the stop-path timeout. This sidesteps the tokio-fork
    /// SIG_IGN issue documented in the recording.rs tests: dash inherits
    /// SIGINT with SIG_IGN from the parent and cannot re-enable it, so
    /// `kill -INT` delivered by `stop_recording_process` is a no-op.
    /// Instead we rely on the child exiting naturally. `stop_recording_process`
    /// succeeds when (a) the wait returns before its 5s timeout and
    /// (b) the audio_file is on disk with content — both are true here.
    fn fake_successful_spawner(
        path: &Path,
        backend: RecorderBackend,
    ) -> Result<ActiveRecording, RecordingError> {
        std::fs::write(path, b"fake-audio-bytes")?;
        let child = Command::new("sh")
            .arg("-c")
            .arg("sleep 0.2")
            .spawn()
            .expect("spawn fake recorder child");
        Ok(ActiveRecording {
            backend,
            audio_file: path.to_path_buf(),
            child,
        })
    }

    #[test]
    fn segment_file_name_zero_pads_to_five_digits() {
        assert_eq!(segment_file_name(0), "segment-00000.wav");
        assert_eq!(segment_file_name(7), "segment-00007.wav");
        assert_eq!(segment_file_name(12345), "segment-12345.wav");
    }

    #[test]
    fn segment_file_name_allows_six_digits_when_exceeded() {
        assert_eq!(segment_file_name(100_000), "segment-100000.wav");
    }

    #[tokio::test]
    async fn start_with_spawner_rejects_zero_segment_secs() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let err = SegmentedRecording::start_with_spawner(
            tempdir.path(),
            RecorderBackend::Pipewire,
            0,
            0,
            fake_successful_spawner,
        )
        .expect_err("zero-second segments must be rejected up front");
        assert!(
            matches!(err, RecordingError::NoSupportedBackend),
            "unexpected error variant: {err:?}",
        );
    }

    #[tokio::test]
    async fn start_with_spawner_creates_segment_dir_and_first_segment_file() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let segment_dir = tempdir.path().join("segments");
        // Directory does not exist before start — start must mkdir it.
        assert!(!segment_dir.exists());

        let recorder = SegmentedRecording::start_with_spawner(
            &segment_dir,
            RecorderBackend::Pipewire,
            5,
            0,
            fake_successful_spawner,
        )
        .expect("start must succeed with a working spawner");

        assert!(segment_dir.is_dir(), "segment_dir should be created");
        assert_eq!(recorder.segment_secs(), 5);
        assert_eq!(recorder.overlap_secs(), 0);
        assert_eq!(recorder.segment_dir(), segment_dir);

        // First segment file should already be on disk with the expected
        // name, and no segments should yet be in the finalized list.
        let expected_first = segment_dir.join("segment-00000.wav");
        assert!(
            expected_first.is_file(),
            "first segment {expected_first:?} should be created by the spawner",
        );

        // Drop the recorder without going through stop — the fake child is
        // a `sleep 60` that tokio::process::Child will kill on drop
        // because ActiveRecording has no custom Drop. We need kill_on_drop
        // semantics to keep the test suite clean; rely on the OS reaping
        // the orphan via process group cleanup when the test binary exits.
        drop(recorder);
    }

    #[tokio::test]
    async fn finalize_current_rolls_to_next_segment_with_monotonic_ids() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mut recorder = SegmentedRecording::start_with_spawner(
            tempdir.path(),
            RecorderBackend::Pipewire,
            5,
            0,
            fake_successful_spawner,
        )
        .expect("start");

        let (id0, path0) = recorder
            .finalize_current()
            .await
            .expect("finalize 0 returns Ok")
            .expect("a segment was in flight");
        assert_eq!(id0, 0, "first finalized segment must be id 0");
        assert_eq!(path0, tempdir.path().join("segment-00000.wav"));

        let (id1, path1) = recorder
            .finalize_current()
            .await
            .expect("finalize 1 returns Ok")
            .expect("second segment was in flight");
        assert_eq!(id1, 1, "segment ids must increase monotonically");
        assert_eq!(path1, tempdir.path().join("segment-00001.wav"));

        // Clean shutdown.
        let all = recorder.stop().await.expect("stop");
        assert_eq!(
            all.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![0, 1, 2],
            "stop must include the in-flight segment after the rolls",
        );
    }

    #[tokio::test]
    async fn stop_returns_every_segment_in_recording_order() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mut recorder = SegmentedRecording::start_with_spawner(
            tempdir.path(),
            RecorderBackend::Pipewire,
            5,
            0,
            fake_successful_spawner,
        )
        .expect("start");

        recorder
            .finalize_current()
            .await
            .expect("finalize 0")
            .expect("segment 0 exists");
        recorder
            .finalize_current()
            .await
            .expect("finalize 1")
            .expect("segment 1 exists");

        let segments = recorder.stop().await.expect("stop succeeds");
        let ids: Vec<usize> = segments.iter().map(|(id, _)| *id).collect();
        assert_eq!(
            ids,
            vec![0, 1, 2],
            "stop should preserve recording order and include the tail segment",
        );
        for (id, path) in &segments {
            let expected_name = format!("segment-{id:05}.wav");
            assert_eq!(
                path.file_name().and_then(|n| n.to_str()),
                Some(&*expected_name)
            );
        }
    }

    #[tokio::test]
    async fn stop_after_finalize_emits_only_finalized_and_new_tail() {
        let tempdir = tempfile::tempdir().expect("tempdir");
        let mut recorder = SegmentedRecording::start_with_spawner(
            tempdir.path(),
            RecorderBackend::Pipewire,
            5,
            0,
            fake_successful_spawner,
        )
        .expect("start");

        recorder
            .finalize_current()
            .await
            .expect("finalize 0")
            .expect("segment 0 exists");

        let segments = recorder.stop().await.expect("stop");
        assert_eq!(
            segments.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            vec![0, 1],
            "exactly two segments should emerge: the finalized one and the in-flight one",
        );
    }
}
