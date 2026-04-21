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
        if segment_secs == 0 {
            return Err(RecordingError::NoSupportedBackend);
        }
        fs::create_dir_all(segment_dir)?;
        let resolved = resolve_recorder_backend(backend)?;
        let mut recorder = Self {
            segment_dir: segment_dir.to_path_buf(),
            backend: resolved,
            segment_secs,
            overlap_secs,
            next_id: 0,
            current_id: 0,
            current: None,
            finalized: Vec::new(),
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
        let path = stop_recording_process(active).await?;
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
        let recording = start_recording_process(&path, self.backend)?;
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
    use super::segment_file_name;

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
}
