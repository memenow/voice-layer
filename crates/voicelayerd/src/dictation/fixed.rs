//! Fixed-duration segmented dictation pipeline.

use std::time::Duration;

use tokio::{
    sync::oneshot,
    time::{MissedTickBehavior, interval},
};
use voicelayer_core::{CaptureSession, DictationCaptureResult};

use crate::{api::error::ApiError, audio::AudioCapture};

use super::{
    MIN_SEGMENT_SECS,
    segmented::{SegmentedContext, SegmentedRunner},
};

pub(super) async fn run(
    session: CaptureSession,
    capture: AudioCapture,
    segment_secs: u32,
    context: SegmentedContext,
    mut stop_rx: oneshot::Receiver<()>,
) -> Result<DictationCaptureResult, ApiError> {
    let mut runner = SegmentedRunner::new(session.session_id, context)?;
    let mut ticker = interval(Duration::from_secs(u64::from(segment_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // Discard the immediate initial tick so the first cut waits a full interval.
    ticker.tick().await;

    let mut cut_start_secs = 0.0_f64;

    loop {
        tokio::select! {
            biased;
            _ = &mut stop_rx => break,
            _ = ticker.tick() => {
                let target_end = cut_start_secs + f64::from(segment_secs);
                cut_chunk(&capture, &mut runner, &mut cut_start_secs, target_end)?;
            }
        }
    }

    let elapsed = capture.elapsed_secs();
    cut_chunk(&capture, &mut runner, &mut cut_start_secs, elapsed)?;
    drop(capture);

    runner.finalize(session).await
}

fn cut_chunk(
    capture: &AudioCapture,
    runner: &mut SegmentedRunner,
    cut_start_secs: &mut f64,
    cut_end_secs: f64,
) -> Result<(), ApiError> {
    let end = cut_end_secs.min(capture.elapsed_secs());
    if end - *cut_start_secs < MIN_SEGMENT_SECS {
        return Ok(());
    }

    let id = runner.alloc_id();
    let path = runner.segment_dir().join(format!("segment-{id:05}.wav"));
    capture
        .cut_wav(&path, *cut_start_secs, end)
        .map_err(|error| ApiError::Recording(error.to_string()))?;
    *cut_start_secs = end;
    runner.spawn_transcribe(id, path);
    Ok(())
}
