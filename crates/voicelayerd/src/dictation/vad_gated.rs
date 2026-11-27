//! Silero-VAD gated dictation pipeline.

use std::time::Duration;

use tokio::{
    sync::oneshot,
    time::{MissedTickBehavior, interval},
};
use voicelayer_core::{
    CaptureSession, DaemonEvent, DictationCaptureResult, SegmentProbeRequest, SegmentProbeResult,
};

use crate::{api::error::ApiError, audio::AudioCapture, worker::WorkerCommand};

use super::{
    MIN_SEGMENT_SECS,
    segmented::{SegmentedContext, SegmentedRunner},
};

pub(super) struct VadParams {
    pub probe_secs: u32,
    pub max_segment_secs: u32,
    pub silence_gap_probes: u32,
}

pub(super) async fn run(
    session: CaptureSession,
    capture: AudioCapture,
    params: VadParams,
    context: SegmentedContext,
    mut stop_rx: oneshot::Receiver<()>,
) -> Result<DictationCaptureResult, ApiError> {
    let mut runner = SegmentedRunner::new(session.session_id, context)?;
    let mut ticker = interval(Duration::from_secs(u64::from(params.probe_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.tick().await;

    let mut probe_index: u32 = 0;
    let mut pending: Option<(f64, f64)> = None;
    let mut silent_streak: u32 = 0;
    let mut stopped = false;

    while !stopped {
        tokio::select! {
            biased;
            _ = &mut stop_rx => { stopped = true; }
            _ = ticker.tick() => {}
        }

        let probe_id = probe_index;
        let probe_start = f64::from(probe_id) * f64::from(params.probe_secs);
        let probe_end = probe_start + f64::from(params.probe_secs);
        probe_index += 1;

        let elapsed = capture.elapsed_secs();
        if elapsed < probe_end - MIN_SEGMENT_SECS && stopped {
            if elapsed - probe_start >= MIN_SEGMENT_SECS {
                pending = Some(match pending {
                    Some((start, _)) => (start, elapsed),
                    None => (probe_start, elapsed),
                });
            }
            flush_unit(&capture, &mut runner, &mut pending)?;
            break;
        }

        let probe_path = runner
            .segment_dir()
            .join(format!("probe-{probe_id:05}.wav"));
        capture
            .cut_wav(&probe_path, probe_start, probe_end.min(elapsed))
            .map_err(|error| ApiError::Recording(error.to_string()))?;

        let verdict = if stopped {
            None
        } else {
            Some(classify_probe(runner.worker(), &probe_path).await?)
        };
        let _ = std::fs::remove_file(&probe_path);

        let has_speech = verdict.as_ref().map(|v| v.has_speech).unwrap_or(true);
        if let Some(verdict) = &verdict {
            runner.events().emit(DaemonEvent::ProbeAnalyzed {
                session_id: runner.session_id(),
                probe_id,
                has_speech: verdict.has_speech,
                speech_ratio: verdict.speech_ratio,
            });
        }

        if has_speech {
            silent_streak = 0;
            pending = Some(match pending {
                Some((start, _)) => (start, probe_end),
                None => (probe_start, probe_end),
            });
            if let Some((start, end)) = pending
                && end - start >= f64::from(params.max_segment_secs)
            {
                flush_unit(&capture, &mut runner, &mut pending)?;
            }
        } else {
            silent_streak += 1;
            if pending.is_some() && silent_streak >= params.silence_gap_probes.max(1) {
                flush_unit(&capture, &mut runner, &mut pending)?;
            }
        }

        if stopped {
            flush_unit(&capture, &mut runner, &mut pending)?;
            break;
        }
    }
    drop(capture);

    runner.finalize(session).await
}

fn flush_unit(
    capture: &AudioCapture,
    runner: &mut SegmentedRunner,
    pending: &mut Option<(f64, f64)>,
) -> Result<(), ApiError> {
    let Some((start, end)) = pending.take() else {
        return Ok(());
    };
    if end - start < MIN_SEGMENT_SECS {
        return Ok(());
    }
    let id = runner.alloc_id();
    let path = runner.segment_dir().join(format!("unit-{id:05}.wav"));
    capture
        .cut_wav(&path, start, end.min(capture.elapsed_secs()))
        .map_err(|error| ApiError::Recording(error.to_string()))?;
    runner.events().emit(DaemonEvent::SpeechUnitFlushed {
        session_id: runner.session_id(),
        unit_id: id,
    });
    runner.spawn_transcribe(id, path);
    Ok(())
}

async fn classify_probe(
    worker: &WorkerCommand,
    path: &std::path::Path,
) -> Result<SegmentProbeResult, ApiError> {
    worker
        .segment_probe(&SegmentProbeRequest {
            audio_file: path.display().to_string(),
        })
        .await
        .map_err(ApiError::Worker)
}
