//! Dictation pipelines over the in-process capture buffer.
//!
//! All capture flows share one continuous [`AudioCapture`] per session.
//! Fixed and VAD-gated segmentation cut WAV chunks out of that buffer — no
//! recorder subprocess is ever rolled, so no audio is lost at boundaries
//! and probes/units carry exact time windows.

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use tokio::{
    sync::{Mutex, oneshot},
    task::JoinSet,
    time::{MissedTickBehavior, interval},
};
use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, DaemonEvent, DictationCaptureRequest, DictationCaptureResult, LanguageProfile,
    SegmentProbeRequest, SegmentationMode, SessionMode, SessionState, StartDictationRequest,
    TranscribeRequest, TranscriptionResult, default_runtime_dir,
};

use crate::{
    api::{AppState, error::ApiError},
    audio::{AudioCapture, write_wav},
    events::EventBus,
    session::SessionStore,
    worker::WorkerCommand,
};

/// Minimum chunk worth transcribing; shorter tails are skipped.
const MIN_SEGMENT_SECS: f64 = 0.2;

pub enum ActiveDictation {
    OneShot(OneShotActive),
    Segmented(SegmentedActive),
}

pub struct OneShotActive {
    pub capture: AudioCapture,
    pub keep_audio: bool,
    pub translate_to_english: bool,
    pub language: Option<String>,
    pub provider_id: Option<String>,
}

pub struct SegmentedActive {
    pub stop_tx: oneshot::Sender<()>,
    pub result_rx: oneshot::Receiver<Result<DictationCaptureResult, ApiError>>,
}

pub type ActiveDictations = Arc<Mutex<HashMap<Uuid, ActiveDictation>>>;

pub fn dictation_dir(session_id: Uuid) -> PathBuf {
    default_runtime_dir()
        .join("dictation")
        .join(session_id.to_string())
}

fn locked_language(profile: &Option<LanguageProfile>) -> Option<String> {
    profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    })
}

fn transition(mut session: CaptureSession, state: SessionState) -> CaptureSession {
    session.state = state;
    session
}

pub async fn start_session(
    state: &AppState,
    request: StartDictationRequest,
) -> Result<CaptureSession, ApiError> {
    let language = locked_language(&request.language_profile);
    let provider_id = request.provider_id.clone();
    let keep_audio = request.keep_audio;
    let translate_to_english = request.translate_to_english;
    let session = CaptureSession::new(
        SessionMode::Dictation,
        request.trigger,
        request.language_profile.unwrap_or_default(),
    );
    let session = transition(session, SessionState::Listening);
    let session_id = session.session_id;

    match request.segmentation {
        SegmentationMode::OneShot => {
            let capture = spawn_capture(state).await?;
            state.active.lock().await.insert(
                session_id,
                ActiveDictation::OneShot(OneShotActive {
                    capture,
                    keep_audio,
                    translate_to_english,
                    language,
                    provider_id,
                }),
            );
            state
                .events
                .emit(DaemonEvent::DictationSessionCreated { session_id });
        }
        SegmentationMode::Fixed { segment_secs } => {
            if segment_secs == 0 {
                return Err(ApiError::BadRequest(
                    "segmentation.segment_secs must be greater than zero".to_owned(),
                ));
            }
            let capture = spawn_capture(state).await?;
            let plan = SegmentationPlan::Fixed { segment_secs };
            spawn_segmented(
                state,
                session.clone(),
                capture,
                plan,
                keep_audio,
                translate_to_english,
                language.clone(),
                provider_id.clone(),
            )
            .await?;
            state.events.emit(DaemonEvent::DictationSegmentedStarted {
                session_id,
                segment_secs,
            });
            state
                .events
                .emit(DaemonEvent::DictationSessionCreated { session_id });
        }
        SegmentationMode::VadGated {
            probe_secs,
            max_segment_secs,
            silence_gap_probes,
        } => {
            if probe_secs == 0 || max_segment_secs == 0 {
                return Err(ApiError::BadRequest(
                    "vad_gated segmentation requires probe_secs and max_segment_secs above zero"
                        .to_owned(),
                ));
            }
            let capture = spawn_capture(state).await?;
            let plan = SegmentationPlan::VadGated {
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
            };
            spawn_segmented(
                state,
                session.clone(),
                capture,
                plan,
                keep_audio,
                translate_to_english,
                language.clone(),
                provider_id.clone(),
            )
            .await?;
            state.events.emit(DaemonEvent::DictationVadGatedStarted {
                session_id,
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
            });
            state
                .events
                .emit(DaemonEvent::DictationSessionCreated { session_id });
        }
    }

    state.sessions.upsert(session.clone()).await;
    Ok(session)
}

enum SegmentationPlan {
    Fixed {
        segment_secs: u32,
    },
    VadGated {
        probe_secs: u32,
        max_segment_secs: u32,
        silence_gap_probes: u32,
    },
}

#[allow(clippy::too_many_arguments)]
async fn spawn_segmented(
    state: &AppState,
    session: CaptureSession,
    capture: AudioCapture,
    plan: SegmentationPlan,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
    provider_id: Option<String>,
) -> Result<(), ApiError> {
    let session_id = session.session_id;
    let (stop_tx, stop_rx) = oneshot::channel();
    let (result_tx, result_rx) = oneshot::channel();
    state.active.lock().await.insert(
        session_id,
        ActiveDictation::Segmented(SegmentedActive { stop_tx, result_rx }),
    );

    let worker = state.config.worker_command.clone();
    let events = state.events.clone();
    let sessions = state.sessions.clone();
    let task_session = session.clone();
    tokio::spawn(async move {
        let outcome = match plan {
            SegmentationPlan::Fixed { segment_secs } => {
                run_fixed_session(FixedSessionArgs {
                    session: task_session,
                    capture,
                    segment_secs,
                    keep_audio,
                    translate_to_english,
                    language,
                    provider_id,
                    worker,
                    events,
                    sessions,
                    stop_rx,
                })
                .await
            }
            SegmentationPlan::VadGated {
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
            } => {
                run_vad_gated_session(VadGatedSessionArgs {
                    session: task_session,
                    capture,
                    probe_secs,
                    max_segment_secs,
                    silence_gap_probes,
                    keep_audio,
                    translate_to_english,
                    language,
                    provider_id,
                    worker,
                    events,
                    sessions,
                    stop_rx,
                })
                .await
            }
        };
        let _ = result_tx.send(outcome);
    });
    Ok(())
}

pub async fn stop_session(
    state: &AppState,
    session_id: Uuid,
) -> Result<DictationCaptureResult, ApiError> {
    let active = state.active.lock().await.remove(&session_id);
    let session = state.sessions.get(session_id).await;

    let (mut session, active) = match (session, active) {
        (Some(session), Some(active)) => (session, active),
        _ => return Err(ApiError::SessionNotFound(session_id)),
    };

    match active {
        ActiveDictation::OneShot(active) => {
            session = transition(session, SessionState::Transcribing);
            state.sessions.upsert(session.clone()).await;
            state
                .events
                .emit(DaemonEvent::DictationTranscribing { session_id });

            let result = finish_oneshot(state, session, active).await;
            record_terminal(state, session_id, &result).await;
            result
        }
        ActiveDictation::Segmented(active) => {
            let _ = active.stop_tx.send(());
            let result = active.result_rx.await.map_err(|_| {
                ApiError::Internal("segmented dictation task ended without a result".to_owned())
            })?;
            record_terminal(state, session_id, &result).await;
            result
        }
    }
}

async fn record_terminal(
    state: &AppState,
    session_id: Uuid,
    result: &Result<DictationCaptureResult, ApiError>,
) {
    if let Some(mut session) = state.sessions.get(session_id).await {
        session = match result {
            Ok(_) => transition(session, SessionState::Completed),
            Err(_) => transition(session, SessionState::Failed),
        };
        state.sessions.upsert(session).await;
    }
    match result {
        Ok(result) => state.events.emit(DaemonEvent::DictationCompleted {
            session_id,
            transcript_chars: result.transcription.text.chars().count(),
        }),
        Err(error) => state.events.emit(DaemonEvent::DictationFailed {
            session_id,
            detail: error.to_string(),
        }),
    }
}

async fn finish_oneshot(
    state: &AppState,
    session: CaptureSession,
    active: OneShotActive,
) -> Result<DictationCaptureResult, ApiError> {
    let session_id = session.session_id;
    let audio_file = dictation_dir(session_id).join("capture.wav");
    let samples = spawn_stop(active.capture).await?;
    write_wav(&audio_file, &samples).map_err(|error| ApiError::Recording(error.to_string()))?;

    let transcription = match transcribe_file(
        &state.config.worker_command,
        &audio_file,
        active.language.clone(),
        active.translate_to_english,
        active.provider_id.clone(),
    )
    .await
    {
        Ok(transcription) => transcription,
        Err(error) => {
            cleanup_audio(&audio_file, active.keep_audio);
            return Err(error);
        }
    };

    let audio_file = cleanup_audio(&audio_file, active.keep_audio).map(|_| audio_file);
    Ok(DictationCaptureResult {
        session,
        transcription,
        audio_file: audio_file.map(|path| path.display().to_string()),
    })
}

/// `POST /v1/dictation/capture`: bounded-duration record + transcribe.
pub async fn capture_once(
    state: &AppState,
    request: DictationCaptureRequest,
) -> Result<DictationCaptureResult, ApiError> {
    if request.duration_seconds == 0 {
        return Err(ApiError::BadRequest(
            "duration_seconds must be greater than zero".to_owned(),
        ));
    }
    let language = locked_language(&request.language_profile);
    let session = CaptureSession::new(
        SessionMode::Dictation,
        request.trigger,
        request.language_profile.unwrap_or_default(),
    );
    let session_id = session.session_id;
    state.sessions.upsert(session.clone()).await;
    state
        .events
        .emit(DaemonEvent::DictationSessionCreated { session_id });

    let session = transition(session, SessionState::Listening);
    state.sessions.upsert(session.clone()).await;
    state
        .events
        .emit(DaemonEvent::DictationListening { session_id });

    let capture = spawn_capture(state).await?;
    tokio::time::sleep(Duration::from_secs(u64::from(request.duration_seconds))).await;
    let samples = spawn_stop(capture).await?;

    let session = transition(session, SessionState::Transcribing);
    state.sessions.upsert(session.clone()).await;
    state
        .events
        .emit(DaemonEvent::DictationTranscribing { session_id });

    let audio_file = dictation_dir(session_id).join("capture.wav");
    write_wav(&audio_file, &samples).map_err(|error| ApiError::Recording(error.to_string()))?;
    let transcription = match transcribe_file(
        &state.config.worker_command,
        &audio_file,
        language,
        request.translate_to_english,
        request.provider_id.clone(),
    )
    .await
    {
        Ok(transcription) => transcription,
        Err(error) => {
            let session = transition(session, SessionState::Failed);
            state.sessions.upsert(session).await;
            state.events.emit(DaemonEvent::DictationFailed {
                session_id,
                detail: error.to_string(),
            });
            cleanup_audio(&audio_file, request.keep_audio);
            return Err(error);
        }
    };

    let session = transition(session, SessionState::Completed);
    state.sessions.upsert(session.clone()).await;
    state.events.emit(DaemonEvent::DictationCompleted {
        session_id,
        transcript_chars: transcription.text.chars().count(),
    });

    let audio_file = cleanup_audio(&audio_file, request.keep_audio).map(|_| audio_file);
    Ok(DictationCaptureResult {
        session,
        transcription,
        audio_file: audio_file.map(|path| path.display().to_string()),
    })
}

struct SegmentRecord {
    id: u32,
    transcript: Option<String>,
    detected_language: Option<String>,
}

struct SegmentOutcome {
    segment_id: u32,
    result: Result<TranscriptionResult, crate::worker::WorkerCallError>,
}

struct FixedSessionArgs {
    session: CaptureSession,
    capture: AudioCapture,
    segment_secs: u32,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
    provider_id: Option<String>,
    worker: WorkerCommand,
    events: EventBus,
    sessions: SessionStore,
    stop_rx: oneshot::Receiver<()>,
}

async fn run_fixed_session(args: FixedSessionArgs) -> Result<DictationCaptureResult, ApiError> {
    let FixedSessionArgs {
        mut session,
        capture,
        segment_secs,
        keep_audio,
        translate_to_english,
        language,
        provider_id,
        worker,
        events,
        sessions,
        mut stop_rx,
    } = args;
    let session_id = session.session_id;
    let segment_dir = dictation_dir(session_id);
    std::fs::create_dir_all(&segment_dir)
        .map_err(|error| ApiError::Recording(format!("unable to create segment dir: {error}")))?;

    let mut segments: Vec<SegmentRecord> = Vec::new();
    let mut tasks: JoinSet<SegmentOutcome> = JoinSet::new();
    let mut ticker = interval(Duration::from_secs(u64::from(segment_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    // The first tick fires immediately; skip it so the first cut happens
    // after `segment_secs` seconds of actual capture.
    ticker.tick().await;

    let mut next_segment_id: u32 = 0;
    let mut cut_start_secs = 0.0_f64;

    loop {
        tokio::select! {
            biased;
            _ = &mut stop_rx => break,
            _ = ticker.tick() => {
                let cut_end_secs = cut_start_secs + f64::from(segment_secs);
                cut_chunk(
                    &capture,
                    &segment_dir,
                    &mut tasks,
                    &worker,
                    &events,
                    session_id,
                    &mut next_segment_id,
                    &mut segments,
                    &mut cut_start_secs,
                    cut_end_secs,
                    language.clone(),
                    translate_to_english,
                    provider_id.clone(),
                )?;
            }
        }
    }

    // Final cut: everything still in the buffer since the last boundary.
    let elapsed = capture.elapsed_secs();
    cut_chunk(
        &capture,
        &segment_dir,
        &mut tasks,
        &worker,
        &events,
        session_id,
        &mut next_segment_id,
        &mut segments,
        &mut cut_start_secs,
        elapsed,
        language.clone(),
        translate_to_english,
        provider_id.clone(),
    )?;
    drop(capture);

    session = transition(session, SessionState::Transcribing);
    sessions.upsert(session.clone()).await;
    events.emit(DaemonEvent::DictationTranscribing { session_id });

    while let Some(outcome) = tasks.join_next().await {
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => continue,
        };
        let segment = segments
            .iter_mut()
            .find(|segment| segment.id == outcome.segment_id);
        match outcome.result {
            Ok(result) => {
                if let Some(segment) = segment {
                    segment.transcript = Some(result.text);
                    segment.detected_language = result.detected_language;
                }
            }
            Err(error) => {
                return Err(ApiError::Worker(error));
            }
        }
    }

    finish_segmented(session, segments, &segment_dir, keep_audio)
}

struct VadGatedSessionArgs {
    session: CaptureSession,
    capture: AudioCapture,
    probe_secs: u32,
    max_segment_secs: u32,
    silence_gap_probes: u32,
    keep_audio: bool,
    translate_to_english: bool,
    language: Option<String>,
    provider_id: Option<String>,
    worker: WorkerCommand,
    events: EventBus,
    sessions: SessionStore,
    stop_rx: oneshot::Receiver<()>,
}

/// VAD-gated segmentation over the continuous capture buffer.
///
/// Every `probe_secs` a probe WAV is cut from the buffer and classified by
/// the worker's `segment_probe` RPC. Speech probes extend the pending
/// window; after `silence_gap_probes` consecutive silent probes the pending
/// window is flushed as one speech unit; a window reaching
/// `max_segment_secs` flushes immediately. Flushed units are transcribed in
/// the background like `Fixed` chunks.
async fn run_vad_gated_session(
    args: VadGatedSessionArgs,
) -> Result<DictationCaptureResult, ApiError> {
    let VadGatedSessionArgs {
        mut session,
        capture,
        probe_secs,
        max_segment_secs,
        silence_gap_probes,
        keep_audio,
        translate_to_english,
        language,
        provider_id,
        worker,
        events,
        sessions,
        mut stop_rx,
    } = args;
    let session_id = session.session_id;
    let segment_dir = dictation_dir(session_id);
    std::fs::create_dir_all(&segment_dir)
        .map_err(|error| ApiError::Recording(format!("unable to create segment dir: {error}")))?;

    let mut segments: Vec<SegmentRecord> = Vec::new();
    let mut tasks: JoinSet<SegmentOutcome> = JoinSet::new();
    let mut ticker = interval(Duration::from_secs(u64::from(probe_secs)));
    ticker.set_missed_tick_behavior(MissedTickBehavior::Delay);
    ticker.tick().await;

    let mut next_id: u32 = 0;
    let mut probe_index: u32 = 0;
    // Window covered by the pending speech run, in capture-buffer seconds.
    let mut pending: Option<(f64, f64)> = None;
    let mut silent_streak: u32 = 0;
    let mut stopped = false;

    while !stopped {
        tokio::select! {
            biased;
            _ = &mut stop_rx => { stopped = true; }
            _ = ticker.tick() => {}
        }

        let probe_start = f64::from(probe_index) * f64::from(probe_secs);
        let probe_end = probe_start + f64::from(probe_secs);
        probe_index += 1;

        if capture.elapsed_secs() < probe_end - MIN_SEGMENT_SECS && stopped {
            break;
        }
        if capture.elapsed_secs() < probe_end - MIN_SEGMENT_SECS {
            // Stop arrived before a full probe accumulated; take the tail.
        }

        let probe_id = next_id;
        let probe_path = segment_dir.join(format!("probe-{probe_id:05}.wav"));
        capture
            .cut_wav(
                &probe_path,
                probe_start,
                probe_end.min(capture.elapsed_secs()),
            )
            .map_err(|error| ApiError::Recording(error.to_string()))?;

        let verdict = if stopped {
            // No point probing the tail when stopping; treat as speech so
            // the pending window and the tail itself are flushed below.
            None
        } else {
            Some(classify_probe(&worker, &probe_path).await?)
        };

        let has_speech = verdict.as_ref().map(|v| v.has_speech).unwrap_or(true);
        if let Some(verdict) = &verdict {
            events.emit(DaemonEvent::ProbeAnalyzed {
                session_id,
                probe_id,
                has_speech: verdict.has_speech,
                speech_ratio: verdict.speech_ratio,
            });
        }
        let _ = std::fs::remove_file(&probe_path);

        if has_speech {
            silent_streak = 0;
            pending = Some(match pending {
                Some((start, _)) => (start, probe_end),
                None => (probe_start, probe_end),
            });
            if let Some((start, end)) = pending
                && end - start >= f64::from(max_segment_secs)
            {
                flush_unit(
                    &capture,
                    &segment_dir,
                    &mut tasks,
                    &worker,
                    &events,
                    session_id,
                    &mut next_id,
                    &mut segments,
                    &mut pending,
                    language.clone(),
                    translate_to_english,
                    provider_id.clone(),
                )?;
            }
        } else {
            silent_streak += 1;
            if pending.is_some() && silent_streak >= silence_gap_probes.max(1) {
                flush_unit(
                    &capture,
                    &segment_dir,
                    &mut tasks,
                    &worker,
                    &events,
                    session_id,
                    &mut next_id,
                    &mut segments,
                    &mut pending,
                    language.clone(),
                    translate_to_english,
                    provider_id.clone(),
                )?;
            }
        }

        if stopped {
            // Flush any pending speech window at stop.
            flush_unit(
                &capture,
                &segment_dir,
                &mut tasks,
                &worker,
                &events,
                session_id,
                &mut next_id,
                &mut segments,
                &mut pending,
                language.clone(),
                translate_to_english,
                provider_id.clone(),
            )?;
            break;
        }
    }
    drop(capture);

    session = transition(session, SessionState::Transcribing);
    sessions.upsert(session.clone()).await;
    events.emit(DaemonEvent::DictationTranscribing { session_id });

    while let Some(outcome) = tasks.join_next().await {
        let outcome = match outcome {
            Ok(outcome) => outcome,
            Err(_) => continue,
        };
        let segment = segments
            .iter_mut()
            .find(|segment| segment.id == outcome.segment_id);
        match outcome.result {
            Ok(result) => {
                if let Some(segment) = segment {
                    segment.transcript = Some(result.text);
                    segment.detected_language = result.detected_language;
                }
            }
            Err(error) => {
                return Err(ApiError::Worker(error));
            }
        }
    }

    finish_segmented(session, segments, &segment_dir, keep_audio)
}

#[allow(clippy::too_many_arguments)]
fn flush_unit(
    capture: &AudioCapture,
    segment_dir: &std::path::Path,
    tasks: &mut JoinSet<SegmentOutcome>,
    worker: &WorkerCommand,
    events: &EventBus,
    session_id: Uuid,
    next_id: &mut u32,
    segments: &mut Vec<SegmentRecord>,
    pending: &mut Option<(f64, f64)>,
    language: Option<String>,
    translate_to_english: bool,
    provider_id: Option<String>,
) -> Result<(), ApiError> {
    let Some((start, end)) = pending.take() else {
        return Ok(());
    };
    if end - start < MIN_SEGMENT_SECS {
        return Ok(());
    }
    let id = *next_id;
    *next_id += 1;
    let path = segment_dir.join(format!("unit-{id:05}.wav"));
    capture
        .cut_wav(&path, start, end.min(capture.elapsed_secs()))
        .map_err(|error| ApiError::Recording(error.to_string()))?;
    events.emit(DaemonEvent::SpeechUnitFlushed {
        session_id,
        unit_id: id,
    });
    spawn_transcribe(
        tasks,
        worker,
        events,
        session_id,
        id,
        segments,
        path,
        language,
        translate_to_english,
        provider_id,
    );
    Ok(())
}

async fn classify_probe(
    worker: &WorkerCommand,
    path: &std::path::Path,
) -> Result<voicelayer_core::SegmentProbeResult, ApiError> {
    worker
        .segment_probe(&SegmentProbeRequest {
            audio_file: path.display().to_string(),
        })
        .await
        .map_err(ApiError::Worker)
}

#[allow(clippy::too_many_arguments)]
fn cut_chunk(
    capture: &AudioCapture,
    segment_dir: &std::path::Path,
    tasks: &mut JoinSet<SegmentOutcome>,
    worker: &WorkerCommand,
    events: &EventBus,
    session_id: Uuid,
    next_segment_id: &mut u32,
    segments: &mut Vec<SegmentRecord>,
    cut_start_secs: &mut f64,
    cut_end_secs: f64,
    language: Option<String>,
    translate_to_english: bool,
    provider_id: Option<String>,
) -> Result<(), ApiError> {
    let available = capture.elapsed_secs();
    let end = cut_end_secs.min(available);
    if end - *cut_start_secs < MIN_SEGMENT_SECS {
        return Ok(());
    }

    let id = *next_segment_id;
    *next_segment_id += 1;
    let path = segment_dir.join(format!("segment-{id:05}.wav"));
    // Windows are exactly [cut_start, end]: the buffer is one continuous
    // stream, so no overlap is needed to avoid losing audio, and including
    // it would transcribe the boundary audio twice.
    capture
        .cut_wav(&path, *cut_start_secs, end)
        .map_err(|error| ApiError::Recording(error.to_string()))?;
    *cut_start_secs = end;

    spawn_transcribe(
        tasks,
        worker,
        events,
        session_id,
        id,
        segments,
        path,
        language,
        translate_to_english,
        provider_id,
    );
    Ok(())
}

#[allow(clippy::too_many_arguments)]
fn spawn_transcribe(
    tasks: &mut JoinSet<SegmentOutcome>,
    worker: &WorkerCommand,
    events: &EventBus,
    session_id: Uuid,
    id: u32,
    segments: &mut Vec<SegmentRecord>,
    path: PathBuf,
    language: Option<String>,
    translate_to_english: bool,
    provider_id: Option<String>,
) {
    segments.push(SegmentRecord {
        id,
        transcript: None,
        detected_language: None,
    });
    events.emit(DaemonEvent::SegmentRecorded {
        session_id,
        segment_id: id,
    });

    let worker = worker.clone();
    let events = events.clone();
    tasks.spawn(async move {
        let request = TranscribeRequest {
            audio_file: path.display().to_string(),
            language,
            translate_to_english,
            provider_id,
        };
        let result = worker.transcribe(&request).await;
        match &result {
            Ok(transcription) => {
                let transcript_chars = transcription.text.chars().count();
                events.emit(DaemonEvent::SegmentTranscribed {
                    session_id,
                    segment_id: id,
                    transcript_chars,
                });
                events.emit(DaemonEvent::SpeechUnitTranscribed {
                    session_id,
                    unit_id: id,
                    transcript_chars,
                });
            }
            Err(error) => events.emit(DaemonEvent::SegmentTranscribeFailed {
                session_id,
                segment_id: id,
                detail: error.to_string(),
            }),
        }
        SegmentOutcome {
            segment_id: id,
            result,
        }
    });
}

fn finish_segmented(
    session: CaptureSession,
    mut segments: Vec<SegmentRecord>,
    segment_dir: &std::path::Path,
    keep_audio: bool,
) -> Result<DictationCaptureResult, ApiError> {
    segments.sort_by_key(|segment| segment.id);
    // Preserve whisper's native leading whitespace on Latin transcripts
    // (it already separates words) and concatenate CJK transcripts without
    // adding artificial spaces. A final trim strips edge whitespace.
    let combined: String = segments
        .iter()
        .filter_map(|segment| segment.transcript.as_deref())
        .filter(|chunk| !chunk.is_empty())
        .collect();
    let detected_language = segments
        .iter()
        .find_map(|segment| segment.detected_language.clone());

    // With `keep_audio` the operator wants the whole segment set; return
    // the directory path so `ls` lists every segment in order.
    let audio_file = if keep_audio {
        Some(segment_dir.display().to_string())
    } else {
        let _ = std::fs::remove_dir_all(segment_dir);
        None
    };

    Ok(DictationCaptureResult {
        session,
        transcription: TranscriptionResult {
            text: combined.trim().to_owned(),
            detected_language,
            notes: vec![format!("{} segments stitched", segments.len())],
        },
        audio_file,
    })
}

async fn spawn_capture(state: &AppState) -> Result<AudioCapture, ApiError> {
    if let Some((duration, rate)) = state.test_audio_silence {
        return Ok(AudioCapture::synthetic_silence(duration, rate));
    }
    tokio::task::spawn_blocking(AudioCapture::start)
        .await
        .map_err(|_| ApiError::Internal("audio capture task panicked".to_owned()))?
        .map_err(|error| ApiError::Recording(error.to_string()))
}

async fn spawn_stop(capture: AudioCapture) -> Result<Vec<f32>, ApiError> {
    tokio::task::spawn_blocking(move || capture.stop())
        .await
        .map_err(|_| ApiError::Internal("audio capture task panicked".to_owned()))?
        .map_err(|error| ApiError::Recording(error.to_string()))
}

async fn transcribe_file(
    worker: &WorkerCommand,
    audio_file: &std::path::Path,
    language: Option<String>,
    translate_to_english: bool,
    provider_id: Option<String>,
) -> Result<TranscriptionResult, ApiError> {
    worker
        .transcribe(&TranscribeRequest {
            audio_file: audio_file.display().to_string(),
            language,
            translate_to_english,
            provider_id,
        })
        .await
        .map_err(ApiError::Worker)
}

/// Returns `Some(())` when the audio is kept, `None` after deleting it.
fn cleanup_audio(path: &std::path::Path, keep_audio: bool) -> Option<()> {
    if keep_audio {
        Some(())
    } else {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(path.parent().unwrap_or(path));
        None
    }
}
