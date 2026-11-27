//! Dictation pipelines over the in-process capture buffer.
//!
//! Each session owns one continuous [`AudioCapture`]. Fixed and VAD-gated modes
//! slice WAV chunks directly from that buffer without restarting capture.

mod fixed;
mod segmented;
mod vad_gated;

use std::{collections::HashMap, path::PathBuf, sync::Arc, time::Duration};

use tokio::sync::{Mutex, oneshot};
use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, DaemonEvent, DictationCaptureRequest, DictationCaptureResult, LanguageProfile,
    SegmentationMode, SessionMode, SessionState, StartDictationRequest, TranscribeRequest,
    TranscriptionResult, default_runtime_dir,
};

use crate::{
    api::{AppState, error::ApiError},
    audio::{AudioCapture, write_wav},
    worker::WorkerCommand,
};

use self::segmented::{SegmentedContext, SegmentedOptions, SegmentedPlan};

/// Minimum chunk duration worth transcribing; shorter trailing tails are skipped.
pub(super) const MIN_SEGMENT_SECS: f64 = 0.2;

/// Handle held while a dictation capture is running.
pub enum ActiveDictation {
    OneShot(OneShotActive),
    Segmented(SegmentedActive),
}

/// Active state for a one-shot session.
pub struct OneShotActive {
    pub capture: AudioCapture,
    pub keep_audio: bool,
    pub translate_to_english: bool,
    pub language: Option<String>,
    pub provider_id: Option<String>,
}

/// Channel pair controlling a background segmented dictation pipeline.
pub struct SegmentedActive {
    pub stop_tx: oneshot::Sender<()>,
    pub result_rx: oneshot::Receiver<Result<DictationCaptureResult, ApiError>>,
}

/// Active dictation sessions keyed by `session_id`.
pub type ActiveDictations = Arc<Mutex<HashMap<Uuid, ActiveDictation>>>;

/// Per-session runtime directory storing temporary or retained capture WAVs.
pub fn dictation_dir(session_id: Uuid) -> PathBuf {
    default_runtime_dir()
        .join("dictation")
        .join(session_id.to_string())
}

pub(super) fn transition(mut session: CaptureSession, state: SessionState) -> CaptureSession {
    session.state = state;
    session
}

fn locked_language(profile: &Option<LanguageProfile>) -> Option<String> {
    profile.as_ref().and_then(|profile| {
        (profile.input_languages.len() == 1).then(|| profile.input_languages[0].clone())
    })
}

/// Start a dictation session according to `request.segmentation`.
pub async fn start_session(
    state: &AppState,
    request: StartDictationRequest,
) -> Result<CaptureSession, ApiError> {
    let options = SegmentedOptions {
        keep_audio: request.keep_audio,
        translate_to_english: request.translate_to_english,
        language: locked_language(&request.language_profile),
        provider_id: request.provider_id.clone(),
    };
    let session = transition(
        CaptureSession::new(
            SessionMode::Dictation,
            request.trigger,
            request.language_profile.unwrap_or_default(),
        ),
        SessionState::Listening,
    );
    let session_id = session.session_id;

    match request.segmentation {
        SegmentationMode::OneShot => {
            let capture = spawn_capture(state).await?;
            state.active.lock().await.insert(
                session_id,
                ActiveDictation::OneShot(OneShotActive {
                    capture,
                    keep_audio: options.keep_audio,
                    translate_to_english: options.translate_to_english,
                    language: options.language,
                    provider_id: options.provider_id,
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
            spawn_segmented(
                state,
                session.clone(),
                capture,
                SegmentedPlan::Fixed { segment_secs },
                options,
            )
            .await;
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
            spawn_segmented(
                state,
                session.clone(),
                capture,
                SegmentedPlan::VadGated {
                    probe_secs,
                    max_segment_secs,
                    silence_gap_probes,
                },
                options,
            )
            .await;
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

async fn spawn_segmented(
    state: &AppState,
    session: CaptureSession,
    capture: AudioCapture,
    plan: SegmentedPlan,
    options: SegmentedOptions,
) {
    let session_id = session.session_id;
    let (stop_tx, stop_rx) = oneshot::channel();
    let (result_tx, result_rx) = oneshot::channel();
    state.active.lock().await.insert(
        session_id,
        ActiveDictation::Segmented(SegmentedActive { stop_tx, result_rx }),
    );

    let context = SegmentedContext {
        options,
        worker: state.config.worker_command.clone(),
        events: state.events.clone(),
        sessions: state.sessions.clone(),
    };
    tokio::spawn(async move {
        let outcome = match plan {
            SegmentedPlan::Fixed { segment_secs } => {
                fixed::run(session, capture, segment_secs, context, stop_rx).await
            }
            SegmentedPlan::VadGated {
                probe_secs,
                max_segment_secs,
                silence_gap_probes,
            } => {
                vad_gated::run(
                    session,
                    capture,
                    vad_gated::VadParams {
                        probe_secs,
                        max_segment_secs,
                        silence_gap_probes,
                    },
                    context,
                    stop_rx,
                )
                .await
            }
        };
        let _ = result_tx.send(outcome);
    });
}

/// Stop an active dictation session, transcribe pending audio, and record terminal state.
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

/// Record for a fixed duration and return the transcription in one response.
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
            let failed = transition(session, SessionState::Failed);
            state.sessions.upsert(failed).await;
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
    let audio_file = dictation_dir(session.session_id).join("capture.wav");
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

fn cleanup_audio(path: &std::path::Path, keep_audio: bool) -> Option<()> {
    if keep_audio {
        Some(())
    } else {
        let _ = std::fs::remove_file(path);
        let _ = std::fs::remove_dir(path.parent().unwrap_or(path));
        None
    }
}
