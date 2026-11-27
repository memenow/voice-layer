//! Shared state and background transcription runner for segmented dictation modes.

use std::path::{Path, PathBuf};

use tokio::task::JoinSet;
use uuid::Uuid;
use voicelayer_core::{
    CaptureSession, DaemonEvent, DictationCaptureResult, SessionState, TranscribeRequest,
    TranscriptionResult,
};

use crate::{
    api::error::ApiError,
    events::EventBus,
    session::SessionStore,
    worker::{WorkerCallError, WorkerCommand},
};

use super::{dictation_dir, transition};

pub(super) struct SegmentedOptions {
    pub keep_audio: bool,
    pub translate_to_english: bool,
    pub language: Option<String>,
    pub provider_id: Option<String>,
}

/// Shared runtime dependencies passed into a segmented dictation task.
pub(super) struct SegmentedContext {
    pub options: SegmentedOptions,
    pub worker: WorkerCommand,
    pub events: EventBus,
    pub sessions: SessionStore,
}

pub(super) enum SegmentedPlan {
    Fixed {
        segment_secs: u32,
    },
    VadGated {
        probe_secs: u32,
        max_segment_secs: u32,
        silence_gap_probes: u32,
    },
}

struct SegmentRecord {
    id: u32,
    transcript: Option<String>,
    detected_language: Option<String>,
}

struct SegmentOutcome {
    segment_id: u32,
    result: Result<TranscriptionResult, WorkerCallError>,
}

/// Accumulates background transcription tasks and orders their results by segment id.
pub(super) struct SegmentedRunner {
    session_id: Uuid,
    segment_dir: PathBuf,
    options: SegmentedOptions,
    worker: WorkerCommand,
    events: EventBus,
    sessions: SessionStore,
    segments: Vec<SegmentRecord>,
    tasks: JoinSet<SegmentOutcome>,
    next_id: u32,
}

impl SegmentedRunner {
    pub(super) fn new(session_id: Uuid, context: SegmentedContext) -> Result<Self, ApiError> {
        let segment_dir = dictation_dir(session_id);
        std::fs::create_dir_all(&segment_dir).map_err(|error| {
            ApiError::Recording(format!("unable to create segment dir: {error}"))
        })?;
        Ok(Self {
            session_id,
            segment_dir,
            options: context.options,
            worker: context.worker,
            events: context.events,
            sessions: context.sessions,
            segments: Vec::new(),
            tasks: JoinSet::new(),
            next_id: 0,
        })
    }

    pub(super) fn segment_dir(&self) -> &Path {
        &self.segment_dir
    }

    pub(super) fn worker(&self) -> &WorkerCommand {
        &self.worker
    }

    pub(super) fn events(&self) -> &EventBus {
        &self.events
    }

    pub(super) fn session_id(&self) -> Uuid {
        self.session_id
    }

    pub(super) fn alloc_id(&mut self) -> u32 {
        let id = self.next_id;
        self.next_id += 1;
        id
    }

    /// Queue a recorded WAV chunk for asynchronous transcription.
    pub(super) fn spawn_transcribe(&mut self, id: u32, path: PathBuf) {
        self.segments.push(SegmentRecord {
            id,
            transcript: None,
            detected_language: None,
        });
        self.events.emit(DaemonEvent::SegmentRecorded {
            session_id: self.session_id,
            segment_id: id,
        });

        let worker = self.worker.clone();
        let events = self.events.clone();
        let session_id = self.session_id;
        let language = self.options.language.clone();
        let translate_to_english = self.options.translate_to_english;
        let provider_id = self.options.provider_id.clone();

        self.tasks.spawn(async move {
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

    /// Wait for all queued segment transcriptions and stitch transcripts in id order.
    pub(super) async fn finalize(
        mut self,
        session: CaptureSession,
    ) -> Result<DictationCaptureResult, ApiError> {
        let session = transition(session, SessionState::Transcribing);
        self.sessions.upsert(session.clone()).await;
        self.events.emit(DaemonEvent::DictationTranscribing {
            session_id: self.session_id,
        });

        while let Some(joined) = self.tasks.join_next().await {
            let Ok(outcome) = joined else {
                continue;
            };
            match outcome.result {
                Ok(result) => {
                    if let Some(segment) = self
                        .segments
                        .iter_mut()
                        .find(|segment| segment.id == outcome.segment_id)
                    {
                        segment.transcript = Some(result.text);
                        segment.detected_language = result.detected_language;
                    }
                }
                Err(error) => {
                    if !self.options.keep_audio {
                        let _ = std::fs::remove_dir_all(&self.segment_dir);
                    }
                    return Err(ApiError::Worker(error));
                }
            }
        }

        self.segments.sort_by_key(|segment| segment.id);
        let chunks: Vec<&str> = self
            .segments
            .iter()
            .filter_map(|segment| segment.transcript.as_deref())
            .map(str::trim)
            .filter(|chunk| !chunk.is_empty())
            .collect();
        let combined = stitch_transcript_chunks(&chunks);
        let detected_language = self
            .segments
            .iter()
            .find_map(|segment| segment.detected_language.clone());

        let audio_file = if self.options.keep_audio {
            Some(self.segment_dir.display().to_string())
        } else {
            let _ = std::fs::remove_dir_all(&self.segment_dir);
            None
        };

        Ok(DictationCaptureResult {
            session,
            transcription: TranscriptionResult {
                text: combined,
                detected_language,
                notes: vec![format!("{} segments stitched", self.segments.len())],
            },
            audio_file,
        })
    }
}

/// Join trimmed transcript segments, inserting a space across non-CJK boundaries
/// without adding artificial spaces between CJK ideographs.
fn stitch_transcript_chunks(chunks: &[&str]) -> String {
    let mut out = String::new();
    for chunk in chunks {
        if chunk.is_empty() {
            continue;
        }
        if let (Some(prev), Some(next)) = (out.chars().last(), chunk.chars().next())
            && !prev.is_whitespace()
            && !next.is_whitespace()
            && !(is_cjk(prev) && is_cjk(next))
        {
            out.push(' ');
        }
        out.push_str(chunk);
    }
    out
}

fn is_cjk(ch: char) -> bool {
    matches!(
        ch as u32,
        0x3000..=0x303F
            | 0x3040..=0x30FF
            | 0x3400..=0x4DBF
            | 0x4E00..=0x9FFF
            | 0xF900..=0xFAFF
            | 0xFF00..=0xFFEF
    )
}

#[cfg(test)]
mod tests {
    use super::stitch_transcript_chunks;

    #[test]
    fn stitch_inserts_spaces_for_latin_and_preserves_tight_cjk() {
        assert_eq!(
            stitch_transcript_chunks(&["Hello world", "how are you"]),
            "Hello world how are you"
        );
        assert_eq!(stitch_transcript_chunks(&["你好", "世界"]), "你好世界");
        assert_eq!(stitch_transcript_chunks(&["Hello", "世界"]), "Hello 世界");
    }
}
