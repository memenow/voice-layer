//! Typed lifecycle events streamed over `GET /v1/events/stream`.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::now_epoch_millis;

/// Daemon event payload. The `event_type` tag matches the SSE `event:` field.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
#[serde(tag = "event_type", rename_all = "snake_case")]
pub enum DaemonEvent {
    DictationSessionCreated {
        session_id: Uuid,
    },
    DictationListening {
        session_id: Uuid,
    },
    DictationTranscribing {
        session_id: Uuid,
    },
    DictationCompleted {
        session_id: Uuid,
        transcript_chars: usize,
    },
    DictationFailed {
        session_id: Uuid,
        detail: String,
    },
    DictationSegmentedStarted {
        session_id: Uuid,
        segment_secs: u32,
    },
    DictationVadGatedStarted {
        session_id: Uuid,
        probe_secs: u32,
        max_segment_secs: u32,
        silence_gap_probes: u32,
    },
    ProbeAnalyzed {
        session_id: Uuid,
        probe_id: u32,
        has_speech: bool,
        speech_ratio: f32,
    },
    SpeechUnitFlushed {
        session_id: Uuid,
        unit_id: u32,
    },
    SpeechUnitTranscribed {
        session_id: Uuid,
        unit_id: u32,
        transcript_chars: usize,
    },
    SegmentRecorded {
        session_id: Uuid,
        segment_id: u32,
    },
    SegmentTranscribed {
        session_id: Uuid,
        segment_id: u32,
        transcript_chars: usize,
    },
    SegmentTranscribeFailed {
        session_id: Uuid,
        segment_id: u32,
        detail: String,
    },
    ComposeJobCreated {
        title: String,
    },
    RewriteJobCreated {
        title: String,
    },
    TranslateJobCreated {
        title: String,
    },
    TranscriptionCompleted {
        transcript_chars: usize,
    },
    WorkerProvidersUnavailable {
        detail: String,
    },
    /// Synthesized when an SSE subscriber lags behind the broadcast channel.
    EventsLost {
        count: u64,
    },
}

impl DaemonEvent {
    /// Stable wire name used as the SSE `event:` field.
    pub fn name(&self) -> &'static str {
        match self {
            Self::DictationSessionCreated { .. } => "dictation_session_created",
            Self::DictationListening { .. } => "dictation_listening",
            Self::DictationTranscribing { .. } => "dictation_transcribing",
            Self::DictationCompleted { .. } => "dictation_completed",
            Self::DictationFailed { .. } => "dictation_failed",
            Self::DictationSegmentedStarted { .. } => "dictation_segmented_started",
            Self::DictationVadGatedStarted { .. } => "dictation_vad_gated_started",
            Self::ProbeAnalyzed { .. } => "probe_analyzed",
            Self::SpeechUnitFlushed { .. } => "speech_unit_flushed",
            Self::SpeechUnitTranscribed { .. } => "speech_unit_transcribed",
            Self::SegmentRecorded { .. } => "segment_recorded",
            Self::SegmentTranscribed { .. } => "segment_transcribed",
            Self::SegmentTranscribeFailed { .. } => "segment_transcribe_failed",
            Self::ComposeJobCreated { .. } => "compose_job_created",
            Self::RewriteJobCreated { .. } => "rewrite_job_created",
            Self::TranslateJobCreated { .. } => "translate_job_created",
            Self::TranscriptionCompleted { .. } => "transcription_completed",
            Self::WorkerProvidersUnavailable { .. } => "worker_providers_unavailable",
            Self::EventsLost { .. } => "events_lost",
        }
    }
}

/// Timestamped wrapper around a [`DaemonEvent`] emitted on the SSE stream.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct EventEnvelope {
    pub created_at_millis: u64,
    #[serde(flatten)]
    pub event: DaemonEvent,
}

impl EventEnvelope {
    /// Stamp `event` with the current epoch-millisecond timestamp.
    pub fn new(event: DaemonEvent) -> Self {
        Self {
            created_at_millis: now_epoch_millis(),
            event,
        }
    }
}
