//! Dictation capture, segmentation, transcription, and worker audio RPC payloads.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::{CaptureSession, LanguageProfile, TriggerKind};

/// Request body for `POST /v1/sessions/dictation`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StartDictationRequest {
    pub trigger: TriggerKind,
    #[serde(default)]
    pub language_profile: Option<LanguageProfile>,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
    #[serde(default)]
    pub segmentation: SegmentationMode,
    /// Optional ASR provider id shared across every transcription in this session.
    #[serde(default)]
    pub provider_id: Option<String>,
}

/// Strategy for segmenting live microphone capture during dictation.
#[derive(Debug, Clone, Copy, Default, Serialize, Deserialize, PartialEq, Eq)]
#[serde(tag = "mode", rename_all = "snake_case")]
pub enum SegmentationMode {
    /// Record continuously from start to stop and transcribe once on stop.
    #[default]
    OneShot,
    /// Cut continuous capture into fixed-length chunks and transcribe in the background.
    Fixed { segment_secs: u32 },
    /// Probe capture with Silero VAD every `probe_secs` and flush speech units on silence or cap.
    VadGated {
        probe_secs: u32,
        max_segment_secs: u32,
        #[serde(default = "default_silence_gap_probes")]
        silence_gap_probes: u32,
    },
}

fn default_silence_gap_probes() -> u32 {
    1
}

/// Request body for `POST /v1/dictation/capture`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureRequest {
    pub trigger: TriggerKind,
    #[serde(default)]
    pub language_profile: Option<LanguageProfile>,
    pub duration_seconds: u32,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub keep_audio: bool,
    #[serde(default)]
    pub provider_id: Option<String>,
}

/// Response for completed dictation capture and `POST /v1/sessions/dictation/stop`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct DictationCaptureResult {
    pub session: CaptureSession,
    pub transcription: TranscriptionResult,
    pub audio_file: Option<String>,
}

/// Request body for `POST /v1/sessions/dictation/stop`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StopDictationRequest {
    pub session_id: Uuid,
}

/// Request body for `POST /v1/transcriptions`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscribeRequest {
    pub audio_file: String,
    #[serde(default)]
    pub language: Option<String>,
    #[serde(default)]
    pub translate_to_english: bool,
    #[serde(default)]
    pub provider_id: Option<String>,
}

/// Transcription result returned by the Python worker and daemon endpoints.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranscriptionResult {
    pub text: String,
    pub detected_language: Option<String>,
    pub notes: Vec<String>,
}

/// Speech interval in seconds from the start of a probed WAV file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SpeechRegion {
    pub start_secs: f32,
    pub end_secs: f32,
}

/// Request payload for the worker `segment_probe` JSON-RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct SegmentProbeRequest {
    pub audio_file: String,
}

/// Response payload for the worker `segment_probe` JSON-RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct SegmentProbeResult {
    pub has_speech: bool,
    pub speech_ratio: f32,
    pub regions: Vec<SpeechRegion>,
    #[serde(default)]
    pub notes: Vec<String>,
}

/// Request payload for the worker `stitch_wav_segments` JSON-RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct StitchWavSegmentsRequest {
    pub audio_files: Vec<String>,
    pub out_file: String,
}

/// Response payload for the worker `stitch_wav_segments` JSON-RPC method.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct StitchWavSegmentsResult {
    pub audio_file: String,
    pub segment_count: usize,
    pub duration_secs: f32,
}
