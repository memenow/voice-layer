//! Composition, rewrite, translation, injection, and RFC 9457 error types.

use serde::{Deserialize, Serialize};
use uuid::Uuid;

use super::SessionMode;

/// Preset prompt archetype for `POST /v1/sessions/compose`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CompositionArchetype {
    Email,
    CoverLetter,
    DailyReport,
    Issue,
    PullRequestDescription,
    Prompt,
    TechnicalSummary,
    Custom,
}

/// Request body for `POST /v1/sessions/compose`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ComposeRequest {
    pub spoken_prompt: String,
    #[serde(default)]
    pub archetype: Option<CompositionArchetype>,
    #[serde(default)]
    pub output_language: Option<String>,
}

/// Target transformation style for `POST /v1/rewrites`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum RewriteStyle {
    MoreFormal,
    Shorter,
    Politer,
    MoreTechnical,
    Translate,
}

/// Request body for `POST /v1/rewrites`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct RewriteRequest {
    pub source_text: String,
    pub style: RewriteStyle,
    #[serde(default)]
    pub output_language: Option<String>,
}

/// Request body for `POST /v1/translations`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct TranslateRequest {
    pub source_text: String,
    pub target_language: String,
}

/// Availability status of a [`PreviewArtifact`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum PreviewStatus {
    Ready,
    NeedsProvider,
    Rejected,
}

/// Generated text preview returned before host injection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct PreviewArtifact {
    pub artifact_id: Uuid,
    pub status: PreviewStatus,
    pub title: String,
    pub generated_text: Option<String>,
    pub notes: Vec<String>,
}

impl PreviewArtifact {
    /// Build a [`PreviewStatus::NeedsProvider`] artifact naming the unconfigured workflow.
    pub fn needs_provider(title: &str, mode: SessionMode) -> Self {
        let mode_label = match mode {
            SessionMode::Dictation => "dictation",
            SessionMode::Compose => "composition",
            SessionMode::Rewrite => "rewrite",
            SessionMode::Translate => "translation",
        };

        Self {
            artifact_id: Uuid::new_v4(),
            status: PreviewStatus::NeedsProvider,
            title: title.to_owned(),
            generated_text: None,
            notes: vec![
                format!("No configured provider is available for the {mode_label} workflow."),
                "Configure a local or cloud provider before requesting generated text.".to_owned(),
            ],
        }
    }
}

/// Response pairing a job identifier with a generated [`PreviewArtifact`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct CompositionReceipt {
    pub job_id: Uuid,
    pub preview: PreviewArtifact,
}

impl CompositionReceipt {
    /// Wrap [`PreviewArtifact::needs_provider`] with a fresh `job_id`.
    pub fn needs_provider(title: &str, mode: SessionMode) -> Self {
        Self {
            job_id: Uuid::new_v4(),
            preview: PreviewArtifact::needs_provider(title, mode),
        }
    }
}

/// Destination host surface for text injection.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum InjectTarget {
    GuiAccessible,
    GuiClipboard,
    TerminalBracketedPaste,
    TerminalKittyRemote,
}

/// Request body for `POST /v1/inject`.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InjectRequest {
    pub target: InjectTarget,
    pub text: String,
    #[serde(default)]
    pub auto_submit: bool,
}

/// Formatted payload ready to hand to a host injection adapter.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct InjectionPlan {
    pub target: InjectTarget,
    pub payload: String,
    pub auto_submit: bool,
}

/// RFC 9457 problem details returned on non-2xx daemon responses.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProblemDetails {
    #[serde(rename = "type")]
    pub problem_type: String,
    pub title: String,
    pub status: u16,
    pub detail: String,
}

impl ProblemDetails {
    pub fn new(
        problem_type: impl Into<String>,
        title: impl Into<String>,
        status: u16,
        detail: impl Into<String>,
    ) -> Self {
        Self {
            problem_type: problem_type.into(),
            title: title.into(),
            status,
            detail: detail.into(),
        }
    }
}
