//! Generative-workflow form state for the desktop shell.
//!
//! The compose / rewrite / translate workflows share a common shape: an input
//! form, a submitted job, the resulting [`PreviewArtifact`], and an optional
//! injection plan. That shared sub-state is [`JobStage`]; each workflow wraps it
//! with its own typed inputs. This lives apart from [`crate::state`] because the
//! multi-line inputs hold an [`text_editor::Content`], which is interior-mutable
//! and not the pure, unit-testable state that module owns.

use iced::widget::text_editor;

use voicelayer_core::{
    CompositionArchetype, CompositionReceipt, InjectTarget, InjectionPlan, PreviewArtifact,
    RewriteStyle,
};

use crate::state::{Preferences, SharedError};

/// The preview-and-inject sub-state shared by the generative workflows. A
/// successful job yields a [`PreviewArtifact`]; the operator reviews it, chooses
/// an inject target, then asks the daemon to plan the injection. `POST /v1/inject`
/// returns an [`InjectionPlan`] — it prepares the paste, it does not itself type
/// into the focused application.
pub(crate) struct JobStage {
    pub(crate) submitting: bool,
    pub(crate) error: Option<String>,
    pub(crate) preview: Option<PreviewArtifact>,
    pub(crate) inject_target: InjectTarget,
    pub(crate) auto_submit: bool,
    pub(crate) injecting: bool,
    pub(crate) plan: Option<InjectionPlan>,
}

impl JobStage {
    fn seeded(default_target: InjectTarget) -> Self {
        Self {
            submitting: false,
            error: None,
            preview: None,
            inject_target: default_target,
            auto_submit: false,
            injecting: false,
            plan: None,
        }
    }

    /// Clear the previous run's preview and plan as a new submission starts.
    pub(crate) fn begin_submit(&mut self) {
        self.submitting = true;
        self.error = None;
        self.preview = None;
        self.plan = None;
    }
}

/// Store a generative job's outcome on its [`JobStage`]: a preview on success, a
/// user-facing message on failure. Clears `submitting` either way.
pub(crate) fn apply_job_result(
    job: &mut JobStage,
    result: Result<Box<CompositionReceipt>, SharedError>,
) {
    job.submitting = false;
    match result {
        Ok(receipt) => {
            let receipt = *receipt;
            job.preview = Some(receipt.preview);
            job.error = None;
        }
        Err(error) => job.error = Some((*error).clone()),
    }
}

/// Trim a form text field into an optional wire value: blank ⇒ `None`.
pub(crate) fn optional_text(value: &str) -> Option<String> {
    let trimmed = value.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_owned())
    }
}

/// Compose form: a spoken-style prompt plus an optional archetype and output
/// language.
pub(crate) struct ComposeForm {
    pub(crate) prompt: text_editor::Content,
    pub(crate) archetype: Option<CompositionArchetype>,
    pub(crate) language: String,
    pub(crate) job: JobStage,
}

impl ComposeForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            prompt: text_editor::Content::new(),
            archetype: None,
            language: prefs.default_output_language.clone(),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }
}

/// Rewrite form: source text, a required restyle, and an optional output
/// language.
pub(crate) struct RewriteForm {
    pub(crate) source: text_editor::Content,
    pub(crate) style: RewriteStyle,
    pub(crate) language: String,
    pub(crate) job: JobStage,
}

impl RewriteForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            style: RewriteStyle::MoreFormal,
            language: prefs.default_output_language.clone(),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }
}

/// Translate form: source text and a required target language.
pub(crate) struct TranslateForm {
    pub(crate) source: text_editor::Content,
    pub(crate) target: String,
    pub(crate) job: JobStage,
}

impl TranslateForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            target: prefs.default_output_language.clone(),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }
}
