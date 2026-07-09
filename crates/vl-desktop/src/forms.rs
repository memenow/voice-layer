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
    /// Generation counter for the injection currently in flight. Bumped on every
    /// new submission and every new inject so a late `POST /v1/inject` reply from
    /// a superseded preview is dropped instead of landing its plan on the job's
    /// newer (or still-running) request.
    pub(crate) inject_epoch: u64,
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
            inject_epoch: 0,
        }
    }

    /// Begin a new submission: clear the previous run's preview and plan, adopt
    /// the current default inject target, and invalidate any injection still in
    /// flight so its late reply cannot attach to this fresh job.
    pub(crate) fn begin_submit(&mut self, default_target: InjectTarget) {
        self.submitting = true;
        self.error = None;
        self.preview = None;
        self.inject_target = default_target;
        self.invalidate_inject();
    }

    pub(crate) fn set_inject_target(&mut self, target: InjectTarget) {
        if self.inject_target != target {
            self.inject_target = target;
            self.invalidate_inject();
        }
    }

    pub(crate) fn toggle_auto_submit(&mut self) {
        self.auto_submit = !self.auto_submit;
        self.invalidate_inject();
    }

    fn invalidate_inject(&mut self) {
        self.plan = None;
        self.injecting = false;
        self.inject_epoch = self.inject_epoch.wrapping_add(1);
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
    language_edited: bool,
    pub(crate) job: JobStage,
}

impl ComposeForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            prompt: text_editor::Content::new(),
            archetype: None,
            language: prefs.default_output_language.clone(),
            language_edited: false,
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }

    pub(crate) fn edit_language(&mut self, value: String) {
        self.language = value;
        self.language_edited = true;
    }

    pub(crate) fn sync_default_language(&mut self, value: &str) {
        if !self.language_edited {
            self.language = value.to_owned();
        }
    }

    pub(crate) fn output_language(&self) -> Option<String> {
        optional_text(&self.language)
    }
}

/// Rewrite form: source text, a required restyle, and an optional output
/// language.
pub(crate) struct RewriteForm {
    pub(crate) source: text_editor::Content,
    pub(crate) style: RewriteStyle,
    pub(crate) language: String,
    language_edited: bool,
    pub(crate) job: JobStage,
}

impl RewriteForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            style: RewriteStyle::MoreFormal,
            language: prefs.default_output_language.clone(),
            language_edited: false,
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }

    pub(crate) fn edit_language(&mut self, value: String) {
        self.language = value;
        self.language_edited = true;
    }

    pub(crate) fn sync_default_language(&mut self, value: &str) {
        if !self.language_edited {
            self.language = value.to_owned();
        }
    }

    pub(crate) fn output_language(&self) -> Option<String> {
        optional_text(&self.language)
    }
}

/// Translate form: source text and a required target language.
pub(crate) struct TranslateForm {
    pub(crate) source: text_editor::Content,
    pub(crate) target: String,
    target_edited: bool,
    pub(crate) job: JobStage,
}

impl TranslateForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            target: prefs.default_output_language.clone(),
            target_edited: false,
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }

    pub(crate) fn edit_target(&mut self, value: String) {
        self.target = value;
        self.target_edited = true;
    }

    pub(crate) fn sync_default_target(&mut self, value: &str) {
        if !self.target_edited {
            self.target = value.to_owned();
        }
    }

    pub(crate) fn target_language(&self) -> Option<String> {
        optional_text(&self.target)
    }
}

#[cfg(test)]
mod tests {
    use super::{ComposeForm, JobStage, RewriteForm, TranslateForm};
    use crate::state::Preferences;
    use voicelayer_core::{InjectTarget, InjectionPlan};

    fn preferences_with_language(language: &str) -> Preferences {
        Preferences {
            default_output_language: language.to_owned(),
            ..Preferences::default()
        }
    }

    fn prepared_job() -> JobStage {
        let mut job = JobStage::seeded(InjectTarget::GuiAccessible);
        job.injecting = true;
        job.plan = Some(InjectionPlan {
            target: InjectTarget::GuiAccessible,
            payload: "prepared".to_owned(),
            auto_submit: false,
        });
        job
    }

    /// A new submission adopts the current default inject target (so a changed
    /// Settings default takes effect) and cancels any injection still in flight,
    /// advancing the epoch so its late reply is dropped rather than attached.
    #[test]
    fn begin_submit_adopts_default_target_and_invalidates_in_flight_inject() {
        let mut job = JobStage::seeded(InjectTarget::GuiAccessible);
        // An injection from a prior preview is mid-flight when the user resubmits.
        job.injecting = true;
        let before = job.inject_epoch;

        job.begin_submit(InjectTarget::GuiClipboard);

        assert!(job.submitting, "a fresh submission is in flight");
        assert_eq!(
            job.inject_target,
            InjectTarget::GuiClipboard,
            "the new submission adopts the current default inject target",
        );
        assert!(!job.injecting, "any in-flight injection is cancelled");
        assert!(job.plan.is_none());
        assert!(job.preview.is_none());
        assert_eq!(
            job.inject_epoch,
            before + 1,
            "the inject generation advances so a superseded reply is dropped",
        );
    }

    /// Every submission advances the inject epoch, so each supersedes the
    /// previous generation's in-flight reply.
    #[test]
    fn begin_submit_advances_inject_epoch_each_call() {
        let mut job = JobStage::seeded(InjectTarget::GuiAccessible);
        let e0 = job.inject_epoch;
        job.begin_submit(InjectTarget::GuiAccessible);
        let e1 = job.inject_epoch;
        job.begin_submit(InjectTarget::GuiAccessible);
        let e2 = job.inject_epoch;
        assert!(
            e1 > e0 && e2 > e1,
            "each submission supersedes the previous inject generation",
        );
    }

    #[test]
    fn changing_inject_target_invalidates_the_prepared_plan_and_reply_epoch() {
        let mut job = prepared_job();
        let before = job.inject_epoch;

        job.set_inject_target(InjectTarget::GuiClipboard);

        assert_eq!(job.inject_target, InjectTarget::GuiClipboard);
        assert!(!job.injecting);
        assert!(job.plan.is_none());
        assert_eq!(job.inject_epoch, before + 1);
    }

    #[test]
    fn toggling_auto_submit_invalidates_the_prepared_plan_and_reply_epoch() {
        let mut job = prepared_job();
        let before = job.inject_epoch;

        job.toggle_auto_submit();

        assert!(job.auto_submit);
        assert!(!job.injecting);
        assert!(job.plan.is_none());
        assert_eq!(job.inject_epoch, before + 1);
    }

    #[test]
    fn untouched_language_fields_follow_settings_default_changes() {
        let prefs = preferences_with_language("English");
        let mut compose = ComposeForm::new(&prefs);
        let mut rewrite = RewriteForm::new(&prefs);
        let mut translate = TranslateForm::new(&prefs);

        compose.sync_default_language("French");
        rewrite.sync_default_language("French");
        translate.sync_default_target("French");

        assert_eq!(compose.output_language().as_deref(), Some("French"));
        assert_eq!(rewrite.output_language().as_deref(), Some("French"));
        assert_eq!(translate.target_language().as_deref(), Some("French"));
    }

    #[test]
    fn explicitly_cleared_language_fields_do_not_reinherit_the_default() {
        let prefs = preferences_with_language("English");
        let mut compose = ComposeForm::new(&prefs);
        let mut rewrite = RewriteForm::new(&prefs);
        let mut translate = TranslateForm::new(&prefs);

        compose.edit_language(String::new());
        rewrite.edit_language(String::new());
        translate.edit_target(String::new());
        compose.sync_default_language("French");
        rewrite.sync_default_language("French");
        translate.sync_default_target("French");

        assert_eq!(compose.output_language(), None);
        assert_eq!(rewrite.output_language(), None);
        assert_eq!(translate.target_language(), None);
    }
}
