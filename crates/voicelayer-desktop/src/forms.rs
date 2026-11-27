//! Form and preview-to-inject job state for compose, rewrite, and translate panels.

use iced::widget::text_editor;

use voicelayer_core::{
    CompositionArchetype, CompositionReceipt, InjectTarget, InjectionPlan, PreviewArtifact,
    RewriteStyle,
};

use crate::state::{Preferences, SharedError};

/// Tracks a text field that inherits a default value until explicitly edited.
pub(crate) struct LanguageField {
    value: String,
    edited: bool,
}

impl LanguageField {
    fn new(default: &str) -> Self {
        Self {
            value: default.to_owned(),
            edited: false,
        }
    }

    pub(crate) fn as_str(&self) -> &str {
        &self.value
    }

    pub(crate) fn edit(&mut self, value: String) {
        self.value = value;
        self.edited = true;
    }

    pub(crate) fn sync_default(&mut self, value: &str) {
        if !self.edited {
            self.value = value.to_owned();
        }
    }

    pub(crate) fn optional_value(&self) -> Option<String> {
        let trimmed = self.value.trim();
        (!trimmed.is_empty()).then(|| trimmed.to_owned())
    }
}

/// Preview and injection-plan state shared by the generative workflows.
pub(crate) struct JobStage {
    pub(crate) submitting: bool,
    pub(crate) error: Option<String>,
    pub(crate) preview: Option<PreviewArtifact>,
    pub(crate) inject_target: InjectTarget,
    pub(crate) auto_submit: bool,
    pub(crate) injecting: bool,
    pub(crate) plan: Option<InjectionPlan>,
    /// Generation counter incremented on each submission or inject request so
    /// late replies from a superseded preview are ignored.
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

pub(crate) fn apply_job_result(
    job: &mut JobStage,
    result: Result<Box<CompositionReceipt>, SharedError>,
) {
    job.submitting = false;
    match result {
        Ok(receipt) => {
            job.preview = Some(receipt.preview);
            job.error = None;
        }
        Err(error) => job.error = Some((*error).clone()),
    }
}

pub(crate) struct ComposeForm {
    pub(crate) prompt: text_editor::Content,
    pub(crate) archetype: Option<CompositionArchetype>,
    pub(crate) language: LanguageField,
    pub(crate) job: JobStage,
}

impl ComposeForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            prompt: text_editor::Content::new(),
            archetype: None,
            language: LanguageField::new(&prefs.default_output_language),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }
}

pub(crate) struct RewriteForm {
    pub(crate) source: text_editor::Content,
    pub(crate) style: RewriteStyle,
    pub(crate) language: LanguageField,
    pub(crate) job: JobStage,
}

impl RewriteForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            style: RewriteStyle::MoreFormal,
            language: LanguageField::new(&prefs.default_output_language),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
    }
}

pub(crate) struct TranslateForm {
    pub(crate) source: text_editor::Content,
    pub(crate) target: LanguageField,
    pub(crate) job: JobStage,
}

impl TranslateForm {
    pub(crate) fn new(prefs: &Preferences) -> Self {
        Self {
            source: text_editor::Content::new(),
            target: LanguageField::new(&prefs.default_output_language),
            job: JobStage::seeded(prefs.default_inject_target.clone()),
        }
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

    #[test]
    fn begin_submit_adopts_default_target_and_invalidates_in_flight_inject() {
        let mut job = JobStage::seeded(InjectTarget::GuiAccessible);
        job.injecting = true;
        let before = job.inject_epoch;

        job.begin_submit(InjectTarget::GuiClipboard);

        assert!(job.submitting);
        assert_eq!(job.inject_target, InjectTarget::GuiClipboard);
        assert!(!job.injecting);
        assert!(job.plan.is_none());
        assert!(job.preview.is_none());
        assert_eq!(job.inject_epoch, before + 1);
    }

    #[test]
    fn begin_submit_advances_inject_epoch_each_call() {
        let mut job = JobStage::seeded(InjectTarget::GuiAccessible);
        let e0 = job.inject_epoch;
        job.begin_submit(InjectTarget::GuiAccessible);
        let e1 = job.inject_epoch;
        job.begin_submit(InjectTarget::GuiAccessible);
        let e2 = job.inject_epoch;
        assert!(e1 > e0 && e2 > e1);
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

        compose.language.sync_default("French");
        rewrite.language.sync_default("French");
        translate.target.sync_default("French");

        assert_eq!(compose.language.optional_value().as_deref(), Some("French"));
        assert_eq!(rewrite.language.optional_value().as_deref(), Some("French"));
        assert_eq!(translate.target.optional_value().as_deref(), Some("French"));
    }

    #[test]
    fn explicitly_cleared_language_fields_do_not_reinherit_the_default() {
        let prefs = preferences_with_language("English");
        let mut compose = ComposeForm::new(&prefs);
        let mut rewrite = RewriteForm::new(&prefs);
        let mut translate = TranslateForm::new(&prefs);

        compose.language.edit(String::new());
        rewrite.language.edit(String::new());
        translate.target.edit(String::new());
        compose.language.sync_default("French");
        rewrite.language.sync_default("French");
        translate.target.sync_default("French");

        assert_eq!(compose.language.optional_value(), None);
        assert_eq!(rewrite.language.optional_value(), None);
        assert_eq!(translate.target.optional_value(), None);
    }
}
