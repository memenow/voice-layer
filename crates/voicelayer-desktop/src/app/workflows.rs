//! Compose, rewrite, translate, and host-injection handlers for [`App`].

use iced::Task;
use voicelayer_core::{ComposeRequest, InjectRequest, RewriteRequest, TranslateRequest};

use crate::forms::{JobStage, apply_job_result};
use crate::state::{DaemonStatus, WorkflowTab};

use super::{App, Message};

impl App {
    pub(super) fn handle_workflows(&mut self, message: Message) -> Task<Message> {
        match message {
            Message::ComposePromptEdited(action) => {
                self.compose.prompt.perform(action);
                Task::none()
            }
            Message::ComposeArchetypeSelected(archetype) => {
                self.compose.archetype = archetype;
                Task::none()
            }
            Message::ComposeLanguageEdited(value) => {
                self.compose.language.edit(value);
                Task::none()
            }
            Message::ComposeSubmitPressed => self.submit_compose(),
            Message::Composed(result) => {
                apply_job_result(&mut self.compose.job, result);
                Task::none()
            }
            Message::RewriteSourceEdited(action) => {
                self.rewrite.source.perform(action);
                Task::none()
            }
            Message::RewriteStyleSelected(style) => {
                self.rewrite.style = style;
                Task::none()
            }
            Message::RewriteLanguageEdited(value) => {
                self.rewrite.language.edit(value);
                Task::none()
            }
            Message::RewriteSubmitPressed => self.submit_rewrite(),
            Message::Rewritten(result) => {
                apply_job_result(&mut self.rewrite.job, result);
                Task::none()
            }
            Message::TranslateSourceEdited(action) => {
                self.translate.source.perform(action);
                Task::none()
            }
            Message::TranslateTargetEdited(value) => {
                self.translate.target.edit(value);
                Task::none()
            }
            Message::TranslateSubmitPressed => self.submit_translate(),
            Message::Translated(result) => {
                apply_job_result(&mut self.translate.job, result);
                Task::none()
            }
            Message::InjectTargetSelected(tab, target) => {
                if let Some(job) = self.job_mut(tab) {
                    job.set_inject_target(target);
                }
                Task::none()
            }
            Message::InjectAutoSubmitToggled(tab) => {
                if let Some(job) = self.job_mut(tab) {
                    job.toggle_auto_submit();
                }
                Task::none()
            }
            Message::InjectPressed(tab) => self.submit_inject(tab),
            Message::Injected(tab, epoch, result) => {
                if let Some(job) = self.job_mut(tab)
                    && epoch == job.inject_epoch
                {
                    job.injecting = false;
                    match result {
                        Ok(plan) => {
                            job.plan = Some(*plan);
                            job.error = None;
                        }
                        Err(error) => job.error = Some((*error).clone()),
                    }
                }
                Task::none()
            }
            _ => Task::none(),
        }
    }

    fn job_mut(&mut self, tab: WorkflowTab) -> Option<&mut JobStage> {
        match tab {
            WorkflowTab::Compose => Some(&mut self.compose.job),
            WorkflowTab::Rewrite => Some(&mut self.rewrite.job),
            WorkflowTab::Translate => Some(&mut self.translate.job),
            _ => None,
        }
    }

    fn submit_compose(&mut self) -> Task<Message> {
        let Some(prompt) = prepare_workflow_text(
            self.daemon,
            &self.compose.prompt.text(),
            "Enter a prompt to compose from.",
            &mut self.compose.job,
        ) else {
            return Task::none();
        };
        let request = ComposeRequest {
            spoken_prompt: prompt,
            archetype: self.compose.archetype.clone(),
            output_language: self.compose.language.optional_value(),
        };
        self.compose
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
        let client = self.client.clone();
        Task::perform(
            async move { client.compose(request).await.map(Box::new) },
            Message::Composed,
        )
    }

    fn submit_rewrite(&mut self) -> Task<Message> {
        let Some(source) = prepare_workflow_text(
            self.daemon,
            &self.rewrite.source.text(),
            "Enter text to rewrite.",
            &mut self.rewrite.job,
        ) else {
            return Task::none();
        };
        let request = RewriteRequest {
            source_text: source,
            style: self.rewrite.style.clone(),
            output_language: self.rewrite.language.optional_value(),
        };
        self.rewrite
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
        let client = self.client.clone();
        Task::perform(
            async move { client.rewrite(request).await.map(Box::new) },
            Message::Rewritten,
        )
    }

    fn submit_translate(&mut self) -> Task<Message> {
        let Some(source) = prepare_workflow_text(
            self.daemon,
            &self.translate.source.text(),
            "Enter text to translate.",
            &mut self.translate.job,
        ) else {
            return Task::none();
        };
        let Some(target) = self.translate.target.optional_value() else {
            self.translate.job.error = Some("Enter a target language.".into());
            return Task::none();
        };
        let request = TranslateRequest {
            source_text: source,
            target_language: target,
        };
        self.translate
            .job
            .begin_submit(self.preferences.default_inject_target.clone());
        let client = self.client.clone();
        Task::perform(
            async move { client.translate(request).await.map(Box::new) },
            Message::Translated,
        )
    }

    fn submit_inject(&mut self, tab: WorkflowTab) -> Task<Message> {
        if self.daemon != DaemonStatus::Healthy {
            if let Some(job) = self.job_mut(tab) {
                job.error = Some("Daemon is not healthy; start it first.".into());
            }
            return Task::none();
        }
        let Some(job) = self.job_mut(tab) else {
            return Task::none();
        };
        let Some(text) = job
            .preview
            .as_ref()
            .and_then(|preview| preview.generated_text.clone())
            .filter(|text| !text.is_empty())
        else {
            job.error = Some("This preview has no generated text to inject.".into());
            return Task::none();
        };
        let request = InjectRequest {
            target: job.inject_target.clone(),
            text,
            auto_submit: job.auto_submit,
        };
        job.injecting = true;
        job.error = None;
        job.plan = None;
        job.inject_epoch = job.inject_epoch.wrapping_add(1);
        let epoch = job.inject_epoch;
        let client = self.client.clone();
        Task::perform(
            async move { client.inject(request).await.map(Box::new) },
            move |result| Message::Injected(tab, epoch, result),
        )
    }
}

fn prepare_workflow_text(
    daemon: DaemonStatus,
    raw: &str,
    empty_error: &str,
    job: &mut JobStage,
) -> Option<String> {
    if daemon != DaemonStatus::Healthy {
        job.error = Some("Daemon is not healthy; start it first.".into());
        return None;
    }
    let trimmed = raw.trim();
    if trimmed.is_empty() {
        job.error = Some(empty_error.to_owned());
        return None;
    }
    Some(trimmed.to_owned())
}
