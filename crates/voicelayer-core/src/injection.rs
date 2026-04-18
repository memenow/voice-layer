use crate::domain::{
    BRACKETED_PASTE_END, BRACKETED_PASTE_START, InjectRequest, InjectTarget, InjectionPlan,
};

impl InjectionPlan {
    pub fn from_request(request: &InjectRequest) -> Self {
        let payload = match request.target {
            InjectTarget::GuiAccessible | InjectTarget::GuiClipboard => request.text.clone(),
            InjectTarget::TerminalBracketedPaste => {
                let suffix = if request.auto_submit { "\n" } else { "" };
                format!(
                    "{BRACKETED_PASTE_START}{}{BRACKETED_PASTE_END}{suffix}",
                    request.text
                )
            }
            InjectTarget::TerminalKittyRemote => {
                if request.auto_submit {
                    format!("{}\n", request.text)
                } else {
                    request.text.clone()
                }
            }
        };

        Self {
            target: request.target.clone(),
            payload,
            auto_submit: request.auto_submit,
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::domain::{InjectRequest, InjectTarget, InjectionPlan};

    #[test]
    fn terminal_bracketed_paste_wraps_text_without_submit_by_default() {
        let plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::TerminalBracketedPaste,
            text: "analyze auth flow".to_owned(),
            auto_submit: false,
        });

        assert_eq!(
            plan.payload,
            "\u{1b}[200~analyze auth flow\u{1b}[201~".to_owned()
        );
        assert!(!plan.auto_submit);
    }

    #[test]
    fn terminal_bracketed_paste_appends_newline_when_submission_is_explicit() {
        let plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::TerminalBracketedPaste,
            text: "run now".to_owned(),
            auto_submit: true,
        });

        assert_eq!(plan.payload, "\u{1b}[200~run now\u{1b}[201~\n".to_owned());
    }
}
