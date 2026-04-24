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

    #[test]
    fn terminal_kitty_remote_passes_text_verbatim_when_auto_submit_false() {
        let plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::TerminalKittyRemote,
            text: "ls -la".to_owned(),
            auto_submit: false,
        });

        assert_eq!(plan.payload, "ls -la".to_owned());
        assert!(!plan.auto_submit);
    }

    #[test]
    fn terminal_kitty_remote_appends_newline_when_submission_is_explicit() {
        let plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::TerminalKittyRemote,
            text: "ls -la".to_owned(),
            auto_submit: true,
        });

        assert_eq!(plan.payload, "ls -la\n".to_owned());
        assert!(plan.auto_submit);
    }

    #[test]
    fn gui_clipboard_and_gui_accessible_share_verbatim_payload_arm() {
        // `InjectionPlan::from_request` merges these two targets into a
        // single match arm that clones `request.text` as the payload.
        // Pin the shared behavior so a future split of the arm is
        // caught here rather than downstream in the daemon's injection
        // path.
        let text = "Inject into the focused editable surface.".to_owned();

        let accessible_plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::GuiAccessible,
            text: text.clone(),
            auto_submit: false,
        });
        let clipboard_plan = InjectionPlan::from_request(&InjectRequest {
            target: InjectTarget::GuiClipboard,
            text: text.clone(),
            auto_submit: false,
        });

        assert_eq!(accessible_plan.payload, text);
        assert_eq!(clipboard_plan.payload, text);
        assert_eq!(accessible_plan.payload, clipboard_plan.payload);
    }
}
