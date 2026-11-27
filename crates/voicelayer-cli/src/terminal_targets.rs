use std::io::Write;
use std::process::{Command, Stdio};

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) enum ForegroundInjectionTarget {
    None,
    Tmux,
    Wezterm { pane_id: String },
    Kitty { r#match: String },
}

pub(crate) fn paste_into_tmux_pane(
    target_pane: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }

    if tmux_target_conflicts(target_pane, std::env::var("TMUX_PANE").ok().as_deref()) {
        return Err(
            "refusing to paste into the same tmux pane that is currently running the controller"
                .into(),
        );
    }

    let set_status = Command::new("tmux")
        .arg("set-buffer")
        .arg("--")
        .arg(text)
        .status()?;
    if !set_status.success() {
        return Err(format!("tmux set-buffer failed with status {set_status}").into());
    }

    let paste_status = Command::new("tmux")
        .arg("paste-buffer")
        .arg("-dpr")
        .arg("-t")
        .arg(target_pane)
        .status()?;
    if !paste_status.success() {
        return Err(format!("tmux paste-buffer failed with status {paste_status}").into());
    }

    Ok(())
}

pub(crate) fn tmux_target_conflicts(target_pane: &str, current_pane: Option<&str>) -> bool {
    current_pane == Some(target_pane)
}

pub(crate) fn wezterm_target_conflicts(target_pane: &str, current_pane: Option<&str>) -> bool {
    current_pane == Some(target_pane)
}

pub(crate) fn resolve_foreground_injection_target(
    tmux_target_pane: Option<String>,
    wezterm_target_pane_id: Option<String>,
    kitty_match: Option<String>,
) -> Result<ForegroundInjectionTarget, Box<dyn std::error::Error>> {
    let mut selected = Vec::new();
    if tmux_target_pane.is_some() {
        selected.push("tmux");
    }
    if wezterm_target_pane_id.is_some() {
        selected.push("wezterm");
    }
    if kitty_match.is_some() {
        selected.push("kitty");
    }

    if selected.len() > 1 {
        return Err(format!(
            "choose only one foreground injection target, got: {}",
            selected.join(", ")
        )
        .into());
    }

    if let Some(pane_id) = wezterm_target_pane_id {
        return Ok(ForegroundInjectionTarget::Wezterm { pane_id });
    }
    if let Some(r#match) = kitty_match {
        return Ok(ForegroundInjectionTarget::Kitty { r#match });
    }
    if tmux_target_pane.is_some() || std::env::var("TMUX").is_ok() {
        return Ok(ForegroundInjectionTarget::Tmux);
    }
    Ok(ForegroundInjectionTarget::None)
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub(crate) struct TmuxPaneSummary {
    pub(crate) pane_id: String,
    pub(crate) location: String,
    pub(crate) window_name: String,
    pub(crate) current_command: String,
    pub(crate) active: bool,
}

pub(crate) fn resolve_tmux_target_pane(
    explicit_target: Option<String>,
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    if explicit_target.is_some() {
        return Ok(explicit_target);
    }

    if std::env::var("TMUX").is_err() {
        return Ok(None);
    }

    let current_pane = std::env::var("TMUX_PANE").ok();
    let panes = discover_tmux_panes()?;
    let candidates: Vec<TmuxPaneSummary> = panes
        .into_iter()
        .filter(|pane| Some(pane.pane_id.as_str()) != current_pane.as_deref())
        .collect();

    match candidates.len() {
        0 => Ok(None),
        1 => {
            let pane = &candidates[0];
            eprintln!(
                "auto-selected tmux pane {} ({}, cmd: {})",
                pane.pane_id, pane.location, pane.current_command
            );
            Ok(Some(pane.pane_id.clone()))
        }
        _ => prompt_for_tmux_pane(&candidates),
    }
}

fn discover_tmux_panes() -> Result<Vec<TmuxPaneSummary>, Box<dyn std::error::Error>> {
    let output = Command::new("tmux")
        .arg("list-panes")
        .arg("-a")
        .arg("-F")
        .arg("#{pane_id}\t#{session_name}:#{window_index}.#{pane_index}\t#{window_name}\t#{pane_current_command}\t#{pane_active}")
        .output()?;

    if !output.status.success() {
        return Err(format!("tmux list-panes failed with status {}", output.status).into());
    }

    Ok(parse_tmux_panes_output(&String::from_utf8(output.stdout)?))
}

pub(crate) fn parse_tmux_panes_output(output: &str) -> Vec<TmuxPaneSummary> {
    output
        .lines()
        .filter_map(|line| {
            let mut parts = line.splitn(5, '\t');
            let pane_id = parts.next()?.to_owned();
            let location = parts.next()?.to_owned();
            let window_name = parts.next()?.to_owned();
            let current_command = parts.next()?.to_owned();
            let active = matches!(parts.next(), Some("1"));
            Some(TmuxPaneSummary {
                pane_id,
                location,
                window_name,
                current_command,
                active,
            })
        })
        .collect()
}

pub(crate) fn paste_into_wezterm_pane(
    pane_id: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    if wezterm_target_conflicts(pane_id, std::env::var("WEZTERM_PANE").ok().as_deref()) {
        return Err(
            "refusing to send text into the same wezterm pane that is running the controller"
                .into(),
        );
    }

    let status = Command::new("wezterm")
        .arg("cli")
        .arg("send-text")
        .arg("--pane-id")
        .arg(pane_id)
        .arg(text)
        .status()?;
    if !status.success() {
        return Err(format!("wezterm cli send-text failed with status {status}").into());
    }
    Ok(())
}

pub(crate) fn paste_into_kitty_target(
    target_match: &str,
    text: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    if text.is_empty() {
        return Ok(());
    }
    let mut child = Command::new("kitten")
        .arg("@")
        .arg("send-text")
        .arg("--match")
        .arg(target_match)
        .arg("--stdin")
        .arg("--bracketed-paste")
        .arg("auto")
        .stdin(Stdio::piped())
        .spawn()?;
    if let Some(mut stdin) = child.stdin.take() {
        stdin.write_all(text.as_bytes())?;
    }
    let status = child.wait()?;
    if !status.success() {
        return Err(format!("kitten @ send-text failed with status {status}").into());
    }
    Ok(())
}

fn prompt_for_tmux_pane(
    candidates: &[TmuxPaneSummary],
) -> Result<Option<String>, Box<dyn std::error::Error>> {
    eprintln!("select a tmux target pane for transcript injection:");
    for (index, pane) in candidates.iter().enumerate() {
        eprintln!(
            "  [{}] {}  {}  window={}  cmd={}",
            index + 1,
            pane.pane_id,
            pane.location,
            pane.window_name,
            pane.current_command
        );
    }
    eprint!("selection (number, pane id, Enter to skip): ");
    std::io::stderr().flush()?;

    let mut input = String::new();
    std::io::stdin().read_line(&mut input)?;
    let input = input.trim();
    if input.is_empty() || input.eq_ignore_ascii_case("q") {
        return Ok(None);
    }

    if let Ok(index) = input.parse::<usize>()
        && (1..=candidates.len()).contains(&index)
    {
        return Ok(Some(candidates[index - 1].pane_id.clone()));
    }

    if let Some(pane) = candidates.iter().find(|pane| pane.pane_id == input) {
        return Ok(Some(pane.pane_id.clone()));
    }

    Err("invalid tmux pane selection".into())
}

#[cfg(test)]
mod tests {
    use super::{
        ForegroundInjectionTarget, parse_tmux_panes_output, resolve_foreground_injection_target,
        tmux_target_conflicts, wezterm_target_conflicts,
    };

    #[test]
    fn tmux_target_conflict_detects_same_pane() {
        assert!(tmux_target_conflicts("%1", Some("%1")));
        assert!(!tmux_target_conflicts("%1", Some("%2")));
        assert!(!tmux_target_conflicts("%1", None));
    }

    #[test]
    fn wezterm_target_conflict_detects_same_pane() {
        assert!(wezterm_target_conflicts("12", Some("12")));
        assert!(!wezterm_target_conflicts("12", Some("13")));
        assert!(!wezterm_target_conflicts("12", None));
    }

    #[test]
    fn parse_tmux_panes_output_reads_expected_fields() {
        let panes =
            parse_tmux_panes_output("%1\tdev:1.0\teditor\tbash\t1\n%2\tdev:1.1\ttests\tnvim\t0\n");
        assert_eq!(panes.len(), 2);
        assert_eq!(panes[0].pane_id, "%1");
        assert_eq!(panes[1].location, "dev:1.1");
        assert_eq!(panes[1].current_command, "nvim");
        assert!(!panes[1].active);
    }

    #[test]
    fn resolve_foreground_target_rejects_multiple_explicit_targets() {
        let error =
            resolve_foreground_injection_target(Some("%1".to_owned()), Some("12".to_owned()), None)
                .expect_err("multiple explicit targets should fail");
        assert!(
            error
                .to_string()
                .contains("choose only one foreground injection target")
        );
    }

    #[test]
    fn resolve_foreground_target_prefers_explicit_wezterm_target() {
        let target =
            resolve_foreground_injection_target(None, Some("12".to_owned()), None).unwrap();
        assert_eq!(
            target,
            ForegroundInjectionTarget::Wezterm {
                pane_id: "12".to_owned()
            }
        );
    }
}
