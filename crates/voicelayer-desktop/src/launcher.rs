//! Locating and spawning the `voicelayerd` daemon from the desktop shell.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::state::SharedError;

fn resolve_daemon_binary() -> PathBuf {
    if let Some(explicit) = std::env::var_os("VOICELAYER_DAEMON_BIN") {
        return PathBuf::from(explicit);
    }
    if let Some(home) = std::env::var_os("HOME") {
        let candidate = PathBuf::from(home).join(".local/bin/voicelayerd");
        if candidate.is_file() {
            return candidate;
        }
    }
    PathBuf::from("voicelayerd")
}

pub(crate) async fn spawn_daemon() -> Result<(), SharedError> {
    let binary = resolve_daemon_binary();
    let mut command = Command::new(&binary);
    command
        .stdin(Stdio::null())
        .stdout(Stdio::null())
        .stderr(Stdio::null());
    // Detach into a new process group so closing the GUI does not SIGHUP the daemon.
    #[cfg(unix)]
    command.process_group(0);

    match command.spawn() {
        Ok(_) => {
            // Brief grace window so the daemon binds its socket before re-probing.
            tokio::time::sleep(Duration::from_millis(500)).await;
            Ok(())
        }
        Err(error) => Err(Arc::new(format!(
            "could not execute `{}`: {error}. Set VOICELAYER_DAEMON_BIN or add \
             ~/.local/bin to PATH.",
            binary.display(),
        ))),
    }
}

#[cfg(test)]
mod tests {
    use voicelayer_testkit::EnvGuard;

    use super::resolve_daemon_binary;

    #[test]
    fn resolve_daemon_binary_honors_explicit_override() {
        let mut env = EnvGuard::lock();
        env.set("VOICELAYER_DAEMON_BIN", "/custom/voicelayerd");
        assert_eq!(
            resolve_daemon_binary().to_str(),
            Some("/custom/voicelayerd")
        );
    }

    #[test]
    fn resolve_daemon_binary_falls_back_to_local_bin_when_present() {
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        let fake_local = tmp.path().join(".local/bin");
        std::fs::create_dir_all(&fake_local).expect("create fake ~/.local/bin");
        let vl_path = fake_local.join("voicelayerd");
        std::fs::write(&vl_path, b"#!/bin/sh\n").expect("write fake daemon binary");

        let home = tmp.path().to_str().expect("tempdir path should be UTF-8");
        let mut env = EnvGuard::lock();
        env.remove("VOICELAYER_DAEMON_BIN");
        env.set("HOME", home);
        assert_eq!(resolve_daemon_binary(), vl_path);
    }

    #[test]
    fn resolve_daemon_binary_falls_back_to_path_lookup_as_last_resort() {
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        let home = tmp.path().to_str().expect("tempdir path should be UTF-8");
        let mut env = EnvGuard::lock();
        env.remove("VOICELAYER_DAEMON_BIN");
        env.set("HOME", home);
        assert_eq!(resolve_daemon_binary().to_str(), Some("voicelayerd"));
    }
}
