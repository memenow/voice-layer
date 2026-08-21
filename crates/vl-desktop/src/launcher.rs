//! Locating and launching the `voicelayerd` daemon from the desktop shell.
//!
//! Kept apart from the [`crate::app`] controller because it is plain process
//! supervision — no iced, no UI state — and carries its own env-var resolution
//! tests.

use std::path::PathBuf;
use std::process::{Command, Stdio};
use std::sync::Arc;
use std::time::Duration;

#[cfg(unix)]
use std::os::unix::process::CommandExt;

use crate::state::SharedError;

fn resolve_daemon_binary() -> PathBuf {
    // Prefer an explicit override for development setups, then the install.sh
    // target, then $PATH. Falling back to the bare name lets `Command::spawn`
    // surface the usual "program not found" error with a clear hint.
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
    // Detach from the GUI's process group so closing the desktop shell does not
    // SIGHUP the daemon. `process_group(0)` makes the child a new group leader;
    // on non-Unix this degrades to the default attached behavior.
    #[cfg(unix)]
    command.process_group(0);

    match command.spawn() {
        Ok(_) => {
            // Give the daemon a moment to open its socket before the caller
            // re-probes health; without this the probe often loses the race.
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
    use std::ffi::OsString;
    use std::sync::Mutex;

    use super::resolve_daemon_binary;

    /// Serializes env mutation across every test in this module; see
    /// `crates/voicelayerd/src/worker.rs` for the rationale.
    static ENV_LOCK: Mutex<()> = Mutex::new(());

    /// Snapshots an env var at construction and restores it on drop so an
    /// assertion panic never leaks state into the next test.
    struct EnvGuard {
        key: &'static str,
        previous: Option<OsString>,
    }

    impl EnvGuard {
        fn capture(key: &'static str) -> Self {
            Self {
                key,
                previous: std::env::var_os(key),
            }
        }
    }

    impl Drop for EnvGuard {
        fn drop(&mut self) {
            // SAFETY: ENV_LOCK is held by the surrounding test for the lifetime
            // of the guard, so no other thread observes the intermediate state.
            unsafe {
                match &self.previous {
                    Some(value) => std::env::set_var(self.key, value),
                    None => std::env::remove_var(self.key),
                }
            }
        }
    }

    fn set_env(key: &str, value: impl AsRef<std::ffi::OsStr>) {
        // SAFETY: ENV_LOCK must be held by the caller.
        unsafe {
            std::env::set_var(key, value);
        }
    }

    fn unset_env(key: &str) {
        // SAFETY: ENV_LOCK must be held by the caller.
        unsafe {
            std::env::remove_var(key);
        }
    }

    #[test]
    fn resolve_daemon_binary_honors_explicit_override() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_DAEMON_BIN");
        set_env("VOICELAYER_DAEMON_BIN", "/custom/voicelayerd");
        assert_eq!(
            resolve_daemon_binary().to_str(),
            Some("/custom/voicelayerd")
        );
    }

    #[test]
    fn resolve_daemon_binary_falls_back_to_local_bin_when_present() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_DAEMON_BIN");
        let _home_guard = EnvGuard::capture("HOME");
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        let fake_local = tmp.path().join(".local/bin");
        std::fs::create_dir_all(&fake_local).expect("create fake ~/.local/bin");
        let vl_path = fake_local.join("voicelayerd");
        std::fs::write(&vl_path, b"#!/bin/sh\n").expect("write fake daemon binary");
        unset_env("VOICELAYER_DAEMON_BIN");
        set_env("HOME", tmp.path());
        assert_eq!(resolve_daemon_binary(), vl_path);
    }

    #[test]
    fn resolve_daemon_binary_falls_back_to_path_lookup_as_last_resort() {
        let _guard = ENV_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _vl_bin_guard = EnvGuard::capture("VOICELAYER_DAEMON_BIN");
        let _home_guard = EnvGuard::capture("HOME");
        let tmp = tempfile::tempdir().expect("tempdir should be creatable");
        unset_env("VOICELAYER_DAEMON_BIN");
        // Point HOME at an empty directory so `~/.local/bin/voicelayerd` misses.
        set_env("HOME", tmp.path());
        assert_eq!(resolve_daemon_binary().to_str(), Some("voicelayerd"));
    }
}
