# Service Installation (systemd / launchd)

VoiceLayer's daemon (`voicelayerd`) runs as a long-lived per-user service so
the socket is always available when the CLI, desktop shell, or a hotkey
target asks for it.

## One-shot install

```bash
scripts/install.sh
```

The script:

1. Runs `cargo build --release --bin voicelayerd --bin vl --bin vl-desktop`.
2. Installs the three binaries into `~/.local/bin` (override with
   `VOICELAYER_INSTALL_BIN_DIR`).
3. Linux: copies `systemd/voicelayerd.service` to `~/.config/systemd/user/`.
   macOS: renders `launchd/com.memenow.voicelayerd.plist` into
   `~/Library/LaunchAgents/`.
4. Seeds the unified config file (`config/config.toml.example`) unless it
   already exists:
   - Linux: `~/.config/voicelayer/config.toml`
   - macOS: `~/Library/Application Support/com.memenow.voicelayer/config.toml`
5. Finishes with `vl doctor`.

## Linux (systemd user unit)

```bash
systemctl --user enable --now voicelayerd
systemctl --user status voicelayerd
journalctl --user -u voicelayerd
```

The unit runs `%h/.local/bin/voicelayerd` directly.

## macOS (launchd agent)

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memenow.voicelayerd.plist
launchctl print gui/$(id -u)/com.memenow.voicelayerd
```

Logs go to `~/Library/Logs/voicelayerd.log`. Stop with
`launchctl bootout gui/$(id -u)/com.memenow.voicelayerd`.

## Configuration

The service reads the unified TOML config (see above) and honors
`VOICELAYER_*` environment overrides. Two daemon-level overrides matter for
service operation:

- `VOICELAYER_PROJECT_ROOT` — repository root used to locate the Python
  worker (`python/voicelayer_orchestrator`). Set it when the daemon is not
  launched from the repository root.
- `VOICELAYER_SOCKET_PATH` — override for the daemon socket path. Defaults
  to `$XDG_RUNTIME_DIR/voicelayer/daemon.sock` (Linux) or
  `$TMPDIR/voicelayer/daemon.sock` (macOS).

Provider settings (`[llm]`, `[whisper]`, `[whisper_server]`, `[vad]`) are
documented in the provider guides; every key has a `VOICELAYER_*` override.

## Manual install (Linux)

```bash
cargo build --release --bin voicelayerd --bin vl --bin vl-desktop
install -m 0755 target/release/{voicelayerd,vl,vl-desktop} ~/.local/bin/
install -d ~/.config/systemd/user ~/.config/voicelayer
install -m 0644 systemd/voicelayerd.service ~/.config/systemd/user/voicelayerd.service
[[ -f ~/.config/voicelayer/config.toml ]] || \
  install -m 0600 config/config.toml.example ~/.config/voicelayer/config.toml
systemctl --user daemon-reload
systemctl --user enable --now voicelayerd
```

## Uninstall (Linux)

```bash
systemctl --user disable --now voicelayerd
rm -f ~/.config/systemd/user/voicelayerd.service
rm -f ~/.local/bin/{voicelayerd,vl,vl-desktop}
systemctl --user daemon-reload
```

Config files under `~/.config/voicelayer/` are left in place so a later
reinstall can reuse them.
