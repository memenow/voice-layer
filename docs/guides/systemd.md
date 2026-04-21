# Systemd User Unit

VoiceLayer's daemon (`voicelayerd`) is intended to run as a long-lived user-level
systemd service so the socket is always available when the CLI, desktop shell,
or a hotkey target asks for it.

## One-shot install

```bash
scripts/install.sh
```

The script:

1. Runs `cargo build --release --bin vl --bin vl-desktop`.
2. Installs `vl` and `vl-desktop` into `~/.local/bin` (override with
   `VOICELAYER_INSTALL_BIN_DIR`).
3. Copies `systemd/voicelayerd.service` to `~/.config/systemd/user/`
   (override with `VOICELAYER_INSTALL_UNIT_DIR`).
4. Seeds `~/.config/voicelayer/voicelayerd.env` from the shipped
   `voicelayerd.env.example` unless it already exists (override directory with
   `VOICELAYER_INSTALL_ENV_DIR`).
5. Runs `systemctl --user daemon-reload` and finishes with `vl doctor`.

Enabling the daemon afterwards is a single command:

```bash
systemctl --user enable --now voicelayerd
```

`systemctl --user status voicelayerd` and `journalctl --user -u voicelayerd`
are the usual knobs for inspection.

## Environment variables

`~/.config/voicelayer/voicelayerd.env` drives the daemon. The install script
seeds it from `systemd/voicelayerd.env.example`; edit paths before enabling
the unit.

### Core

- `VOICELAYER_PROJECT_ROOT` — repository root the daemon uses to locate the
  Python worker (`python/voicelayer_orchestrator`).
- `VOICELAYER_SOCKET_PATH` — optional override for the daemon socket path.
  Defaults to `$XDG_RUNTIME_DIR/voicelayer/daemon.sock`.

### whisper.cpp (ASR)

- `VOICELAYER_WHISPER_BIN` — path (or `$PATH` binary) for `whisper-cli`.
- `VOICELAYER_WHISPER_MODEL_PATH` — absolute path to a ggml model.
- `VOICELAYER_WHISPER_TIMEOUT_SECONDS` — per-call timeout.
- `VOICELAYER_WHISPER_NO_GPU` — set truthy to force CPU on whisper-cli.
- `VOICELAYER_WHISPER_ARGS` — extra flags forwarded verbatim.
- `VOICELAYER_WHISPER_SERVER_HOST`, `VOICELAYER_WHISPER_SERVER_PORT`,
  `VOICELAYER_WHISPER_SERVER_TIMEOUT_SECONDS` — persistent
  [whisper-server](./local-asr-provider.md) configuration.
- `VOICELAYER_WHISPER_SERVER_AUTO_START`,
  `VOICELAYER_WHISPER_SERVER_BIN`,
  `VOICELAYER_WHISPER_SERVER_ARGS`,
  `VOICELAYER_WHISPER_SERVER_LAUNCH_TIMEOUT_SECONDS`,
  `VOICELAYER_WHISPER_SERVER_POLL_INTERVAL_SECONDS` — autostart a local
  whisper-server subprocess.
- `VOICELAYER_WHISPER_VAD_ENABLED`,
  `VOICELAYER_WHISPER_VAD_MODEL_PATH`,
  `VOICELAYER_WHISPER_VAD_THRESHOLD`,
  `VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS`,
  `VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS`,
  `VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS`,
  `VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS`,
  `VOICELAYER_WHISPER_VAD_SAMPLE_RATE` — optional silero-vad pre-pass.

### OpenAI-compatible LLM

- `VOICELAYER_LLM_ENDPOINT`, `VOICELAYER_LLM_MODEL`,
  `VOICELAYER_LLM_API_KEY`, `VOICELAYER_LLM_TIMEOUT_SECONDS`.
- `VOICELAYER_LLM_AUTO_START`, `VOICELAYER_LLAMA_SERVER_BIN`,
  `VOICELAYER_LLAMA_MODEL_PATH`, `VOICELAYER_LLAMA_HF_REPO`,
  `VOICELAYER_LLAMA_SERVER_ARGS`,
  `VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS`,
  `VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS` — optional llama-server autostart.

## Manual install

If you prefer not to run `scripts/install.sh`:

```bash
cargo build --release --bin vl --bin vl-desktop
install -m 0755 target/release/vl ~/.local/bin/vl
install -m 0755 target/release/vl-desktop ~/.local/bin/vl-desktop
install -d ~/.config/systemd/user ~/.config/voicelayer
install -m 0644 systemd/voicelayerd.service ~/.config/systemd/user/voicelayerd.service
[[ -f ~/.config/voicelayer/voicelayerd.env ]] || \
  install -m 0600 systemd/voicelayerd.env.example ~/.config/voicelayer/voicelayerd.env
systemctl --user daemon-reload
systemctl --user enable --now voicelayerd
```

## Uninstall

```bash
systemctl --user disable --now voicelayerd
rm -f ~/.config/systemd/user/voicelayerd.service
rm -f ~/.local/bin/vl ~/.local/bin/vl-desktop
systemctl --user daemon-reload
```

Environment files under `~/.config/voicelayer/` are left in place so a later
reinstall can reuse them.
