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
2. Installs `vl`, `vl-desktop`, and
   `voicelayer-whisper-server-run.sh` into `~/.local/bin` (override with
   `VOICELAYER_INSTALL_BIN_DIR`).
3. Copies both `systemd/voicelayerd.service` and
   `systemd/voicelayer-whisper-server.service` to `~/.config/systemd/user/`
   (override with `VOICELAYER_INSTALL_UNIT_DIR`). The whisper-server unit
   is installed but left disabled — enable it only if you want server-mode
   ASR (see [Persistent whisper-server](#persistent-whisper-server)).
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
  Defaults to `$XDG_RUNTIME_DIR/voicelayer/daemon.sock`. The shipped unit
  hardcodes `--socket-path %t/voicelayer/daemon.sock`, so this env var is
  only consulted when the daemon is launched outside systemd.
- `VOICELAYER_WORKER_TIMEOUT_SECONDS` — per-call upper bound on worker
  JSON-RPC inference calls (`transcribe`, `compose`, `rewrite`,
  `translate`). Probes (`health`, `list_providers`) use a fixed 15s
  budget. Defaults to 600.

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

## Persistent whisper-server

whisper.cpp's `whisper-server` keeps the ggml model mmapped across
requests, cutting ~260ms/capture compared with spawning `whisper-cli`
per request. You have two ways to keep one running:

| Option | When to pick it | Managed by |
| --- | --- | --- |
| `VOICELAYER_WHISPER_SERVER_AUTO_START=1` in `voicelayerd.env` | You want the daemon to spawn the server lazily on first transcribe and reap it on daemon exit. | `voicelayerd` |
| `systemctl --user enable --now voicelayer-whisper-server` | You want the server up on login independent of the daemon, with its own journald logs and restart policy. | dedicated unit |

Pick **one** — running both simultaneously makes the daemon's autostart
path race the dedicated unit for the same port. When the dedicated unit
is active, leave `VOICELAYER_WHISPER_SERVER_AUTO_START` unset or `false`.

### Enable the dedicated unit

```bash
# Required env keys (uncomment in ~/.config/voicelayer/voicelayerd.env):
#   VOICELAYER_WHISPER_SERVER_BIN=/absolute/path/to/whisper-server
#   VOICELAYER_WHISPER_MODEL_PATH=/absolute/path/to/ggml-base.en.bin
#   VOICELAYER_WHISPER_SERVER_PORT=8188
systemctl --user enable --now voicelayer-whisper-server
systemctl --user status voicelayer-whisper-server
journalctl --user -u voicelayer-whisper-server
```

The unit invokes `voicelayer-whisper-server-run.sh` (installed to
`~/.local/bin/`), which validates the env, reads
`VOICELAYER_WHISPER_SERVER_HOST` / `_PORT` / `_ARGS`, and `exec`s the
binary so systemd tracks the server PID directly.

`vl doctor` reports the current mode under `whisper_mode`:
`"server"` once the daemon successfully probes
`http://$HOST:$PORT/`, otherwise `"cli"` or `"unconfigured"`.

### Docker alternative

The shipped unit runs a native `whisper-server` binary. To use the
prebuilt `ghcr.io/ggml-org/whisper.cpp:main` image instead, add a
drop-in override:

```bash
mkdir -p ~/.config/systemd/user/voicelayer-whisper-server.service.d
cat > ~/.config/systemd/user/voicelayer-whisper-server.service.d/docker.conf <<'EOF'
[Service]
# Clear the native ExecStart and swap in docker.
ExecStart=
ExecStart=/usr/bin/docker run --rm --name voicelayer-whisper-server \
  -p 127.0.0.1:8188:8188 \
  -v %h/.cache/voicelayer/models:/models:ro \
  --entrypoint whisper-server \
  ghcr.io/ggml-org/whisper.cpp:main \
  -m /models/ggml-base.en.bin \
  --host 0.0.0.0 --port 8188 -ng
ExecStop=/usr/bin/docker stop voicelayer-whisper-server
EOF
systemctl --user daemon-reload
systemctl --user restart voicelayer-whisper-server
```

Adjust the model filename and port to match your `voicelayerd.env`.

## Manual install

If you prefer not to run `scripts/install.sh`:

```bash
cargo build --release --bin vl --bin vl-desktop
install -m 0755 target/release/vl ~/.local/bin/vl
install -m 0755 target/release/vl-desktop ~/.local/bin/vl-desktop
install -m 0755 scripts/voicelayer-whisper-server-run.sh \
  ~/.local/bin/voicelayer-whisper-server-run.sh
install -d ~/.config/systemd/user ~/.config/voicelayer
install -m 0644 systemd/voicelayerd.service ~/.config/systemd/user/voicelayerd.service
install -m 0644 systemd/voicelayer-whisper-server.service \
  ~/.config/systemd/user/voicelayer-whisper-server.service
[[ -f ~/.config/voicelayer/voicelayerd.env ]] || \
  install -m 0600 systemd/voicelayerd.env.example ~/.config/voicelayer/voicelayerd.env
systemctl --user daemon-reload
systemctl --user enable --now voicelayerd
# Optional: also enable whisper-server if you want server-mode ASR.
# systemctl --user enable --now voicelayer-whisper-server
```

## Uninstall

```bash
systemctl --user disable --now voicelayerd voicelayer-whisper-server
rm -f ~/.config/systemd/user/voicelayerd.service \
      ~/.config/systemd/user/voicelayer-whisper-server.service
rm -f ~/.local/bin/vl \
      ~/.local/bin/vl-desktop \
      ~/.local/bin/voicelayer-whisper-server-run.sh
systemctl --user daemon-reload
```

Environment files under `~/.config/voicelayer/` are left in place so a later
reinstall can reuse them.
