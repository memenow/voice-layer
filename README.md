# VoiceLayer

VoiceLayer is a local-first voice composition layer for Ubuntu and macOS (Apple Silicon) desktop workflows.
It combines low-latency dictation, structured text composition, rewrite, and translation behind a local daemon, a CLI/TUI, an optional desktop UI, and host injection adapters.

## Scope

VoiceLayer targets:

- Browser text areas and document editors
- IDE input fields and code review forms
- Terminal and TUI environments such as tmux, WezTerm, Kitty, and Neovim
- Drafting workflows that benefit from preview and confirmation before insertion

VoiceLayer is not an IME candidate window, a subtitle-only transcriber, a browser extension, or a cloud-only voice assistant.

## Architecture

- `crates/voicelayer-core`: shared domain types, unified TOML/env config, Unix-socket HTTP client, and injection planning
- `crates/voicelayer-testkit`: dev-only test helpers (`EnvGuard`, doc link/code-span scanners)
- `crates/voicelayer-ui`: shared color, typography, accessibility, and terminal rendering tokens
- `crates/voicelayerd`: Unix-socket daemon, in-process `cpal` audio capture, `/v1` HTTP control API, SSE stream, and worker supervision
- `crates/voicelayer-cli`: `vl` CLI/TUI operator tool and foreground push-to-talk panel
- `crates/voicelayer-desktop`: `vl-desktop` interactive iced GUI shell, capture HUD, global hotkeys, and optional tray icon
- `python/voicelayer_worker`: stdio JSON-RPC 2.0 inference worker and provider adapters
- `openapi/voicelayerd.v1.yaml`: OpenAPI 3.1 contract for the `/v1` daemon API
- `systemd/` and `launchd/`: Linux systemd user units and macOS LaunchAgent template
- `scripts/install.sh`: release installer for binaries, service definitions, and environment defaults
- `docs/`: standalone HTML documentation starting at [`docs/index.html`](docs/index.html)
  - Architecture: [`docs/architecture/overview.html`](docs/architecture/overview.html), [`docs/architecture/worker-protocol.html`](docs/architecture/worker-protocol.html), [`docs/architecture/host-injection.html`](docs/architecture/host-injection.html)
  - Guides: [`docs/guides/development.html`](docs/guides/development.html), [`docs/guides/services.html`](docs/guides/services.html), [`docs/guides/asr-providers.html`](docs/guides/asr-providers.html), [`docs/guides/llm-provider.html`](docs/guides/llm-provider.html), [`docs/guides/desktop.html`](docs/guides/desktop.html)

## Capabilities

Implemented today:

- `/v1` HTTP API over a Unix domain socket with Server-Sent Events at `/v1/events/stream`
- One-shot, fixed-duration segmented, and Silero VAD-gated live dictation (`POST /v1/sessions/dictation` with `segmentation.mode = one_shot | fixed | vad_gated`)
- `vl dictation foreground-ptt` alternate-screen panel with hold-to-record, transcript scrolling, clipboard backup/restore, and tmux / WezTerm / Kitty targeting
- `vl-desktop` GUI shell sharing the daemon socket, session store, and SSE stream, with XDG GlobalShortcuts (Linux) and Carbon global hotkeys (macOS)
- Local ASR via `whisper.cpp` (`whisper-cli` and persistent `whisper-server` with optional autostart), plus opt-in GPU providers (`mimo_v2_5_asr`, `qwen3_asr_1_7b`) selected via `provider_id`
- Optional Silero VAD pre-pass applied uniformly across ASR backends inside the Python worker
- Compose, rewrite, and translate workflows backed by any OpenAI-compatible chat completions endpoint, with optional `llama-server` autostart
- `vl doctor` and `vl providers` diagnostics for ASR mode, LLM reachability, and global hotkey availability

Planned / out of current scope:

- Global shortcut portal binding inside the `vl` CLI (handled today by `vl-desktop` and raw-mode terminal PTT)
- AT-SPI writable target discovery
- Always-on background listening and mid-utterance partial transcripts
- `.deb` packaging

## Development

### Requirements

- Rust 1.100+
- Python 3.12+
- `uv` 0.11+
- Ubuntu with PipeWire, or macOS on Apple Silicon (CoreAudio)

### Verification Chain

Every change must pass the authoritative check sequence (also enforced in CI and documented in `CLAUDE.md`):

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
uv sync --group dev
uv run ruff check python tests/python
uv run ruff format --check python tests/python
uv run pytest -q tests/python
```

Run all Python commands in this repository through `uv`.

### Worker Runtime

`voicelayerd` resolves the Python worker under `VOICELAYER_PROJECT_ROOT` (falling back to the current working directory):

1. `<project_root>/.venv/bin/python -m voicelayer_worker.worker`
2. `uv run --project <project_root> python -m voicelayer_worker.worker`

When starting the daemon outside the repository root, set `VOICELAYER_PROJECT_ROOT`.
If a local LLM endpoint is configured, `vl doctor` probes `/v1/models`. With `VOICELAYER_LLM_AUTO_START=true`, the worker launches `llama-server` on demand for local endpoints.

### Common Commands

Start the daemon (default socket `$XDG_RUNTIME_DIR/voicelayer/daemon.sock` on Linux or `$TMPDIR/voicelayer/daemon.sock` on macOS):

```bash
cargo run -p voicelayerd
```

Inspect runtime health, provider catalog, and hotkey portal status:

```bash
cargo run -p voicelayer-cli -- doctor
cargo run -p voicelayer-cli -- providers
cargo run -p voicelayer-cli -- hotkeys portal-status
```

Transcribe an audio file, record a fixed clip, or render a bracketed paste sequence:

```bash
cargo run -p voicelayer-cli -- transcribe-file /path/to/sample.wav --language auto
cargo run -p voicelayer-cli -- record-transcribe --duration-seconds 8 --language auto
cargo run -p voicelayer-cli -- print-bracketed-paste "Review the authentication changes."
```

Drive live dictation sessions (`POST /v1/sessions/dictation`, `POST /v1/sessions/dictation/stop`, `GET /v1/sessions`):

```bash
cargo run -p voicelayer-cli -- dictation start --language auto
cargo run -p voicelayer-cli -- dictation start --mode fixed --segment-secs 8 --language auto
cargo run -p voicelayer-cli -- dictation start --mode vad-gated --probe-secs 2 --max-segment-secs 30 --language auto
cargo run -p voicelayer-cli -- dictation list
cargo run -p voicelayer-cli -- dictation stop <session-id>
```

Live segmented sessions stream `dictation.segment_recorded` / `dictation.segment_transcribed` (fixed mode) or `dictation.probe_analyzed` / `dictation.speech_unit_flushed` / `dictation.speech_unit_transcribed` (VAD-gated mode) alongside `dictation.completed` / `dictation.failed` on `/v1/events/stream`.

Run the foreground push-to-talk terminal panel:

```bash
cargo run -p voicelayer-cli -- dictation foreground-ptt --language auto
cargo run -p voicelayer-cli -- dictation foreground-ptt \
  --default-stop-action inject \
  --copy-on-stop \
  --restore-clipboard-on-exit \
  --save-dir ~/Documents/voice-layer
```

Panel keybindings: `j`/`k` or arrows and `PageUp`/`PageDown` to scroll, `c` to copy, `r` to restore the clipboard backup, `i` to re-inject, `s` to save, `d` to discard, and `Esc` to exit.
When the terminal supports key release events, the PTT key operates as hold-to-record; otherwise pressing once starts and pressing again stops.

Explicit terminal injection targets:

```bash
cargo run -p voicelayer-cli -- dictation foreground-ptt --tmux-target-pane %2
cargo run -p voicelayer-cli -- dictation foreground-ptt --wezterm-target-pane-id 12
cargo run -p voicelayer-cli -- dictation foreground-ptt --kitty-match 'title:Output'
```

Inside tmux, omitting `--tmux-target-pane` auto-selects a single candidate pane or prompts when multiple panes exist, and never pastes into the controller's own pane.

Manage persistent defaults in `~/.config/voicelayer/config.toml`:

```bash
cargo run -p voicelayer-cli -- config path
cargo run -p voicelayer-cli -- config init-defaults
cargo run -p voicelayer-cli -- config show
cargo run -p voicelayer-cli -- config set foreground_ptt.default_stop_action inject
```

### Guides

- Local ASR (`whisper-cli`, `whisper-server`, Silero VAD, MiMo-V2.5-ASR, Qwen3-ASR-1.7B): [`docs/guides/asr-providers.html`](docs/guides/asr-providers.html)
- Local LLM (OpenAI-compatible endpoint, `llama-server` autostart): [`docs/guides/llm-provider.html`](docs/guides/llm-provider.html)
- Desktop shell (`vl-desktop`, capture HUD, hotkeys, client env vars): [`docs/guides/desktop.html`](docs/guides/desktop.html)
- User services (`scripts/install.sh`, systemd units, launchd agent): [`docs/guides/services.html`](docs/guides/services.html)

## License

Apache License 2.0. See [LICENSE](LICENSE).
