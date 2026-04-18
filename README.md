# VoiceLayer

VoiceLayer is a local-first voice composition layer for Ubuntu desktop workflows.
It combines fast dictation, structured text composition, rewrite, and translation into a single daemon, CLI/TUI, and host-injection stack.

## Scope

VoiceLayer is designed for:

- Browser text areas and document editors
- IDE input surfaces and comment fields
- Terminal and TUI applications such as tmux, Neovim, Claude Code, and Codex CLI
- Drafting workflows that need preview and confirmation before insertion

VoiceLayer is not designed as:

- A traditional IME candidate window
- A subtitle-only transcriber
- A browser-only extension
- A cloud-only voice assistant

## Architecture

- `crates/voicelayer-core`: shared domain types and injection planning
- `crates/voicelayerd`: Unix-socket daemon and `/v1` control API
- `crates/vl`: CLI/TUI entry point and operator tooling
- `python/voicelayer_orchestrator`: JSON-RPC worker protocol and provider orchestration entry point
- `docs/`: architecture, host strategy, and operations documentation
- `openapi/`: local API contract

## Current Status

This repository currently provides the foundation layer:

- Rust workspace scaffolding
- Local daemon API skeleton over a Unix domain socket
- CLI commands for daemon startup, environment inspection, provider discovery, and terminal-safe paste rendering
- CLI commands for daemon startup, environment inspection, provider discovery, terminal-safe paste rendering, and file-based ASR transcription
- A daemon-side dictation capture path that records a short clip, advances session state, and returns a transcript
- A live dictation session lifecycle in the daemon with explicit start/stop semantics
- Python worker protocol scaffolding for future ASR, LLM, and TTS providers
- A live Rust-to-Python stdio JSON-RPC bridge using the `uv`-managed project environment
- A first real LLM integration path for OpenAI-compatible local endpoints such as `llama.cpp server`
- A first real ASR integration path for `whisper.cpp` file transcription
- Architecture and API documentation for the initial product shape

Audio capture, model execution, GNOME accessibility integration, and terminal-specific adapters are documented and bounded, but not yet fully implemented in this scaffold.

## Development

### Requirements

- Rust 1.86+
- Python 3.12+
- `uv` 0.11+
- Ubuntu with PipeWire

### Useful Commands

```bash
cargo fmt --all
cargo test --all
uv sync --group dev
uv run python -m unittest discover -s tests/python
uv run pytest -q tests/python
```

Python commands in this repository should always run through `uv`.

### Worker Runtime

The daemon and CLI launch the Python worker from the project-managed environment.
Resolution order is:

1. `VOICELAYER_PROJECT_ROOT/.venv/bin/python -m voicelayer_orchestrator.worker`
2. `uv run --project <project_root> python -m voicelayer_orchestrator.worker`

If you run the daemon outside the repository root, set `VOICELAYER_PROJECT_ROOT` explicitly.
When a local LLM endpoint is configured, `vl doctor` also probes endpoint reachability through `/v1/models`.
If `VOICELAYER_LLM_AUTO_START=true`, the worker can also auto-launch `llama-server` for local endpoints.

### Run the Daemon

```bash
cargo run -p vl -- daemon run --project-root "$(pwd)"
```

By default the daemon listens on:

```text
$XDG_RUNTIME_DIR/voicelayer/daemon.sock
```

### Inspect the Environment

```bash
cargo run -p vl -- doctor
```

### Inspect Providers

```bash
cargo run -p vl -- providers
```

### Configure a Local LLM Endpoint

See [docs/guides/local-llm-provider.md](/home/billduke/Documents/memenow/voice-layer/docs/guides/local-llm-provider.md) for the `llama.cpp server` path and environment variables.

### Configure a Local ASR Provider

See [docs/guides/local-asr-provider.md](/home/billduke/Documents/memenow/voice-layer/docs/guides/local-asr-provider.md) for the `whisper.cpp` file transcription path and environment variables.

### Render a Bracketed Paste Payload

```bash
cargo run -p vl -- print-bracketed-paste "Analyze the current repository authentication flow."
```

### Transcribe a Local Audio File

```bash
cargo run -p vl -- transcribe-file /path/to/sample.wav --language auto
```

### Record and Transcribe a Short Clip

```bash
cargo run -p vl -- record-transcribe --duration-seconds 8 --language auto
```

The CLI currently prefers `pw-record` with `timeout --signal=INT` and falls back to `arecord`.
Internally this now reuses the same daemon-side dictation capture flow that the future UI and hotkey layer will call.

The daemon now also exposes a live dictation session flow:

- `POST /v1/sessions/dictation` starts recording
- `POST /v1/sessions/dictation/stop` stops recording and returns the transcript

The `vl` CLI now exercises that control plane directly:

```bash
cargo run -p vl -- dictation start --backend pipewire --language auto
cargo run -p vl -- dictation stop <session-id>
```

`foreground-ptt` now uses an alternate-screen status panel instead of streaming JSON on each transition.
The panel shows:

- current dictation status
- active session ID
- last completed session ID
- last transcript preview
- last injection result
- last error
- recent events

Panel controls now include:

- `j` / `k` or Up / Down to scroll the full transcript view
- `PageUp` / `PageDown` for larger transcript jumps
- `c` to copy the last completed transcript to the system clipboard on demand
- `r` to restore the saved text clipboard backup after the tool has overwritten the clipboard
- `i` to re-apply the last injection target
- `s` to save the last transcript to a timestamped text file
- `d` to discard the last transcript from the panel
- `Esc` to exit

If you also want a clipboard fallback after each completed dictation:

```bash
cargo run -p vl -- dictation foreground-ptt --backend pipewire --language auto --copy-on-stop
```

This writes the finished transcript to the system clipboard before any optional terminal-target injection.

You can now change the default stop behavior without leaving the panel:

```bash
cargo run -p vl -- dictation foreground-ptt \
  --default-stop-action inject \
  --restore-clipboard-on-exit \
  --save-dir ~/Documents/voice-layer
```

Available default stop actions are:

- `none`
- `copy`
- `inject`
- `save`

VoiceLayer can also persist these defaults in a local config file:

```bash
cargo run -p vl -- config path
cargo run -p vl -- config init-defaults
cargo run -p vl -- config show
cargo run -p vl -- config set foreground_ptt.default_stop_action inject
```

The config file lives at:

```text
~/.config/voicelayer/config.toml
```

For terminal-focused fallback usage, `vl` also provides a foreground raw-terminal mode:

```bash
cargo run -p vl -- dictation foreground-ptt --backend pipewire --language auto
```

When the terminal reports key release events, this behaves like hold-to-record.
When release events are not available, it degrades to:

- first key press starts dictation
- second key press stops dictation
- `Esc` exits the mode

If you run the controller inside tmux and want the transcript pasted into another pane:

```bash
cargo run -p vl -- dictation foreground-ptt --backend pipewire --language auto --tmux-target-pane %2
```

This uses `tmux set-buffer` plus `tmux paste-buffer -dpr -t <pane>`.
The controller refuses to paste into the same pane that is currently running `foreground-ptt`.

If you omit `--tmux-target-pane` while running inside tmux:

- zero candidate panes: no tmux injection is attempted
- one candidate pane: it is selected automatically
- multiple candidate panes: `vl` prompts you to choose a target pane before entering raw mode

For terminal-specific explicit targets outside tmux:

```bash
cargo run -p vl -- dictation foreground-ptt --wezterm-target-pane-id 12
cargo run -p vl -- dictation foreground-ptt --kitty-match 'title:Output'
```

These routes are explicit-only:

- WezTerm uses `wezterm cli send-text --pane-id`
- Kitty uses `kitten @ send-text --match ... --stdin --bracketed-paste auto`

VoiceLayer does not auto-discover WezTerm or Kitty targets yet.

### Inspect Global Shortcuts Portal Support

```bash
cargo run -p vl -- hotkeys portal-status
```

This checks whether the current desktop session exposes `org.freedesktop.portal.GlobalShortcuts`.

## Product Defaults

- Desktop target: Ubuntu GNOME Wayland
- Local ASR baseline: `whisper.cpp`
- Local LLM baseline: `Gemma 4` via `llama.cpp`-compatible deployment
- GUI insertion priority: AT-SPI, then clipboard, then keyboard simulation fallback
- Terminal insertion priority: bracketed paste, then terminal-specific adapters
- Preview surface: CLI/TUI first, GUI preview later

## License

The repository is intended to ship under the Apache License 2.0.
