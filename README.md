# VoiceLayer

VoiceLayer is a local-first voice composition layer for desktop workflows.
It combines fast dictation, structured text composition, rewrite, and
translation into a single daemon, CLI/TUI, and host-injection stack.

Supported platforms: **Ubuntu GNOME Wayland** (primary) and **macOS (Apple
Silicon)**.

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

- `crates/voicelayer-core`: shared domain types, unified config schema, injection planning
- `crates/voicelayer-client`: typed Unix-socket client used by every consumer
- `crates/voicelayerd`: Unix-socket daemon serving the `/v1` control API
- `crates/vl`: CLI/TUI entry point (pure client of the daemon socket)
- `crates/vl-desktop`: iced desktop shell (pure client; not a default workspace member)
- `python/voicelayer_orchestrator`: JSON-RPC worker protocol and provider orchestration
- `docs/`: architecture, host strategy, and operations documentation
- `openapi/`: local API contract (canonical schema)

Runtime topology:

```text
+------------------+      HTTP/JSON over UDS     +------------------+
| CLI / TUI / GUI  | <-------------------------> | voicelayerd      |
+------------------+                             +------------------+
                                                        |
                                                        | JSON-RPC over stdio
                                                        | (persistent process,
                                                        |  initialize handshake)
                                                        v
                                                 +----------------------+
                                                 | Python orchestrator  |
                                                 +----------------------+
```

Key properties:

- The daemon owns audio capture (in-process, via cpal), the dictation
  session lifecycle, and one persistent Python worker process.
- Configuration lives in a single TOML file; the daemon hands the
  provider-facing sections to the worker in the `initialize` handshake.
  `VOICELAYER_*` environment variables act as an override layer.
- API errors are RFC 9457 `application/problem+json` with semantic status
  codes; see `openapi/voicelayerd.v1.yaml` for the problem type registry.

## Development

### Requirements

- Rust 1.88+
- Python 3.12+
- `uv` 0.11+
- Ubuntu with PipeWire, or macOS on Apple Silicon

### Useful Commands

```bash
cargo fmt --all
cargo clippy --all-targets -- -D warnings
cargo test
uv sync --group dev
uv run ruff check python tests/python
uv run pytest -q tests/python
```

Python commands in this repository should always run through `uv`.

### Install

```bash
scripts/install.sh
```

Builds `voicelayerd`, `vl`, and `vl-desktop`, installs them to
`~/.local/bin`, registers the per-user service (systemd on Linux, launchd on
macOS), and seeds the config file.

### Configuration

One file covers the daemon, providers, and CLI defaults:

- Linux: `~/.config/voicelayer/config.toml`
- macOS: `~/Library/Application Support/com.memenow.voicelayer/config.toml`

```bash
vl config path
vl config init-defaults
vl config show
vl config set llm.endpoint http://127.0.0.1:8080
vl config set whisper.model_path /models/ggml-base.en.bin
vl config set foreground_ptt.default_stop_action inject
```

Provider guides: [docs/guides/local-llm-provider.md](docs/guides/local-llm-provider.md),
[docs/guides/local-asr-provider.md](docs/guides/local-asr-provider.md).

### Run the Daemon

The service managers run `voicelayerd` directly:

```bash
systemctl --user enable --now voicelayerd     # Linux
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.memenow.voicelayerd.plist  # macOS
```

By default the daemon listens on `$XDG_RUNTIME_DIR/voicelayer/daemon.sock`
(macOS: `$TMPDIR/voicelayer/daemon.sock`).

### Inspect the Environment

```bash
vl doctor
```

### Dictation

```bash
vl dictation start --language auto
vl dictation stop <session-id>

# segmented capture: transcribe every 15s chunk while recording continues
vl dictation start --segment-secs 15 --overlap-secs 1
```

The foreground push-to-talk panel:

```bash
vl dictation foreground-ptt --language auto
vl dictation foreground-ptt --default-stop-action inject --tmux-target-pane %2
```

Panel controls: `j`/`k`/arrow keys scroll the transcript, `c` copies it, `r`
restores the clipboard backup, `i` re-applies injection, `s` saves to a
file, `d` discards, `Esc` exits. On macOS, `--default-stop-action inject`
without a terminal target pastes into the focused application via
clipboard + synthetic Cmd+V (requires Input Monitoring permission).

### Record and Transcribe a Short Clip

```bash
vl record-transcribe --duration-seconds 8 --language auto
vl transcribe-file /path/to/sample.wav --language auto
```

### Global Hotkey Status

```bash
vl hotkeys status
```

## Product Defaults

- Desktop targets: Ubuntu GNOME Wayland (primary), macOS Apple Silicon
- Local ASR baseline: `whisper.cpp` (CUDA on NVIDIA Linux, Metal on macOS)
- Local LLM baseline: `Gemma` via `llama.cpp`-compatible deployment
- GUI insertion priority: AT-SPI (Linux), clipboard + Cmd+V (macOS),
  then keyboard simulation fallback
- Terminal insertion priority: bracketed paste, then terminal-specific adapters
- Preview surface: CLI/TUI first, GUI preview later

## License

The repository ships under the Apache License 2.0.
