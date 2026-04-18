# Development Guide

## Repository Layout

- `crates/voicelayer-core`: shared domain and contracts
- `crates/voicelayerd`: daemon and local HTTP API over a Unix domain socket
- `crates/vl`: operator CLI and future TUI entry point
- `python/voicelayer_orchestrator`: worker protocol and provider runtime
- `openapi/`: local API contract
- `systemd/`: user service templates

## Local Commands

```bash
cargo fmt --all
cargo test --all
uv sync --group dev
uv run python -m unittest discover -s tests/python
uv run pytest -q tests/python
cargo run -p vl -- providers
cargo run -p vl -- preview compose "Write a concise technical summary."
cargo run -p vl -- transcribe-file /path/to/sample.wav --language auto
cargo run -p vl -- record-transcribe --duration-seconds 8 --language auto
cargo run -p vl -- dictation start --backend pipewire --language auto
cargo run -p vl -- dictation stop <session-id>
cargo run -p vl -- dictation foreground-ptt --backend pipewire --language auto
cargo run -p vl -- hotkeys portal-status
```

## Daemon Socket

The default socket path is:

```text
$XDG_RUNTIME_DIR/voicelayer/daemon.sock
```

## Worker Project Root

The Rust control plane discovers the Python worker from `VOICELAYER_PROJECT_ROOT`.
If the variable is unset, the daemon falls back to the current working directory.

For source-based development, start the daemon from the repository root or pass:

```bash
cargo run -p vl -- daemon run --project-root "$(pwd)"
```

If you want a real composition provider instead of the current `needs_provider` fallback, configure `VOICELAYER_LLM_ENDPOINT` and `VOICELAYER_LLM_MODEL` as described in `docs/guides/local-llm-provider.md`.
If you also set `VOICELAYER_LLM_AUTO_START=true`, VoiceLayer can auto-launch `llama-server` for a local endpoint.
For file-based ASR, configure `VOICELAYER_WHISPER_BIN` and `VOICELAYER_WHISPER_MODEL_PATH` as described in `docs/guides/local-asr-provider.md`.

## Operational Notes

- The current scaffold exposes a stable local API shape before provider wiring is complete.
- Generation and transcription requests already cross the Rust/Python process boundary. `compose`, `rewrite`, and `translate` can now call a configured OpenAI-compatible endpoint; they still return provider-unavailable until that endpoint is configured.
- `transcribe` can now invoke `whisper-cli` for local file transcription when `VOICELAYER_WHISPER_MODEL_PATH` is configured.
- `vl record-transcribe` records a short WAV clip with `pw-record` or `arecord`, then passes the file into the same `whisper-cli` transcription path.
- `voicelayerd` now exposes a daemon-side dictation capture path at `/v1/dictation/capture`, which advances session state from `listening` to `transcribing` to `completed` or `failed`.
- `voicelayerd` also exposes a live dictation session flow at `/v1/sessions/dictation` and `/v1/sessions/dictation/stop` for future push-to-talk integrations.
- `vl dictation start/stop` now uses the daemon UDS client instead of calling capture helpers directly, so the CLI exercises the same control plane that future hotkey/UI integrations will use.
- `vl dictation foreground-ptt` provides a terminal-local fallback for push-to-talk. It enables raw mode and keyboard enhancement flags; if the terminal reports release events it behaves like hold-to-record, otherwise it falls back to press-to-start / press-again-to-stop.
- `vl dictation foreground-ptt` now renders an alternate-screen status panel with session visibility, transcript preview, recent events, last injection result, and last error, instead of printing full JSON after every stop.
- The alternate-screen panel now preserves the full last transcript for scrolling review and supports `j/k`, arrow keys, and `PageUp/PageDown` for navigation. Press `c` to copy the last completed transcript to the clipboard on demand.
- Press `r` to restore the saved clipboard text backup after any clipboard overwrite performed by the panel.
- The panel also supports lightweight result actions: `i` re-applies the last injection, `s` saves the last transcript to a timestamped file in the current directory, and `d` discards the last transcript from the panel state.
- `vl dictation foreground-ptt --copy-on-stop` adds an explicit clipboard fallback using the system clipboard after each completed dictation.
- `vl dictation foreground-ptt --default-stop-action <none|copy|inject|save>` configures what the tool should try immediately after a stop.
- `--restore-clipboard-on-exit` restores the saved text clipboard snapshot when the tool exits.
- `--save-dir <path>` changes where the `save` action writes transcript files.
- `cargo run -p vl -- config init-defaults` writes a default config file to `~/.config/voicelayer/config.toml`.
- `cargo run -p vl -- config show` prints the effective on-disk config file content.
- `cargo run -p vl -- config path` prints the config file location.
- `cargo run -p vl -- config set foreground_ptt.default_stop_action inject` updates one supported config key in place.
- `vl dictation foreground-ptt --tmux-target-pane <pane>` can paste the finished transcript into another tmux pane using bracketed paste mode. It intentionally refuses to paste back into the controller's own pane.
- When `--tmux-target-pane` is omitted inside tmux, `vl` auto-discovers candidate panes. One candidate is auto-selected; multiple candidates trigger an interactive selection prompt before raw mode starts.
- `vl dictation foreground-ptt --wezterm-target-pane-id <id>` and `--kitty-match <match>` provide explicit terminal-specific injection routes for WezTerm and Kitty. These do not auto-discover targets yet.
- `vl hotkeys portal-status` probes `org.freedesktop.portal.GlobalShortcuts` through D-Bus so you can tell early whether GNOME/portal-based push-to-talk is viable on the current session.
- `cargo run -p vl -- doctor` now reports whether the configured LLM endpoint is reachable via `/v1/models`.
- `cargo run -p vl -- doctor` also reports whether `whisper-cli` and the configured whisper model are available.
- If auto-start is enabled, provider runtime files are written under `$XDG_RUNTIME_DIR/voicelayer/providers`.
- Bracketed paste rendering is implemented now because it is deterministic and safe to validate in isolation.
- Python commands must run through `uv`, not the system interpreter.
