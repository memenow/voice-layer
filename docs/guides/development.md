# Development Guide

## Repository Layout

- `crates/voicelayer-core`: shared domain and contracts
- `crates/voicelayerd`: daemon and local HTTP API over a Unix domain socket
- `crates/vl`: operator CLI and TUI entry point
- `crates/vl-desktop`: interactive GUI shell that shares the daemon socket
- `python/voicelayer_orchestrator`: worker protocol and provider runtime (`providers/` holds one file per provider surface — whisper-cli, whisper-server, silero-vad, llama autostart, OpenAI-compatible LLM)
- `openapi/`: local API contract
- `systemd/`: user service templates for the daemon and the optional persistent `whisper-server`
- `scripts/`: installer, whisper-server wrapper, and cold-start benchmark helpers

## Verification Chain

The authoritative list every change must pass before merge (also mirrored in `README.md` and `CLAUDE.md`):

```bash
cargo fmt --all
cargo clippy --all-targets --all-features -- -D warnings
cargo test --all
uv sync --group dev
uv run ruff check python tests/python
uv run ruff format --check python tests/python
uv run pytest -q tests/python
```

## Useful Runtime Commands

```bash
cargo run -p vl -- providers
cargo run -p vl -- preview compose "Write a concise technical summary."
cargo run -p vl -- transcribe-file /path/to/sample.wav --language auto
cargo run -p vl -- record-transcribe --duration-seconds 8 --language auto
cargo run -p vl -- dictation start --backend pipewire --language auto
cargo run -p vl -- dictation start --mode fixed --segment-secs 8 --language auto
cargo run -p vl -- dictation start --mode vad-gated --probe-secs 2 --max-segment-secs 30 --language auto
cargo run -p vl -- dictation stop <session-id>
cargo run -p vl -- dictation foreground-ptt --backend pipewire --language auto
cargo run -p vl -- hotkeys portal-status
cargo run -p vl-desktop
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

For real composition / rewrite / translation, configure `VOICELAYER_LLM_ENDPOINT` and `VOICELAYER_LLM_MODEL` as described in `docs/guides/local-llm-provider.md`. When the endpoint is unset or unreachable, the worker returns a `provider-unavailable` JSON-RPC error instead of fabricating output.
If you also set `VOICELAYER_LLM_AUTO_START=true`, VoiceLayer can auto-launch `llama-server` for a local endpoint.
For file-based ASR, configure `VOICELAYER_WHISPER_BIN` and `VOICELAYER_WHISPER_MODEL_PATH` as described in `docs/guides/local-asr-provider.md`. For warm-model reuse across transcribe calls, point at a persistent `VOICELAYER_WHISPER_SERVER_*` endpoint (see the same guide).

## Operational Notes

- The daemon exposes a `/v1` API over a Unix domain socket with Server-Sent Events at `/v1/events/stream`. Every handler is covered by in-process integration tests under `crates/voicelayerd/src/lib.rs`'s `#[cfg(test)]` modules (`http_api_tests`, `sse_stream_tests`, `segmented_orchestration_tests`).
- `compose`, `rewrite`, and `translate` call the configured OpenAI-compatible chat completion endpoint through the Python worker's `providers/llm_openai_compatible.py`. When no endpoint is configured they surface JSON-RPC `-32004` (provider unavailable); when the endpoint fails they surface `-32005` (provider request failed).
- `transcribe` prefers a persistent `whisper-server` endpoint (see `providers/whisper_server.py`) and falls back to one-shot `whisper-cli` (`providers/whisper_cli.py`). When `VOICELAYER_WHISPER_VAD_ENABLED=true` with a valid silero-vad ONNX model, a pre-pass in `providers/vad_segmenter.py` trims non-speech before whisper — failure downgrades to the raw WAV without losing the request.
- `vl record-transcribe` records a short WAV clip with `pw-record` or `arecord`, then reuses the daemon's one-shot capture path into the same transcription flow.
- `voicelayerd` serves the daemon-side one-shot capture path at `/v1/dictation/capture`, which advances session state from `listening` to `transcribing` to `completed` or `failed`, setting `DictationFailureKind` on the unhappy paths.
- `voicelayerd` also serves live dictation sessions at `/v1/sessions/dictation` (start) and `/v1/sessions/dictation/stop`. `StartDictationRequest.segmentation` selects between `{"mode": "one_shot"}`, `{"mode": "fixed", "segment_secs": N, "overlap_secs": 0}`, and `{"mode": "vad_gated", "probe_secs": P, "max_segment_secs": M, "silence_gap_probes": G}`. In `fixed` mode the recorder rolls every `N` seconds, each chunk is transcribed in the background, and per-segment events (`dictation.segment_recorded`, `dictation.segment_transcribed`) stream on `/v1/events/stream` alongside the lifecycle events. In `vad_gated` mode the recorder rolls at `probe_secs` cadence, each probe is classified by silero-vad (via the worker's `segment_probe` RPC), pending speech probes flush on silence or at `max_segment_secs`, and per-probe / per-unit events (`dictation.probe_analyzed`, `dictation.speech_unit_flushed`, `dictation.speech_unit_transcribed`) stream alongside the lifecycle events. VAD-gated mode requires `VOICELAYER_WHISPER_VAD_ENABLED=true` and a silero-vad ONNX model path.
- `vl dictation start/stop` uses the daemon UDS client, so the CLI exercises the same control plane hotkey and UI integrations use. `--mode {one-shot,fixed,vad-gated}` selects the segmentation strategy; `fixed` requires `--segment-secs`, `vad-gated` requires `--probe-secs` and `--max-segment-secs` (with an optional `--silence-gap-probes`, default 1). Omitting `--mode` keeps the pre-Phase-3 one-shot behavior.
- `vl dictation foreground-ptt` provides a terminal-local fallback for push-to-talk. It enables raw mode and keyboard enhancement flags; if the terminal reports release events it behaves like hold-to-record, otherwise it falls back to press-to-start / press-again-to-stop.
- The foreground-ptt alternate-screen panel preserves the full last transcript for scrolling review and supports `j/k`, arrow keys, and `PageUp/PageDown` for navigation. Press `c` to copy the last completed transcript to the clipboard on demand.
- Press `r` to restore the saved clipboard text backup after any clipboard overwrite performed by the panel.
- The panel also supports lightweight result actions: `i` re-applies the last injection, `s` saves the last transcript to a timestamped file under `$XDG_STATE_HOME/voicelayer/transcripts/` (default `~/.local/state/voicelayer/transcripts/`, override with `--save-dir`), and `d` discards the last transcript from the panel state.
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
- `cargo run -p vl -- doctor` reports recorder diagnostics, whisper mode (`cli` / `server` / `unconfigured`), LLM reachability, portal availability, and systemd unit state (`installed` / `active`).
- When whisper-server or llama-server autostart is enabled, provider runtime files are written under `$XDG_RUNTIME_DIR/voicelayer/providers`.
- `vl-desktop` is an interactive GUI shell; see `docs/guides/desktop.md` for its two client-side env vars. It shares the same daemon socket, session store, and SSE stream as the CLI.
- To install VoiceLayer as a systemd user service, run `scripts/install.sh` (see `docs/guides/systemd.md`). The installer builds release binaries, copies units into `~/.config/systemd/user/`, and leaves the optional `voicelayer-whisper-server` unit disabled by default.
- Python commands must run through `uv`, not the system interpreter.
