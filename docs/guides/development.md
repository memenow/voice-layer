# Development Guide

## Repository Layout

- `crates/voicelayer-core`: shared domain types, unified config schema, injection planning
- `crates/voicelayer-client`: typed Unix-socket client (the only way consumers talk to the daemon)
- `crates/voicelayerd`: daemon; modules: `api/`, `dictation/`, `audio/`, `worker/`, `session/`, `events/`, `platform/`
- `crates/vl`: operator CLI (pure client)
- `crates/vl-desktop`: iced desktop shell (pure client; excluded from default workspace members)
- `python/voicelayer_orchestrator`: worker protocol (`worker.py`, protocol-only) and provider orchestration (`providers/pipeline.py` + adapters)
- `openapi/`: local API contract (canonical schema, linted in CI with spectral)
- `systemd/`, `launchd/`: per-user service templates

## Verification Chain

```bash
cargo fmt --all \
  && cargo clippy --all-targets -- -D warnings \
  && cargo test \
  && uv run ruff check python tests/python \
  && uv run ruff format --check python tests/python \
  && uv run pytest -q tests/python
```

`cargo test` covers the default workspace members (core, client, daemon,
CLI). `vl-desktop` is checked separately (`cargo check -p vl-desktop`) and
needs the Linux GUI system packages listed in `.github/workflows/ci.yml` on
Ubuntu; on macOS it builds without extra packages.

## Local Commands

```bash
uv sync --group dev
cargo run -p vl -- providers
cargo run -p vl -- preview compose "Write a concise technical summary."
cargo run -p vl -- transcribe-file /path/to/sample.wav --language auto
cargo run -p vl -- record-transcribe --duration-seconds 8 --language auto
cargo run -p vl -- dictation start --language auto
cargo run -p vl -- dictation start --segment-secs 15
cargo run -p vl -- dictation stop <session-id>
cargo run -p vl -- dictation foreground-ptt --language auto
cargo run -p vl -- hotkeys status
```

## Daemon Socket

The default socket path is `$XDG_RUNTIME_DIR/voicelayer/daemon.sock` on
Linux and `$TMPDIR/voicelayer/daemon.sock` on macOS. The socket is created
owner-only (`0600`).

## Worker Project Root

The daemon discovers the Python worker from `VOICELAYER_PROJECT_ROOT`
(falling back to the current working directory). For source-based
development, start the daemon from the repository root or:

```bash
cargo run -p voicelayerd -- --project-root "$(pwd)"
```

## Configuration

One TOML file covers daemon, providers, and CLI defaults. The daemon parses
it (plus `VOICELAYER_*` overrides) and hands provider sections to the worker
in the `initialize` handshake; the worker never reads provider settings from
its own environment.

```bash
vl config path / show / init-defaults
vl config set llm.endpoint http://127.0.0.1:8080
vl config set whisper.model_path /models/ggml-base.en.bin
vl config set foreground_ptt.default_stop_action inject
```

`vl config set <key> none` removes a key.

## Operational Notes

- All daemon access goes through `voicelayer-client`; the CLI and desktop
  shell hold no daemon internals.
- API failures are RFC 9457 problem+json; the problem type registry lives in
  `openapi/voicelayerd.v1.yaml`.
- The Python worker is a persistent process; `initialize` is the first
  JSON-RPC frame and carries the provider config.
- Provider autostart state files live under the runtime directory
  (`voicelayer/providers`).
- `vl dictation foreground-ptt` renders an alternate-screen status panel
  (scroll with `j/k`/arrows/PageUp/PageDown, `c` copy, `r` restore clipboard,
  `i` re-inject, `s` save, `d` discard, `Esc` exit).
- On macOS, `--default-stop-action inject` without a terminal target pastes
  via clipboard + synthetic Cmd+V (requires Input Monitoring permission).
- Python commands must run through `uv`, not the system interpreter.
