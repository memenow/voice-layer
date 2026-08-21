# VoiceLayer Contributor Instructions

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.
`AGENTS.md` is a symbolic link to this file — only edit `CLAUDE.md`.

## Product Focus

VoiceLayer is a local-first voice composition layer for Ubuntu desktop environments.
It is not a traditional input method editor. The product must support:

- Low-latency dictation into the focused application.
- Structured composition workflows for longer text such as email, issue descriptions, prompts, and technical notes.
- Text rewrite and translation workflows.
- GUI and terminal/TUI targets with the same domain model.

## Engineering Rules

- Research dependencies and prior art before adding new code paths or third-party packages.
- Prefer readable, direct implementations over clever abstractions.
- Default to Google-style API design and OpenAPI documentation for public interfaces.
- Use inclusive language in code, docs, and user-visible text.
- Keep the core repository Apache-2.0 friendly. Any restrictive or copyleft component must stay outside the required runtime path.
- Treat local execution as the default. Cloud providers are optional enhancements.

## Architecture Defaults

- Rust owns the long-running daemon, CLI/TUI, desktop integration, process supervision, and host adapters.
- Python owns model orchestration, experimentation, and provider-specific worker implementations.
- Inter-process communication between Rust and Python uses JSON-RPC 2.0 over stdio with one persistent worker process; the first frame is `initialize`, which carries the provider configuration from the daemon.
- The daemon exposes a local `/v1` API over a Unix domain socket and documents it with OpenAPI 3.1 (`openapi/voicelayerd.v1.yaml`). Failures use RFC 9457 problem+json.
- Every daemon consumer (CLI, desktop shell) goes through the `voicelayer-client` crate; no consumer links daemon internals.
- Configuration lives in a single TOML file; `VOICELAYER_*` environment variables are an override layer applied only by Rust.

## Host Strategy

- Ubuntu GNOME Wayland is the primary desktop target; macOS (Apple Silicon) is the second supported platform.
- Global shortcuts: XDG Global Shortcuts portal on Linux, `global-hotkey` (Carbon) on macOS.
- GUI text injection: AT-SPI editable text operations on Linux; clipboard + synthetic Cmd+V (CoreGraphics) on macOS.
- Terminal injection should prefer bracketed paste and must not auto-submit by default.
- Keyboard simulation tools such as `ydotool` or `wtype` are fallbacks, not the primary strategy.
- Audio capture is in-process via `cpal` (PipeWire ALSA shim on Linux, CoreAudio on macOS); no recorder subprocesses.
- Platform differences live in `voicelayerd::platform` and the `vl-desktop` hotkeys module; business code must stay free of `cfg(target_os)`.

## Workflow

- Before large feature work, complete the Discovery Gate in Serena memories under `features/<feature>/`.
- If implementation diverges from `docs/`, update the docs before or alongside the code change.
- Use `uv` for Python commands and isolated environments. Do not rely on the system interpreter for project tasks.
- Before closing a task, run the verification chain in the Commands section below.
- Keep README, OpenAPI, and architecture docs aligned with shipped behavior.
- Use GitHub flow: feature branch per change, open PRs with `gh pr create`, and follow conventional commits.

## Commands

- Verify before closing a task (authoritative chain):

  ```bash
  cargo fmt --all \
    && cargo clippy --all-targets -- -D warnings \
    && cargo test \
    && uv run ruff check python tests/python \
    && uv run ruff format --check python tests/python \
    && uv run pytest -q tests/python
  ```

  `cargo test` covers the default workspace members (core, client, daemon,
  CLI). `vl-desktop` is excluded from default members (GUI system packages
  on Linux); run `cargo check -p vl-desktop` when touching it.

- Sync Python dev environment: `uv sync --group dev`
- Run the daemon from source: `cargo run -p voicelayerd`
- Inspect runtime environment and provider reachability: `vl doctor`
- List host adapters and worker providers: `vl providers`
- If the daemon is launched outside the repo root, set `VOICELAYER_PROJECT_ROOT` so the Python worker resolves.

## Domain Vocabulary

Public types use the domain terms `CaptureSession`, `PreviewArtifact`, `InjectionPlan`, and `ProviderDescriptor`.
The canonical schema for the local daemon API lives at `openapi/voicelayerd.v1.yaml`.
