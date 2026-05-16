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
- Inter-process communication between Rust and Python uses JSON-RPC over stdio.
- The daemon exposes a local `/v1` API over a Unix domain socket and documents it with OpenAPI 3.1.

## Host Strategy

- Ubuntu GNOME Wayland is the primary desktop target.
- Global shortcuts should prefer the XDG Global Shortcuts portal.
- GUI text injection should prefer AT-SPI editable text operations.
- Terminal injection should prefer bracketed paste and must not auto-submit by default.
- Keyboard simulation tools such as `ydotool` or `wtype` are fallbacks, not the primary strategy.

## Workflow

- Before large feature work, complete the Discovery Gate in Serena memories under `features/<feature>/`.
- If implementation diverges from `docs/`, update the docs before or alongside the code change.
- Use `uv` for Python commands and isolated environments. Do not rely on the system interpreter for project tasks.
- Before closing a task, run the verification chain in the Commands section below.
- Keep README, OpenAPI, and architecture docs aligned with shipped behavior.
- Documentation layout: standalone pages under `docs/` are maintained directly as HTML. The guard tests scan `docs/**/*.html` (excluding `docs/assets/`).
  They also scan the repo-root `README.md`; update the HTML page directly whenever shipped behavior changes.
- Use GitHub flow: feature branch per change, open PRs with `gh pr create`, and follow conventional commits.

## Commands

- Verify before closing a task (authoritative chain):

  ```bash
  cargo fmt --all \
    && cargo clippy --all-targets --all-features -- -D warnings \
    && cargo test --all \
    && uv run ruff check python tests/python \
    && uv run ruff format --check python tests/python \
    && uv run pytest -q tests/python
  ```

- Sync Python dev environment: `uv sync --group dev`
- Run the daemon from source: `cargo run -p vl -- daemon run --project-root "$(pwd)"`
- Inspect runtime environment and provider reachability: `cargo run -p vl -- doctor`
- List host adapters and worker providers: `cargo run -p vl -- providers`
- If the daemon is launched outside the repo root, set `VOICELAYER_PROJECT_ROOT` so the Python worker resolves.

## Domain Vocabulary

Public types use the domain terms `CaptureSession`, `PreviewArtifact`, `InjectionPlan`, and `ProviderDescriptor`.
The canonical schema for the local daemon API lives at `openapi/voicelayerd.v1.yaml`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **voice-layer** (2767 symbols, 5797 relationships, 244 execution flows). Use the GitNexus MCP tools to understand code, assess impact, and navigate safely.

> If any GitNexus tool warns the index is stale, run `npx gitnexus analyze` in terminal first.

## Always Do

- **MUST run impact analysis before editing any symbol.** Before modifying a function, class, or method, run `gitnexus_impact({target: "symbolName", direction: "upstream"})` and report the blast radius (direct callers, affected processes, risk level) to the user.
- **MUST run `gitnexus_detect_changes()` before committing** to verify your changes only affect expected symbols and execution flows.
- **MUST warn the user** if impact analysis returns HIGH or CRITICAL risk before proceeding with edits.
- When exploring unfamiliar code, use `gitnexus_query({query: "concept"})` to find execution flows instead of grepping. It returns process-grouped results ranked by relevance.
- When you need full context on a specific symbol — callers, callees, which execution flows it participates in — use `gitnexus_context({name: "symbolName"})`.

## Never Do

- NEVER edit a function, class, or method without first running `gitnexus_impact` on it.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis.
- NEVER rename symbols with find-and-replace — use `gitnexus_rename` which understands the call graph.
- NEVER commit changes without running `gitnexus_detect_changes()` to check affected scope.

## Resources

| Resource | Use for |
|----------|---------|
| `gitnexus://repo/voice-layer/context` | Codebase overview, check index freshness |
| `gitnexus://repo/voice-layer/clusters` | All functional areas |
| `gitnexus://repo/voice-layer/processes` | All execution flows |
| `gitnexus://repo/voice-layer/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
|------|---------------------|
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
