# VoiceLayer Contributor Instructions

This file provides contributor and agent instructions for this repository.
`AGENTS.md` is a symbolic link to this file; edit `CLAUDE.md` only.

## Product Focus

VoiceLayer is a local-first voice composition layer for Ubuntu and macOS (Apple Silicon) desktop workflows; Ubuntu GNOME Wayland is the primary target.
It is not a traditional input method editor. The product supports:

- Low-latency dictation into the focused application.
- Structured composition workflows for longer text such as email, issue descriptions, prompts, and technical notes.
- Text rewrite and translation workflows.
- GUI and terminal/TUI targets sharing the same domain model.

## Engineering Rules

- Research dependencies and prior art before adding new code paths or third-party packages.
- Prefer readable, direct implementations over unnecessary abstractions.
- Default to Google-style API design and OpenAPI documentation for public interfaces.
- Use inclusive language in code, docs, and user-visible text.
- Keep the core repository Apache-2.0 compatible. Restrictive or copyleft components must stay outside the required runtime path.
- Keep local execution as the default. Cloud providers are optional enhancements.

## Architecture Defaults

- Rust owns the daemon, CLI/TUI, desktop UI, process supervision, audio capture, and host adapters.
- Python (`python/voicelayer_worker`) owns model orchestration and provider-specific worker adapters.
- Inter-process communication between Rust and Python uses line-delimited JSON-RPC 2.0 over stdio.
- The daemon exposes a local `/v1` HTTP API over a Unix domain socket, documented in `openapi/voicelayerd.v1.yaml`.

## Host Strategy

- Ubuntu GNOME Wayland is the primary desktop target; macOS Apple Silicon is supported.
- Global shortcuts prefer the XDG Global Shortcuts portal on Linux and Carbon `RegisterEventHotKey` on macOS.
- GUI text injection prefers AT-SPI editable text operations, then clipboard paste with restoration.
- Terminal injection prefers bracketed paste and never auto-submits by default.
- Keyboard simulation tools (`ydotool`, `wtype`) remain fallbacks.

## Workflow

- Keep README, OpenAPI, and HTML documentation aligned with shipped behavior.
- Use `uv` for all Python commands and isolated environments; do not depend on the system interpreter.
- Standalone docs under `docs/` are maintained directly as HTML. Guard tests scan `docs/**/*.html` (excluding `docs/assets/`), `README.md`, `CLAUDE.md`, `openapi/voicelayerd.v1.yaml`, and `systemd/voicelayerd.env.example`.
- Before closing a task, run the verification chain in the Commands section below.
- Use GitHub flow: feature branch per change, open PRs with `gh pr create`, and follow conventional commits.

## Commands

- Authoritative verification chain:

  ```bash
  cargo fmt --all \
    && cargo clippy --all-targets --all-features -- -D warnings \
    && cargo test --all \
    && uv run ruff check python tests/python \
    && uv run ruff format --check python tests/python \
    && uv run pytest -q tests/python
  ```

- Sync Python dev environment: `uv sync --group dev`
- Run the daemon from source: `cargo run -p voicelayerd` (add `--project-root "$(pwd)"` or set `VOICELAYER_PROJECT_ROOT` when starting outside the repo root)
- Inspect runtime environment and provider reachability: `cargo run -p voicelayer-cli -- doctor`
- List host adapters and worker providers: `cargo run -p voicelayer-cli -- providers`

## Domain Vocabulary

Public types use the domain terms `CaptureSession`, `PreviewArtifact`, `InjectionPlan`, and `ProviderDescriptor`.
The canonical schema for the local daemon API lives at `openapi/voicelayerd.v1.yaml`.

<!-- gitnexus:start -->
# GitNexus — Code Intelligence

This project is indexed by GitNexus as **voice-layer** (2941 symbols, 7538 relationships, 241 execution flows).

> Index stale? Run `node .gitnexus/run.cjs analyze --index-only` from the project root — it auto-selects an available runner. No `.gitnexus/run.cjs` yet? Bootstrap with `npx`, `bunx`, or `pnpm dlx` — e.g. `bunx gitnexus@latest analyze` (npm 11 npx crash; #1939).

## Always Do

- **MUST run impact before editing.** Use `impact({target: "symbolName", direction: "upstream"})` or `node .gitnexus/run.cjs impact "symbolName" --direction upstream --repo .`; report callers, processes, and risk. Never substitute grep for graph analysis.
- **MUST analyze graph changes before committing.** Use `detect_changes({scope: "all"})` (MCP) or `node .gitnexus/run.cjs detect-changes --scope all --repo .` (CLI fallback). `partial: true` or `truncated: true` is not a clean check — a zero means unseen, not unaffected; re-run it. For regression review: `detect_changes({scope: "compare", base_ref: "main"})` or `node .gitnexus/run.cjs detect-changes --scope compare --base-ref "main" --repo .`.
- MUST warn on HIGH/CRITICAL `risk` pre-edit; never use `riskSharedAxes` to waive a HIGH/CRITICAL `risk` warning. Compare File/symbol: MCP File omits axes; Graph-RAG expands File.
- **MUST treat `risk: UNKNOWN` as unresolved, not as low.** An empty caller set is not evidence the symbol is unused — it can also mean the callers are not resolvable by the index (plain-object property access, dynamic dispatch, cross-language calls). `impact` pairs `UNKNOWN` with a `riskNote` saying so. Confirm with a text search before treating the symbol as safe to change or delete; do not proceed on the strength of a zero.
- **MUST use `query({search_query: "concept"})` for concepts/flows, `context({name: "symbolName"})` for a named symbol, or `impact` for blast radius, on read-only callers, dependencies, imports, or execution flow.** Graph first; text search only for empty/`UNKNOWN`/literals.
- For security review, `explain({target: "fileOrSymbol"})` lists taint findings (source→sink flows; needs `analyze --pdg`).

## Never Do

- NEVER edit a function, class, or method before MCP/CLI impact analysis.
- NEVER ignore HIGH or CRITICAL risk warnings from impact analysis, and never read `UNKNOWN` as an all-clear — it means the walk could not answer, which is the one verdict that requires confirming by other means.
- NEVER rename symbols with find-and-replace — use `rename` which understands the call graph.
- NEVER commit before MCP/CLI graph change analysis.

## Resources

| Resource | Use for |
| --- | --- |
| `gitnexus://repo/voice-layer/context` | Codebase overview, check index freshness |
| `gitnexus://repo/voice-layer/clusters` | All functional areas |
| `gitnexus://repo/voice-layer/processes` | All execution flows |
| `gitnexus://repo/voice-layer/process/{name}` | Step-by-step execution trace |

## CLI

| Task | Read this skill file |
| --- | --- |
| Understand architecture / "How does X work?" | `.claude/skills/gitnexus-exploring/SKILL.md` |
| Blast radius / "What breaks if I change X?" | `.claude/skills/gitnexus-impact-analysis/SKILL.md` |
| Trace bugs / "Why is X failing?" | `.claude/skills/gitnexus-debugging/SKILL.md` |
| Rename / extract / split / refactor | `.claude/skills/gitnexus-refactoring/SKILL.md` |
| Tools, resources, schema reference | `.claude/skills/gitnexus-guide/SKILL.md` |
| Index, status, clean, wiki CLI commands | `.claude/skills/gitnexus-cli/SKILL.md` |

<!-- gitnexus:end -->
