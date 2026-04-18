---
name: verify
description: Run the authoritative VoiceLayer verification chain — cargo fmt, clippy with -D warnings, cargo test --all, and pytest. Invoke before closing a task or handing off a change.
---

Execute the chain end-to-end and stop on the first failure. Do not skip or weaken a step; it matches the Commands block in `CLAUDE.md`.

```bash
cargo fmt --all \
  && cargo clippy --all-targets --all-features -- -D warnings \
  && cargo test --all \
  && uv run ruff check python tests/python \
  && uv run ruff format --check python tests/python \
  && uv run pytest -q tests/python
```

Guidance:

- If `uv` is missing, report it to the operator and stop; do not fall back to the system Python.
- If clippy reports warnings, fix them rather than relaxing `-D warnings` or trimming `--all-features`.
- If Ruff reports format drift, run `uv run ruff format python tests/python` to apply the fix and rerun the chain; do not disable rules to silence output.
- If `tests/python` is absent or empty, surface that fact instead of silently skipping pytest or Ruff.
- Report results grouped by step: format, clippy, cargo test, ruff check, ruff format, pytest. Call out the first failing step clearly.
