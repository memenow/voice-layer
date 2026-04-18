---
name: daemon-run
description: Start voicelayerd from source with the correct project root, then probe it with `vl doctor`. Use for local smoke checks during development.
disable-model-invocation: true
---

Preconditions:

- `uv sync --group dev` has been run in this repository at least once.
- The current working directory is the repository root.
- No other `voicelayerd` instance is listening on `$XDG_RUNTIME_DIR/voicelayer/daemon.sock`. If one is, stop and surface that to the operator instead of starting a second instance.

Steps:

1. Start the daemon in the background and capture its stdout/stderr to a temporary log file so the session can continue:

   ```bash
   cargo run -p vl -- daemon run --project-root "$(pwd)"
   ```

2. Wait briefly for the Unix domain socket at `$XDG_RUNTIME_DIR/voicelayer/daemon.sock` to appear, then run the inspector:

   ```bash
   cargo run -p vl -- doctor
   ```

3. Report the doctor output verbatim, highlighting:
   - Worker command resolution (direct `.venv/bin/python` vs `uv run --project`).
   - LLM endpoint reachability (when `VOICELAYER_LLM_ENDPOINT` is set).
   - ASR binary status (when `VOICELAYER_WHISPER_BIN` is set).

4. Remind the operator how to stop the daemon (SIGTERM on the reported pid) instead of leaving the shutdown implicit.
