# vl-desktop

`vl-desktop` is the GUI shell that sits alongside `vl` and reaches the
daemon through the same `/v1` Unix socket. Unlike `voicelayerd`, it
runs as an interactive user process (not a systemd user service), so
its operator knobs live outside `~/.config/voicelayer/voicelayerd.env`.

## Launch

```bash
vl-desktop
```

Starting the daemon on demand: if `vl-desktop` cannot reach the socket
it offers a "Start daemon" button that spawns `vl daemon run` and
waits for readiness.

## Environment variables

| Variable | Purpose | Default |
| --- | --- | --- |
| `VOICELAYER_VL_BIN` | Absolute path to the `vl` binary that `vl-desktop` invokes when starting the daemon or issuing CLI fallbacks. | `$HOME/.local/bin/vl`, then `$PATH` lookup |
| `VOICELAYER_LOG` | `tracing_subscriber` env-filter for `vl-desktop`'s logs. Accepts the usual `target=level[,target=level]*` syntax. | `vl_desktop=info` |

Both knobs are client-side only — the daemon does not read them. If
you need to override `VOICELAYER_VL_BIN` or `VOICELAYER_LOG`
persistently, set them in your shell profile, desktop entry, or
launcher wrapper rather than the daemon env file. The coverage test
in `tests/python/test_env_example.py` enforces this separation: these
keys are listed in `CLIENT_ONLY_ENV_VARS` and intentionally excluded
from `systemd/voicelayerd.env.example`.

## Relationship to the daemon

- `VOICELAYER_SOCKET_PATH` (documented in
  [systemd.md](./systemd.md)) is consulted by `vl-desktop` through the
  same helper as the CLI; overriding it here makes `vl-desktop` talk
  to a non-default socket without touching the daemon unit.
- `vl-desktop` consumes the same SSE stream (`/v1/events/stream`) the
  CLI does — portal-registered hotkeys, overlay state, and
  `DictationFailureKind` all flow through the shared event envelope.
