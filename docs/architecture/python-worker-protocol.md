# Python Worker Protocol

## Purpose

The Python worker boundary exists so provider-specific logic can move faster
without destabilizing the Rust daemon.

## Transport

- Protocol: JSON-RPC 2.0
- Medium: stdio
- Ownership: the daemon spawns **one persistent worker process** (lazily, on
  the first inference call) and owns its lifecycle, including respawn after
  a crash and `kill_on_drop` at daemon shutdown.
- Runtime environment: the worker executes through the repository's `uv`
  environment, not the system interpreter.

Every response's `id` is validated against the request; a worker that
answers with a mismatched id, a wrong protocol version, or a closed pipe is
discarded and respawned on the next call.

## The `initialize` Handshake

`initialize` is the first frame after spawn and carries the provider
configuration payload:

```json
{
  "llm": {"endpoint": "...", "model": "...", "auto_start": false, ...},
  "whisper": {"model_path": "...", "binary": "whisper-cli", ...},
  "whisper_server": {"host": "127.0.0.1", "port": 8188, ...},
  "vad": {"enabled": false, ...}
}
```

The worker answers `{"status": "ok", "worker": "voicelayer_orchestrator",
"protocol": "2.0"}` and rejects every other method with `-32002` until
initialized. The worker never reads provider settings from its own
environment; the daemon builds the payload from the unified TOML config
(with `VOICELAYER_*` overrides applied).

## Methods

- `initialize`
- `health`
- `list_providers`
- `compose`
- `rewrite`
- `translate`
- `transcribe`

## Behavior

- `health` reports ASR and LLM readiness, probing the configured LLM
  endpoint, and surfaces `asr_configured`, `asr_error`, `llm_configured`,
  `llm_reachable`, and `llm_error`.
- `list_providers` returns the whisper.cpp ASR descriptor plus the
  configured LLM descriptor (or a stub LLM descriptor when unconfigured).
- `transcribe` prefers a persistent `whisper-server` endpoint when
  `[whisper_server]` is configured, falling back to one-shot `whisper-cli`.
  With the optional silero-vad pre-pass enabled (`[vad]`), the worker trims
  non-speech before dispatch and short-circuits with an empty transcript
  when no speech is detected. VAD failure falls back to the raw WAV.
- `compose`, `rewrite`, and `translate` call the configured
  OpenAI-compatible chat completion endpoint, optionally auto-starting
  `llama-server` when `[llm] auto_start = true`.

## Error Policy

- Invalid JSON-RPC request: `-32600`
- Method not found: `-32601`
- Parse error: `-32700`
- Not initialized: `-32002`
- Provider unavailable: `-32004`
- Provider request failed: `-32005`

The daemon maps these onto RFC 9457 problem types: `-32004` becomes
`urn:voicelayer:problem:provider_unavailable` (503), `-32005` becomes
`provider_request_failed` (502); worker timeouts become 504.

## Preview Payload Shape

`compose`, `rewrite`, and `translate` return:

```json
{
  "title": "string",
  "generated_text": "string",
  "notes": ["string"]
}
```

## Module Layout

- `worker.py` — protocol-only: JSON-RPC constants, `initialize`, dispatch,
  `serve`, `main`.
- `config.py` — dataclasses built from the `initialize` payload; no
  environment reads.
- `providers/pipeline.py` — orchestration: VAD pre-pass, whisper server/cli
  dispatch with fallback, LLM readiness, health assembly.
- `providers/llm_openai_compatible.py` — chat completion HTTP client
  (httpx), endpoint URL normalization, preview payload builders.
- `providers/llama_autostart.py` — background launch and readiness polling
  for `llama-server`.
- `providers/whisper_cli.py` — validation and invocation of `whisper-cli`.
- `providers/whisper_server.py` — httpx client, readiness probe, optional
  autostart for the persistent `whisper-server` path.
- `providers/vad_segmenter.py` — silero-vad pre-pass (v4 or v5 ONNX);
  lazy-imports `numpy` and `onnxruntime` so the `vad` extra stays optional.
- `providers/__init__.py` — `ProviderInvocationError` /
  `ProviderUnavailableError`, runtime dir, cross-platform stale-lock
  reclamation (psutil), `supported_providers`.
