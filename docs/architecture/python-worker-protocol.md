# Python Worker Protocol

## Purpose

The Python worker boundary exists so provider-specific logic can move faster without destabilizing the Rust daemon.

## Transport

- Protocol: JSON-RPC 2.0
- Medium: stdio
- Ownership: Rust starts the worker process on demand and owns the call lifecycle
- Runtime environment: the worker must execute through the repository's `uv` environment, not the system interpreter

## Required Methods

- `health`
- `list_providers`
- `compose`
- `rewrite`
- `translate`
- `transcribe`

## Current Behavior

The worker implements every required method with real providers:

- `health` reports ASR and LLM readiness, probes the configured LLM endpoint, and surfaces
  `asr_configured`, `asr_error`, `llm_configured`, `llm_reachable`, and `llm_error`.
- `list_providers` returns the whisper.cpp ASR descriptor plus the configured LLM descriptor, or a
  stub LLM descriptor when no endpoint is set.
- `transcribe` prefers a persistent `whisper-server` endpoint when
  `VOICELAYER_WHISPER_SERVER_*` is configured (see
  `providers/whisper_server.py`), falling back to one-shot `whisper-cli`
  via `providers/whisper_cli.py` when the server is unreachable or
  returns an error. Both paths are independent; configuring only the
  cli variables keeps the legacy behavior. When the optional silero-vad
  pre-pass is enabled (`VOICELAYER_WHISPER_VAD_ENABLED=true`, see
  `providers/vad_segmenter.py`), the worker runs VAD on the input WAV
  before dispatching to whisper, trims non-speech, and short-circuits
  with an empty-transcript response when no speech is detected.
  VAD failure falls back to the raw WAV without losing the request.
- `compose`, `rewrite`, and `translate` call the configured OpenAI-compatible chat completion
  endpoint through `providers/llm_openai_compatible.py`, optionally auto-starting `llama-server`
  via `providers/llama_autostart.py` when `VOICELAYER_LLM_AUTO_START=true`.

The Rust daemon and CLI invoke every method through a live stdio bridge. `health` and
`list_providers` are exercised end-to-end in Rust integration tests; transcription, chat
completion, llama-server auto-start, and URL normalization are covered by Python unit tests in
`tests/python/test_worker.py`.

When a provider is not configured, the worker returns `-32004` (provider unavailable) rather than
fabricating output. When a configured provider fails at runtime, it returns `-32005`
(provider request failed).

## Error Policy

- Invalid JSON-RPC request: `-32600`
- Method not found: `-32601`
- Parse error: `-32700`
- Provider unavailable: `-32004`
- Provider request failed: `-32005`

## Preview Payload Shape

`compose`, `rewrite`, and `translate` return a preview payload shaped as:

```json
{
  "title": "string",
  "generated_text": "string",
  "notes": ["string"]
}
```

## Module Layout

Provider-specific logic lives under `python/voicelayer_orchestrator/providers/`:

- `llm_openai_compatible.py` — chat completion HTTP client, endpoint URL normalization, preview
  payload builders.
- `llama_autostart.py` — optional background launch and readiness polling for `llama-server`.
- `whisper_cli.py` — validation and invocation of the `whisper-cli` subprocess.
- `whisper_server.py` — HTTP client, readiness probe, optional autostart, and `/inference`
  multipart encoder for the persistent `whisper-server` path.
- `vad_segmenter.py` — optional silero-vad pre-pass (v4 or v5 ONNX) that trims non-speech out of
  the input WAV before transcription. Lazy-imports `numpy` and `onnxruntime` so the `vad` extra
  stays optional.

Shared utilities live in `providers/__init__.py` (`ProviderInvocationError`,
`provider_runtime_dir`, `supported_providers`). Environment-backed dataclasses and loaders live
in `config.py`. `worker.py` is protocol-only: JSON-RPC constants, `handle_request`, `serve`,
and `main`.
