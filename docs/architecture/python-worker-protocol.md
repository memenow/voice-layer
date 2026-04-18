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

## Current Scaffold Behavior

The worker currently implements:

- health reporting
- provider listing
- explicit provider-unavailable errors for generation and transcription methods

The Rust daemon and CLI now invoke these methods through a live stdio bridge.
`health` and `list_providers` are exercised end-to-end in Rust tests.

The worker does not fabricate generated text.
That preserves production-safe behavior while the provider adapters are still being wired.

## Error Policy

- Invalid JSON-RPC request: `-32600`
- Method not found: `-32601`
- Parse error: `-32700`
- Provider unavailable: `-32004`

## Next Implementation Step

The first real provider integration should add a `compose` execution path without changing the protocol shape.
That keeps the daemon, CLI, and future UI surfaces stable while model runtime work evolves.

When `compose`, `rewrite`, or `translate` eventually succeed, the worker should return a preview payload shaped as:

```json
{
  "title": "string",
  "generated_text": "string",
  "notes": ["string"]
}
```
