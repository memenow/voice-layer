# Architecture Overview

## Goals

VoiceLayer must support two primary workflows:

- Dictation: low-latency speech-to-text insertion into the focused target
- Composition: convert spoken intent into a structured, polished text artifact before insertion

The system must work across GUI applications and terminal/TUI surfaces without splitting the domain model.

## Runtime Topology

```text
+------------------+      JSON over UDS      +------------------+
| CLI / TUI / UI   | <---------------------> | voicelayerd      |
+------------------+                         +------------------+
                                                     |
                                                     | JSON-RPC over stdio
                                                     v
                                            +----------------------+
                                            | Python orchestrator  |
                                            +----------------------+
```

## Component Responsibilities

### Rust

- Capture session lifecycle
- Local control API
- Host targeting and injection planning
- CLI/TUI entry points
- Daemon reliability, logging, and supervision

### Python

- Provider orchestration
- Model-family-specific integration logic
- Rapid experimentation for ASR, LLM, and TTS backends

## Domain Objects

The first-class domain language is:

- `CaptureSession`
- `TranscriptChunk`
- `CompositionJob`
- `PreviewArtifact`
- `InjectionPlan`
- `ProviderDescriptor`
- `HotkeyBinding`
- `LanguageProfile`

These names should remain stable across Rust, Python, and OpenAPI surfaces.

## Initial Transport Decisions

- External control plane: HTTP over Unix domain socket
- Streaming events: Server-Sent Events
- Rust/Python bridge: JSON-RPC 2.0 over stdio
- Worker launch policy: on-demand Python worker invocation through the `uv`-managed project environment

## Deferred Work

This scaffold deliberately defers:

- Real audio capture
- ASR runtime wiring
- GNOME portal implementation details
- AT-SPI writable target discovery
- Terminal-specific adapters such as kitty remote control

The repository already defines the seams for those additions so the first implementation does not need to redesign the public interfaces.
