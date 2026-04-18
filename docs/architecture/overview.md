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

## Recorder Backends

The daemon captures microphone audio to a temporary WAV file through a subprocess backend:

- `pw-record` (PipeWire, preferred when available) wrapped by `timeout` so one-shot capture can
  honor a fixed duration.
- `arecord` (ALSA) when PipeWire is not reachable.
- `RecorderBackend::Auto` selects PipeWire when both `pw-record` and `timeout` are present,
  otherwise falls back to `arecord`.

Live sessions (`POST /v1/sessions/dictation` and `POST /v1/sessions/dictation/stop`) start
the recorder as a long-running child process and send SIGINT on stop; the audio is then handed
to the Python worker for transcription via `whisper-cli`. The daemon does not process PCM in
process; all capture flows through the recorder subprocess to a WAV file.

## Deferred Work

This scaffold deliberately defers:

- GNOME portal hotkey binding beyond availability probing
- AT-SPI writable target discovery
- Terminal-specific adapters such as kitty remote control beyond the existing `kitten @` client
- Desktop GUI entry point

The repository already defines the seams for those additions so the first implementation does not need to redesign the public interfaces.
