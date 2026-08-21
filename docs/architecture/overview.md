# Architecture Overview

## Goals

VoiceLayer must support two primary workflows:

- Dictation: low-latency speech-to-text insertion into the focused target
- Composition: convert spoken intent into a structured, polished text artifact before insertion

The system works across GUI applications and terminal/TUI surfaces without
splitting the domain model, on Ubuntu GNOME Wayland (primary) and macOS
(Apple Silicon).

## Runtime Topology

```text
+------------------+      HTTP/JSON over UDS     +------------------+
| CLI / TUI / UI   | <-------------------------> | voicelayerd      |
| (voicelayer-     |                             |                  |
|  client crate)   |                             |                  |
+------------------+                             +------------------+
                                                        |
                                                        | JSON-RPC 2.0 over stdio
                                                        | (one persistent process;
                                                        |  first frame = initialize)
                                                        v
                                                 +----------------------+
                                                 | Python orchestrator  |
                                                 +----------------------+
```

## Component Responsibilities

### Rust

- Capture session lifecycle and in-process audio capture (cpal: PipeWire
  ALSA shim on Linux, CoreAudio on macOS)
- Local control API (`api/`), typed RFC 9457 errors
- Persistent worker process supervision (`worker/`)
- Host targeting and injection planning (`voicelayer-core`)
- CLI/TUI entry points (`vl`) and desktop shell (`vl-desktop`) as pure
  clients of `voicelayer-client`
- Platform differences confined to `voicelayerd::platform` and the desktop
  shell's `hotkeys` module

### Python

- Provider orchestration (`providers/pipeline.py`)
- Model-family-specific integration logic (whisper-cli, whisper-server,
  OpenAI-compatible LLM, silero-vad)
- The worker is protocol-only at the surface (`worker.py`) and holds no
  environment-derived configuration

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

These names remain stable across Rust, Python, and OpenAPI surfaces.

## Transport Decisions

- External control plane: HTTP over Unix domain socket (owner-only, `0600`)
- Errors: RFC 9457 `application/problem+json` with semantic status codes
- Streaming events: Server-Sent Events with typed event envelopes
  (`DaemonEvent`); lagging subscribers receive a synthetic `events_lost`
  marker
- Rust/Python bridge: JSON-RPC 2.0 over stdio with a persistent worker and
  an `initialize` handshake
- Worker launch: on first inference call, through the `uv`-managed project
  environment

## Configuration

A single TOML file is the source of truth (Linux:
`~/.config/voicelayer/config.toml`; macOS: `~/Library/Application
Support/com.memenow.voicelayer/config.toml`). Precedence, highest first:

1. CLI flags (per command)
2. `VOICELAYER_*` environment overrides (applied by the daemon/CLI only)
3. TOML config file
4. Struct defaults

The daemon serializes `[llm]`, `[whisper]`, `[whisper_server]`, and `[vad]`
into the worker's `initialize` payload; the Python worker never reads
provider settings from its environment.

`vl config set <dotted.key> <value>` edits the file with schema validation
(unknown keys are rejected); `<value>` of `none` removes a key.

## Audio Capture

Capture runs in-process: a cpal input stream on a dedicated thread converts
the default input device to mono f32 at the device rate, and chunks are cut
from the continuous buffer, resampled to 16 kHz, and written as 16-bit PCM
WAV under the runtime directory.

- `segmentation: {"mode": "one_shot"}` (default): one transcription at stop.
- `segmentation: {"mode": "fixed", "segment_secs": N}`: a chunk is cut
  every N seconds and transcribed in the background while capture
  continues. On stop the daemon
  waits for in-flight transcriptions and returns the stitched transcript.
  With `keep_audio`, `DictationCaptureResult.audio_file` is the segment
  directory under the runtime dir (one zero-padded WAV per segment).

Live sessions stream typed events on `GET /v1/events/stream`:
`dictation_segmented_started`, `segment_recorded`, `segment_transcribed` /
`segment_transcribe_failed`, and the terminal `dictation_completed` /
`dictation_failed`.

## Optional silero-vad Pre-pass

The worker can run silero-vad (ONNX v4/v5) before whisper: speech regions
are concatenated into a trimmed WAV and fed to the transcriber. The pre-pass
applies uniformly to every transcribe-bearing endpoint because it sits
inside the worker's `transcribe` handler. Enabled via `[vad]` config;
runtime dependencies ship as the optional `vad` extra and import failures
degrade to raw-WAV transcription. See `docs/guides/local-asr-provider.md`.

## Health Model

`GET /v1/health` is cheap liveness: it returns the last cached snapshot. The
daemon refreshes the snapshot in the background (30 s) once the worker has
been spawned, and `POST /v1/health/refresh` forces an immediate deep probe
(worker health + platform hotkey probe).

## Session Store

Sessions are process-local, bounded (256 entries), and terminal states
expire after 10 minutes. A daemon restart starts a clean table.

## Deferred Work

- AT-SPI writable target discovery (Linux GUI injection)
- kitty remote control auto-discovery beyond the existing explicit `kitten @` route
- Rich desktop shell features (tray, settings center, multi-window)
