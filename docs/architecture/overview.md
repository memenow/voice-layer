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

## Configuration Precedence

The `vl` CLI reads foreground-PTT options from three sources. Sources higher in the list win when
a value is set at multiple layers:

1. **Command-line flag** — every `vl dictation foreground-ptt` option can be set explicitly on the
   command line (for example `--language en`, `--key f9`, `--default-stop-action inject`).
2. **TOML config file** — persisted with `vl config set <key> <value>`. The file lives at the path
   reported by `vl config path`, typically
   `~/.config/voicelayer/config.toml` on Linux. Entries under `[foreground_ptt]` populate the same
   fields the CLI flags set.
3. **Struct defaults** — the values baked into `ForegroundPttConfig::default` (space key,
   `RecorderBackend::Auto`, stop action `none`, and so on) apply when neither the file nor the CLI
   provides a value.

Supported keys for `vl config set` today:

```
foreground_ptt.language                    (BCP-47 tag; "none" clears)
foreground_ptt.backend                     auto | pipewire | alsa
foreground_ptt.translate_to_english        bool
foreground_ptt.keep_audio                  bool
foreground_ptt.key                         space | enter | tab | f8 | f9 | f10
foreground_ptt.tmux_target_pane            tmux pane id (%N); "none" clears
foreground_ptt.wezterm_target_pane_id      wezterm pane id; "none" clears
foreground_ptt.kitty_match                 kitten match expression; "none" clears
foreground_ptt.copy_on_stop                bool
foreground_ptt.default_stop_action         none | copy | inject | save
foreground_ptt.restore_clipboard_on_exit   bool
foreground_ptt.save_dir                    directory path; "none" clears
```

Boolean values accept `true`/`false`, `1`/`0`, `yes`/`no`, or `on`/`off`.

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

## Dictation Segmentation (Phase 3B, 3D)

`StartDictationRequest` carries a `segmentation` field that controls how audio is captured for
transcription:

- `segmentation: {"mode": "one_shot"}` (default) — the pre-Phase-3 behavior: a single
  recorder subprocess runs from `create` to `stop` and the full WAV is transcribed once.
- `segmentation: {"mode": "fixed", "segment_secs": N, "overlap_secs": 0}` — the recorder is
  rolled every `N` seconds; each finalized chunk is transcribed in the background via the
  configured whisper provider while the next chunk records. On stop the daemon waits for all
  in-flight transcription tasks and returns a single concatenated transcript. When
  `keep_audio` is true, `DictationCaptureResult.audio_file` points at the *segment directory*
  under `$XDG_RUNTIME_DIR/voicelayer/dictation/<session_id>/` (one WAV per segment, sorted by
  numeric id) rather than a single recording; listing the directory reconstructs the capture
  in order. `keep_audio=false` deletes each segment and removes the directory.
- `segmentation: {"mode": "vad_gated", "probe_secs": P, "max_segment_secs": M, "silence_gap_probes": G}`
  — the recorder rolls at the shorter `probe_secs` cadence; each probe runs through the Python
  worker's `segment_probe` RPC (silero-vad classification). Speech probes accumulate into a
  pending buffer and flush as a single logical speech unit when either `G` consecutive silent
  probes arrive or the buffered duration reaches `M` seconds. The daemon concatenates the
  speech probes via the `stitch_wav_segments` RPC before handing the merged WAV to
  `transcribe`, so a speaker's natural pauses (not the fixed wall-clock) bound each
  transcription input. `G` defaults to 1 (flush on the first silent probe after speech).
  Requires VAD to be configured (`VOICELAYER_WHISPER_VAD_ENABLED=true`,
  `VOICELAYER_WHISPER_VAD_MODEL_PATH`); when VAD is unreachable the orchestrator falls back to
  a pessimistic "speech" verdict per probe so no audio is silently dropped.

While a segmented or VAD-gated session is live the daemon streams per-segment events on
`GET /v1/events/stream` in addition to the existing session-level events:

- `dictation.segmented_started` / `dictation.vad_gated_started` — emitted once when the
  respective orchestrator task begins.
- `dictation.segment_recorded` — emitted each time a fixed-mode segment's recording
  finalizes successfully.
- `dictation.segment_transcribed` — emitted each time a fixed-mode segment's background
  transcription task completes, carrying the character count on success or the worker error
  on failure.
- `dictation.probe_analyzed` — VAD-gated mode only; emitted once per physical probe with the
  speech/silence classification and `speech_ratio`.
- `dictation.speech_unit_flushed` — VAD-gated mode only; emitted when the pending speech
  buffer is flushed (either on silence or on max-segment-duration) and handed to transcribe.
- `dictation.speech_unit_transcribed` — VAD-gated mode only; emitted when a flushed speech
  unit's background transcribe task completes.
- `dictation.completed` / `dictation.failed` — unchanged; sent once at session end with the
  stitched transcript or the classified failure. `DictationFailureKind::RecordingFailed` and
  `AsrFailed` are populated by the daemon; `InjectionFailed` is only ever set by the client.

Segmentation never applies to `POST /v1/dictation/capture` (one-shot bounded duration) or
`POST /v1/transcriptions` (file transcription); those continue to use `whisper-cli` directly
against a single WAV.

## Optional silero-vad Pre-pass (Phase 3D)

A silero-vad pre-pass can run inside the Python worker before any whisper call. When enabled it
detects speech regions in the captured WAV, writes a trimmed 16-bit mono WAV containing only the
concatenated speech spans, and feeds that file to the configured whisper provider. The pre-pass
applies uniformly to all transcribe-bearing endpoints — live dictation (one-shot and segmented),
`POST /v1/dictation/capture`, and `POST /v1/transcriptions` — because the VAD layer sits inside
the `transcribe` JSON-RPC call and is invisible to the daemon and to the OpenAPI contract.

The pre-pass is gated on `VOICELAYER_WHISPER_VAD_ENABLED=true` plus a valid
`VOICELAYER_WHISPER_VAD_MODEL_PATH` pointing at a silero-vad ONNX export (v4 or v5). Runtime
dependencies (`numpy`, `onnxruntime`) ship as the optional `vad` extra; import failures are
caught and downgraded to "transcribe the raw WAV" so no request is lost. See
`docs/guides/local-asr-provider.md` for the full list of tunables.

## Deferred Work

This scaffold deliberately defers:

- GNOME portal hotkey binding beyond availability probing
- AT-SPI writable target discovery
- Terminal-specific adapters such as kitty remote control beyond the existing `kitten @` client
- Desktop GUI entry point

The repository already defines the seams for those additions so the first implementation does not need to redesign the public interfaces.
