# Local ASR Provider Guide

## Recommended Baseline

The first real ASR provider in VoiceLayer targets `whisper.cpp`.
The current implementation uses the official `whisper-cli` file transcription path.

The official `whisper.cpp` CLI supports:

- `flac`
- `mp3`
- `ogg`
- `wav`

The current VoiceLayer integration invokes `whisper-cli` directly and expects a local model file.

## Example `whisper.cpp` Startup

Build and run manually:

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav
```

## VoiceLayer Environment Variables

VoiceLayer reads the following ASR variables inside the Python worker:

```bash
VOICELAYER_WHISPER_BIN=whisper-cli
VOICELAYER_WHISPER_MODEL_PATH=/absolute/path/to/ggml-base.en.bin
VOICELAYER_WHISPER_TIMEOUT_SECONDS=300
VOICELAYER_WHISPER_NO_GPU=false
VOICELAYER_WHISPER_ARGS=
```

`VOICELAYER_WHISPER_MODEL_PATH` is required for the `whisper-cli` path.

### Persistent `whisper-server` (preferred for Phase 3 segmented workflows)

Launching `whisper-cli` once per transcription pays the full model-init cost every call
(measured at ~0.84 s on CPU, see Phase 3 Pre-flight below). A long-lived `whisper-server` process
keeps the model mmapped and drops per-request latency by roughly the cold-start cost — the
Warm-server baseline subsection below documents the reproduction command and the
maintainer-reported saving. The worker auto-selects the server whenever the configured endpoint
is reachable and falls back to `whisper-cli` otherwise.

```bash
VOICELAYER_WHISPER_SERVER_HOST=127.0.0.1
VOICELAYER_WHISPER_SERVER_PORT=8188
VOICELAYER_WHISPER_SERVER_TIMEOUT_SECONDS=60
VOICELAYER_WHISPER_SERVER_AUTO_START=false
VOICELAYER_WHISPER_SERVER_BIN=/abs/path/to/whisper-server
VOICELAYER_WHISPER_SERVER_ARGS=-t 4
VOICELAYER_WHISPER_SERVER_LAUNCH_TIMEOUT_SECONDS=30
VOICELAYER_WHISPER_SERVER_POLL_INTERVAL_SECONDS=0.5
```

Set any of those variables to opt into the server path; the worker returns to
`whisper-cli` when none are set. When `VOICELAYER_WHISPER_SERVER_AUTO_START=true` the worker
launches a background `whisper-server` process against `VOICELAYER_WHISPER_MODEL_PATH` and
waits up to `LAUNCH_TIMEOUT_SECONDS` for `GET /` to respond. PID, command, and endpoint are
written under `$XDG_RUNTIME_DIR/voicelayer/providers/` so doctor tooling can recover state.

You can also run the server manually:

```bash
# Host binary (requires a local whisper.cpp build):
whisper-server -m /path/to/ggml-base.en.bin --host 127.0.0.1 --port 8188 -t 4

# Docker (no host build):
docker run -d --name voicelayer-whisper-server \
  -v /path/to/ggml-base.en.bin:/model.bin:ro \
  -p 127.0.0.1:8188:8080 \
  --entrypoint whisper-server \
  ghcr.io/ggml-org/whisper.cpp:main \
  -m /model.bin --host 0.0.0.0 --port 8080 -t 4
```

After either path, configure VoiceLayer to point at it:

```bash
export VOICELAYER_WHISPER_SERVER_HOST=127.0.0.1
export VOICELAYER_WHISPER_SERVER_PORT=8188
```

Both paths use the same `POST /inference` multipart contract (fields `file`, `language`,
`translate`, `response_format=json`). The worker sends `response_format=json` and parses
`{"text": "...", "language": "..."}`.

## Example Source Workflow

```bash
export VOICELAYER_PROJECT_ROOT="$(pwd)"
export VOICELAYER_WHISPER_BIN="/path/to/whisper-cli"
export VOICELAYER_WHISPER_MODEL_PATH="/path/to/ggml-base.en.bin"

cargo run -p vl -- doctor
cargo run -p vl -- transcribe-file /path/to/sample.wav --language auto
cargo run -p vl -- record-transcribe --duration-seconds 8 --language auto
```

## Recorder Backends

The current CLI recording path is intentionally minimal:

- preferred backend: `pw-record` wrapped by `timeout --signal=INT`
- fallback backend: `arecord -d`

This recording path writes a temporary WAV file and then forwards that file to `whisper-cli`.
The same path is now exposed through the daemon-side `dictation capture` flow, so future global hotkey and tray integrations can reuse it without re-implementing recording logic.

### Optional silero-vad Pre-pass (Phase 3D)

The Python worker can run a silero-vad pre-pass before handing audio to whisper. When enabled it
detects speech regions in the captured WAV, concatenates them into a trimmed 16-bit mono WAV, and
feeds that file to the configured transcription provider. The JSON-RPC `transcribe` contract is
unchanged — VAD is invisible to the daemon.

VAD pulls in `onnxruntime` and `numpy`, which are shipped as the optional `vad` extra to keep the
default worker stdlib-only. Install the extra once and download a silero-vad ONNX model:

```bash
uv sync --extra vad
curl -L -o /abs/path/to/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

Environment variables:

```bash
VOICELAYER_WHISPER_VAD_ENABLED=true
VOICELAYER_WHISPER_VAD_MODEL_PATH=/abs/path/to/silero_vad.onnx
VOICELAYER_WHISPER_VAD_THRESHOLD=0.5
VOICELAYER_WHISPER_VAD_MIN_SPEECH_MS=250
VOICELAYER_WHISPER_VAD_MIN_SILENCE_MS=100
VOICELAYER_WHISPER_VAD_SPEECH_PAD_MS=30
VOICELAYER_WHISPER_VAD_MAX_SEGMENT_SECS=30
VOICELAYER_WHISPER_VAD_SAMPLE_RATE=16000
```

The worker only enables the pre-pass when both `VOICELAYER_WHISPER_VAD_ENABLED` is truthy and
`VOICELAYER_WHISPER_VAD_MODEL_PATH` points at an existing ONNX file. If import of `onnxruntime`
or `numpy` fails at runtime, the worker annotates the transcribe response with the error and
falls back to transcribing the raw WAV — no request is lost. Silero-vad v4 (`h` + `c` state) and
v5 (`state` tensor) ONNX exports are both supported; the sample rate must be 16000 Hz or 8000 Hz.

### VAD-gated Live Dictation (Phase 3D-B)

The pre-pass above runs *after* whisper has already received a fixed-duration WAV. The
**VAD-gated** segmentation mode goes further: the recorder rolls at a short `probe_secs` cadence,
each probe is classified by silero-vad through the worker's `segment_probe` JSON-RPC method, and
the orchestrator only flushes a buffered speech unit to `transcribe` when it sees `silence_gap_probes`
consecutive silent probes (default `1`) or when the buffered duration would exceed
`max_segment_secs`. Speaker pauses, not the wall-clock, bound each transcription input.

```bash
export VOICELAYER_WHISPER_VAD_ENABLED=true
export VOICELAYER_WHISPER_VAD_MODEL_PATH=/abs/path/to/silero_vad.onnx
# Same VAD tunables as the pre-pass apply.

cargo run -p vl -- daemon run --project-root "$(pwd)"           # in one shell
cargo run -p vl -- dictation start \
  --mode vad-gated --probe-secs 2 --max-segment-secs 30 \
  --language auto                                               # in another
# (speak, pausing naturally between phrases)
cargo run -p vl -- dictation stop <session-id-from-start>
```

While a VAD-gated session is live the daemon streams these per-probe / per-unit events on
`GET /v1/events/stream` alongside the existing lifecycle events:

- `dictation.vad_gated_started` — emitted once when the orchestrator begins.
- `dictation.probe_analyzed` — once per physical probe with the speech/silence verdict and
  `speech_ratio`.
- `dictation.speech_unit_flushed` — once per flushed pending buffer (silence-triggered or
  max-duration-triggered).
- `dictation.speech_unit_failed` — emitted instead of `speech_unit_flushed` when the
  `stitch_wav_segments` RPC errors during a flush; the session ultimately surfaces as
  `dictation.failed` with `failure_kind: asr_failed`.
- `dictation.speech_unit_transcribed` — once per flushed unit's background transcribe task.
- `dictation.completed` / `dictation.failed` — same shape as fixed mode; `text` is the
  concatenated transcripts of every flushed speech unit.

If the `segment_probe` RPC itself errors (silero-vad crash, ONNX import failure, etc.) the
orchestrator falls back to a pessimistic "speech" verdict per probe so live dictation never
silently drops audio. The synthetic verdict carries a note explaining the failure, and the
operator can inspect it in the SSE stream.

When VAD is **not** configured (`VOICELAYER_WHISPER_VAD_ENABLED` unset or
`VOICELAYER_WHISPER_VAD_MODEL_PATH` missing), the daemon still accepts `--mode vad-gated`
requests but every probe falls back to "speech". Silence-triggered flushes never fire, so
`max_segment_secs` becomes the only remaining flush trigger — the effective cadence is the
buffered-duration cap, not `probe_secs`. Configure VAD properly before relying on the gating
to bound transcribe inputs.

## Current Scope

The current ASR integration covers file transcription, short-recording transcription, Phase 3B
fixed-duration segmented live dictation (recorder rolled every *N* seconds with background
transcription), the optional silero-vad pre-pass, and Phase 3D-B VAD-gated segmentation
(probe-cadence recording with silence-triggered flush). It does not yet cover:

- always-on live microphone capture
- partial transcripts streamed mid-utterance
- background ASR daemonization

Those belong to later stages of the VoiceLayer runtime.

## Phase 3 Pre-flight: Cold-start Measurement

Phase 3 of the roadmap (near-real-time ASR) depends on a concrete measurement of `whisper-cli`
cold-start latency on the operator's own hardware and model. The decision gate is:

- mean cold-start under ~0.25 s → proceed directly with fixed-duration segment transcription
  (Phase 3B).
- mean cold-start ≥ ~0.25 s → introduce a persistent `whisper-server` provider first (new
  `python/voicelayer_orchestrator/providers/whisper_server.py`); segmenting without it is
  strictly slower than the existing one-shot transcription.

The repository ships three helpers for running that measurement reproducibly:

```bash
# 1. Generate a deterministic 3-second silent fixture (96 KB, 16 kHz mono PCM).
python3 scripts/generate_silent_fixture.py

# 2a. Measure cold-start against a host whisper-cli binary.
export VOICELAYER_WHISPER_MODEL_PATH=/abs/path/to/ggml-base.en.bin
RUNS=5 scripts/benchmark-whisper-cold-start.sh

# 2b. OR measure against the official container without installing anything on the host.
#     Pulls ghcr.io/ggml-org/whisper.cpp:main and caches the model under
#     ~/.cache/voicelayer/models.
RUNS=5 scripts/benchmark-whisper-cold-start-docker.sh
```

Both scripts print each run's wall-clock seconds plus mean/min/max and a decision line. If a
different threshold is wanted for the call, override `THRESHOLD_SECONDS` on the command line.

The fixture contains only silence, so its transcription is empty; that is intentional — cold-start
here measures how long whisper-cli spends mapping the model and initializing the graph before
producing any tokens.

### Baseline measurement (CPU, ggml-base.en.bin)

Recorded via `scripts/benchmark-whisper-cold-start-docker.sh` with `IMAGE=ghcr.io/ggml-org/whisper.cpp:main`,
`ggml-base.en.bin`, 5 runs, on an Ubuntu 24 workstation with an RTX 5090 laptop GPU (GPU was not
enabled for this measurement because the CPU image was used):

- mean: 0.8446 s
- min : 0.8083 s
- max : 0.8962 s

The mean is ~3.4× the 0.25 s gate, so the roadmap takes the `whisper-server` detour before
Phase 3B. The figure includes `docker exec` overhead (~10–30 ms); host `whisper-cli` cold-start
is at most that much faster and still far above the threshold. Operators with a different
hardware/model combination should re-run the script and update this section when the value
changes meaningfully.

### Warm-server baseline

The cold-start row above is only half the Phase 3 decision. The other half — how much per-call
wall-clock the persistent server actually saves once the ggml context stays warm across
requests — is measured against an already-running `whisper-server`. Start one with the Docker
command earlier in this guide, or let the worker auto-start it via
`VOICELAYER_WHISPER_SERVER_AUTO_START=true`, then:

```bash
COLD_START_SECONDS=0.8446 \
  HOST=127.0.0.1 PORT=8188 \
  RUNS=10 WARMUP_RUNS=2 \
  scripts/benchmark-whisper-warm-server.sh
```

The script sends `WARMUP_RUNS` untimed `POST /inference` requests so per-connection caches
settle, times the next `RUNS`, and reports mean/min/max. When `COLD_START_SECONDS` is passed
(use the mean printed by the cold-start script above), the output also includes the absolute and
relative saving per transcribe call. There is no proceed/stop gate here because the tradeoff is
configuration-specific: even a small warm-path saving compounds inside segmented dictation (one
transcribe per segment), while one-shot transcription amortises the cold-start cost across a
single call.

On the maintainer's reference workstation (Ubuntu 24, CPU-only, `ggml-base.en.bin`,
`ghcr.io/ggml-org/whisper.cpp:main` server container on port 8188) real voice captures measured
~0.65 s/call via the warm server and ~0.98 s/call via one-shot `whisper-cli-docker`, for a
~0.26 s daemon-mediated saving per transcribe call. The saving is what drives VoiceLayer
preferring the server path whenever an endpoint is configured: segmented live dictation runs
one transcribe per segment, so a ~260 ms per-call delta flows straight into user-visible
end-to-end latency. Operators with different hardware or model sizes will see different deltas;
rerun the script to confirm before relying on the ratio.
