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
(measured at ~0.84s on CPU, see Phase 3 Pre-flight below). A long-lived `whisper-server` process
keeps the model mmapped and drops per-request latency by roughly the cold-start cost. The worker
auto-selects the server whenever the configured endpoint is reachable and falls back to
`whisper-cli` otherwise.

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

## Current Scope

The current ASR integration covers file transcription, short-recording transcription, Phase 3B
fixed-duration segmented live dictation (recorder rolled every *N* seconds with background
transcription), and the optional silero-vad pre-pass above. It does not yet cover:

- always-on live microphone capture
- dynamic VAD-driven segmentation boundaries at the recorder layer
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
