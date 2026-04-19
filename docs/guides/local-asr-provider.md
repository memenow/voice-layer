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

`VOICELAYER_WHISPER_MODEL_PATH` is required for the current implementation.

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

## Current Scope

The current ASR integration is intentionally limited to file transcription and short-recording transcription.
It does not yet cover:

- always-on live microphone capture
- VAD-driven chunking
- partial transcripts
- background ASR daemonization

Those belong to the next stage of the VoiceLayer runtime.

## Phase 3 Pre-flight: Cold-start Measurement

Phase 3 of the roadmap (near-real-time ASR) depends on a concrete measurement of `whisper-cli`
cold-start latency on the operator's own hardware and model. The decision gate is:

- mean cold-start under ~0.25 s → proceed directly with fixed-duration segment transcription
  (Phase 3B).
- mean cold-start ≥ ~0.25 s → introduce a persistent `whisper-server` provider first (new
  `python/voicelayer_orchestrator/providers/whisper_server.py`); segmenting without it is
  strictly slower than the existing one-shot transcription.

The repository ships two helpers for running that measurement reproducibly:

```bash
# 1. Generate a deterministic 3-second silent fixture (96 KB, 16 kHz mono PCM).
python3 scripts/generate_silent_fixture.py

# 2. Measure cold-start. Requires VOICELAYER_WHISPER_MODEL_PATH and `whisper-cli` on PATH.
export VOICELAYER_WHISPER_MODEL_PATH=/abs/path/to/ggml-base.en.bin
RUNS=5 scripts/benchmark-whisper-cold-start.sh
```

The script prints each run's wall-clock seconds plus mean/min/max and a decision line. If a
different threshold is wanted for the call, override `THRESHOLD_SECONDS` on the command line.

The fixture contains only silence, so its transcription is empty; that is intentional — cold-start
here measures how long whisper-cli spends mapping the model and initializing the graph before
producing any tokens.
