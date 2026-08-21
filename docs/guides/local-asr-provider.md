# Local ASR Provider Guide

## Recommended Baseline

VoiceLayer's ASR baseline is `whisper.cpp`. Two provider paths exist:

- `whisper-cli` one-shot file transcription (simplest).
- `whisper-server`, a persistent HTTP server that keeps the ggml model
  mmapped across requests — preferred for segmented dictation, where paying
  the model cold-start per chunk would dominate latency.

The official `whisper.cpp` CLI supports `flac`, `mp3`, `ogg`, and `wav`
inputs.

## Example `whisper.cpp` Build

```bash
git clone https://github.com/ggml-org/whisper.cpp.git
cd whisper.cpp
cmake -B build
cmake --build build -j --config Release
./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav
```

On the NVIDIA Linux box, build with CUDA (`-DGGML_CUDA=ON`); on Apple
Silicon, Metal acceleration is enabled by default.

## Configuration

Set in the unified config file (`vl config path`):

```toml
[whisper]
model_path = "/absolute/path/to/ggml-base.en.bin"   # required
# binary = "whisper-cli"
# timeout_seconds = 300
# no_gpu = false
# extra_args = ""

# Optional persistent server path (preferred):
[whisper_server]
# host = "127.0.0.1"
# port = 8188
# timeout_seconds = 60
# auto_start = false
# server_bin = "/abs/path/to/whisper-server"
# extra_args = "-t 4"
# launch_timeout_seconds = 30
# poll_interval_seconds = 0.5
```

Setting any `[whisper_server]` key opts into the server path; the worker
auto-selects the server whenever the endpoint is reachable and falls back to
`whisper-cli` otherwise. With `auto_start = true` the worker launches a
background `whisper-server` against `whisper.model_path` and waits for
readiness; PID, command, and endpoint are written under the runtime
directory.

You can also run the server manually:

```bash
# Host binary:
whisper-server -m /path/to/ggml-base.en.bin --host 127.0.0.1 --port 8188 -t 4

# Docker (no host build):
docker run -d --name voicelayer-whisper-server \
  -v /path/to/ggml-base.en.bin:/model.bin:ro \
  -p 127.0.0.1:8188:8080 \
  --entrypoint whisper-server \
  ghcr.io/ggml-org/whisper.cpp:main \
  -m /model.bin --host 0.0.0.0 --port 8080 -t 4
```

Both paths share the `POST /inference` multipart contract (fields `file`,
`language`, `translate`, `response_format=json`).

Environment override mapping: `VOICELAYER_WHISPER_BIN`/`_MODEL_PATH`/
`_TIMEOUT_SECONDS`/`_NO_GPU`/`_ARGS` for `[whisper]`, and the
`VOICELAYER_WHISPER_SERVER_*` family for `[whisper_server]`.

## Audio Capture

The daemon captures audio in-process via `cpal` (the PipeWire ALSA shim on
Linux, CoreAudio on macOS), resamples to 16 kHz mono, and writes 16-bit PCM
WAV files under the runtime directory. There is no external recorder
subprocess; segmented dictation cuts chunks out of the continuous capture
buffer, so no audio is lost at chunk boundaries. Word-level stitching of
boundary-clipped words (via whisper timestamps) is deferred.

## Optional silero-vad Pre-pass

The worker can run a silero-vad pre-pass before handing audio to whisper:
speech regions are detected in the captured WAV, concatenated into a trimmed
16-bit mono WAV, and fed to the transcription provider. The JSON-RPC
`transcribe` contract is unchanged — VAD is invisible to the daemon.

VAD pulls in `onnxruntime` and `numpy`, shipped as the optional `vad` extra:

```bash
uv sync --extra vad
curl -L -o /abs/path/to/silero_vad.onnx \
  https://github.com/snakers4/silero-vad/raw/master/src/silero_vad/data/silero_vad.onnx
```

```toml
[vad]
enabled = true
model_path = "/abs/path/to/silero_vad.onnx"
# threshold = 0.5
# min_speech_ms = 250
# min_silence_ms = 100
# speech_pad_ms = 30
# max_segment_secs = 30
# sample_rate = 16000
```

If importing `onnxruntime` or `numpy` fails at runtime, the worker annotates
the transcribe response and falls back to the raw WAV — no request is lost.
Silero-vad v4 and v5 ONNX exports are both supported; the sample rate must
be 16000 or 8000 Hz.

## Cold-start Measurement

The persistent `whisper-server` path exists because `whisper-cli` pays the
full model-init cost per call. Measured baseline (5 runs,
`ggml-base.en.bin`, CPU Docker image, Ubuntu 24 workstation with an RTX 5090
laptop GPU): mean 0.8446 s, min 0.8083 s, max 0.8962 s — roughly 3.4× the
0.25 s budget that would make per-call CLI launches acceptable. Reproduce
with:

```bash
python3 scripts/generate_silent_fixture.py
RUNS=5 scripts/benchmark-whisper-cold-start.sh          # host binary
RUNS=5 scripts/benchmark-whisper-cold-start-docker.sh   # container
```

## Verify

```bash
vl doctor
vl transcribe-file /path/to/sample.wav --language auto
vl record-transcribe --duration-seconds 8 --language auto
```

## Current Scope

The ASR integration covers file transcription, fixed-duration captures, and
segmented live dictation (chunks cut from the continuous buffer with
background transcription), plus the optional VAD pre-pass. It does not yet
cover:

- dynamic VAD-driven segmentation boundaries at the capture layer
- partial transcripts streamed mid-utterance
