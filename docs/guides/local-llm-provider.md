# Local LLM Provider Guide

## Recommended Baseline

The first real composition provider in VoiceLayer targets an OpenAI-compatible chat endpoint.
For local-first development, the intended baseline is `llama.cpp server`.

The official `llama.cpp` project documents an OpenAI-compatible endpoint at:

```text
http://localhost:8080/v1/chat/completions
```

## Example `llama.cpp` Startup

```bash
llama-server -m /path/to/model.gguf --port 8080
```

If you want to start directly from Hugging Face in a development environment:

```bash
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --port 8080
```

## VoiceLayer Environment Variables

VoiceLayer reads the following variables inside the Python worker:

```bash
VOICELAYER_LLM_ENDPOINT=http://127.0.0.1:8080
VOICELAYER_LLM_MODEL=gemma-3-1b-it
VOICELAYER_LLM_API_KEY=
VOICELAYER_LLM_TIMEOUT_SECONDS=60
VOICELAYER_LLM_AUTO_START=true
VOICELAYER_LLAMA_SERVER_BIN=llama-server
VOICELAYER_LLAMA_MODEL_PATH=/absolute/path/to/model.gguf
VOICELAYER_LLAMA_HF_REPO=
VOICELAYER_LLAMA_SERVER_ARGS=
VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS=45
VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS=0.5
```

`VOICELAYER_LLM_ENDPOINT` may be provided as:

- `http://127.0.0.1:8080`
- `http://127.0.0.1:8080/v1`
- `http://127.0.0.1:8080/v1/chat/completions`

VoiceLayer normalizes all of them to the chat completions endpoint.
For health checks, VoiceLayer probes the corresponding `/v1/models` endpoint.

When `VOICELAYER_LLM_AUTO_START=true`, VoiceLayer will try to launch `llama-server` automatically if:

- the configured endpoint is local
- the endpoint is currently unreachable
- either `VOICELAYER_LLAMA_MODEL_PATH` or `VOICELAYER_LLAMA_HF_REPO` is configured

Automatic startup writes provider state files under:

```text
$XDG_RUNTIME_DIR/voicelayer/providers
```

## Example Source Workflow

```bash
export VOICELAYER_PROJECT_ROOT="$(pwd)"
export VOICELAYER_LLM_ENDPOINT="http://127.0.0.1:8080"
export VOICELAYER_LLM_MODEL="gemma-3-1b-it"
export VOICELAYER_LLM_AUTO_START="true"
export VOICELAYER_LLAMA_SERVER_BIN="llama-server"
export VOICELAYER_LLAMA_HF_REPO="ggml-org/gemma-3-1b-it-GGUF"

cargo run -p vl -- doctor
cargo run -p vl -- providers
cargo run -p vl -- preview compose "Write a concise professional status update for today's backend work."
```

## systemd User Service

The provided user service reads an optional env file at:

```text
~/.config/voicelayer/voicelayerd.env
```

Populate it with the project root and provider variables before enabling the service.
