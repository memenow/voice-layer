# Local LLM Provider Guide

## Recommended Baseline

VoiceLayer's composition/rewrite/translation workflows target an
OpenAI-compatible chat endpoint. For local-first development the baseline is
`llama.cpp server`, which serves:

```text
http://localhost:8080/v1/chat/completions
```

## Example `llama.cpp` Startup

```bash
llama-server -m /path/to/model.gguf --port 8080

# or straight from Hugging Face in a development environment
llama-server -hf ggml-org/gemma-3-1b-it-GGUF --port 8080
```

On the NVIDIA Linux box, use a CUDA build of `llama-server`; on Apple
Silicon the Metal build is the default.

## VoiceLayer Configuration

The daemon owns provider configuration and hands it to the worker in the
`initialize` handshake. Set it in the unified config file (see
`vl config path`):

```toml
[llm]
endpoint = "http://127.0.0.1:8080"
model = "gemma-3-1b-it"
# api_key = ""
# timeout_seconds = 60

# Optional llama-server autostart:
# auto_start = true
# server_bin = "llama-server"
# model_path = "/absolute/path/to/model.gguf"
# hf_repo = "ggml-org/gemma-3-1b-it-GGUF"
# server_args = "--ctx-size 8192"
# launch_timeout_seconds = 45
# poll_interval_seconds = 0.5
```

or via `vl config set`:

```bash
vl config set llm.endpoint http://127.0.0.1:8080
vl config set llm.model gemma-3-1b-it
vl config set llm.auto_start true
vl config set llm.hf_repo ggml-org/gemma-3-1b-it-GGUF
```

`endpoint` may be provided as `http://127.0.0.1:8080`, `.../v1`, or
`.../v1/chat/completions`; VoiceLayer normalizes all of them. Health checks
probe the corresponding `/v1/models` endpoint.

When `auto_start` is true, VoiceLayer launches `llama-server` automatically
if the configured endpoint is local, currently unreachable, and either
`model_path` or `hf_repo` is set. Provider state files are written under the
runtime directory (`$XDG_RUNTIME_DIR/voicelayer/providers` on Linux,
`$TMPDIR/voicelayer/providers` on macOS).

Every `[llm]` key maps to a `VOICELAYER_*` environment override
(`VOICELAYER_LLM_ENDPOINT`, `VOICELAYER_LLM_MODEL`,
`VOICELAYER_LLM_API_KEY`, `VOICELAYER_LLM_TIMEOUT_SECONDS`,
`VOICELAYER_LLM_AUTO_START`, `VOICELAYER_LLAMA_SERVER_BIN`,
`VOICELAYER_LLAMA_MODEL_PATH`, `VOICELAYER_LLAMA_HF_REPO`,
`VOICELAYER_LLAMA_SERVER_ARGS`, `VOICELAYER_LLAMA_LAUNCH_TIMEOUT_SECONDS`,
`VOICELAYER_LLAMA_POLL_INTERVAL_SECONDS`).

## Verify

```bash
vl doctor        # reports llm_configured / llm_reachable
vl providers
vl preview compose "Write a concise professional status update for today's backend work."
```
