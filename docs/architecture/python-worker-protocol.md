# Python Worker Protocol

## Purpose

The Python worker boundary exists so provider-specific logic can move faster without destabilizing the Rust daemon.

## Transport

- Protocol: JSON-RPC 2.0
- Medium: stdio
- Ownership: Rust starts the worker process on demand and owns the call lifecycle
- Runtime environment: the worker must execute through the repository's `uv` environment, not the system interpreter

## Required Methods

- `health`
- `list_providers`
- `compose`
- `rewrite`
- `translate`
- `transcribe`
- `segment_probe`
- `stitch_wav_segments`

## Current Behavior

The worker implements every required method with real providers:

- `health` reports ASR and LLM readiness, probes the configured LLM endpoint, and surfaces
  `asr_configured`, `asr_error`, `llm_configured`, `llm_reachable`, and `llm_error`.
- `list_providers` returns the whisper.cpp ASR descriptor plus the configured LLM descriptor, or a
  stub LLM descriptor when no endpoint is set.
- `transcribe` prefers a persistent `whisper-server` endpoint when
  `VOICELAYER_WHISPER_SERVER_*` is configured (see
  `providers/whisper_server.py`), falling back to one-shot `whisper-cli`
  via `providers/whisper_cli.py` when the server is unreachable or
  returns an error. Both paths are independent; configuring only the
  cli variables keeps the legacy behavior. When the optional silero-vad
  pre-pass is enabled (`VOICELAYER_WHISPER_VAD_ENABLED=true`, see
  `providers/vad_segmenter.py`), the worker runs VAD on the input WAV
  before dispatching to whisper, trims non-speech, and short-circuits
  with an empty-transcript response when no speech is detected.
  VAD failure falls back to the raw WAV without losing the request.
  When the request body sets `provider_id="mimo_v2_5_asr"` and the
  optional Xiaomi MiMo-V2.5-ASR provider is configured (see
  `providers/mimo_asr.py` and `VOICELAYER_MIMO_*`), the worker bypasses
  the whisper chain entirely and dispatches to the MiMo backend; an
  unknown `provider_id` returns `-32004` rather than silently routing
  to whisper, and an explicit MiMo failure returns `-32005` without
  whisper fallback. The same `provider_id` is plumbed through every
  transcribe call a dictation session emits — the daemon stores it on
  the active session at `POST /v1/sessions/dictation` (or
  `POST /v1/dictation/capture`) and forwards it on every fixed-segment,
  vad-gated speech-unit, and one-shot stop transcribe call, so a
  caller that selects MiMo at session start never silently falls back
  to whisper for a subset of segments. The silero-vad pre-pass also
  applies on the MiMo path: when `VOICELAYER_WHISPER_VAD_ENABLED=true`
  the same trim/short-circuit/fall-back-to-raw-WAV behavior runs
  before MiMo inference, which avoids paying the cold-load cost on
  pure-silence captures and prevents MiMo's causal LM from
  hallucinating transcripts on detected-silence audio.
  After `transcribe` returns — on success, on whisper / MiMo failure,
  and on the no-speech short-circuit — the dispatcher unlinks the
  per-call sidecar files it produced under `runtime_dir/`: the
  `.vad-trimmed.wav` / `.vad-empty.wav` written by the silero-vad
  pre-pass, plus the `mimo-segment-<ts>-<idx>.wav` chunks emitted by
  the long-audio splitter. The original capture is left intact for the
  caller (the dictation pipeline owns its lifetime via `keep_audio`).
- `compose`, `rewrite`, and `translate` call the configured OpenAI-compatible chat completion
  endpoint through `providers/llm_openai_compatible.py`, optionally auto-starting `llama-server`
  via `providers/llama_autostart.py` when `VOICELAYER_LLM_AUTO_START=true`.
- `segment_probe` runs a silero-vad pass on a single WAV file (typically a
  1–2 s probe) and returns `{has_speech, speech_ratio, regions, notes}`
  where `regions` is a list of `{start_secs, end_secs}` speech spans.
  Intended as the classification primitive for VAD-gated segmentation:
  the daemon orchestrator feeds each physical probe through this method
  and uses the response to decide whether to keep accumulating or to
  flush the pending buffer to `transcribe`. Returns `-32004` when VAD is
  not configured and `-32005` when silero-vad itself fails. See
  `providers/vad_segmenter.py::probe_audio_file`.
- `stitch_wav_segments` concatenates a list of WAV files into a single
  output WAV and returns `{audio_file, segment_count, duration_secs}`.
  Used by the VAD-gated orchestrator to merge the pending speech probes
  before handing the unit to `transcribe`. All inputs must share sample
  rate, sample width, and channel count; mismatches raise `-32005`.
  Invalid requests (empty `audio_files`, non-string entries, missing
  `out_file`) return `-32600`. See `providers/audio_stitch.py`.

The Rust daemon and CLI invoke every method through a live stdio bridge. `health` and
`list_providers` are exercised end-to-end in Rust integration tests; transcription, chat
completion, llama-server auto-start, and URL normalization are covered by Python unit tests in
`tests/python/test_worker.py`.

When a provider is not configured, the worker returns `-32004` (provider unavailable) rather than
fabricating output. When a configured provider fails at runtime, it returns `-32005`
(provider request failed).

## Error Policy

- Invalid JSON-RPC request: `-32600`
- Method not found: `-32601`
- Parse error: `-32700`
- Provider unavailable: `-32004`
- Provider request failed: `-32005`

## Preview Payload Shape

`compose`, `rewrite`, and `translate` return a preview payload shaped as:

```json
{
  "title": "string",
  "generated_text": "string",
  "notes": ["string"]
}
```

## Module Layout

Provider-specific logic lives under `python/voicelayer_orchestrator/providers/`:

- `llm_openai_compatible.py` — chat completion HTTP client, endpoint URL normalization, preview
  payload builders.
- `llama_autostart.py` — optional background launch and readiness polling for `llama-server`.
- `whisper_cli.py` — validation and invocation of the `whisper-cli` subprocess.
- `whisper_server.py` — HTTP client, readiness probe, optional autostart, and `/inference`
  multipart encoder for the persistent `whisper-server` path.
- `mimo_asr.py` — optional Xiaomi MiMo-V2.5-ASR provider. Lazy-loads the `MimoAudio` wrapper
  on first call and caches it in a module-level dict keyed by `(model_path, tokenizer_path,
  device, dtype)`. Splits inputs longer than `VOICELAYER_MIMO_LONG_AUDIO_SPLIT_SECONDS` into
  WAV chunks via the stdlib `wave` module so upstream issue #6 (decoder repetition past ~3
  minutes) does not surface in operator-facing transcripts.
- `vad_segmenter.py` — optional silero-vad pre-pass (v4 or v5 ONNX) that trims non-speech out of
  the input WAV before transcription, plus `probe_audio_file` for the `segment_probe` RPC that
  backs VAD-gated segmentation. Lazy-imports `numpy` and `onnxruntime` so the `vad` extra stays
  optional.
- `audio_stitch.py` — stdlib `wave`-based WAV concatenator backing the `stitch_wav_segments`
  RPC. Rejects format mismatches between inputs so the orchestrator never silently produces a
  malformed output WAV.

Shared utilities live in `providers/__init__.py` (`ProviderInvocationError`,
`provider_runtime_dir`, `supported_providers`). Environment-backed dataclasses and loaders live
in `config.py`. `worker.py` is protocol-only: JSON-RPC constants, `handle_request`, `serve`,
and `main`.
