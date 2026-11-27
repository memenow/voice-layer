#!/usr/bin/env bash
# Measure `whisper-cli` cold-start latency to choose between whisper-cli and whisper-server.
#
# Requirements:
#   - whisper-cli on PATH (or VOICELAYER_WHISPER_BIN set to an absolute path)
#   - VOICELAYER_WHISPER_MODEL_PATH set to an absolute path to a ggml whisper model
#   - tests/fixtures/small.wav (create with `uv run python scripts/generate-silent-fixture.py`)
#
# Usage:
#   VOICELAYER_WHISPER_MODEL_PATH=/abs/path/to/ggml-base.en.bin \
#     scripts/benchmark-whisper-cold-start.sh
#
# Optional environment overrides:
#   RUNS=5                           Number of timed runs (default: 5)
#   FIXTURE=path.wav                 Input audio (default: tests/fixtures/small.wav)
#   VOICELAYER_WHISPER_BIN=...       Path to whisper-cli
#   VOICELAYER_WHISPER_NO_GPU=true   Pass -ng to measure CPU cold-start only
#   THRESHOLD_SECONDS=0.25           Latency threshold for recommending whisper-server

set -euo pipefail

BIN="${VOICELAYER_WHISPER_BIN:-whisper-cli}"
MODEL="${VOICELAYER_WHISPER_MODEL_PATH:-}"
FIXTURE="${FIXTURE:-tests/fixtures/small.wav}"
RUNS="${RUNS:-5}"
THRESHOLD_SECONDS="${THRESHOLD_SECONDS:-0.25}"

if [[ -z "$MODEL" ]]; then
  echo "error: VOICELAYER_WHISPER_MODEL_PATH is not set" >&2
  echo "       export VOICELAYER_WHISPER_MODEL_PATH=/abs/path/to/ggml-base.en.bin" >&2
  exit 64
fi

if ! command -v "$BIN" >/dev/null 2>&1 && [[ ! -x "$BIN" ]]; then
  echo "error: cannot find whisper-cli (VOICELAYER_WHISPER_BIN=$BIN)" >&2
  exit 64
fi

if [[ ! -f "$FIXTURE" ]]; then
  echo "error: fixture not found at $FIXTURE" >&2
  echo "       generate one with: uv run python scripts/generate-silent-fixture.py" >&2
  exit 64
fi

if [[ ! -f "$MODEL" ]]; then
  echo "error: model file not found at $MODEL" >&2
  exit 64
fi

extra_args=()
if [[ "${VOICELAYER_WHISPER_NO_GPU:-false}" =~ ^(1|true|yes|on)$ ]]; then
  extra_args+=("-ng")
fi

tmp_out="$(mktemp -d)"
trap 'rm -rf "$tmp_out"' EXIT

printf 'bin     : %s\n' "$BIN"
printf 'model   : %s\n' "$MODEL"
printf 'fixture : %s\n' "$FIXTURE"
printf 'runs    : %d\n' "$RUNS"
printf 'args    : %s\n' "${extra_args[*]:-(none)}"
printf -- '---\n'

# Measure one invocation and print wall-clock seconds to stdout.
time_one() {
  local out_stem="$tmp_out/out-$1"
  local start_ns end_ns
  start_ns=$(date +%s%N)
  "$BIN" -m "$MODEL" -f "$FIXTURE" -otxt -of "$out_stem" -np -l auto "${extra_args[@]}" \
    >"$tmp_out/stdout-$1" 2>"$tmp_out/stderr-$1"
  end_ns=$(date +%s%N)
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.4f", (e - s) / 1e9 }'
}

total_ms=0
min_s=""
max_s=""
for i in $(seq 1 "$RUNS"); do
  seconds="$(time_one "$i")"
  printf 'run %d: %s s\n' "$i" "$seconds"
  total_ms=$(awk -v t="$total_ms" -v s="$seconds" 'BEGIN { printf "%.0f", t + s * 1000 }')
  if [[ -z "$min_s" ]]; then
    min_s="$seconds"
    max_s="$seconds"
  else
    min_s=$(awk -v a="$min_s" -v b="$seconds" 'BEGIN { printf "%.4f", (a < b ? a : b) }')
    max_s=$(awk -v a="$max_s" -v b="$seconds" 'BEGIN { printf "%.4f", (a > b ? a : b) }')
  fi
done

mean_s=$(awk -v t="$total_ms" -v n="$RUNS" 'BEGIN { printf "%.4f", t / 1000 / n }')

printf -- '---\n'
printf 'mean : %s s\n' "$mean_s"
printf 'min  : %s s\n' "$min_s"
printf 'max  : %s s\n' "$max_s"

decision=$(awk -v m="$mean_s" -v t="$THRESHOLD_SECONDS" 'BEGIN { print (m < t ? "below" : "above") }')
if [[ "$decision" == "below" ]]; then
  echo ""
  echo "recommendation: mean cold-start ($mean_s s) < threshold ($THRESHOLD_SECONDS s)"
  echo "                one-shot whisper-cli is fast enough for segmented dictation."
else
  echo ""
  echo "recommendation: mean cold-start ($mean_s s) >= threshold ($THRESHOLD_SECONDS s)"
  echo "                prefer persistent whisper-server (see docs/guides/asr-providers.html)"
  echo "                to avoid paying model load latency on every segment."
fi
