#!/usr/bin/env bash
# Measure warm `whisper-server` /inference latency.
#
# Pair with scripts/benchmark-whisper-cold-start{,-docker}.sh to see how much
# per-call latency a persistent whisper-server saves once the ggml model stays
# memory-mapped across requests.
#
# Requirements:
#   - curl on PATH
#   - A whisper-server instance reachable at HOST:PORT (see docs/guides/asr-providers.html,
#     or enable VOICELAYER_WHISPER_SERVER_AUTO_START inside the worker)
#   - tests/fixtures/small.wav (create with `uv run python scripts/generate-silent-fixture.py`)
#
# Usage:
#   HOST=127.0.0.1 PORT=8188 RUNS=10 WARMUP_RUNS=2 \
#     scripts/benchmark-whisper-warm-server.sh
#
# Optional environment overrides:
#   HOST=127.0.0.1                    whisper-server bind host
#   PORT=8188                         whisper-server bind port
#   FIXTURE=tests/fixtures/small.wav  input audio file
#   RUNS=10                           timed requests after warmup
#   WARMUP_RUNS=2                     untimed requests to settle caches
#   LANGUAGE=auto                     forwarded as -F language=...
#   TRANSLATE=false                   forwarded as -F translate=true/false
#   COLD_START_SECONDS=               optional cold-start baseline for delta output

set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8188}"
FIXTURE="${FIXTURE:-tests/fixtures/small.wav}"
RUNS="${RUNS:-10}"
WARMUP_RUNS="${WARMUP_RUNS:-2}"
LANGUAGE="${LANGUAGE:-auto}"
TRANSLATE="${TRANSLATE:-false}"
COLD_START_SECONDS="${COLD_START_SECONDS:-}"

# Validate RUNS early to avoid divide-by-zero in awk mean calculation.
if ! [[ "$RUNS" =~ ^[1-9][0-9]*$ ]]; then
  echo "error: RUNS must be a positive integer (got: $RUNS)" >&2
  exit 64
fi

if ! [[ "$WARMUP_RUNS" =~ ^(0|[1-9][0-9]*)$ ]]; then
  echo "error: WARMUP_RUNS must be a non-negative integer (got: $WARMUP_RUNS)" >&2
  exit 64
fi

if ! command -v curl >/dev/null 2>&1; then
  echo "error: curl is not on PATH" >&2
  exit 64
fi

if [[ ! -f "$FIXTURE" ]]; then
  echo "error: fixture not found at $FIXTURE" >&2
  echo "       generate one with: uv run python scripts/generate-silent-fixture.py" >&2
  exit 64
fi

base_url="http://$HOST:$PORT"
inference_url="$base_url/inference"

# Treat any HTTP status below 500 on GET / as ready.
if ! status=$(curl -s -o /dev/null -w '%{http_code}' --max-time 5 "$base_url/"); then
  echo "error: curl failed to reach $base_url/" >&2
  exit 64
fi
if [[ "$status" -ge 500 || "$status" -eq 000 ]]; then
  echo "error: $base_url/ returned HTTP $status" >&2
  echo "       start whisper-server first (see docs/guides/asr-providers.html)" >&2
  exit 64
fi

printf 'endpoint  : %s\n' "$inference_url"
printf 'fixture   : %s\n' "$FIXTURE"
printf 'language  : %s\n' "$LANGUAGE"
printf 'translate : %s\n' "$TRANSLATE"
printf 'warmup    : %d\n' "$WARMUP_RUNS"
printf 'runs      : %d\n' "$RUNS"
printf -- '---\n'

tmp_body="$(mktemp)"
trap 'rm -f "$tmp_body"' EXIT

# Send one /inference request and print wall-clock seconds to stdout.
post_one() {
  local start_ns end_ns status
  start_ns=$(date +%s%N)
  status=$(curl -s -o "$tmp_body" -w '%{http_code}' \
    --max-time 60 \
    -X POST "$inference_url" \
    -F "response_format=json" \
    -F "language=$LANGUAGE" \
    -F "translate=$TRANSLATE" \
    -F "file=@$FIXTURE")
  end_ns=$(date +%s%N)
  if [[ "$status" -lt 200 || "$status" -ge 300 ]]; then
    echo "error: $inference_url returned HTTP $status" >&2
    head -c 200 "$tmp_body" >&2 || true
    echo >&2
    exit 70
  fi
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.4f", (e - s) / 1e9 }'
}

if (( WARMUP_RUNS > 0 )); then
  for i in $(seq 1 "$WARMUP_RUNS"); do
    elapsed="$(post_one)"
    printf 'warmup %d: %s s (discarded)\n' "$i" "$elapsed"
  done
  printf -- '---\n'
fi

total_ms=0
min_s=""
max_s=""
for i in $(seq 1 "$RUNS"); do
  seconds="$(post_one)"
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

if [[ -n "$COLD_START_SECONDS" ]]; then
  echo ""
  awk -v warm="$mean_s" -v cold="$COLD_START_SECONDS" 'BEGIN {
    if (cold <= 0) {
      printf "warning: COLD_START_SECONDS=%s is not positive; skipping delta.\n", cold
      exit
    }
    delta = cold - warm
    ratio = delta / cold * 100
    printf "cold start baseline : %.4f s (supplied)\n", cold
    printf "warm-server mean    : %.4f s\n", warm
    if (delta >= 0) {
      printf "per-call saving     : %.4f s (%.1f%% vs cold start)\n", delta, ratio
    } else {
      printf "per-call saving     : %.4f s (warm path is slower; check your config)\n", delta
    }
  }'
fi
