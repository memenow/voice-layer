#!/usr/bin/env bash
# Docker variant of scripts/benchmark-whisper-cold-start.sh.
#
# Measures whisper-cli cold-start latency by running it inside an
# ephemeral long-lived container and timing `docker exec` invocations.
# This path never installs whisper.cpp on the host; only the ggml model
# is downloaded to the host cache and bind-mounted read-only.
#
# Requirements:
#   - Docker daemon reachable without sudo
#   - Network access on first run to pull the image and model
#
# Usage:
#   RUNS=5 scripts/benchmark-whisper-cold-start-docker.sh
#
# Optional environment overrides:
#   IMAGE=ghcr.io/ggml-org/whisper.cpp:main          (CPU default)
#   IMAGE=ghcr.io/ggml-org/whisper.cpp:main-cuda     (use --gpus all)
#   GPUS='all'                                       (pass to `docker run`)
#   MODEL_NAME=base.en                               (default; any ggml-*.bin name)
#   MODELS_DIR=$HOME/.cache/voicelayer/models        (host model cache)
#   FIXTURE=tests/fixtures/small.wav                 (input audio)
#   RUNS=5                                           (timed exec invocations)
#   THRESHOLD_SECONDS=0.25                           (decision threshold)
#
# The measurement includes `docker exec` overhead (~10-30ms on most hosts),
# so a mean below ~0.20s safely clears the 0.25s threshold for Phase 3B
# even after subtracting exec overhead. Host `whisper-cli` cannot be slower
# than the measured docker-exec number; this is a conservative upper bound.

set -euo pipefail

IMAGE="${IMAGE:-ghcr.io/ggml-org/whisper.cpp:main}"
GPUS="${GPUS:-}"
MODEL_NAME="${MODEL_NAME:-base.en}"
MODELS_DIR="${MODELS_DIR:-$HOME/.cache/voicelayer/models}"
FIXTURE="${FIXTURE:-tests/fixtures/small.wav}"
RUNS="${RUNS:-5}"
THRESHOLD_SECONDS="${THRESHOLD_SECONDS:-0.25}"
CONTAINER_NAME="voicelayer-whisper-bench-$$"

if ! command -v docker >/dev/null 2>&1; then
  echo "error: docker is not on PATH" >&2
  exit 64
fi

if ! docker info >/dev/null 2>&1; then
  echo "error: cannot reach docker daemon (check service / group)" >&2
  exit 64
fi

if [[ ! -f "$FIXTURE" ]]; then
  echo "error: fixture not found at $FIXTURE" >&2
  echo "       generate one with: python3 scripts/generate_silent_fixture.py" >&2
  exit 64
fi

model_path="$MODELS_DIR/ggml-$MODEL_NAME.bin"
if [[ ! -f "$model_path" ]]; then
  echo "downloading ggml-$MODEL_NAME.bin into $MODELS_DIR..."
  mkdir -p "$MODELS_DIR"
  curl -L --fail --progress-bar \
    -o "$model_path" \
    "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-$MODEL_NAME.bin"
fi

abs_fixture="$(readlink -f "$FIXTURE")"
abs_model="$(readlink -f "$model_path")"

docker_run_args=(-d
  --name "$CONTAINER_NAME"
  --entrypoint sleep
  -v "$abs_model":/model.bin:ro
  -v "$abs_fixture":/fixture.wav:ro
)
if [[ -n "$GPUS" ]]; then
  docker_run_args+=(--gpus "$GPUS")
fi

trap 'docker rm -f "$CONTAINER_NAME" >/dev/null 2>&1 || true' EXIT

printf 'image     : %s\n' "$IMAGE"
printf 'model     : %s\n' "$model_path"
printf 'fixture   : %s\n' "$FIXTURE"
printf 'gpus      : %s\n' "${GPUS:-(none)}"
printf 'runs      : %d\n' "$RUNS"
printf 'container : %s\n' "$CONTAINER_NAME"
printf -- '---\n'

echo "pulling image if needed..."
docker pull "$IMAGE" >/dev/null

echo "starting container..."
docker run "${docker_run_args[@]}" "$IMAGE" infinity >/dev/null

# Quick sanity: the image must expose whisper-cli.
if ! docker exec "$CONTAINER_NAME" sh -c 'command -v whisper-cli >/dev/null'; then
  echo "error: whisper-cli not found inside $IMAGE" >&2
  exit 70
fi

time_one() {
  local start_ns end_ns
  start_ns=$(date +%s%N)
  docker exec "$CONTAINER_NAME" \
    whisper-cli -m /model.bin -f /fixture.wav -otxt -of /tmp/out -np -l auto \
    >/dev/null 2>&1
  end_ns=$(date +%s%N)
  awk -v s="$start_ns" -v e="$end_ns" 'BEGIN { printf "%.4f", (e - s) / 1e9 }'
}

total_ms=0
min_s=""
max_s=""
for i in $(seq 1 "$RUNS"); do
  seconds="$(time_one)"
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
echo ""
if [[ "$decision" == "below" ]]; then
  echo "decision: mean cold-start ($mean_s s) < threshold ($THRESHOLD_SECONDS s)"
  echo "          Includes docker exec overhead (~10-30ms). Host whisper-cli"
  echo "          will be at least as fast. Phase 3B (fixed segment recording)"
  echo "          can proceed."
else
  echo "decision: mean cold-start ($mean_s s) >= threshold ($THRESHOLD_SECONDS s)"
  echo "          Even before subtracting docker exec overhead this exceeds"
  echo "          the gate. Introduce providers/whisper_server.py first"
  echo "          (persistent whisper-server); fixed-segment transcription"
  echo "          would be slower than today's one-shot path otherwise."
fi
