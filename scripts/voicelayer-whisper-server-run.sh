#!/usr/bin/env bash
#
# Wrapper for launching whisper.cpp's whisper-server under systemd.
#
# Reads VOICELAYER_WHISPER_SERVER_* and VOICELAYER_WHISPER_MODEL_PATH from
# the environment (typically via EnvironmentFile=~/.config/voicelayer/voicelayerd.env
# in systemd/voicelayer-whisper-server.service), validates required variables,
# and execs the server so systemd tracks the server PID directly.
#
# Mirrors the argv layout produced by
# python/voicelayer_worker/providers/whisper_server.py::build_whisper_server_command.
# Keep both sides in sync; tests/python/test_whisper_server.py checks parity.
#
# Docker alternative:
#   Override ExecStart in ~/.config/systemd/user/voicelayer-whisper-server.service.d/
#   docker.conf; see docs/guides/services.html for the drop-in example.
#
# VOICELAYER_WHISPER_SERVER_ARGS is split with Python's shlex.split so quoting
# matches the Python autostart path and avoids bash glob expansion.

set -euo pipefail

: "${VOICELAYER_WHISPER_SERVER_BIN:?VOICELAYER_WHISPER_SERVER_BIN must point at a whisper-server binary}"
: "${VOICELAYER_WHISPER_MODEL_PATH:?VOICELAYER_WHISPER_MODEL_PATH must point at a ggml model}"

host="${VOICELAYER_WHISPER_SERVER_HOST:-127.0.0.1}"
port="${VOICELAYER_WHISPER_SERVER_PORT:-8188}"

extra_args=()
if [[ -n "${VOICELAYER_WHISPER_SERVER_ARGS:-}" ]]; then
  if ! command -v python3 >/dev/null 2>&1; then
    echo "error: python3 is required to parse VOICELAYER_WHISPER_SERVER_ARGS" >&2
    exit 1
  fi
  # Write NUL-delimited tokens to a tempfile so shlex.split errors (such as
  # unmatched quotes) fail explicitly instead of silently running with no args.
  tokens_file=$(mktemp -t voicelayer-whisper-server-args.XXXXXX)
  trap 'rm -f "${tokens_file}"' EXIT
  if ! python3 - <<'PY' > "${tokens_file}"
import os
import shlex
import sys

for token in shlex.split(os.environ.get("VOICELAYER_WHISPER_SERVER_ARGS", "")):
    sys.stdout.write(token + "\0")
PY
  then
    echo "error: failed to parse VOICELAYER_WHISPER_SERVER_ARGS (check quoting)" >&2
    exit 1
  fi
  while IFS= read -r -d '' token; do
    extra_args+=("${token}")
  done < "${tokens_file}"
  # Remove the tempfile before exec because EXIT traps do not run after exec.
  rm -f "${tokens_file}"
  trap - EXIT
fi

exec "${VOICELAYER_WHISPER_SERVER_BIN}" \
  -m "${VOICELAYER_WHISPER_MODEL_PATH}" \
  --host "${host}" \
  --port "${port}" \
  ${extra_args[@]+"${extra_args[@]}"}
