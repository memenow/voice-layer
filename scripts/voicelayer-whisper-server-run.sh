#!/usr/bin/env bash
#
# Wrapper for launching whisper.cpp's whisper-server under systemd.
#
# Reads VOICELAYER_WHISPER_SERVER_* plus the shared
# VOICELAYER_WHISPER_MODEL_PATH from the environment — typically supplied
# through EnvironmentFile=~/.config/voicelayer/voicelayerd.env in
# systemd/voicelayer-whisper-server.service — validates the required
# knobs, then exec's the server so systemd gets the server's PID and
# exit code directly (no shell in between).
#
# Mirrors the argv layout produced by the Python autostart path
# (python/voicelayer_orchestrator/providers/whisper_server.py::
# _build_whisper_server_command). The pairing is locked by the
# test_whisper_server_launch regression test; if you change argv here,
# update the Python helper in the same PR.
#
# Docker alternative:
#   For operators who prefer the ghcr.io/ggml-org/whisper.cpp:main
#   container, override ExecStart in the drop-in
#   ~/.config/systemd/user/voicelayer-whisper-server.service.d/
#   override.conf, e.g.:
#
#     [Service]
#     ExecStart=
#     ExecStart=/usr/bin/docker run --rm --name voicelayer-whisper-server \
#       -p 127.0.0.1:${VOICELAYER_WHISPER_SERVER_PORT}:${VOICELAYER_WHISPER_SERVER_PORT} \
#       -v %h/.cache/voicelayer/models:/models:ro \
#       --entrypoint whisper-server \
#       ghcr.io/ggml-org/whisper.cpp:main \
#       -m /models/ggml-base.en.bin \
#       --host 0.0.0.0 --port ${VOICELAYER_WHISPER_SERVER_PORT} -ng
#     ExecStop=/usr/bin/docker stop voicelayer-whisper-server
#
#   See docs/guides/systemd.md for the full docker recipe.
#
# VOICELAYER_WHISPER_SERVER_ARGS is split on whitespace with default
# bash IFS; embedded-space arguments are not supported here. Operators
# needing complex flags should edit ExecStart directly.

set -euo pipefail

: "${VOICELAYER_WHISPER_SERVER_BIN:?VOICELAYER_WHISPER_SERVER_BIN must point at a whisper-server binary}"
: "${VOICELAYER_WHISPER_MODEL_PATH:?VOICELAYER_WHISPER_MODEL_PATH must point at a ggml model}"

host="${VOICELAYER_WHISPER_SERVER_HOST:-127.0.0.1}"
port="${VOICELAYER_WHISPER_SERVER_PORT:-8188}"

extra_args=()
if [[ -n "${VOICELAYER_WHISPER_SERVER_ARGS:-}" ]]; then
  # shellcheck disable=SC2206  # intentional word-splitting
  extra_args=(${VOICELAYER_WHISPER_SERVER_ARGS})
fi

exec "${VOICELAYER_WHISPER_SERVER_BIN}" \
  -m "${VOICELAYER_WHISPER_MODEL_PATH}" \
  --host "${host}" \
  --port "${port}" \
  "${extra_args[@]}"
