#!/usr/bin/env bash
# Build and install VoiceLayer binaries and user service definitions.
#
# Builds `voicelayerd` (daemon), `vl` (CLI/TUI), and `vl-desktop` (GUI),
# installs them into $HOME/.local/bin alongside the whisper-server wrapper,
# installs systemd user units on Linux or a LaunchAgent plist on macOS,
# seeds ~/.config/voicelayer/voicelayerd.env if absent, and runs `vl doctor`.
# Rerunning is safe: binaries and service files are updated in place while
# preserving any existing env file.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null && pwd)"

BIN_DIR="${VOICELAYER_INSTALL_BIN_DIR:-${HOME}/.local/bin}"
UNIT_DIR="${VOICELAYER_INSTALL_UNIT_DIR:-${HOME}/.config/systemd/user}"
ENV_DIR="${VOICELAYER_INSTALL_ENV_DIR:-${HOME}/.config/voicelayer}"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required on PATH (install rustup)" >&2
  exit 1
fi

echo ">> Building voice-layer (release) ..."
(cd "${REPO_ROOT}" && cargo build --release --bin voicelayerd --bin vl --bin vl-desktop)

echo ">> Installing binaries to ${BIN_DIR}"
install -d "${BIN_DIR}"
install -m 0755 "${REPO_ROOT}/target/release/voicelayerd" "${BIN_DIR}/voicelayerd"
install -m 0755 "${REPO_ROOT}/target/release/vl" "${BIN_DIR}/vl"
install -m 0755 "${REPO_ROOT}/target/release/vl-desktop" "${BIN_DIR}/vl-desktop"
install -m 0755 "${REPO_ROOT}/scripts/voicelayer-whisper-server-run.sh" \
  "${BIN_DIR}/voicelayer-whisper-server-run.sh"

OS="$(uname -s)"
if [[ "${OS}" == "Darwin" ]]; then
  AGENT_DIR="${HOME}/Library/LaunchAgents"
  echo ">> Installing launchd agent to ${AGENT_DIR}"
  install -d "${AGENT_DIR}"
  sed "s|__HOME__|${HOME}|g" \
    "${REPO_ROOT}/launchd/com.memenow.voicelayerd.plist" \
    > "${AGENT_DIR}/com.memenow.voicelayerd.plist"
elif [[ "${OS}" == "Linux" ]]; then
  echo ">> Installing user-level systemd units to ${UNIT_DIR}"
  install -d "${UNIT_DIR}"
  install -m 0644 "${REPO_ROOT}/systemd/voicelayerd.service" "${UNIT_DIR}/voicelayerd.service"
  # voicelayer-whisper-server.service stays disabled by default. Enable it
  # explicitly with `systemctl --user enable --now voicelayer-whisper-server`
  # when using server-mode ASR. See docs/guides/services.html.
  install -m 0644 "${REPO_ROOT}/systemd/voicelayer-whisper-server.service" \
    "${UNIT_DIR}/voicelayer-whisper-server.service"
else
  echo "error: unsupported OS ${OS} (expected Linux or Darwin)" >&2
  exit 1
fi

echo ">> Seeding environment file (existing file is preserved)"
install -d "${ENV_DIR}"
if [[ ! -f "${ENV_DIR}/voicelayerd.env" ]]; then
  install -m 0600 "${REPO_ROOT}/systemd/voicelayerd.env.example" "${ENV_DIR}/voicelayerd.env"
  echo "   wrote ${ENV_DIR}/voicelayerd.env (edit model paths before enabling the service)"
else
  echo "   kept ${ENV_DIR}/voicelayerd.env (no overwrite)"
fi

if [[ "${OS}" == "Linux" ]] && command -v systemctl >/dev/null 2>&1; then
  echo ">> Reloading user-level systemd manager"
  systemctl --user daemon-reload || echo "   (daemon-reload failed; safe to ignore if systemd --user is unavailable)"
  for unit in voicelayerd voicelayer-whisper-server; do
    if systemctl --user is-active --quiet "${unit}"; then
      echo ">> Restarting active ${unit}.service to pick up new binary"
      systemctl --user try-restart "${unit}" \
        || echo "   (try-restart failed; run systemctl --user restart ${unit} manually)"
    fi
  done
fi

if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
  echo ">> NOTE: ${BIN_DIR} is not on your PATH; add it before running vl or vl-desktop"
fi

echo ">> Running vl doctor"
"${BIN_DIR}/vl" doctor || true

if [[ "${OS}" == "Darwin" ]]; then
  cat <<'EOM'

Done.

Next steps (macOS):
  1. Edit ~/Library/Application Support/com.memenow.voicelayer/config.toml
     (or add EnvironmentVariables in ~/Library/LaunchAgents/com.memenow.voicelayerd.plist)
     to configure your model paths.
  2. Load the LaunchAgent:
       launchctl bootstrap "gui/$(id -u)" ~/Library/LaunchAgents/com.memenow.voicelayerd.plist
     See docs/guides/services.html for service management details.
  3. Launch the desktop shell:
       vl-desktop
EOM
else
  cat <<'EOM'

Done.

Next steps (Linux):
  1. Edit ~/.config/voicelayer/voicelayerd.env to configure your model paths.
  2. Enable the daemon on login:
       systemctl --user enable --now voicelayerd
  3. (Optional) Run whisper-server as a persistent unit instead of
     one-shot whisper-cli per capture:
       systemctl --user enable --now voicelayer-whisper-server
     See docs/guides/services.html for trade-offs vs VOICELAYER_WHISPER_SERVER_AUTO_START.
  4. Launch the desktop shell:
       vl-desktop
EOM
fi
