#!/usr/bin/env bash
# Install VoiceLayer binaries and the per-user service definition.
#
# Builds `voicelayerd` (daemon), `vl` (CLI), and `vl-desktop` (GUI shell),
# copies them to $HOME/.local/bin, installs the service definition (systemd
# user unit on Linux, launchd agent on macOS), seeds the unified config
# file, and finishes with `vl doctor`. Rerunning is safe: binaries and unit
# files are overwritten in place; an existing config file is preserved.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"

BIN_DIR="${VOICELAYER_INSTALL_BIN_DIR:-${HOME}/.local/bin}"
OS="$(uname -s)"

if ! command -v cargo >/dev/null 2>&1; then
  echo "error: cargo is required on PATH (install rustup)" >&2
  exit 1
fi

echo ">> Building voice-layer (release) ..."
(cd "${REPO_ROOT}" && cargo build --release --bin voicelayerd --bin vl --bin vl-desktop)

echo ">> Installing binaries to ${BIN_DIR}"
install -d "${BIN_DIR}"
for bin in voicelayerd vl vl-desktop; do
  install -m 0755 "${REPO_ROOT}/target/release/${bin}" "${BIN_DIR}/${bin}"
done

case "${OS}" in
  Linux)
    UNIT_DIR="${VOICELAYER_INSTALL_UNIT_DIR:-${HOME}/.config/systemd/user}"
    ENV_DIR="${HOME}/.config/voicelayer"
    echo ">> Installing user-level systemd unit to ${UNIT_DIR}"
    install -d "${UNIT_DIR}"
    install -m 0644 "${REPO_ROOT}/systemd/voicelayerd.service" "${UNIT_DIR}/voicelayerd.service"
    SERVICE_HINT="systemctl --user enable --now voicelayerd"
    CONFIG_PATH="${ENV_DIR}/config.toml"
    ;;
  Darwin)
    AGENT_DIR="${HOME}/Library/LaunchAgents"
    echo ">> Installing launchd agent to ${AGENT_DIR}"
    install -d "${AGENT_DIR}"
    sed "s|__HOME__|${HOME}|g" \
      "${REPO_ROOT}/launchd/com.memenow.voicelayerd.plist" \
      > "${AGENT_DIR}/com.memenow.voicelayerd.plist"
    SERVICE_HINT="launchctl bootstrap gui/\$(id -u) ~/Library/LaunchAgents/com.memenow.voicelayerd.plist"
    CONFIG_PATH="${HOME}/Library/Application Support/com.memenow.voicelayer/config.toml"
    ;;
  *)
    echo "error: unsupported OS ${OS} (expected Linux or Darwin)" >&2
    exit 1
    ;;
esac

echo ">> Seeding config file (existing file is preserved)"
install -d "$(dirname "${CONFIG_PATH}")"
if [[ ! -f "${CONFIG_PATH}" ]]; then
  install -m 0600 "${REPO_ROOT}/config/config.toml.example" "${CONFIG_PATH}"
  echo "   wrote ${CONFIG_PATH} — edit paths before enabling the service"
else
  echo "   kept ${CONFIG_PATH} (no overwrite)"
fi

if [[ "${OS}" == "Linux" ]] && command -v systemctl >/dev/null 2>&1; then
  echo ">> Reloading user-level systemd manager"
  systemctl --user daemon-reload || echo "   (daemon-reload failed; safe to ignore if systemd --user is unavailable)"
  if systemctl --user is-active --quiet voicelayerd; then
    echo ">> Restarting active voicelayerd.service to pick up new binary"
    systemctl --user try-restart voicelayerd || echo "   (try-restart failed; run systemctl --user restart voicelayerd manually)"
  fi
fi

if [[ ":${PATH}:" != *":${BIN_DIR}:"* ]]; then
  echo ">> NOTE: ${BIN_DIR} is not on your PATH — add it before running vl/vl-desktop"
fi

echo ">> Running vl doctor"
"${BIN_DIR}/vl" doctor || true

cat <<EOM

Done.

Next steps:
  1. Edit ${CONFIG_PATH} to point at your models.
  2. Enable the daemon on login:
       ${SERVICE_HINT}
  3. Launch the desktop shell:
       vl-desktop
EOM
