#!/usr/bin/env bash
# Install VoiceLayer binaries and the user-level systemd unit.
#
# The script builds `vl` (CLI + TUI + daemon) and `vl-desktop` (GUI shell),
# copies them to $HOME/.local/bin, renders the user-unit template under
# $HOME/.config/systemd/user/, and finishes by running `vl doctor` so the
# operator sees the environment probe immediately. Rerunning the script is
# safe: binaries and unit files are overwritten in place.

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
(cd "${REPO_ROOT}" && cargo build --release --bin vl --bin vl-desktop)

echo ">> Installing binaries to ${BIN_DIR}"
install -d "${BIN_DIR}"
install -m 0755 "${REPO_ROOT}/target/release/vl" "${BIN_DIR}/vl"
install -m 0755 "${REPO_ROOT}/target/release/vl-desktop" "${BIN_DIR}/vl-desktop"

echo ">> Installing user-level systemd unit to ${UNIT_DIR}"
install -d "${UNIT_DIR}"
install -m 0644 "${REPO_ROOT}/systemd/voicelayerd.service" "${UNIT_DIR}/voicelayerd.service"

echo ">> Seeding environment file (existing file is preserved)"
install -d "${ENV_DIR}"
if [[ ! -f "${ENV_DIR}/voicelayerd.env" ]]; then
  install -m 0600 "${REPO_ROOT}/systemd/voicelayerd.env.example" "${ENV_DIR}/voicelayerd.env"
  echo "   wrote ${ENV_DIR}/voicelayerd.env — edit paths before enabling the unit"
else
  echo "   kept ${ENV_DIR}/voicelayerd.env (no overwrite)"
fi

if command -v systemctl >/dev/null 2>&1; then
  echo ">> Reloading user-level systemd manager"
  systemctl --user daemon-reload || echo "   (daemon-reload failed; safe to ignore if systemd --user is unavailable)"
  # If the unit was already running before the install, bounce it so the
  # running daemon picks up the freshly copied binary. `try-restart` is a
  # no-op when the unit isn't active, so this is safe on first install.
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

cat <<'EOM'

Done.

Next steps:
  1. Edit ~/.config/voicelayer/voicelayerd.env to point at your models.
  2. Enable the daemon on login:
       systemctl --user enable --now voicelayerd
  3. Launch the desktop shell:
       vl-desktop
EOM
