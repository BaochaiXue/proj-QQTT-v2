#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RULE_SRC="${SCRIPT_DIR}/99-qqtt-realsense-wsl.rules"
RULE_DST="/etc/udev/rules.d/99-qqtt-realsense-wsl.rules"
TARGET_USER="${SUDO_USER:-${USER:-}}"

usage() {
    cat <<'EOF'
Install or remove the persistent WSL RealSense udev rule used by the RSUSB backend.

Usage:
  bash env_install/install_wsl_realsense_udev.sh
  bash env_install/install_wsl_realsense_udev.sh --print-rule
  bash env_install/install_wsl_realsense_udev.sh --remove

Notes:
  - This script needs sudo to write /etc/udev/rules.d/.
  - Windows-side usbipd attach/auto-attach is still required separately.
EOF
}

require_group_membership_note() {
    if [[ -n "${TARGET_USER}" ]] && id -nG "${TARGET_USER}" 2>/dev/null | tr ' ' '\n' | grep -qx 'plugdev'; then
        return 0
    fi
    cat <<EOF
[warn] user '${TARGET_USER:-<unknown>}' is not currently in the 'plugdev' group.
[warn] the installed rule will grant the USB device nodes to group 'plugdev'.
EOF
}

reload_and_trigger() {
    sudo udevadm control --reload-rules
    sudo udevadm trigger --subsystem-match=usb --attr-match=idVendor=8086 --attr-match=idProduct=0b5c || true
}

main() {
    case "${1:-}" in
        -h|--help)
            usage
            ;;
        --print-rule)
            cat "${RULE_SRC}"
            ;;
        --remove)
            sudo rm -f "${RULE_DST}"
            reload_and_trigger
            echo "[ok] removed ${RULE_DST}"
            ;;
        "")
            require_group_membership_note
            sudo install -m 0644 "${RULE_SRC}" "${RULE_DST}"
            reload_and_trigger
            echo "[ok] installed ${RULE_DST}"
            echo "[ok] reloaded udev rules and retriggered matching Intel D455 USB devices"
            echo "[note] Windows-side usbipd auto-attach is still configured separately"
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
