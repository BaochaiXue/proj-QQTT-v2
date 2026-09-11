#!/usr/bin/env bash
# Install the Demo v7 clickable launcher: app menu entry + desktop icon.
# Idempotent; re-run after moving the repo (paths in the .desktop are absolute).
set -eu

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${HERE}/../.." && pwd)"
DESKTOP_SRC="${HERE}/demo_v7.desktop"
APPS_DIR="${HOME}/.local/share/applications"
DESKTOP_DIR="$(xdg-user-dir DESKTOP 2>/dev/null || echo "${HOME}/Desktop")"

chmod +x "${HERE}/demo_v7.sh"

# The committed .desktop carries absolute paths from the machine it was
# authored on. Copying it verbatim made the documented "re-run after moving
# the repo" a no-op — the installed entry kept pointing at the OLD path.
# Generate the installed copy against THIS checkout instead.
GENERATED="$(mktemp)"
trap 'rm -f "${GENERATED}"' EXIT
sed -e "s|^Exec=.*|Exec=${HERE}/demo_v7.sh|" \
    -e "s|^Icon=.*|Icon=${HERE}/demo_v7.png|" \
    -e "s|^Path=.*|Path=${REPO_ROOT}|" \
    "${DESKTOP_SRC}" > "${GENERATED}"

mkdir -p "${APPS_DIR}"
install -m 644 "${GENERATED}" "${APPS_DIR}/demo_v7.desktop"

if [ -d "${DESKTOP_DIR}" ]; then
    install -m 755 "${GENERATED}" "${DESKTOP_DIR}/demo_v7.desktop"
    # GNOME requires the trusted bit for a double-clickable desktop launcher.
    gio set "${DESKTOP_DIR}/demo_v7.desktop" metadata::trusted true 2>/dev/null || true
fi

update-desktop-database "${APPS_DIR}" 2>/dev/null || true
command -v desktop-file-validate >/dev/null 2>&1 \
    && desktop-file-validate "${APPS_DIR}/demo_v7.desktop"

echo "installed:"
echo "  app menu : ${APPS_DIR}/demo_v7.desktop"
[ -d "${DESKTOP_DIR}" ] && echo "  desktop  : ${DESKTOP_DIR}/demo_v7.desktop"
echo "self test : ${HERE}/demo_v7.sh --check"
