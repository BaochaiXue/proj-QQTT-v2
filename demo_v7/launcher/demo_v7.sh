#!/usr/bin/env bash
# demo_v7 clickable launcher: reproduces the terminal environment the demo has
# always run in (full conda activation, repo-root cwd), logs to a per-run file
# so a windowless double-click failure is never silent, and pops a dialog on
# non-zero exit. Also supports `--check` (env self-test, no GUI).
# NOTE deliberately no `set -u`: conda activate.d scripts (cuda-nvcc) read
# variables that may be unset and would abort activation under nounset.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
# Resolve conda.sh instead of pinning one machine's install: honour an
# explicit DEMO_V7_CONDA_SH, then the active install ($CONDA_EXE / conda on
# PATH), then the usual prefixes. The original hardcoded miniforge3 path
# stays first in the fallback list so this box behaves identically.
_conda_sh_candidates=(
    "${DEMO_V7_CONDA_SH}"
    "${HOME}/miniforge3/etc/profile.d/conda.sh"
    "${HOME}/miniconda3/etc/profile.d/conda.sh"
    "${HOME}/anaconda3/etc/profile.d/conda.sh"
    "/opt/conda/etc/profile.d/conda.sh"
)
if [ -n "${CONDA_EXE}" ]; then
    _conda_sh_candidates=(
        "${DEMO_V7_CONDA_SH}"
        "$(dirname "$(dirname "${CONDA_EXE}")")/etc/profile.d/conda.sh"
        "${_conda_sh_candidates[@]:1}"
    )
fi
CONDA_SH=""
for _candidate in "${_conda_sh_candidates[@]}"; do
    if [ -n "${_candidate}" ] && [ -f "${_candidate}" ]; then
        CONDA_SH="${_candidate}"
        break
    fi
done
if [ -z "${CONDA_SH}" ] && command -v conda >/dev/null 2>&1; then
    CONDA_SH="$(conda info --base 2>/dev/null)/etc/profile.d/conda.sh"
fi
ENV_NAME="demo_2_max"
LOG_DIR="${HOME}/.local/state/demo_v7/logs"
mkdir -p "${LOG_DIR}"
LOG_FILE="${LOG_DIR}/demo_v7_$(date +%Y%m%d_%H%M%S).log"
# keep the newest 20 logs
ls -1t "${LOG_DIR}"/demo_v7_*.log 2>/dev/null | tail -n +21 | xargs -r rm -f --

fail_dialog() {
    local message="$1"
    if command -v zenity >/dev/null 2>&1; then
        zenity --error --title "demo_v7 启动失败" \
            --text "${message}\n\n日志: ${LOG_FILE}" 2>/dev/null || true
    elif command -v notify-send >/dev/null 2>&1; then
        notify-send "demo_v7 启动失败" "${message} (日志: ${LOG_FILE})" || true
    fi
}

{
    echo "[launcher] $(date -Is) repo=${REPO_ROOT} env=${ENV_NAME}"
    if [ ! -f "${CONDA_SH}" ]; then
        echo "[launcher] conda.sh not found: ${CONDA_SH}"
        exit 90
    fi
    # Full activation (not just the env python): the pipeline has always run
    # from an activated terminal; activation-provided vars (CONDA_PREFIX,
    # PATH, LD_LIBRARY_PATH hooks) are part of the known-good environment.
    # shellcheck disable=SC1090
    source "${CONDA_SH}"
    conda activate "${ENV_NAME}" || { echo "[launcher] conda activate failed"; exit 91; }
    cd "${REPO_ROOT}" || exit 92
    # Pin the interpreter to the activated env: the caller's PATH (or an
    # auto-activated repo .venv) must never shadow it.
    PYBIN="${CONDA_PREFIX}/bin/python"
    echo "[launcher] python=${PYBIN}"

    if [ "${1:-}" = "--check" ]; then
        "${PYBIN}" - << 'EOF'
import sys
print("python:", sys.executable)
import PySide6  # noqa: F401
from demo_v7.ipc import protocol  # noqa: F401
from demo_v7.gui import main_window  # noqa: F401
print("demo_v7 imports OK; PySide6", PySide6.__version__)
EOF
        exit $?
    fi

    # NOT exec: it would replace this shell, making the exit-code
    # check and fail_dialog below unreachable — a windowless crash
    # (e.g. a CUDA driver mismatch) would then look like a clean exit.
    "${PYBIN}" demo_v7/app.py "$@"
} >>"${LOG_FILE}" 2>&1
STATUS=$?
if [ "${STATUS}" -ne 0 ]; then
    fail_dialog "demo_v7 退出码 ${STATUS}"
fi
exit "${STATUS}"
