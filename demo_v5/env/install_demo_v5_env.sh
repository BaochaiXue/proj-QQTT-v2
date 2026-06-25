#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAIN_ENV="${DEMO_V5_MAIN_ENV:-demo_2_max}"
SHAPE_ENV="${DEMO_V5_SHAPE_PRIOR_ENV:-phystwin-max}"
MODE="${1:-update}"

usage() {
    cat <<'EOF'
Install or update Demo v5 conda environments.

Usage:
  bash demo_v5/env/install_demo_v5_env.sh [update|create|check]

Environment variables:
  DEMO_V5_MAIN_ENV         default: demo_2_max
  DEMO_V5_SHAPE_PRIOR_ENV  default: phystwin-max

Notes:
  - update: update existing envs or create missing envs, then pip-install extras.
  - create: create both envs; fails if they already exist.
  - check: only run environment and asset checks.
EOF
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

create_or_update_env() {
    local env_name="$1"
    local yaml_path="$2"
    if [[ "${MODE}" == "create" ]]; then
        conda env create -f "${yaml_path}"
    elif env_exists "${env_name}"; then
        conda env update -n "${env_name}" -f "${yaml_path}" --prune
    else
        conda env create -f "${yaml_path}"
    fi
}

install_pip_extras() {
    local env_name="$1"
    local requirements_path="$2"
    conda run -n "${env_name}" --no-capture-output \
        python -m pip install -r "${requirements_path}"
}

run_checks() {
    conda run -n "${MAIN_ENV}" --no-capture-output \
        python "${REPO_ROOT}/demo_v5/env/check_demo_v5_env.py" --role main --require-cuda
    conda run -n "${SHAPE_ENV}" --no-capture-output \
        python "${REPO_ROOT}/demo_v5/env/check_demo_v5_env.py" --role shape-prior --require-cuda
}

main() {
    case "${MODE}" in
        -h|--help|help)
            usage
            ;;
        check)
            run_checks
            ;;
        update|create)
            create_or_update_env "${MAIN_ENV}" "${SCRIPT_DIR}/environment-demo-v5-main.yml"
            install_pip_extras "${MAIN_ENV}" "${SCRIPT_DIR}/requirements-demo-v5-main.txt"
            create_or_update_env "${SHAPE_ENV}" "${SCRIPT_DIR}/environment-demo-v5-shape-prior.yml"
            install_pip_extras "${SHAPE_ENV}" "${SCRIPT_DIR}/requirements-demo-v5-shape-prior.txt"
            run_checks
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
