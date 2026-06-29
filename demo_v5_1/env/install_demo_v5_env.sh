#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

MAIN_ENV="${DEMO_V5_MAIN_ENV:-demo_2_max}"
SHAPE_ENV="${DEMO_V5_SHAPE_PRIOR_ENV:-phystwin-max}"
MODE="${1:-update}"
SAM3_PACKAGE_REF="${DEMO_V5_SAM3_PACKAGE_REF:-sam3==0.1.4}"
PYTORCH3D_SOURCE="${DEMO_V5_PYTORCH3D_SOURCE:-${HOME}/external/pytorch3d-demo-v5-cu126}"
PYTORCH3D_REF="${DEMO_V5_PYTORCH3D_REF:-v0.7.9}"

usage() {
    cat <<'EOF'
Install or update Demo v5 conda environments.

Usage:
  bash demo_v5/env/install_demo_v5_env.sh [update|create|check]

Environment variables:
  DEMO_V5_MAIN_ENV         default: demo_2_max
  DEMO_V5_SHAPE_PRIOR_ENV  default: phystwin-max
  DEMO_V5_SAM3_PACKAGE_REF default: sam3==0.1.4
  DEMO_V5_PYTORCH3D_SOURCE default: ~/external/pytorch3d-demo-v5-cu126
  DEMO_V5_PYTORCH3D_REF    default: v0.7.9

Notes:
  - update: update existing envs or create missing envs, then pip-install extras.
  - create: create both envs; fails if they already exist.
  - check: only run environment and asset checks.
EOF
}

env_exists() {
    conda env list | awk '{print $1}' | grep -qx "$1"
}

env_prefix() {
    local env_name="$1"
    conda env list | awk -v name="${env_name}" '$1 == name {print $NF; exit}'
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

install_main_no_deps_extras() {
    conda run -n "${MAIN_ENV}" --no-capture-output \
        python -m pip install --no-deps "${SAM3_PACKAGE_REF}"
}

ensure_pytorch3d_source() {
    if [[ -d "${PYTORCH3D_SOURCE}/.git" ]]; then
        git -C "${PYTORCH3D_SOURCE}" fetch --tags --quiet
    else
        mkdir -p "$(dirname "${PYTORCH3D_SOURCE}")"
        git clone --recursive https://github.com/facebookresearch/pytorch3d.git "${PYTORCH3D_SOURCE}"
    fi
    git -C "${PYTORCH3D_SOURCE}" checkout --quiet "${PYTORCH3D_REF}"
    git -C "${PYTORCH3D_SOURCE}" submodule update --init --recursive --quiet
}

install_main_compiled_extras() {
    local main_prefix
    main_prefix="$(env_prefix "${MAIN_ENV}")"
    if [[ -z "${main_prefix}" ]]; then
        echo "Could not resolve conda prefix for ${MAIN_ENV}" >&2
        exit 1
    fi
    local cuda_target="${main_prefix}/targets/x86_64-linux"
    local build_env=(
        CUDA_HOME="${main_prefix}"
        CUDACXX="${main_prefix}/bin/nvcc"
        CUB_HOME="${cuda_target}/include"
        CPATH="${cuda_target}/include:${CPATH:-}"
        CPLUS_INCLUDE_PATH="${cuda_target}/include:${CPLUS_INCLUDE_PATH:-}"
        LIBRARY_PATH="${cuda_target}/lib:${LIBRARY_PATH:-}"
        LD_LIBRARY_PATH="${cuda_target}/lib:${LD_LIBRARY_PATH:-}"
        FORCE_CUDA=1
        TORCH_CUDA_ARCH_LIST="${DEMO_V5_TORCH_CUDA_ARCH_LIST:-8.9}"
        MAX_JOBS="${DEMO_V5_BUILD_MAX_JOBS:-8}"
    )
    ensure_pytorch3d_source
    rm -rf "${PYTORCH3D_SOURCE}/build"
    conda run -n "${MAIN_ENV}" --no-capture-output \
        env "${build_env[@]}" \
            PYTORCH3D_DISABLE_PULSAR=1 \
            python -m pip install --no-build-isolation --no-deps "${PYTORCH3D_SOURCE}"

}

install_shape_prior_compiled_extras() {
    local shape_prefix
    shape_prefix="$(env_prefix "${SHAPE_ENV}")"
    if [[ -z "${shape_prefix}" ]]; then
        echo "Could not resolve conda prefix for ${SHAPE_ENV}" >&2
        exit 1
    fi
    local cuda_target="${shape_prefix}/targets/x86_64-linux"
    local nvcc_flags="-I${cuda_target}/include"
    if [[ -n "${NVCC_FLAGS:-}" ]]; then
        nvcc_flags="${nvcc_flags} ${NVCC_FLAGS}"
    fi
    ensure_pytorch3d_source
    rm -rf "${PYTORCH3D_SOURCE}/build"
    conda run -n "${SHAPE_ENV}" --no-capture-output \
        env CUDA_HOME="${shape_prefix}" \
            CUDACXX="${shape_prefix}/bin/nvcc" \
            CUB_HOME="${cuda_target}/include" \
            FORCE_CUDA=1 \
            PYTORCH3D_DISABLE_PULSAR=1 \
            NVCC_FLAGS="${nvcc_flags}" \
            TORCH_CUDA_ARCH_LIST="${DEMO_V5_TORCH_CUDA_ARCH_LIST:-8.9}" \
            MAX_JOBS="${DEMO_V5_BUILD_MAX_JOBS:-8}" \
            python -m pip install --no-build-isolation --no-deps "${PYTORCH3D_SOURCE}"
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
            install_main_no_deps_extras
            install_main_compiled_extras
            create_or_update_env "${SHAPE_ENV}" "${SCRIPT_DIR}/environment-demo-v5-shape-prior.yml"
            install_pip_extras "${SHAPE_ENV}" "${SCRIPT_DIR}/requirements-demo-v5-shape-prior.txt"
            install_shape_prior_compiled_extras
            run_checks
            ;;
        *)
            usage >&2
            exit 2
            ;;
    esac
}

main "$@"
