#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="demo3_trackers"
BASE_ENV="demo_2_max"
ALLOW_MUTATE_CURRENT_ENV=0
INSTALL_LOCOTRACK=0
INSTALL_TAPNET=0
INSTALL_NVOFA=0
INSTALL_ORT_TRT_PROBE=0
EXTERNAL_ROOT="/home/zhangxinjie/external_tracking_backends"
LOG_ROOT="data/experiments/demo3_tracking_backend_install_logs"

usage() {
  cat <<'EOF'
Usage:
  bash scripts/harness/experiments/demo3_tracking_backend_install/install_tracking_backends_optional.sh \
    --env demo3_trackers \
    --base-env demo_2_max \
    --install-locotrack \
    --install-tapnet \
    --install-nvofa \
    --install-onnxruntime-trt-probe

This is a manual optional installer. It does not run from check_all.py.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --env)
      ENV_NAME="$2"; shift 2 ;;
    --base-env)
      BASE_ENV="$2"; shift 2 ;;
    --allow-mutate-current-env)
      ALLOW_MUTATE_CURRENT_ENV=1; shift ;;
    --install-locotrack)
      INSTALL_LOCOTRACK=1; shift ;;
    --install-tapnet)
      INSTALL_TAPNET=1; shift ;;
    --install-nvofa)
      INSTALL_NVOFA=1; shift ;;
    --install-onnxruntime-trt-probe)
      INSTALL_ORT_TRT_PROBE=1; shift ;;
    --external-root)
      EXTERNAL_ROOT="$2"; shift 2 ;;
    --log-root)
      LOG_ROOT="$2"; shift 2 ;;
    -h|--help)
      usage; exit 0 ;;
    *)
      echo "Unknown option: $1" >&2
      usage
      exit 2 ;;
  esac
done

mkdir -p "$EXTERNAL_ROOT" "$LOG_ROOT"
LOG_FILE="$LOG_ROOT/install_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG_FILE") 2>&1

echo "[demo3-install] env=$ENV_NAME base_env=$BASE_ENV external_root=$EXTERNAL_ROOT"
echo "[demo3-install] log=$LOG_FILE"

if [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  # shellcheck source=/dev/null
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
else
  echo "conda.sh not found; activate conda before running this script." >&2
  exit 1
fi

if [[ "$ALLOW_MUTATE_CURRENT_ENV" -ne 1 ]]; then
  if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
    echo "[demo3-install] creating env $ENV_NAME by cloning $BASE_ENV"
    conda create -y -n "$ENV_NAME" --clone "$BASE_ENV"
  fi
  conda activate "$ENV_NAME"
else
  echo "[demo3-install] mutating current env because --allow-mutate-current-env was passed"
fi

python - <<'PY'
import sys
try:
    import torch
    print(f"[demo3-install] python={sys.version.split()[0]} torch={torch.__version__} cuda={torch.version.cuda}")
except Exception as exc:
    print(f"[demo3-install] torch probe failed: {exc}")
PY

clone_if_missing() {
  local url="$1"
  local dir="$2"
  if [[ -d "$dir/.git" ]]; then
    echo "[demo3-install] repo exists: $dir"
  else
    echo "[demo3-install] cloning $url -> $dir"
    git clone "$url" "$dir"
  fi
}

if [[ "$INSTALL_LOCOTRACK" -eq 1 ]]; then
  clone_if_missing "https://github.com/cvlab-kaist/locotrack" "$EXTERNAL_ROOT/locotrack"
  if [[ -f "$EXTERNAL_ROOT/locotrack/requirements.txt" ]]; then
    python -m pip install -r "$EXTERNAL_ROOT/locotrack/requirements.txt" || echo "[demo3-install] locotrack requirements install failed; continuing with probe"
  fi
  python -m pip install -e "$EXTERNAL_ROOT/locotrack" || echo "[demo3-install] editable locotrack install failed; leaving repo clone for manual setup"
fi

if [[ "$INSTALL_TAPNET" -eq 1 ]]; then
  clone_if_missing "https://github.com/google-deepmind/tapnet" "$EXTERNAL_ROOT/tapnet"
  if [[ -f "$EXTERNAL_ROOT/tapnet/requirements.txt" ]]; then
    python -m pip install -r "$EXTERNAL_ROOT/tapnet/requirements.txt" || echo "[demo3-install] tapnet requirements install failed; continuing with probe"
  fi
  python -m pip install -e "$EXTERNAL_ROOT/tapnet" || echo "[demo3-install] editable tapnet install failed; leaving repo clone for manual setup"
fi

if [[ "$INSTALL_NVOFA" -eq 1 ]]; then
  clone_if_missing "https://github.com/NVIDIA/NVIDIAOpticalFlowSDK" "$EXTERNAL_ROOT/NVIDIAOpticalFlowSDK"
  echo "[demo3-install] NVOFA SDK cloned. Build a flow helper manually if CMake/CUDA SDK requirements are not already configured."
fi

if [[ "$INSTALL_ORT_TRT_PROBE" -eq 1 ]]; then
  python -m pip install onnx onnxruntime-gpu || echo "[demo3-install] onnxruntime-gpu install failed; stack probe will record current status"
fi

python scripts/harness/experiments/check_demo3_tracking_backend_stack.py
echo "[demo3-install] done"
