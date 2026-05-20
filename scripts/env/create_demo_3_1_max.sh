#!/usr/bin/env bash
set -euo pipefail

SOURCE_ENV="${SOURCE_ENV:-}"
TARGET_ENV="${TARGET_ENV:-demo_3_1_max}"
CONDA_BIN="${CONDA_BIN:-conda}"

ENV_NAMES="$("${CONDA_BIN}" env list | awk '{print $1}')"
if [[ -z "${SOURCE_ENV}" ]]; then
  if printf '%s\n' "${ENV_NAMES}" | grep -qx "demo_3_max"; then
    SOURCE_ENV="demo_3_max"
  else
    SOURCE_ENV="demo3-max"
  fi
fi

if printf '%s\n' "${ENV_NAMES}" | grep -qx "${TARGET_ENV}"; then
  echo "Conda environment ${TARGET_ENV} already exists; leaving it unchanged."
else
  "${CONDA_BIN}" create -y --name "${TARGET_ENV}" --clone "${SOURCE_ENV}"
fi

"${CONDA_BIN}" run -n "${TARGET_ENV}" --no-capture-output python -m pip install -U pip
"${CONDA_BIN}" run -n "${TARGET_ENV}" --no-capture-output python -m pip install onnx onnxruntime-gpu onnxscript

cat <<'MSG'

demo_3_1_max has been cloned. External tracker repos and weights remain manual:

Track-On2:
  git clone https://github.com/gorkaydemir/track_on.git third_party/track_on
  conda run -n demo_3_1_max --no-capture-output python -m pip install -r third_party/track_on/requirements.txt

LiteTracker:
  git clone https://github.com/ImFusionGmbH/lite-tracker.git third_party/lite-tracker
  cd third_party/lite-tracker && uv sync
  # Optional serial ONNX-CUDA profiling path:
  conda run -n demo_3_1_max --no-capture-output python -m pip install onnx onnxruntime-gpu onnxscript

Notes:
  - Do not replace the core PyTorch stack unless a tracker repo explicitly requires it.
  - Track-On2 may require Hugging Face access for DINOv3 weights.
  - Track-On2 mmcv CUDA ops on RTX 4090 may require TORCH_CUDA_ARCH_LIST=8.9 and a source build.
  - LiteTracker needs its tracker weights and may reuse CoTracker3 online weights depending on checkout.
MSG
