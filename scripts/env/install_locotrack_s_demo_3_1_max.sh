#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-demo_3_1_max}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT}/external"
LOCOTRACK_DIR="${EXTERNAL_DIR}/locotrack"
LOCOTRACK_PYTORCH_DIR="${LOCOTRACK_DIR}/locotrack_pytorch"
CKPT_DIR="${ROOT}/checkpoints/locotrack"
CKPT_PATH="${CKPT_DIR}/locotrack_small.ckpt"
FULL=0

usage() {
  cat <<EOF
Install LocoTrack-S into existing conda env: ${ENV_NAME}

Usage:
  scripts/env/install_locotrack_s_demo_3_1_max.sh [--full]

Default installs live-inference dependencies only and does not reinstall torch.
--full also installs LocoTrack eval/training dependencies.
EOF
}

for arg in "$@"; do
  case "$arg" in
    --full)
      FULL=1
      ;;
    --help|-h)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: ${arg}" >&2
      usage >&2
      exit 2
      ;;
  esac
done

mkdir -p "${EXTERNAL_DIR}" "${CKPT_DIR}"

if [ ! -d "${LOCOTRACK_DIR}/.git" ]; then
  git clone https://github.com/cvlab-kaist/locotrack.git "${LOCOTRACK_DIR}"
else
  git -C "${LOCOTRACK_DIR}" pull --ff-only
fi

# Do not reinstall torch/torchvision/torchaudio; demo_3_1_max already owns CUDA torch.
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install \
    einops==0.8.0 \
    mediapy==1.2.2 \
    opencv-python==4.10.0.84 \
    matplotlib

if [ "${FULL}" = "1" ]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install \
      lightning==2.3.3 \
      tensorflow_datasets \
      tensorflow \
      tensorflow_graphics \
      wandb
fi

if [ ! -f "${CKPT_PATH}" ]; then
  conda run --no-capture-output -n "${ENV_NAME}" python - <<PY
import torch

url = "https://huggingface.co/datasets/hamacojr/LocoTrack-pytorch-weights/resolve/main/locotrack_small.ckpt"
dst = r"${CKPT_PATH}"
torch.hub.download_url_to_file(url, dst, progress=True)
print(dst)
PY
fi

conda run --no-capture-output -n "${ENV_NAME}" python - <<PY
import sys
from pathlib import Path

repo = Path(r"${LOCOTRACK_PYTORCH_DIR}")
ckpt = Path(r"${CKPT_PATH}")
assert repo.is_dir(), repo
assert ckpt.is_file(), ckpt
sys.path.insert(0, str(repo))
from models.locotrack_model import load_model

model = load_model(str(ckpt), model_size="small")
print("LocoTrack-S import/load OK:", type(model).__name__)
PY

cat <<EOF

LocoTrack-S installed.

Use Demo 3.1 flags:
  --cotracker-backend locotrack \\
  --tracking-backend-execution-mode batch-views \\
  --locotrack-repo-dir ${LOCOTRACK_PYTORCH_DIR} \\
  --locotrack-checkpoint ${CKPT_PATH} \\
  --locotrack-model-size small \\
  --locotrack-window-frames 8 \\
  --locotrack-query-chunk-size 256
EOF
