#!/usr/bin/env bash
set -euo pipefail

ENV_NAME="${ENV_NAME:-demo_3_1_max}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EXTERNAL_DIR="${ROOT}/external"
TAPNET_DIR="${EXTERNAL_DIR}/tapnet"
CKPT_DIR="${ROOT}/checkpoints/tapnextpp"
CKPT_PATH="${CKPT_DIR}/tapnextpp_ckpt.pt"
ALLOW_NUMPY_DOWNGRADE=0
CUDA_SMOKE=0
CUDA_BATCH_SMOKE=0

usage() {
  cat <<EOF
Install TAPNext++ into existing conda env: ${ENV_NAME}

Usage:
  scripts/env/install_tapnextpp_demo_3_1_max.sh [--allow-numpy-downgrade] [--cuda-smoke] [--cuda-batch-smoke]

Default behavior:
  - clone/update google-deepmind/tapnet into external/tapnet
  - install TapNet editable with --no-deps
  - install recurrentgemma with --no-deps
  - install lightweight runtime deps only if missing
  - download checkpoints/tapnextpp/tapnextpp_ckpt.pt
  - do not reinstall torch/torchvision/torchaudio
  - do not downgrade numpy unless --allow-numpy-downgrade is passed
EOF
}

for arg in "$@"; do
  case "${arg}" in
    --allow-numpy-downgrade) ALLOW_NUMPY_DOWNGRADE=1 ;;
    --cuda-smoke) CUDA_SMOKE=1 ;;
    --cuda-batch-smoke) CUDA_BATCH_SMOKE=1 ;;
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

if [ ! -d "${TAPNET_DIR}/.git" ]; then
  git clone https://github.com/google-deepmind/tapnet.git "${TAPNET_DIR}"
else
  git -C "${TAPNET_DIR}" pull --ff-only
fi

# The upstream top-level tapnet/__init__.py eagerly imports legacy JAX/TF
# modules. Demo 3.1 only needs the PyTorch tapnext package, so keep top-level
# imports lightweight in this local external checkout.
python - <<PY
from pathlib import Path
path = Path(r"${TAPNET_DIR}") / "tapnet" / "__init__.py"
text = '''# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""TAP package.

This local checkout is used by the Demo 3.1 PyTorch TAPNext++ backend. Keep
top-level imports lightweight so ``tapnet.tapnext`` does not require the
legacy JAX/TensorFlow evaluation stack.
"""
'''
path.write_text(text, encoding="utf-8")
print(path)
PY

# Do not reinstall torch/torchvision/torchaudio; demo_3_1_max owns CUDA torch.
conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install -e "${TAPNET_DIR}" --no-deps

conda run --no-capture-output -n "${ENV_NAME}" \
  python -m pip install --no-deps \
    "git+https://github.com/google-deepmind/recurrentgemma.git@main"

conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
import importlib.util
import subprocess
import sys

missing = []
for module, package in [
    ("tqdm", "tqdm"),
    ("einops", "einops"),
    ("mediapy", "mediapy"),
    ("cv2", "opencv-python"),
    ("matplotlib", "matplotlib"),
]:
    if importlib.util.find_spec(module) is None:
        missing.append(package)
if missing:
    subprocess.check_call([sys.executable, "-m", "pip", "install", *missing])
else:
    print("TAPNext++ lightweight runtime deps already present")
PY

if [ "${ALLOW_NUMPY_DOWNGRADE}" = "1" ]; then
  conda run --no-capture-output -n "${ENV_NAME}" \
    python -m pip install "numpy<2.1.0"
else
  conda run --no-capture-output -n "${ENV_NAME}" python - <<'PY'
from importlib.metadata import version
from packaging.version import Version

np_version = Version(version("numpy"))
if np_version >= Version("2.1.0"):
    print(
        "WARNING: upstream TAPNext++ colab pins numpy<2.1.0; keeping current "
        f"numpy {np_version}. If the smoke test fails with numpy errors, rerun "
        "with --allow-numpy-downgrade."
    )
PY
fi

if [ ! -f "${CKPT_PATH}" ]; then
  conda run --no-capture-output -n "${ENV_NAME}" python - <<PY
import torch
url = "https://storage.googleapis.com/dm-tapnet/tapnextpp/tapnextpp_ckpt.pt"
dst = r"${CKPT_PATH}"
torch.hub.download_url_to_file(url, dst, progress=True)
print(dst)
PY
fi

conda run --no-capture-output -n "${ENV_NAME}" python - <<PY
from pathlib import Path
import torch
from tapnet.tapnext.tapnext_torch import TAPNext
from tapnet.tapnext.tapnext_torch_utils import tracker_certainty

ckpt_path = Path(r"${CKPT_PATH}")
model = TAPNext(image_size=(256, 256))
ckpt = torch.load(ckpt_path, map_location="cpu")
state = {k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()}
missing, unexpected = model.load_state_dict(state, strict=False)
assert not missing, missing
assert not unexpected, unexpected
print("TAPNext++ import/load OK:", type(model).__name__, ckpt_path)
print("tracker_certainty:", callable(tracker_certainty))
PY

if [ "${CUDA_SMOKE}" = "1" ] || [ "${CUDA_BATCH_SMOKE}" = "1" ]; then
  BATCHES=()
  if [ "${CUDA_SMOKE}" = "1" ]; then
    BATCHES+=(1)
  fi
  if [ "${CUDA_BATCH_SMOKE}" = "1" ]; then
    BATCHES+=(3)
  fi
  conda run --no-capture-output -n "${ENV_NAME}" python - "${CKPT_PATH}" "${BATCHES[@]}" <<'PY'
import sys
import time
from pathlib import Path

import torch
from tapnet.tapnext.tapnext_torch import TAPNext

ckpt_path = Path(sys.argv[1])
batches = [int(arg) for arg in sys.argv[2:]]
if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available for TAPNext++ smoke")

def load_model():
    model = TAPNext(image_size=(256, 256))
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = {k.replace("tapnext.", ""): v for k, v in ckpt["state_dict"].items()}
    model.load_state_dict(state)
    model.to("cuda").eval()
    return model

for batch_size in batches:
    model = load_model()
    video0 = torch.rand((batch_size, 1, 256, 256, 3), device="cuda") * 255.0
    video1 = torch.rand((batch_size, 1, 256, 256, 3), device="cuda") * 255.0
    yx = torch.rand((batch_size, 8, 2), device="cuda") * 255.0
    query = torch.cat([torch.zeros((batch_size, 8, 1), device="cuda"), yx], dim=-1)
    torch.cuda.synchronize()
    started = time.perf_counter()
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.float16, enabled=True):
        tracks0, _track_logits0, visible0, state = model(video=video0, query_points=query)
        tracks1, _track_logits1, visible1, state = model(video=video1, state=state)
    torch.cuda.synchronize()
    print(
        f"CUDA smoke B={batch_size} N=8 OK",
        tuple(tracks0.shape),
        tuple(tracks1.shape),
        tuple(visible1.shape),
        "state_step",
        state.step,
        "elapsed_ms",
        round((time.perf_counter() - started) * 1000.0, 2),
    )
    del model
    torch.cuda.empty_cache()
PY
fi

cat <<EOF

TAPNext++ installed.

Use Demo 3.1 flags:
  --cotracker-backend tapnextpp \\
  --tracking-backend-execution-mode batch-views \\
  --tapnet-repo-dir ${TAPNET_DIR} \\
  --tapnextpp-checkpoint ${CKPT_PATH} \\
  --tapnextpp-image-size 256,256 \\
  --tapnextpp-autocast-dtype fp16
EOF
