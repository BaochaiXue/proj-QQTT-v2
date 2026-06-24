# Demo 4 Realtime PhysTwin Environment Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use systematic debugging for build/import failures. Execute this plan directly under the repo autonomous goal policy; do not wait for separate approval.

**Goal:** Make the `demo_2_max` conda environment able to run the real `realtime_phystwin` Demo v4 optimization/inference stack without downgrading the existing Demo 3/4 camera environment and without placeholders.

**Architecture:** Keep the existing Python 3.12, Torch 2.11.0+cu130, CUDA 13 runtime in `demo_2_max`. Install missing PhysTwin dependencies and CUDA extensions from known working external source trees, patching source compatibility issues if a build fails. Validate by importing the real realtime_phystwin entry points and exercising CUDA-backed extension smoke checks.

**Tech Stack:** Conda `demo_2_max`, PyTorch 2.11 CUDA 13, Warp, PyTorch3D, Kaolin, gsplat, mip-splatting `diff_gaussian_rasterization`, `simple_knn`, FuturePhysTwin/realtime_phystwin.

---

## Baseline

- Branch must be `single-camera`.
- Run `git pull --ff-only origin main` before edits.
- Existing `demo_2_max` baseline keeps:
  - `python 3.12.13`
  - `torch 2.11.0+cu130`
  - `torchvision 0.26.0+cu130`
  - CUDA device available on RTX 4090
- Baseline `pip check` currently reports the pre-existing `sam3` metadata conflict with `numpy 2.4.5`; do not solve this by downgrading numpy.

## Task 1: Install Missing Non-Compiled Dependencies

**Commands:**

```bash
conda run -n demo_2_max --no-capture-output python -m pip install \
  --upgrade-strategy only-if-needed \
  warp-lang==1.13.0 \
  trimesh==4.12.2 \
  wandb==0.27.0 \
  plyfile \
  lpips \
  cma \
  termcolor \
  fvcore \
  stannum \
  moviepy \
  kornia \
  pyrender \
  Rtree \
  'pyglet<2' \
  usd-core
```

**Validation:**

```bash
conda run -n demo_2_max --no-capture-output python - <<'PY'
import cma, fvcore, kornia, lpips, moviepy, plyfile, pyrender, trimesh, wandb, warp
print("non_compiled_physwin_deps_ok")
PY
```

## Task 2: Build CUDA Extensions From Real Sources

Use CUDA 13 from `/usr/local/cuda` and preserve the current torch build.

**Environment:**

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:${LD_LIBRARY_PATH:-}
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
```

**Commands:**

```bash
conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/external/pytorch3d-phystwin-max

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/external/kaolin-phystwin-max

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/external/mip-splatting-phystwin-max/submodules/diff-gaussian-rasterization

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/FuturePhysTwin/gaussian_splatting/submodules/simple-knn

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  gsplat==1.4.0
```

**Validation:**

```bash
conda run -n demo_2_max --no-capture-output python - <<'PY'
import torch
import kaolin
import pytorch3d
import diff_gaussian_rasterization
import simple_knn
import gsplat
print(torch.__version__, torch.version.cuda)
print("compiled_physwin_deps_ok")
PY
```

## Task 3: Validate Realtime PhysTwin Imports

Run from the `realtime_phystwin` checkout so its `qqtt` package wins import resolution.

```bash
cd /home/xinjie/single_proj_qqtt/realtime_phystwin
conda run -n demo_2_max --no-capture-output python - <<'PY'
import optimize_cma
import train_warp
import inference_warp
from qqtt import InvPhyTrainerWarp, OptimizerCMA
from qqtt.model.diff_simulator.spring_mass_warp import SpringMassSystemWarp
print("realtime_phystwin_imports_ok")
PY
```

## Task 4: Run CUDA Smoke Checks

```bash
conda run -n demo_2_max --no-capture-output python - <<'PY'
import torch
from pytorch3d.ops import knn_points

x = torch.rand(1, 8, 3, device="cuda")
y = torch.rand(1, 8, 3, device="cuda")
dist, idx, _ = knn_points(x, y, K=2)
assert dist.is_cuda and idx.is_cuda
print("pytorch3d_cuda_ok")
PY
```

```bash
cd /home/xinjie/single_proj_qqtt/realtime_phystwin
conda run -n demo_2_max --no-capture-output python - <<'PY'
import torch
import warp as wp

wp.init()
print("warp_cuda_devices", wp.get_devices())
print("torch_cuda_device", torch.cuda.get_device_name(0))
PY
```

## Task 5: Record Outcome

Create `docs/generated/demo4_realtime_phystwin_env_2026-06-24.md` with:

- Exact install commands that succeeded.
- Torch/CUDA versions after install.
- Import and CUDA smoke results.
- Any pre-existing conflicts that remain because fixing them would require a downgrade.
- Any source patches applied under external dependency trees.

## Task 6: Commit And Push Scoped Docs

Only stage files created for this environment task. Do not stage the nested `realtime_phystwin/` checkout unless explicitly requested.

```bash
git add docs/exec-plans/active/2026-06-24-demo4-realtime-phystwin-env.md \
  docs/generated/demo4_realtime_phystwin_env_2026-06-24.md
git commit -m "docs: record demo4 realtime phystwin env setup"
git push origin single-camera
```

## Execution Outcome

- Installed the realtime PhysTwin dependency stack into `demo_2_max` while preserving Python 3.12, Torch 2.11.0+cu130, torchvision 0.26.0, and Pillow 12.2.0.
- Built PyTorch3D, Kaolin, mip-splatting `diff_gaussian_rasterization`, FuturePhysTwin `simple_knn`, and `gsplat` against the existing CUDA/Torch stack.
- Patched MoviePy distribution metadata in the conda environment so Pillow 12.2.0 is accepted instead of forcing a Pillow downgrade.
- Added real `realtime_phystwin` online-stream support and patched optional W&B video logging so missing H.264 writer output does not abort training.
- Validation completed:
  - realtime PhysTwin import and CUDA smoke passed.
  - `python -m unittest tests.test_demo_v4_futurephystwin_chunks` passed with 39 tests.
  - `optimize_cma.py --max_iter 1` completed on Demo v4 chunk 0031 and wrote `optimal_params.pkl`.
  - `train_warp.py` completed through iteration 199 on GPU0 and wrote `best_199.pth`.
  - `scripts/harness/validation/run.py --profile smoke` passed with 301 tests.
- `pip check` still reports the pre-existing `sam3` vs NumPy 2.4.5 metadata conflict; it was not resolved by downgrading NumPy.

Detailed command transcript and artifact paths are recorded in `docs/generated/demo4_realtime_phystwin_env_2026-06-24.md`.
