# Demo 4 realtime_phystwin environment setup

Date: 2026-06-24

Branch: `single-camera`

Conda environment: `demo_2_max`

Goal: install the real Demo v4 / `realtime_phystwin` optimization and training stack into the existing Demo 4 conda environment without downgrading the current camera/demo stack and without placeholder implementations.

## Baseline preserved

- Python: `3.12.13`
- Torch: `2.11.0+cu130`
- Torch CUDA runtime: `13.0`
- GPU smoke device: `NVIDIA GeForce RTX 4090`
- Torchvision: `0.26.0`
- Pillow restored and kept at `12.2.0`

`pip check` in `demo_2_max` still reports the pre-existing `sam3` metadata conflict:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.5.
```

This was intentionally not fixed by downgrading NumPy.

## Packages installed

Non-compiled runtime dependencies:

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

Additional build/runtime dependencies used by Kaolin and gsplat:

```bash
conda run -n demo_2_max --no-capture-output python -m pip install \
  wget pygltflib ipycanvas ipyevents jupyter_client flask tornado comm pybind11 jaxtyping
```

CUDA/source extension builds used the existing Torch stack:

```bash
export CUDA_HOME=/usr/local/cuda
export CUDACXX=/usr/local/cuda/bin/nvcc
export FORCE_CUDA=1
export TORCH_CUDA_ARCH_LIST=8.9
export MAX_JOBS=8
export PYTORCH3D_DISABLE_PULSAR=1

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps /home/xinjie/external/pytorch3d-phystwin-max

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps /home/xinjie/external/kaolin-phystwin-max

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps /home/xinjie/external/mip-splatting-phystwin-max/submodules/diff-gaussian-rasterization

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps /home/xinjie/FuturePhysTwin/gaussian_splatting/submodules/simple-knn

conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps gsplat==1.4.0
```

Validated installed package versions:

```text
torch 2.11.0+cu130 cuda 13.0
torchvision 0.26.0
pillow 12.2.0
moviepy 2.2.1 distribution metadata
warp-lang 1.13.0
trimesh 4.12.2
wandb 0.27.0
cma 4.4.4
pytorch3d 0.7.9
kaolin 0.18.0
gsplat 1.4.0
```

Note: `moviepy.__version__` reports `2.1.2`, while `pip show moviepy` reports `2.2.1`.

## Compatibility patches

Environment metadata patch:

- Patched `/home/xinjie/miniforge3/envs/demo_2_max/lib/python3.12/site-packages/moviepy-2.2.1.dist-info/METADATA`
- Changed MoviePy's Pillow requirement from `pillow<12.0,>=9.2.0` to `pillow>=9.2.0`
- This preserved Pillow `12.2.0` and avoided a downgrade.

Source patches in `realtime_phystwin`:

- `realtime_phystwin/.gitignore`
  - Changed `data/` to `/data/` so package source under `qqtt/data/` can be tracked.
- `realtime_phystwin/qqtt/data/online_stream.py`
  - Added real `OnlineChunkReader` and `OnlineFrameBuffer` support for the existing online optimization/training entry points.
  - Reads committed online chunks, appends time-series arrays, builds device tensors, and constructs `structure_points` from object, surface, and interior points.
- `realtime_phystwin/qqtt/engine/trainer_warp.py`
  - Added `_log_wandb_video_if_present`.
  - If OpenCV cannot create an H.264 mp4 on this machine, training logs a warning and W&B metadata instead of aborting when the video file is absent.
  - This does not replace or skip physical optimization; it only prevents optional video logging from killing training.

## Validation

CUDA/import smoke:

```text
torch 2.11.0+cu130 cuda 13.0
gpu0 NVIDIA GeForce RTX 4090
warp_devices ['cpu', 'cuda:0', 'cuda:1']
kaolin 0.18.0
gsplat 1.4.0
diff_gaussian_signature (... kernel_size, subpixel_offset ...)
realtime_phystwin_env_cuda_smoke_ok
```

`realtime_phystwin` import smoke:

```text
realtime_phystwin_imports_ok
```

Demo v4 unit tests:

```text
Ran 39 tests in 1.367s
OK
```

Repo smoke validation:

```text
Ran 301 tests in 3.962s
OK
[validation] smoke checks passed
```

CMA smoke on a real Demo v4 chunk:

```bash
CUDA_VISIBLE_DEVICES=0 conda run -n demo_2_max --no-capture-output \
  python optimize_cma.py \
    --base_path /home/xinjie/single_proj_qqtt/result/demo_v4/full_fake_realtime_native_single_gpu_unlimited_20260624_130952/cases \
    --case_name demo_v4_native_single_gpu_unlimited_chunk_0031 \
    --train_frame 25 \
    --max_iter 1
```

Result:

```text
Optimal error: 0.0007854745072108926
Artifact: realtime_phystwin/experiments_optimization/demo_v4_native_single_gpu_unlimited_chunk_0031/optimal_params.pkl
```

Full `train_warp.py` run on GPU0:

```bash
CUDA_VISIBLE_DEVICES=0 WANDB_MODE=offline WANDB_SILENT=true \
conda run -n demo_2_max --no-capture-output \
  python train_warp.py \
    --base_path /home/xinjie/single_proj_qqtt/result/demo_v4/full_fake_realtime_native_single_gpu_unlimited_20260624_130952/cases \
    --case_name demo_v4_native_single_gpu_unlimited_chunk_0031 \
    --train_frame 25
```

Result:

```text
Iteration 199 loss: 7.215352070488734e-05
Latest best model saved: epoch 199
Artifact: realtime_phystwin/experiments/demo_v4_native_single_gpu_unlimited_chunk_0031/train/best_199.pth
Artifact: realtime_phystwin/experiments/demo_v4_native_single_gpu_unlimited_chunk_0031/train/iter_199.pth
Timing rows: 200 iterations plus CSV header
```

OpenCV still reports H.264 device warnings:

```text
Could not find a valid device
VIDEOIO/FFMPEG: Failed to initialize VideoWriter
```

After the `trainer_warp.py` compatibility patch, these warnings no longer abort training.

## Notes

- `taichi` and `rerun` were not installed because exact import searches in `realtime_phystwin` and `/home/xinjie/FuturePhysTwin` did not find direct runtime imports.
- Generated optimization/training artifacts remain ignored outputs and are not part of the source change.

## Supplemental audit after realtime_phystwin update check

After a later check for `realtime_phystwin` updates, `git fetch origin --prune`
inside `realtime_phystwin` showed no divergence between local `online` and
`origin/online`:

```text
git rev-list --left-right --count HEAD...origin/online
0 0
```

The updated dependency audit found that the core Demo v4 realtime PhysTwin path
was already covered, but full-repo optional imports still needed real packages
or real source paths for shape-prior and Gaussian rendering/evaluation scripts.

Additional packages installed into `demo_2_max`:

```bash
conda run -n demo_2_max --no-capture-output python -m pip install \
  --upgrade-strategy only-if-needed \
  diffusers \
  flow-vis \
  ipdb \
  pytorch-msssim \
  supervision \
  addict \
  yapf \
  pycocotools \
  easydict \
  rembg
```

Additional compiled/source packages:

```bash
CUDA_HOME=/usr/local/cuda \
CUDACXX=/usr/local/cuda/bin/nvcc \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST=8.9 \
MAX_JOBS=8 \
conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/single_proj_qqtt/realtime_phystwin/gaussian_splatting/submodules/fused-ssim
```

```bash
CUDA_HOME=/usr/local/cuda \
CUDACXX=/usr/local/cuda/bin/nvcc \
FORCE_CUDA=1 \
TORCH_CUDA_ARCH_LIST=8.9 \
MAX_JOBS=8 \
conda run -n demo_2_max --no-capture-output python -m pip install \
  --no-build-isolation --no-deps \
  /home/xinjie/external/GroundingDINO-phystwin-max
```

`utils3d==0.1.3` was installed with `--no-deps` because its package metadata
requires `open3d<0.14`, while the active environment uses Open3D `0.19.0`.
The installed `utils3d` metadata was patched from:

```text
Requires-Dist: open3d (>=0.13.0,<0.14.0)
```

to:

```text
Requires-Dist: open3d (>=0.13.0)
```

This avoided downgrading Open3D.

External source paths were registered with:

```text
/home/xinjie/miniforge3/envs/demo_2_max/lib/python3.12/site-packages/realtime_phystwin_external_paths.pth
```

Contents:

```text
/home/xinjie/FuturePhysTwin/data_process
/home/xinjie/single_proj_qqtt/realtime_phystwin/data_process
/home/xinjie/single_proj_qqtt/realtime_phystwin/gaussian_splatting
/home/xinjie/single_proj_qqtt/realtime_phystwin/gaussian_splatting/utils
```

This lets old scripts resolve real local modules such as `TRELLIS`,
`match_pairs`, `models`, `lpipsPyTorch`, and `read_write_model` without copying
or faking modules.

Supplemental import validation:

```text
diffusers ok
flow_vis ok
ipdb ok
pytorch_msssim ok
supervision ok
fused_ssim ok
groundingdino ok
easydict ok
rembg ok
utils3d ok
TRELLIS.trellis.pipelines ok
lpipsPyTorch ok
match_pairs ok
models.matching ok
read_write_model ok
supplemental_realtime_phystwin_imports_ok
```

CUDA extension validation:

```text
fused_ssim_cuda_ok
groundingdino_cuda_ext_ok
```

Static import audit after the supplemental install:

```text
missing_imports
missing_count 0
```

Post-supplement validation:

```text
python -m unittest tests.test_demo_v4_futurephystwin_chunks
Ran 40 tests
OK
```

```text
python scripts/harness/validation/run.py --profile smoke
Ran 301 tests
OK
[validation] smoke checks passed
```

`pip check` after the supplemental install reports only the pre-existing `sam3`
metadata conflict:

```text
sam3 0.1.0 has requirement numpy<2,>=1.26, but you have numpy 2.4.5.
```

The TRELLIS import emits a Numba/TBB warning because the installed TBB is older
than the optional parallel threading layer wants. Import still succeeds.
