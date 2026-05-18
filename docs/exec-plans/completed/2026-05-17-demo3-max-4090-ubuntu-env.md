# demo3-max 4090 Ubuntu Environment

## Goal

Create and validate a native Ubuntu RTX 4090 `demo3-max` conda environment for
Demo 3 by cloning the existing `demo_2_max` environment, then adding only the
CoTracker3 online repo/dependencies and checkpoint needed by the Demo 3
RealSense CoTracker overlay path.

## Scope

- Keep `demo_2_max` unchanged.
- Use native Ubuntu RTX 4090 assumptions, including `TORCH_CUDA_ARCH_LIST=8.9`.
- Do not install WSL usbipd helpers or RTX 5090/SM 12.0 settings.
- Do not build or validate a new FFS TensorRT engine for Demo 3.
- Keep external CoTracker code and checkpoints outside this repo.
- Record validation under `docs/generated/`.

## Plan

1. Confirm the repo is up to date with `origin/main`.
2. Locate the local conda installation and confirm `demo_2_max` exists.
3. Clone `demo_2_max` to `demo3-max`.
4. Add `demo3-max` activation hooks for repo paths, CUDA, SAM 3.1 checkpoint,
   CoTracker repo, and `TORCH_CUDA_ARCH_LIST=8.9`.
5. Clone/update `~/co-tracker`, install it editable in `demo3-max`, and download
   CoTracker3 online checkpoints.
6. Validate Python, PyTorch/CUDA, RealSense, Open3D, HF EdgeTAM, SAM 3.1,
   CoTracker import, repo Demo 3 dry-run, deterministic harness checks, and any
   available RealSense hardware probes.
7. Write `docs/generated/demo3_max_4090_ubuntu_env_validation.md` and
   `docs/generated/demo3_max_4090_ubuntu_env_validation.json` with exact
   commands and outcomes.

## Validation Targets

- `conda run --no-capture-output -n demo3-max python ...` base stack probe.
- `conda run --no-capture-output -n demo3-max python demo_v3/realtime_three_view_cotracker3_realsense_overlay.py --dry-run --camera-ids 0,1,2`.
- `conda run --no-capture-output -n demo3-max python scripts/harness/check_harness_catalog.py`.
- `conda run --no-capture-output -n demo3-max python scripts/harness/check_all.py`.
- Live mask-only / q30 / q128 runs when three RealSense cameras and display
  access are available.

## Results

- Repo sync: `git pull --ff-only origin main` reported already up to date.
- Created `demo3-max` by cloning `/home/xinjie/miniforge3/envs/demo_2_max`.
- Replaced the cloned activation hook with `demo3_max_paths.sh` and set
  `TORCH_CUDA_ARCH_LIST=8.9`.
- Cloned CoTracker to `/home/xinjie/co-tracker` at
  `82e02e8029753ad4ef13cf06be7f4fc5facdda4d`.
- Installed CoTracker editable into `demo3-max`, added the README supplement
  dependencies, and downloaded `scaled_online.pth` plus `baseline_online.pth`.
- Added `pycocotools==2.0.11` so `sam3` imports in the cloned env, and restored
  `setuptools==81.0.0` to satisfy Torch's `<82` metadata.
- PASS: base stack import probe with `torch==2.11.0+cu130`, CUDA `13.0`, RTX
  4090 capability `(8, 9)`, `pyrealsense2`, and `open3d==0.19.0`.
- PASS: HF EdgeTAM imports, SAM 3.1 import, CoTracker import, and CoTracker
  `scaled_online.pth` model load probe.
- WARN: `python -m pip check` still reports the inherited `sam3` metadata
  mismatch with `numpy==2.4.5`.
- PASS: Demo 3 dry-run contract with `depth_source=realsense`,
  `uses_ffs=false`, `mask_source=hf_edgetam`, and
  `cotracker_backend=cotracker3_online`.
- PASS: RealSense probe sees exactly three D455 devices.
- PASS: `scripts/harness/check_harness_catalog.py`.
- PASS: `scripts/harness/check_all.py` quick profile, including 253 unittest
  tests.
- BLOCKED: mask-only/q30/q128/profile live runs require a real root-level
  `calibrate.pkl`; no fake calibration was substituted.
- Reports written:
  - `docs/generated/demo3_max_4090_ubuntu_env_validation.md`
  - `docs/generated/demo3_max_4090_ubuntu_env_validation.json`
