# Demo v5 CUDA 12.8 Workstation Environment Rerun

## Goal

Install Demo v5 completely and run the end-to-end fake-live experiment on the
current `single-camera` workstation.

## Constraints

- Keep source changes on `single-camera`.
- The current NVIDIA driver reports CUDA 12.8, so CUDA 13 PyTorch wheels cannot
  initialize CUDA here.
- Keep external source builds outside this repo.
- Preserve the Demo v5 runtime contract; this plan only adjusts environment
  installation materials and validation commands.

## Plan

- [x] Confirm branch and sync state.
- [x] Run the existing Demo v5 installer and capture the failure mode.
- [x] Verify the local CUDA driver/toolkit matrix.
- [x] Patch Demo v5 environment requirements for a CUDA 12.6 runtime stack that
      runs on the CUDA 12.8 driver.
- [x] Install/update `demo_2_max` and `phystwin-max`.
- [x] Build/install PyTorch3D for the shape-prior worker from an external
      source checkout if no wheel is available.
- [x] Run Demo v5 environment checks.
- [ ] Run Demo v5 dry-run and end-to-end fake-live experiment.

## Validation Log

- `git branch --show-current` -> `single-camera`.
- `git pull --ff-only origin single-camera` -> already up to date.
- `git pull --ff-only origin main` -> already up to date.
- `bash demo_v5/env/install_demo_v5_env.sh update` -> failed while installing
  `phystwin-max`: pip could not resolve `kaolin==0.18.0` from the default
  indexes.
- `nvidia-smi` -> driver `570.211.01`, CUDA version `12.8`.
- `conda run -n demo_2_max ... torch.cuda.is_available()` -> `False` after the
  original installer put `torch 2.11.0+cu130` in `demo_2_max`.
- Patched Demo v5 environment files to use CUDA 12.6 PyTorch wheels on this
  CUDA 12.8 driver workstation. `demo_2_max` now uses `torch 2.11.0+cu126` and
  `torchvision 0.26.0+cu126`.
- Patched the shape-prior stack to use `torch 2.8.0+cu126`, NVIDIA's
  `kaolin 0.18.0` wheel index for `torch-2.8.0_cu126`, `xformers
  0.0.32.post2`, and MoGe from the SAM 3D Objects pinned Git commit.
- Added conda CUDA 12.6 dev libraries for `phystwin-max` and compiled
  `pytorch3d 0.7.9` from the external checkout
  `/home/shen/external/pytorch3d-demo-v5-cu126`.
- `bash demo_v5/env/install_demo_v5_env.sh update` -> passed. Both Demo v5
  environment checks passed with `torch.cuda.is_available() == True` and
  `device_count == 2`.
- `conda run -n demo_2_max --no-capture-output python
  demo_v5/realtime_futurephystwin_chunks.py --dry-run` -> passed. The resolved
  contract is fake-live input, camera/realtime on physical GPU0, managed
  shape-prior worker on physical GPU1, and continuous optimization on physical
  GPU1 after the worker is released.
- First quality fake-live attempt with output base
  `result/demo_v5/full_fake_live_20260625_022746` failed before chunk
  generation because the main environment segmentation worker could not import
  `transformers`.
- Added `transformers==5.9.0` to the main Demo v5 requirements and environment
  check. `conda run -n demo_2_max --no-capture-output python -m pip install -r
  demo_v5/env/requirements-demo-v5-main.txt && bash
  demo_v5/env/install_demo_v5_env.sh check` -> passed.
- Second quality fake-live attempt with output base
  `result/demo_v5/full_fake_live_20260625_022904` failed before chunk
  generation because the main environment TAPNext++ worker could not import
  `einops`.
- Added `einops==0.8.1` and strengthened the main environment check to import
  the vendor TAPNext++ torch modules and EdgeTAM transformer classes. The main
  and shape-prior checks pass.
- Third quality fake-live attempt with output base
  `result/demo_v5/full_fake_live_20260625_023047` failed before chunk
  generation because EdgeTAM config loading required `timm`.
- Added `timm==1.0.27` to the main requirements and strengthened the main
  check to load the local EdgeTAM config. The main and shape-prior checks pass.
