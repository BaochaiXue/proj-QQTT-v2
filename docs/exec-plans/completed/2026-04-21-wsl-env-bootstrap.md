# 2026-04-21 WSL Environment Bootstrap

## Goal

Bootstrap this migrated WSL workspace so the QQTT repo and external Fast-FoundationStereo
repo can both run from named conda environments that match the repo's current validated
CUDA / torch stack.

The bootstrap should leave behind:

- `ffs-standalone`
- `qqtt-ffs-compat`
- a local external FFS repo path under WSL
- the selected checkpoint under that external repo
- generated validation notes with exact commands and outcomes

## Non-Goals

- no production feature changes
- no repo scope changes
- no fake RealSense hardware validation in CI
- no TensorRT runtime setup in this pass
- no vendoring of Fast-FoundationStereo source or weights into this repo

## Local Path Assumptions

- QQTT repo root: `/home/zhangxinjie/proj-QQTT-v2`
- External FFS repo root: `/home/zhangxinjie/Fast-FoundationStereo`
- Miniconda base: `/home/zhangxinjie/miniconda3`
- Existing CUDA-capable reference env: `rtx5090`

## Files To Touch

- `scripts/harness/verify_ffs_demo.py`
- new `docs/generated/wsl_env_bootstrap_validation.md`
- `docs/generated/README.md`
- `docs/envs.md`
- `docs/external-deps.md`
- this exec plan

## Implementation Plan

1. verify the current WSL host facts:
   - conda availability
   - visible NVIDIA GPU
   - external FFS repo path
   - current checkpoint presence or absence
2. create `ffs-standalone` and `qqtt-ffs-compat` from the local CUDA-capable base env,
   then install any missing packages needed by QQTT and FFS command surfaces
3. fetch the selected external FFS checkpoint under `/home/zhangxinjie/Fast-FoundationStereo/weights/`
   if it is not already present
4. validate both environments with command-level proof-of-life:
   - import / version checks
   - QQTT `--help` commands in `qqtt-ffs-compat`
   - FFS demo or equivalent repo-facing proof-of-life in `ffs-standalone`
5. document the exact WSL commands, paths, and results under `docs/generated/`
6. update environment and external-dependency docs so the WSL location is explicit

## Validation Plan

- `conda run -n ffs-standalone python -c "import torch, torchvision, timm, open3d, turbojpeg"`
- `conda run -n qqtt-ffs-compat python -c "import pyrealsense2, torch, open3d"`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n ffs-standalone python scripts/harness/verify_ffs_demo.py --ffs_repo /home/zhangxinjie/Fast-FoundationStereo --model_path /home/zhangxinjie/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth`

## Risks

- the WSL host may not have a USB-attached RealSense device available right now
- Google Drive checkpoint download may fail or throttle
- cloned environments may inherit extra packages from the reference env
- `check_all.py` may expose additional missing optional packages not yet documented

## Completion Checklist

- [x] verify the current WSL host facts
- [x] create `ffs-standalone`
- [x] create `qqtt-ffs-compat`
- [x] install missing QQTT / FFS helper packages
- [x] fetch the selected FFS checkpoints under the external repo
- [x] validate the official FFS demo in WSL
- [x] run repo deterministic checks in `qqtt-ffs-compat`
- [x] update environment / dependency docs

## Progress Log

- 2026-04-21: confirmed WSL2 Ubuntu 24.04.2, conda 25.7.0, and visible RTX 5090 GPU
- 2026-04-21: confirmed the external FFS repo already existed at `/home/zhangxinjie/Fast-FoundationStereo`
- 2026-04-21: created `ffs-standalone` and `qqtt-ffs-compat` by cloning the local `rtx5090` env
- 2026-04-21: installed `timm`, `scikit-image`, `pytest`, `PyTurboJPEG`, and `gdown`; installed `rerun-sdk==0.31.2` in `qqtt-ffs-compat`
- 2026-04-21: downloaded checkpoints `23-36-37`, `20-26-39`, and `20-30-48` into the external FFS repo
- 2026-04-21: fixed `scripts/harness/verify_ffs_demo.py` fallback interpreter resolution for non-Windows conda envs
- 2026-04-21: resolved the cloned `ffs-standalone` WSL `scipy.special` import failure by reinstalling `numpy` / `scipy`
- 2026-04-21: validated the official FFS demo and ran `scripts/harness/check_all.py` successfully in WSL

## Completion Summary

This WSL bootstrap completed successfully for repo-side deterministic validation and external FFS proof-of-life.

- `ffs-standalone`: passed import sanity and official FFS demo validation
- `qqtt-ffs-compat`: passed import sanity and `scripts/harness/check_all.py`
- external checkpoints are present under `/home/zhangxinjie/Fast-FoundationStereo/weights/`
- live RealSense capture remains a manual hardware validation step outside this bootstrap pass
