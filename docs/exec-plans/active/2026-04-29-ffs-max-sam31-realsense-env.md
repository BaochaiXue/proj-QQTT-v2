# 2026-04-29 FFS-max SAM 3.1 RealSense Env

Create a cloned conda environment from `FFS-max` that keeps the existing FFS torch/CUDA/TensorRT stack intact while adding QQTT-compatible SAM 3.1 and RealSense support.

## Scope

- Source env: `FFS-max`
- Target env: `FFS-max-sam31-rs`
- Preserve source torch/CUDA stack:
  - `torch==2.11.0`
  - `torchvision==0.26.0`
  - `cuda-toolkit==13.0.2`
  - TensorRT cu13 packages already present in `FFS-max`
- Add runtime support:
  - official `facebookresearch/sam3` code for SAM 3.1 checkpoint compatibility
  - `pyrealsense2` for QQTT RealSense entrypoints
  - Hugging Face checkpoint download or cache resolution for `facebook/sam3.1`

## Non-Goals

- Do not alter the original `FFS-max` environment.
- Do not vendor SAM 3.1 or FFS weights into this repo.
- Do not change camera CLI defaults, aligned output layout, or formal recording/alignment behavior.
- Do not claim hardware validation unless a physical D455 probe is actually run.

## Plan

1. Snapshot the source env's torch/CUDA/TensorRT package versions.
2. Clone `FFS-max` to `FFS-max-sam31-rs`.
3. Install SAM 3.1 code from the official `facebookresearch/sam3` repo without forcing torch/CUDA changes.
4. Install `pyrealsense2` and verify QQTT camera imports.
5. Resolve or download `sam3.1_multiplex.pt` from Hugging Face into an external cache.
6. Run deterministic import/helper checks:
   - torch/CUDA version invariance
   - `pyrealsense2` import
   - `sam3` import and builder availability
   - `tests.test_sam31_mask_helper_smoke`
   - `scripts/harness/check_all.py`
7. Record exact commands and outcomes in `docs/generated/`.
8. Update `docs/envs.md` and `docs/external-deps.md` with the new environment and external checkpoint location if validation succeeds.

## Validation

- `conda run -n FFS-max-sam31-rs python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- `conda run -n FFS-max-sam31-rs python -c "import pyrealsense2 as rs; print(rs.__version__)"`
- `conda run -n FFS-max-sam31-rs python -c "import sam3; print(getattr(sam3, '__version__', 'unknown'))"`
- `conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_sam31_mask_helper_smoke`
- `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py`

## Progress

- [x] Confirmed `FFS-max` exists locally.
- [x] Confirmed `FFS-max` currently has `torch==2.11.0`, `torchvision==0.26.0`, and `cuda-toolkit==13.0.2`.
- [x] Confirmed `FFS-max` does not currently have `pyrealsense2` or `sam3`.
- [x] Clone target env.
- [x] Install add-ons.
- [x] Download or resolve SAM 3.1 checkpoint.
- [x] Validate and document results.

## Outcome

- Created `FFS-max-sam31-rs`.
- Preserved `torch==2.11.0+cu130`, torch CUDA `13.0`, `cuda-toolkit==13.0.2`, and TensorRT cu13 packages.
- Added RealSense / QQTT camera runtime imports with `pyrealsense2==2.56.5.9235` plus `atomics`, `pynput`, and `threadpoolctl`.
- Installed official `facebookresearch/sam3` from commit `c97c893969003d3e6803fd5d679f21e515aef5ce`.
- Downloaded `sam3.1_multiplex.pt` to `/home/xinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`.
- Set `QQTT_SAM31_CHECKPOINT` in the target conda env.
- `Sam3MultiplexVideoPredictor` initialization succeeded.
- `conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py` passed.
- Detailed validation is recorded in `docs/generated/ffs_max_sam31_realsense_env_validation.md`.
