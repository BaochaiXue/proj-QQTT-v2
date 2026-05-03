# demo_2_max Four-In-One Environment

## Goal

Create and validate a single `demo_2_max` conda environment for local demo work
that can import and run the current QQTT FFS, RealSense, SAM 3.1, and EdgeTAM
stacks.

## Scope

- Clone an already validated CUDA 13 / torch 2.11 local environment instead of
  reinstalling CUDA or PyTorch from scratch.
- Keep external repos and weights external:
  - `/home/zhangxinjie/Fast-FoundationStereo`
  - `/home/zhangxinjie/EdgeTAM`
  - `/home/zhangxinjie/sam3`
  - `/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Add conda activation hooks for EdgeTAM `PYTHONPATH`, shared CUDA 13 toolkit,
  PyTorch shared libraries, RTX 5090 `sm_120`, and the SAM 3.1 checkpoint.
- Do not modify `edgetam-max`, `FFS-max`, `FFS-SAM-RS`, `SAM21-max`, or camera
  runtime defaults.

## Base Environment Choice

Use `FFS-SAM-RS` as the clone source because it exists locally and already
contains the FFS, RealSense, TensorRT, Open3D, and SAM 3.1 runtime packages. The
documented `FFS-max-sam31-rs` target is not currently present in the local conda
environment list.

## Validation

- Confirm `demo_2_max` is created and GPU-enabled.
- Confirm imports for:
  - `torch`, `torchvision`, `cv2`
  - `pyrealsense2`
  - `sam3`
  - `tensorrt`, `triton`, `open3d`
  - `sam2._C` from `/home/zhangxinjie/EdgeTAM`
- Run the EdgeTAM local verifier from `/home/zhangxinjie/EdgeTAM`.
- Run a SAM 3.1 helper smoke test from this repo.
- Run deterministic harness checks after docs are updated.
- Record exact commands and outcomes in `docs/generated/demo_2_max_env_validation.md`.

## Outcome

- Created `demo_2_max` by cloning `FFS-SAM-RS`.
- Added conda activation/deactivation hooks under
  `/home/zhangxinjie/miniconda3/envs/demo_2_max/etc/conda/`.
- Confirmed a single Python process can import `torch`, `torchvision`, `cv2`,
  `pyrealsense2`, `sam3`, `tensorrt`, `triton`, `open3d`, `sam2._C`, and
  `CameraSystem`.
- Confirmed the local SAM 3.1 checkpoint is
  `/home/zhangxinjie/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`.
- `verify_edgetam_max.py` passed in `demo_2_max`.
- `tests.test_sam31_mask_helper_smoke` passed in `demo_2_max`.
- SAM 3.1 `Sam3MultiplexVideoPredictor` construction passed in `demo_2_max`.
- `cameras_viewer_FFS.py --help` and `verify_ffs_tensorrt_wsl.py --help`
  passed in `demo_2_max`.
- `scripts/harness/check_all.py` quick profile passed in `demo_2_max`.
- `python -m pip check` still reports the inherited `sam3` / `numpy==2.4.4`
  metadata conflict; runtime validation succeeded, so the numpy stack was left
  unchanged.
