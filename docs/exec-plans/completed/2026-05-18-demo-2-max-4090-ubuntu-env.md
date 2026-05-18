# demo_2_max RTX 4090 Native Ubuntu Environment

Status: completed with external-asset blockers recorded.

## Goal

Install and validate the `demo_2_max` conda environment on the native Ubuntu RTX
4090 desktop for QQTT Demo 2 / Demo 2.1 local runtime work.

## Scope

- Target native Ubuntu RTX 4090, not WSL RTX 5090 laptop.
- Use environment name `demo_2_max`.
- Keep external repos, checkpoints, and TensorRT engines outside vendored QQTT
  source paths:
  - `$HOME/Fast-FoundationStereo`
  - `$HOME/EdgeTAM`
  - `$HOME/.cache/huggingface/qqtt_sam31/sam3.1_multiplex.pt`
- Use `TORCH_CUDA_ARCH_LIST=8.9` for RTX 4090 native CUDA extension builds.
- Do not apply WSL `usbipd`, WSLg, or WSL udev helper rules.
- Do not reuse RTX 5090 TensorRT engine artifacts as formal RTX 4090 results.

## Steps

1. Confirm machine prerequisites: native Ubuntu, RTX 4090, R580+ driver, CUDA
   reported by `nvidia-smi`, and existing CUDA/TensorRT/conda tools.
2. Ensure QQTT repo is up to date with `git pull --ff-only origin main`.
3. Install or expose conda, create `demo_2_max`, and install PyTorch CUDA 13.
4. Install native Ubuntu system dependencies and Python RealSense/Open3D/demo
   runtime packages.
5. Clone or update external Fast-FoundationStereo and EdgeTAM repos.
6. Install TensorRT Python bindings and provide or validate `trtexec`.
7. Install EdgeTAM with CUDA extension for `sm_89`.
8. Install SAM3.1 code/checkpoint path when credentials and network allow it.
9. Build or validate a 4090-local FFS TensorRT engine when required inputs and
   builder scripts are present.
10. Run deterministic repo checks and demo smoke tests.
11. Write generated Markdown and JSON validation reports under `docs/generated/`.

## Validation

- `check_harness_catalog.py`
- `check_all.py`
- `demo_v2/realtime_masked_edgetam_pcd.py --help`
- `demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run --preset official-lowfps`
- RealSense device probe
- TensorRT / EdgeTAM / SAM3.1 / FFS path and version probes

## Outcome

- `demo_2_max` was created on native Ubuntu RTX 4090 with PyTorch CUDA 13.0,
  TensorRT 10.16.1.11, RealSense, Open3D, Fast-FoundationStereo, EdgeTAM, and
  SAM3 code installed.
- EdgeTAM CUDA extension was built for `TORCH_CUDA_ARCH_LIST=8.9`.
- FFS TensorRT batch=1 and isolated batch=3 engines were built locally under
  `data/experiments/ffs_trt_4090_848x480_pad864_builderopt5*`.
- Repo deterministic checks passed after updating default FFS TensorRT paths to
  the 4090-native artifacts and making one unit test use fixture calibration
  instead of depending on a real repo-root `calibrate.pkl`.
- Remaining operator-provided assets are the SAM3.1 gated checkpoint and real
  repo-root `calibrate.pkl`; the formal three-camera live smoke fail-fast is
  recorded in `docs/generated/demo_2_max_4090_ubuntu_env_validation.md`.
