# 2026-04-22 Cameras Viewer FFS Default TRT

## Goal

Make `cameras_viewer_FFS.py` default to the repo's TensorRT-backed FFS viewer path instead of the older PyTorch path.

## Non-Goals

- no new ONNX Runtime backend
- no change to aligned-case depth-backend entrypoints
- no new engine build automation in the main viewer path
- no change to the existing explicit `--ffs_backend pytorch` fallback behavior

## Files To Touch

- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- `docs/WORKFLOWS.md`
- `docs/HARDWARE_VALIDATION.md`
- current TRT validation docs

## Implementation Plan

1. change the viewer CLI default backend from `pytorch` to `tensorrt`
2. default `--ffs_trt_model_dir` to the repo-local WSL proof-of-life engine directory under `data/ffs_proof_of_life/trt_two_stage_864x480_wsl`
3. keep `--ffs_repo` explicit so the external FFS repo dependency remains caller-visible
4. keep explicit `--ffs_backend pytorch --ffs_model_path ...` as the fallback path for PyTorch-only runs
5. add deterministic tests for the new CLI defaults
6. update user-facing workflow and hardware-validation docs so commands that still intend PyTorch say so explicitly

## Validation Plan

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
- manual live smoke with default TensorRT selection:
  - `python cameras_viewer_FFS.py --max-cams 1 --width 848 --height 480 --duration-s 10 --stats-log-interval-s 5 --ffs_repo /home/zhangxinjie/Fast-FoundationStereo`
