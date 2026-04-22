# 2026-04-21 Cameras Viewer FFS TensorRT

## Goal

Add an optional TensorRT-backed Fast-FoundationStereo live path to `cameras_viewer_FFS.py` so the existing RealSense RGB + FFS preview can switch between the current PyTorch runner and a Windows TensorRT runner.

## Non-Goals

- no change to the default PyTorch viewer path
- no QQTT aligned-case depth-backend TRT integration in this pass
- no single-ONNX / single-engine upstream upgrade
- no change to canonical aligned `depth/` outputs

## Files To Touch

- `data_process/depth_backends/fast_foundation_stereo.py`
- `data_process/depth_backends/__init__.py`
- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- viewer validation docs

## Implementation Plan

1. add a TensorRT runner alongside the existing PyTorch runner in `data_process/depth_backends/`
2. preserve the existing `run_pair(...)` contract so viewer-side reprojection and overlays remain unchanged
3. add viewer CLI flags to select `pytorch` vs `tensorrt` and pass TensorRT model/runtime paths only when needed
4. keep TensorRT fixed-shape behavior explicit:
   - load engine metadata from `onnx.yaml`
   - resize live IR frames to the engine input size before inference
   - keep returning `K_ir_left_used` so downstream geometry uses the actual inference intrinsics
5. add deterministic tests for backend argument resolution and TensorRT engine metadata helpers
6. run software checks plus at least one manual live-camera TensorRT smoke

## Validation Plan

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
- manual live smoke with `cameras_viewer_FFS.py --ffs_backend tensorrt ...`
