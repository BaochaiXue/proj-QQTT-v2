# 2026-04-21 Cameras Viewer FFS TensorRT

## Goal

Keep the optional TensorRT-backed Fast-FoundationStereo live path in `cameras_viewer_FFS.py`, but change the non-matching viewer default from `640x480 + resize` to `848x480` capture with `864x480` TRT engines and pad/unpad-aware preprocessing.

## Non-Goals

- no change to the default PyTorch viewer path
- no QQTT aligned-case depth-backend TRT integration in this pass
- no single-ONNX / single-engine upstream upgrade
- no change to canonical aligned `depth/` outputs
- no attempt to make the upstream two-stage ONNX export accept native `848x480`

## Files To Touch

- `data_process/depth_backends/fast_foundation_stereo.py`
- `data_process/depth_backends/__init__.py`
- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- viewer validation docs
- hardware validation checklist

## Implementation Plan

1. add a TensorRT runner alongside the existing PyTorch runner in `data_process/depth_backends/`
2. preserve the existing `run_pair(...)` contract so viewer-side reprojection and overlays remain unchanged
3. add viewer CLI flags to select `pytorch` vs `tensorrt` and pass TensorRT model/runtime paths only when needed
4. keep TensorRT fixed-shape behavior explicit:
   - load engine metadata from `onnx.yaml`
   - when capture is `848x480` and engine size is `864x480`, symmetrically replicate-pad left/right by `8 px` before inference
   - crop the disparity output back to `848x480` before depth reprojection
   - keep `K_ir_left_used` in the original capture coordinates for the pad/unpad path
   - preserve the existing resize fallback for other engine/capture mismatches
5. add deterministic tests for backend argument resolution and TensorRT engine metadata helpers
6. update validation docs so the intended follow-up engine build target is `864x480`, not `640x480`
7. run software checks plus at least one manual live-camera TensorRT smoke

## Validation Plan

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
- manual live smoke with `cameras_viewer_FFS.py --width 848 --height 480 --ffs_backend tensorrt ...`
