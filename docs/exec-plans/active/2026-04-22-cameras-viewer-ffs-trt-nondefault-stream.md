# 2026-04-22 Cameras Viewer FFS TRT Non-Default Stream

## Goal

Remove the TensorRT `enqueueV3()` default-stream warning from the live FFS viewer path and move TRT execution onto an explicit non-default CUDA stream so the runtime does not pay unnecessary synchronization overhead.

## Non-Goals

- no change to the viewer's TensorRT engine format
- no change to the PyTorch FFS path
- no change to aligned-case depth-backend entrypoints
- no upstream external repo edits

## Files To Touch

- `data_process/depth_backends/fast_foundation_stereo.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- `scripts/harness/verify_ffs_tensorrt_wsl.py`
- TRT validation docs

## Implementation Plan

1. add a small helper that runs TensorRT model forward calls on an explicit non-default CUDA stream and synchronizes correctly with the caller stream
2. use that helper inside `FastFoundationStereoTensorRTRunner` so live viewer TRT inference no longer executes on the default stream
3. reuse the same non-default-stream strategy in the WSL TensorRT proof-of-life harness for demo/profile runs
4. add deterministic tests for the stream helper call order using a fake `torch.cuda` implementation
5. rerun WSL headless TRT validation plus live viewer smoke to confirm the warning disappears and the path still works

## Validation Plan

- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
- `conda run -n ffs-standalone python scripts/harness/verify_ffs_tensorrt_wsl.py`
- `conda run -n qqtt-ffs-compat python cameras_viewer_FFS.py --max-cams 1 --width 848 --height 480 --duration-s 10 --stats-log-interval-s 5 --ffs_repo /home/zhangxinjie/Fast-FoundationStereo`
