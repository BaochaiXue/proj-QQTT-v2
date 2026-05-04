# Demo V2 Default FFS Depth

## Goal

Change `demo_v2/realtime_masked_edgetam_pcd.py` so the realtime masked PCD
demo defaults to FFS depth instead of native RealSense depth.

Required runtime contract:

- FFS depth uses the repo default TensorRT two-stage engine:
  `20-30-48 / valid_iters=4 / 848x480 -> 864x480 / builderOptimizationLevel=5`.
- HF EdgeTAM remains compiled-only via `--compile-mode vision-reduce-overhead`.
- Point colors remain live RGB by default.
- The demo still renders only controller/object masked PCD, not full-scene PCD.

## Implementation Notes

- Reuse FFS runtime helpers from `demo_v2/realtime_single_camera_pointcloud.py`.
- In FFS mode, capture color + IR1 + IR2, run the TensorRT runner in the PCD
  worker for the latest mask packet, align FFS metric depth to color, then
  backproject masked pixels only.
- Keep `--depth-source realsense` available as an explicit fallback.

## Validation

- `python -m py_compile demo_v2/realtime_masked_edgetam_pcd.py`
- Focused realtime smoke tests
- `python scripts/harness/check_all.py`
- Short live smoke when a D455 is attached

