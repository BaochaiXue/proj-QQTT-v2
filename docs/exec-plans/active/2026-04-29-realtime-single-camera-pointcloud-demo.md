# Realtime Single-D455 Camera-Frame Point Cloud Demo

## Goal

Add an operator-facing realtime single-camera D455 demo that streams aligned `color + depth`, backprojects into `camera_color_frame`, and renders the live point cloud in Open3D with render FPS and receive-to-render latency visible in the window.

## Scope

- Add `scripts/harness/realtime_single_camera_pointcloud.py`.
- Keep the point cloud in the RealSense color camera coordinate frame:
  - frame: `camera_color_frame`
  - units: meters
  - axes: `x` right, `y` down, `z` forward
- Do not load `calibrate.pkl` or apply multi-camera/world transforms.
- Keep default quality fixed with `--stride 1` and `--max-points 0`.
- Keep the far-depth clip disabled by default with `--depth-max-m 0.0`; positive values opt into far clipping.
- Default to `--view-mode camera`, which uses the D455 color intrinsics for a first-person camera projection; keep `--view-mode orbit` available for third-person inspection.

## Implementation Plan

1. Add a CLI with explicit supported capture profiles:
   - `--fps {5,15,30}`, default `30`
   - `--profile {848x480,640x480}`, default `848x480`
   - serial, emitter, clipping, stride, point-size, latency-target, and duration controls.
2. Implement a three-stage async pipeline:
   - capture thread waits on RealSense, aligns depth to color, and publishes the newest RGB-D frame.
   - point-cloud worker consumes only the newest RGB-D frame and drops stale work.
   - Open3D GUI thread consumes only the newest point cloud and renders it.
3. Use single-slot latest-wins buffers between stages and expose capture overwrite drops.
4. Optimize postprocessing without automatic quality reduction:
   - precompute pixel grids once per profile/stride.
   - vectorized NumPy `float32` backprojection.
   - one combined depth validity mask.
   - keep `uint8` color until the Open3D upload boundary.
5. Add deterministic smoke tests for CLI help, profile parsing, synthetic backprojection, latest-slot drops, and FPS/latency stats.
6. Add the CLI to `scripts/harness/check_all.py` help checks and document the run command in `scripts/harness/README.md` and `docs/WORKFLOWS.md`.

## Validation

Run:

```bash
conda run -n FFS-max-sam31-rs python scripts/harness/realtime_single_camera_pointcloud.py --help
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py
```

Manual D455 validation remains separate; do not claim hardware capture results unless the demo is actually run with a camera attached.

## Result

- Implemented `scripts/harness/realtime_single_camera_pointcloud.py`.
- Added deterministic smoke coverage in `tests/test_realtime_single_camera_pointcloud_smoke.py`.
- Added the CLI to quick/full `scripts/harness/check_all.py` help checks.
- Documented the run command in `scripts/harness/README.md` and `docs/WORKFLOWS.md`.
- Updated the depth range default so `--depth-max-m 0.0` disables far clipping unless the operator opts in.
- Added `--view-mode {camera,orbit}` and made `camera` the default.

Validation completed on 2026-04-29:

```bash
conda run -n FFS-max-sam31-rs python scripts/harness/realtime_single_camera_pointcloud.py --help
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py
```

All commands exited successfully. No live D455 run was performed in this validation pass.
