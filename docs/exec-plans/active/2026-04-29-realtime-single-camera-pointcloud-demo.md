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
- Keep camera-view default quality fixed with `--stride 1` and `--max-points 0`; default orbit view to `--max-points 200000` for responsive 3D inspection unless the operator explicitly passes `--max-points 0`.
- Keep the far-depth clip disabled by default with `--depth-max-m 0.0`; positive values opt into far clipping.
- Default to `--view-mode camera`, which uses the D455 color intrinsics for a first-person camera projection; keep `--view-mode orbit` available for third-person inspection.
- Default `--render-backend auto` to the fast `image` backend for camera view and the full `pointcloud` backend for orbit view.
- Keep profiler output opt-in through `--debug` so normal operation is not slowed by console logging.

## Implementation Plan

1. Add a CLI with explicit supported capture profiles:
   - `--fps {5,15,30,60}`, default `30`
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
- Added `--debug` profiler HUD and `1 Hz` console timing for wait, align, copy, backprojection, Open3D conversion/update, and receive-to-render latency.
- Added `--render-backend {auto,image,pointcloud}`. Camera view defaults to the image backend, which preserves aligned valid depth pixels while skipping XYZ/Open3D geometry work; pointcloud remains available explicitly and for orbit view.
- Added `--image-splat-px` for optional image-space splatting in the fast camera-view backend.
- Updated the true 3D pointcloud backend to use Open3D tensor point clouds with `float32` positions/colors, reuse the RGB float conversion buffer, and print a one-time warning if `update_geometry()` falls back to remove/add.
- Expanded `--fps` choices to `{5,15,30,60}` after a live D455 profile probe confirmed `848x480` RGB-D capture with depth-to-color alignment at about `59.8 FPS`.
- Optimized the camera-view image backend mask path by converting metric depth bounds to raw `uint16` thresholds, using OpenCV `inRange`/`cvtColor`/`bitwise_and` when available, and keeping a NumPy fallback with identical valid-pixel semantics.
- On a synthetic `848x480` frame in `FFS-max-sam31-rs`, the mask path median dropped from `8.61 ms` to `0.36 ms` while matching the previous float32 predicate output.
- Updated orbit view defaults so omitted `--max-points` resolves to `200000`; camera view still resolves to uncapped `0`, and explicit `--max-points 0` keeps orbit uncapped.
- Updated the point-cloud renderer to track Open3D geometry capacity and proactively re-add geometry when a later frame exceeds the current capacity, avoiding repeated `point count exceeds the existing point count` warnings after orbit capping ramps up to `200000`.

Validation completed on 2026-04-29:

```bash
conda run -n FFS-max-sam31-rs python scripts/harness/realtime_single_camera_pointcloud.py --help
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke
conda run -n FFS-max-sam31-rs python scripts/harness/check_all.py
```

All commands exited successfully. The full interactive demo was not hardware-validated in this validation pass.

Manual D455 profile probe completed on 2026-04-29 with serial `338122303713`:

```text
active stream=depth format=z16 848x480@60
active stream=color format=bgr8 848x480@60
demo_path=wait+align
framesets=299 wall_s=5.011 wall_fps=59.7
unique_color_frames=299 unique_depth_frames=299
color_ts_fps=59.82 depth_ts_fps=59.82
```
