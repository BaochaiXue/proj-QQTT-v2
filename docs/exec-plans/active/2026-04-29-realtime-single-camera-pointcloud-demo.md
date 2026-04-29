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
   - `--fps {5,15,30,60}`, default `60`
   - `--profile {848x480,640x480}`, default `848x480`
   - serial, emitter, clipping, stride, point-size, backproject-backend, latency-target, and duration controls.
2. Implement a three-stage async pipeline:
   - capture thread waits on RealSense, aligns depth to color, and publishes the newest RGB-D frame.
   - point-cloud worker consumes only the newest RGB-D frame and drops stale work.
   - Open3D GUI thread consumes only the newest point cloud and renders it.
3. Use single-slot latest-wins buffers between stages and expose capture overwrite drops.
4. Optimize postprocessing without automatic quality reduction:
   - precompute pixel grids once per profile/stride.
   - vectorized NumPy `float32` backprojection.
   - optional Numba fused rank-sampling backprojection for the stride-1 projection-grid point-cloud path.
   - one combined depth validity mask.
   - keep `uint8` color until the Open3D upload boundary.
5. Add deterministic smoke tests for CLI help, profile parsing, synthetic backprojection, latest-slot drops, and FPS/latency stats.
6. Add the CLI to `scripts/harness/check_all.py` help checks and document the run command in `scripts/harness/README.md` and `docs/WORKFLOWS.md`.

## Validation

Run:

```bash
conda run -n FFS-max-sam31-rs python scripts/harness/realtime_single_camera_pointcloud.py --help
conda run -n FFS-max-sam31-rs python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke tests.test_cameras_viewer_ffs_smoke
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
- Updated the default RGB-D capture rate to `60 FPS`; operators can still pass `--fps 30` for the older lower-rate capture path.
- Optimized the camera-view image backend mask path by converting metric depth bounds to raw `uint16` thresholds, using OpenCV `inRange`/`cvtColor`/`bitwise_and` when available, and keeping a NumPy fallback with identical valid-pixel semantics.
- On a synthetic `848x480` frame in `FFS-max-sam31-rs`, the mask path median dropped from `8.61 ms` to `0.36 ms` while matching the previous float32 predicate output.
- Updated orbit view defaults so omitted `--max-points` resolves to `200000`; camera view still resolves to uncapped `0`, and explicit `--max-points 0` keeps orbit uncapped.
- Updated orbit point rendering defaults so omitted `--point-size` resolves to `1.0` in orbit view to reduce point rasterization load at `200000` points, while explicit `--point-size` values still override the default.
- Updated the point-cloud renderer to track Open3D geometry capacity and proactively re-add geometry when a later frame exceeds the current capacity, avoiding repeated `point count exceeds the existing point count` warnings after orbit capping ramps up to `200000`.
- Optimized point-cloud backprojection so `--max-points` sampling happens before XYZ/RGB materialization, preserving the existing `linspace` valid-pixel sampling order while avoiding full valid-point allocation when orbit view is capped.
- Installed `numba==0.65.1` and `llvmlite==0.47.0` into `FFS-max-sam31-rs` with pip after saving pre-install conda/pip snapshots under `docs/generated/`.
- Added `--backproject-backend {auto,numpy,numba}`. `auto` uses Numba when available for the stride-1 projection-grid point-cloud path, otherwise NumPy; the HUD and debug log show the effective backend.
- Added `--depth-source {realsense,ffs}`. The FFS path captures live `color + infrared(1) + infrared(2)`, runs the repo-local two-stage TensorRT FFS artifact, aligns `depth_ir_left_m` into `camera_color_frame`, and renders it through the existing camera/orbit backends.
- Added `--ffs-repo`, `--ffs-trt-model-dir`, and `--ffs-trt-root`; explicit CLI overrides are preserved, while defaults use the central repo-relative FFS defaults.
- Split realtime processing into three latest-wins stage boundaries: capture slot, depth/FFS slot, and render-prep slot. The native RealSense path remains `color + depth`, while FFS mode uses a separate TensorRT worker and does not require the native depth stream.
- Split the FFS worker further so it publishes raw IR-left metric depth immediately after TensorRT. The render-prep worker now owns `IR-left depth -> color depth` alignment plus image mask / point-cloud packet generation, allowing FFS inference and color alignment to overlap in the live pipeline.
- Added vectorized `align_ir_depth_to_color_fast()` using nearest-depth z-buffering via `np.minimum.at`, plus float-depth camera-image and point-cloud backprojection helpers so FFS depth stays metric `float32` until rendering.
- Optimized the float-depth camera image backend with an OpenCV `float32` `inRange`/`cvtColor`/`bitwise_and` path for the non-splat case, keeping NumPy fallback semantics for invalid, NaN, Inf, and zero-depth pixels.
- Precomputed and cached the IR-left to color-frame projection coefficients for live FFS depth packets, so color alignment reuses fixed ray/transform terms and the z-buffer output allocation instead of rebuilding them every frame.
- Added a single-thread Numba fused IR-left to color-frame z-buffer align kernel for the cached FFS aligner, with the existing NumPy `np.minimum.at` path retained as fallback. The Numba path intentionally stays non-parallel to avoid races when multiple IR pixels project to the same color pixel.
- Fixed the steady-state two-stage TensorRT runner allocations for live batch-1 FFS: prepared IR images now reuse pinned host staging buffers and fixed CUDA input tensors, TensorRT feature/post outputs and stable tensor addresses are cached, and disparity D2H download reuses a pinned host buffer. The live two-stage path now keeps H2D, `execute_async_v3()`, post kernels, D2H, and the single required publish-time stream sync on the same inference stream. This is the low-risk runner-internal step toward full CUDA-event ring buffering; it keeps capture/render pipeline behavior and model outputs unchanged.
- Expanded the HUD and debug line with `depth_source`, FFS TensorRT timing, FFS color-align timing, depth-stage drops, and render-slot drops.
- Split realtime drop statistics into total, after-warmup, and last-debug-window deltas. The viewer resets steady-state drop counters after a short startup warmup while preserving total counters, so live logs distinguish launch/warmup drops from ongoing pipeline drops.
- Changed Open3D GUI rendering to a coalesced `post_to_main_thread` request when render-prep publishes a packet. The UI still pulls only the latest packet and avoids callback burst buildup, but no longer depends on a fixed timer or `set_on_tick_event` cadence.
- Marked fixed-60Hz UI schedulers as the wrong design for this Open3D orbit path. Manual validation showed that a timer-driven pull can be throttled by GUI/SceneWidget draw cadence and produce steady render-slot drops, even when render-prep and Open3D CPU update timings are small. The correct design is packet-arrival coalesced main-thread rendering.

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
