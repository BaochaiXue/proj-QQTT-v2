# Demo 3.1 Dual-4090 RealSense Point-Tracker Overlay

Demo 3.1 is a dual-GPU high-FPS visualization runtime cloned from the Demo 3
RealSense tracking-overlay lineage.

- GPU0 owns three RealSense capture, SAM3.1/HF EdgeTAM masks, RealSense-depth
  fusion, and Open3D/render.
- GPU1 owns a point-tracker backend in a separate child process.
- Supported backend names are `cotracker3_online`, `trackon2`, `litetracker`,
  and `locotrack`; `cotracker3_online` is the default and currently the
  validated live backend.
- The point tracker receives CPU RGB/mask latest-wins packets only.
- The point tracker returns small CPU 2D track/visibility packets.
- The main process lifts tracks to world with group-aligned cached RealSense depth,
  intrinsics, and camera-to-world transforms using the same projection-grid
  backprojection convention as the semantic PCD fusion path.
- Camera/mask/PCD work stays asynchronous, but rendered results are gated by
  the point tracker: a PCD packet is published only when the matching tracking
  result for that group can be lifted into red tracking points.
- Demo 3.1 does not use FFS.
- Demo 3.1 inherits Demo 3.0's online-only object/controller union tracking
  semantics: `--mode exp|demo`, object/controller union masks,
  and `phystwin_dense` query sampling. The Demo 3.1 dual-4090 batch path
  defaults to `--cotracker-query-count 4096` per camera; live profiling showed
  that full batch=3 at 5000/view exceeds the RTX 4090 24GB memory budget. The
  controller/towel mask is capped first with
  `--controller-pcd-max-points-per-camera 4999`; query points and fused PCD
  then use the requested per-view query budget from the capped
  object/controller union.
- Raw tracked queries are separate from display overlays. The backend may track
  up to 4096 capped-union points per camera by default while first-frame mask labels decide
  what is rendered. The default overlay scope is `controller`; Demo 3.1 renders
  all visible controller-labeled tracks by default with
  `--overlay-max-points-per-camera 0`, shown as high-contrast red tracking
  points. Use `--overlay-debug-color-by-camera` to color those lifted controller
  overlay points by source camera when diagnosing alignment. The lift mask
  follows the display scope, so controller overlays must land inside the current
  controller mask rather than the broader object/controller union.
- When CoTracker is not ready, the Open3D window keeps the last valid rendered
  result instead of publishing a new semantic-only PCD frame. Rendered FPS
  therefore measures track-ready results, not camera/mask-only throughput.
- Rendered object PCD density is controlled by FuturePhysTwin-style 5mm world
  voxel sampling by default. `--object-volume-points-per-voxel` can retain more
  representatives inside each occupied voxel without changing CoTracker query
  counts or overlay caps.
- Rendered PCD color defaults to live RGB via `--pcd-color-mode rgb`. Use
  `--pcd-color-mode class` only when you want semantic object/controller solid
  colors for debugging.
- The shared Open3D window/profile label is overridden to `Demo 3.1`.
- Demo 3.1 forwards Demo 2.3 fusion diagnostics and GPU sampling flags to the
  shared three-view runtime.

Dry-run:

```bash
conda run --no-capture-output -n demo3-max \
  python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py \
  --dry-run \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl
```

Live validation still requires a real root-level `calibrate.pkl` that covers the
active three RealSense serials.

Rendered FPS profiling must use `--render-mode pointcloud`; no-render runs are
upstream isolation only. For finite-duration rendered profiles, the shared
three-view runtime writes its profile before Open3D teardown. If the GUI still
hangs or crashes during teardown on the local workstation, run through
`scripts/harness/run_wslg_open3d.sh` or set `QQTT_WSLG_OPEN3D_FAST_EXIT=1`.

Useful rendered/debug profiling flags:

```bash
--cotracker-backend cotracker3_online \
--tracking-backend-execution-mode batch-views \
--cotracker-update-mode batch \
--tracker-batch-query-count-policy fixed \
--gpu-sampling \
--gpu-sampling-device-indexes 0,1 \
--overlay-display-scope controller \
--overlay-max-points-per-camera 0 \
--overlay-debug-color-by-camera \
--wait-for-tracking-overlay \
--debug-color-by-camera \
--debug-save-per-camera-pcd \
--debug-save-mask-overlays
```

If `--gpu-sampling` is enabled without explicit indexes, Demo 3.1 samples the
configured mask and CoTracker physical GPUs, `0,1` by default.

Track-On2 and LiteTracker are exposed through the same child-process contract,
and LocoTrack-S is exposed as the `locotrack` backend. These external
repos/weights stay outside this repository. Use
`scripts/env/create_demo_3_1_max.sh` to clone the current Demo 3 environment.
For LocoTrack-S, install the live-inference dependency set without replacing
the existing CUDA Torch:

```bash
scripts/env/install_locotrack_s_demo_3_1_max.sh
```

Dry-run LocoTrack-S batch-views:

```bash
conda run --no-capture-output -n demo_3_1_max \
  python demo_v3_1/realtime_three_view_cotracker3_realsense_overlay_dual4090.py \
  --dry-run \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl \
  --cotracker-backend locotrack \
  --tracking-backend-execution-mode batch-views \
  --locotrack-repo-dir external/locotrack/locotrack_pytorch \
  --locotrack-checkpoint checkpoints/locotrack/locotrack_small.ckpt \
  --locotrack-model-size small \
  --locotrack-window-frames 8 \
  --locotrack-query-chunk-size 256
```

LocoTrack is treated as a rolling-window backend, not frame-by-frame online
tracking. The contract reports `tracking_backend_online_semantics=windowed`;
serial mode creates one adapter per camera, while batch-views creates one
adapter/model and calls inference once over `[3,T,H,W,3]`. Rendered profile
targets are available through:

```bash
python scripts/harness/run_demo31_locotrack_s_profiles.py --print-commands
python scripts/harness/summarize_demo31_locotrack_s_profiles.py
```

The adapter layer fails early with a clear message if the required external
repo/checkpoint is not configured.
