# Demo 3.1 Dual-4090 RealSense CoTracker Overlay

Demo 3.1 is a dual-GPU high-FPS visualization runtime cloned from the Demo 3
RealSense CoTracker overlay lineage.

- GPU0 owns three RealSense capture, SAM3.1/HF EdgeTAM masks, RealSense-depth
  fusion, and Open3D/render.
- GPU1 owns CoTracker3 online in a separate child process.
- CoTracker receives CPU RGB/mask latest-wins packets only.
- CoTracker returns small CPU 2D track/visibility packets.
- The main process lifts tracks to world with group-aligned cached RealSense depth,
  intrinsics, and camera-to-world transforms.
- Camera/mask/PCD work stays asynchronous, but rendered results are gated by
  CoTracker: a PCD packet is published only when the matching CoTracker result
  for that group can be lifted into red tracking points.
- Demo 3.1 does not use FFS.
- Demo 3.1 inherits Demo 3.0's online-only FuturePhysTwin-compatible tracking
  semantics: `--mode exp|demo`, object/controller union masks, CoTracker3
  online, `phystwin_dense` query sampling, and default query count `auto`
  (`min(union_mask_pixels, 5000)` per camera).
- Raw tracked queries are separate from display overlays. CoTracker may track up
  to 5000 union points per camera while first-frame mask labels decide what is
  rendered. The default overlay scope is `controller`; Demo 3.1 renders all
  visible controller-labeled tracks by default with
  `--overlay-max-points-per-camera 0`, shown as high-contrast red tracking
  points.
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
--gpu-sampling \
--gpu-sampling-device-indexes 0,1 \
--overlay-display-scope controller \
--overlay-max-points-per-camera 0 \
--wait-for-tracking-overlay \
--debug-color-by-camera \
--debug-save-per-camera-pcd \
--debug-save-mask-overlays
```

If `--gpu-sampling` is enabled without explicit indexes, Demo 3.1 samples the
configured mask and CoTracker physical GPUs, `0,1` by default.
