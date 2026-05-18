# Demo 3.1 Dual-4090 RealSense CoTracker Overlay

Demo 3.1 is a dual-GPU high-FPS visualization runtime cloned from the Demo 3
RealSense CoTracker overlay lineage.

- GPU0 owns three RealSense capture, SAM3.1/HF EdgeTAM masks, RealSense-depth
  fusion, and Open3D/render.
- GPU1 owns CoTracker3 online in a separate child process.
- CoTracker receives CPU RGB/mask latest-wins packets only.
- CoTracker returns small CPU 2D track/visibility packets.
- The main process lifts tracks to world with the latest RealSense depth,
  intrinsics, and camera-to-world transforms.
- The renderer never waits for CoTracker.
- Demo 3.1 does not use FFS.
- Demo 3.1 inherits Demo 3.0's online-only FuturePhysTwin-compatible tracking
  semantics: `--mode exp|demo`, object/controller union masks, CoTracker3
  online, `phystwin_dense` query sampling, and default query count `auto`
  (`min(union_mask_pixels, 5000)` per camera).
- Raw tracked queries are separate from display overlays. CoTracker may track up
  to 5000 points per camera while the rendered overlay remains capped at 30
  points per camera by default.

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
