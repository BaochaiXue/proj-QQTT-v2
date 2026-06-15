# Demo 3.2: Single-Camera FFS Masked PCD

Demo 3.2 is the single-camera FFS-depth masked point-cloud runtime. It uses one
camera or the shared fake-live source, runs Fast-FoundationStereo from the IR
stereo pair, propagates SAM3.1/HF EdgeTAM masks, and renders masked PCD plus
TAPNext++ 3D marker overlay.

Dry-run:

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Fake-live replay defaults to tracking visualization: filtered RGB PCD plus
PhysTwin-style rainbow query points. The PCD and query markers are rendered only
from strict same-seq pairs.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --mode demo \
  --replay-fps 30
```

For PCD-only inspection, keep EdgeTAM masks and enhanced-pt filtered RGB PCD but
disable TAPNext++ query markers:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode pcd \
  --mode demo \
  --replay-fps 30
```

If the default TensorRT engine is not present in this checkout, pass the local
engine explicitly:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --mode demo \
  --replay-fps 30 \
  --enable-pcd-filter \
  --ffs-trt-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864
```

The default fake-live case
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543` includes
`color/`, `depth/`, `ir_left/`, `ir_right/`, and IR calibration metadata. Demo
3.2 ignores native depth for the FFS path and computes color-aligned depth from
the replayed IR stereo frames, matching the live camera contract. Fake-live runs
in demo mode. Local FFS TensorRT depth execution is serialized inside the runtime
and cached by frame sequence so point-cloud rendering and TAPNext++ marker lift
can share depth without concurrent TensorRT context use.

Headless enhanced-pt capture keeps the fake-live realtime pipeline running but
does not open Open3D. It saves only sync enhanced-pt filtered PCD, color-aligned
FFS depth, EdgeTAM controller/object masks, and TAPNext++ query trajectory
artifacts:

Demo 3.2 keeps the FFS object mask cleanup at 3px erosion by default, but the
controller PCD mask defaults to 0px erosion so small hand/controller regions are
not eaten before point-cloud generation. Passing `--pcd-mask-erode-pixels`
explicitly still applies the legacy common value to both classes unless
`--object-pcd-mask-erode-pixels` or `--controller-pcd-mask-erode-pixels` is also
provided.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --render-mode none \
  --duration-s 5 \
  --headless-capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke
```

Render the saved artifacts offline. In `tracking` mode, the helper overlays
current-frame query points only, using the saved PhysTwin-style rainbow identity
colors; no historical trajectory lines are drawn. It only uses exact same-seq
query trajectory files, so missing query frames are counted rather than silently
matched to an older tracker output. In `pcd` mode, no query points are drawn.

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video.mp4 \
  --fps 30 \
  --demo-visual-mode tracking
```
