# Demo 3.2: Single-Camera FFS Masked PCD

Demo 3.2 is the single-camera FFS-depth masked point-cloud runtime. It uses one
camera or the shared fake-live source, runs Fast-FoundationStereo from the IR
stereo pair, propagates SAM3.1/HF EdgeTAM masks, and renders masked PCD plus
TAPNext++ 3D marker overlay.

Dry-run:

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Demo 3.2 uses repo-root `table_calibrate.pkl` by default. If that file or its
`table_calibrate_metadata.json` sidecar is missing or invalid, the wrapper fails
before live or fake-live execution. Pass `--table-calibrate <path>` only when
using an alternate single-camera table-world calibration.

Fake-live replay defaults to live tracking visualization: filtered RGB PCD plus
PhysTwin-style rainbow query points. The live PCD and query markers are rendered
only from strict same-seq pairs.

In `--mode demo` with the default `human hand` controller prompt, Demo 3.x now
tracks three EdgeTAM identities: `hand_a`, `object`, and `hand_b`. `hand_a` and
`hand_b` are the two frame-0 hand instances sorted by image x coordinate. The
controller PCD is still the union `hand_a | hand_b`, but query labels,
visibility gating, saved masks, and HUD/headless counts keep the two hand
identities separate. Frame 0 must contain two separable hands; otherwise the
demo fails fast instead of silently collapsing the controller to one mask.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --mode demo \
  --replay-fps 5
```

For PCD-only inspection, keep the full FFS + EdgeTAM + TAPNext++ tracking
pipeline running, but hide the query markers in the render. This makes the
displayed FPS reflect the same full pipeline cost as tracking mode. The PCD
inspection view defaults both object and controller rendering filters to
`pt-filter`; tracking mode keeps the stricter `enhanced-pt` defaults for its
overlay path.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode pcd \
  --mode demo \
  --replay-fps 5
```

If the default TensorRT engine is not present in this checkout, pass the local
engine explicitly:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --mode demo \
  --replay-fps 5 \
  --enable-pcd-filter \
  --ffs-trt-model-dir /home/xinjie/proj-QQTT-v2/data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864
```

The default fake-live case
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543` includes
`color/`, `depth/`, `ir_left/`, `ir_right/`, and IR calibration metadata. Demo
3.2 ignores native depth for the FFS path and computes color-aligned depth from
the replayed IR stereo frames, matching the live camera contract. Fake-live runs
in demo mode and defaults to 5 FPS unless `--replay-fps` is explicitly set.
Use `--replay-fps 0` to replay at metadata FPS. Local FFS TensorRT depth
execution is serialized inside the runtime and cached by frame sequence so
point-cloud rendering and TAPNext++ marker lift can share depth without
concurrent TensorRT context use.

Headless capture keeps the fake-live realtime pipeline running but does not
open Open3D. It saves the sync filtered PCD selected by the visual mode
(`pt-filter` for `pcd`, `enhanced-pt` for `tracking`), RGB frames,
color-aligned FFS depth, EdgeTAM `hand_a`/`hand_b`/`object` masks plus legacy
controller/object masks, and TAPNext++ query trajectory artifacts:

Demo 3.2 defaults both object and controller PCD mask erosion to 0px so small
target regions are not eaten before point-cloud generation. Passing
`--pcd-mask-erode-pixels` explicitly still applies the legacy common value to
both classes unless `--object-pcd-mask-erode-pixels` or
`--controller-pcd-mask-erode-pixels` is also provided.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --render-mode none \
  --duration-s 5 \
  --headless-capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke
```

Render the saved artifacts offline. In `pcd` mode, the helper draws only the
saved filtered RGB point cloud. In `tracking` mode, the helper follows the
FuturePhysTwin 2D tracker view: same-frame RGB target regions plus current-frame
query points only, with stable `gist_rainbow` colors assigned from each query
point's initial y coordinate. By default the tracking renderer applies
`object_mask | controller_mask` to the RGB frame and blacks out table/background
pixels before drawing query points. No PCD and no historical trajectory lines
are drawn in the offline tracking video. It only uses exact same-seq query
trajectory files, so missing query frames are counted rather than silently
matched to an older tracker output.

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_query_phystwin.mp4 \
  --fps 30 \
  --demo-visual-mode tracking
```

For comparison with the old full-RGB tracking background, pass:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_query_full_rgb_compare.mp4 \
  --fps 30 \
  --demo-visual-mode tracking \
  --tracking-background-mask rgb
```

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_pcd_only.mp4 \
  --fps 30 \
  --demo-visual-mode pcd
```
