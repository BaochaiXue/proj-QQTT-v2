# Demo 3.2: Single-Camera Dual-Depth Masked PCD

Demo 3.2 is the single-camera masked point-cloud runtime with a per-run depth
backend. The default `--depth-backend ir-ffs` path runs Fast-FoundationStereo
from the D455 IR stereo pair. The optional `--depth-backend native-realsense`
path uses D455 native depth aligned to color. Both paths feed the same
color-aligned float depth contract into EdgeTAM, TAPNext++, PCD filtering,
table-world transforms, headless capture, panel rendering, and PhysTwin-like
strict product generation.

Dry-run:

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Choose the depth backend for each run:

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --depth-backend ir-ffs
```

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --depth-backend native-realsense
```

Demo 3.2 uses repo-root `table_calibrate.pkl` by default. If that file or its
`table_calibrate_metadata.json` sidecar is missing or invalid, the wrapper fails
before live or fake-live execution. Pass `--table-calibrate <path>` only when
using an alternate single-camera table-world calibration. With table calibration
enabled, runtime PCD and lifted TAPNext++ marker output are in `table_world_z0`;
the tabletop is reported as `table_z_m = 0.0`. The current table-world
calibration treats points above the tabletop as negative Z
(`table_z_above_direction = negative`), so table-Z filtering uses signed
clearance from the table instead of assuming positive Z is up.

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
displayed FPS reflect the same full pipeline cost as tracking mode. Both PCD
inspection and tracking views default object and controller rendering filters
to `none`, with table-Z deletion still enabled by default. Use
`--pcd-filter-preset {original,pt,enhanced-pt}` to override the visual-mode
default for both object and controller; the default is `original`. The same
preset drives rendered/saved PCD and TAPNext++ initialization: query points are
sampled from the preset's residual PCD pixels, not the raw object/controller
union mask.
Displayed tracking query markers are also strict current-frame residual markers:
if a tracked query drifts outside its class residual mask, or the residual point
is removed by the active table-Z filter, that marker is hidden instead of being
lifted from raw target-mask depth. By default this is only a per-frame display
gate, so a marker can reappear if TAPNext++ later tracks it back into the
filtered residual. Add `--tracker-retire-filtered-markers` only when testing the
stricter once-false retirement policy; with that opt-in, permanent retirement
starts after the initialization marker frame. The realtime and offline
side-by-side panels show the remaining live query count in the top-left legend.

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

The default Demo 3.2 fake-live case is
`data_collect/sloth_both_eval_3min_e70_g60_20260621_202627`. It is a 3-minute
`both_eval` recording with `color/`, `depth/`, `ir_left/`, `ir_right/`,
`calibrate.pkl`, `calibrate_metadata.json`, and IR calibration metadata. Demo
3.2 ignores native depth for `--depth-backend ir-ffs` and computes
color-aligned depth from the replayed IR stereo frames, matching the live camera
contract. `--depth-backend native-realsense` reads the recording's native depth
stream and uses librealsense-style color alignment. Table-world output still
uses repo-root `table_calibrate.pkl` unless `--table-calibrate` is passed
explicitly. Fake-live runs in demo mode and defaults to 5 FPS unless
`--replay-fps` is explicitly set. Use `--replay-fps 0` to replay at metadata
FPS.
Local FFS TensorRT depth execution is serialized inside the runtime and cached
by frame sequence so point-cloud rendering and TAPNext++ marker lift can share
depth without concurrent TensorRT context use.

World-Z diagnostics are always reported for table-calibrated PCD. After the current
PCD preset output is transformed into table world, the runtime records
object/controller Z quantiles plus hand_a/hand_b stats when those masks are
available, with table-band candidate counts at 5, 10, 20, and 30 mm. Demo 3.2
`--demo-visual-mode pcd|tracking`, including headless captures, enables runtime
table-Z deletion by default at 0 mm signed clearance; use
`--disable-table-z-filter` for unfiltered ablations, or pass a larger threshold
explicitly:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --mode demo \
  --table-z-filter-threshold-m 0.01 \
  --table-z-filter-classes both
```

Headless capture keeps the fake-live realtime pipeline running but does not
open Open3D. It saves the sync PCD selected by `--pcd-filter-preset`
(`none` by default, or the explicit `pt`/`enhanced-pt` preset), RGB frames,
color-aligned depth in `depth_color_m/`, EdgeTAM `hand_a`/`hand_b`/`object`
masks plus legacy controller/object masks, TAPNext++ query trajectory
artifacts, capture metadata with `depth_backend`, `depth_source_internal`, and
`camera_to_world_c2w`, and per-frame `world_z_stats.jsonl` diagnostics. Older
headless captures that used `ffs_depth_path` remain readable by the strict
finalizer and offline helpers.
By default these saved PCD artifacts have the 0 mm table-Z filter applied; add
`--disable-table-z-filter` to capture the no-table-Z ablation:

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

### PhysTwin-like strict product

Demo 3.2 can also generate a finite-window PhysTwin-compatible product from the
current single-camera stack:

- tracker backend remains TAPNext++;
- mask backend remains EdgeTAM;
- depth backend is the selected `ir-ffs` or `native-realsense` backend;
- strictness applies to the PhysTwin querying, postprocessing, data contract,
  sampling, and visualization semantics.

This mode is headless/offline in P0. It does not run CoTracker and it does not
replace the live side-by-side TAPNext++ overlay.

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --demo-visual-mode tracking \
  --render-mode none \
  --duration-s 5 \
  --headless-capture-dir result/single_demo_v3_2_ffs_masked_pcd/phystwin_strict_smoke \
  --tracking-product-backend phystwin-strict-tracking
```

The strict output defaults to
`<headless-capture-dir>/phystwin_like/` and includes:

```text
manifest.json
mask/processed_masks.pkl
tracking/0.npz
cotracker/0.npz
pcd/<frame_idx>.npz
track_process_data.pkl
final_data.pkl
tracking_2d.mp4
track_process_data.mp4
final_data.mp4
final_pcd.mp4
```

The `cotracker/0.npz` path is compatibility naming only; the manifest records
`tracker_backend=tapnextpp` and `not_actual_cotracker=true`. Query initialization
uses the first-frame raw `object | controller` EdgeTAM mask union and exports
`queries_txy=[0,x,y]`, while the internal/runtime tracker coordinates remain
`tracks_yx`.

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

### Side-by-side panel

Demo 3.2 can render a 1x3 panel for fake-live review:

1. original latest RGB input
2. filtered PCD projected into the camera view
3. tracking overlay with current-frame query markers

The left RGB column follows the latest fake-live input frame and may lead the
processed output, including during startup warmup before the first strict pair is
available. The PCD and tracking columns always use the same strict same-seq
paired frame. The HUD reports `rgb_seq`, `paired_seq`, `rgb_ahead`, source input
time, pipeline latency, display latency, startup hold, filter preset, marker
count, and remaining live tracking query counts.

Offline from a headless capture:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_side_by_side.mp4 \
  --fps 30 \
  --panel-mode side-by-side \
  --tracking-background-mask target-union
```

Realtime fake-live panel:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --demo-visual-mode tracking \
  --render-mode panel \
  --panel-layout side-by-side \
  --tracking-background-mask target-union \
  --panel-video-output result/single_demo_v3_2_ffs_masked_pcd/realtime_panel.mp4
```

For a table-Z filter experiment without rerunning the demo, render RGB
before/after/removed overlays from one headless capture. Removed projected PCD
points are red; the helper also writes
`table_z_filter_overlay_summary.json` with removed counts and ratios per frame
and threshold. The helper reads `table_z_above_direction` from capture metadata
and defaults to `negative` for older captures; pass `--table-z-above-direction`
only when inspecting a capture made with a different table-world convention:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_unused.mp4 \
  --table-z-overlay-sweep \
  --table-z-overlay-output-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/table_z_overlay
```
