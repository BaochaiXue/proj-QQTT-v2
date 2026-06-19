# Workflows

## 1. Preview

```bash
python cameras_viewer.py
```

Use this to verify that the active D455 enumerates and streams correctly before
calibration or recording. This `single-camera` branch defaults to one camera;
pass `--max-cams` or `--serials` only for explicit multi-camera validation.

Single-D455 realtime point-cloud demo:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 60 \
  --view-mode camera
```

FFS depth mode for the same demo:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py \
  --profile 848x480 \
  --fps 30 \
  --depth-source ffs \
  --view-mode camera \
  --debug
```

Versioned single-camera masked PCD demos:

```bash
python demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
python demo_v3_3/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Replay the default fake-live camera case through Demo 3.1 as the camera stream:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --replay-fps 5
```

The default fake-live case is
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543`. The first
numerically sorted camera-0 step becomes demo `seq=0` for SAM3.1
initialization. Demo 3 / 3.1 consume replayed RGB-D; Demo 3.2 / 3.3 consume
replayed RGB plus IR stereo and compute FFS depth. Playback stops cleanly at
the end of the recording. Fake-live runs in demo mode and defaults to 5 FPS;
pass `--replay-fps 0` to replay at metadata FPS. Demo 3.2/3.3 tracking views
default object/controller filters to enhanced-pt, while PCD-only inspection
defaults both layers to pt-filter but still runs TAPNext++ so the displayed FPS
reflects the full pipeline. Pass `--pcd-filter-preset {original,pt,enhanced-pt}`
to control object/controller PCD and TAPNext++ query initialization together;
tracking query points are sampled from that preset's residual PCD pixels. When
enhanced PCD component filtering is enabled,
object filtering keeps one main component while controller filtering keeps two
main components so two-hand controllers are not dropped as disconnected noise.
Demo 3.1, Demo 3.2, and Demo 3.3 use repo-root `table_calibrate.pkl` by default,
so runtime PCD and lifted TAPNext++ markers are in `table_world_z0` with the
tabletop at `table_z_m = 0.0`. The runtime reports world-Z quantiles and
table-band candidate counts at 5, 10, 20, and 30 mm for object/controller PCD,
plus hand_a/hand_b stats when those masks are available, after the current
PT/enhanced-PT filter. The current table calibration uses negative Z as the
direction above the tabletop (`table_z_above_direction = negative`), and the
filter uses signed clearance from the table plane. Demo 3.1/3.2/3.3
`--demo-visual-mode pcd|tracking` windows, plus Demo 3.2/3.3 headless captures,
delete table-plane candidates by default at `--table-z-filter-threshold-m 0.0`;
pass `--disable-table-z-filter` for unfiltered ablations. Demo 3.x Open3D
tracks demo-mode `human hand` controllers as three EdgeTAM identities
(`hand_a`, `object`, `hand_b`) while keeping the controller PCD/depth mask as
`hand_a | hand_b`; frame-0 needs two separable hands for that mode. Demo 3.x
Open3D windows start in a third-person orbit view; pass `--view-mode camera`
for the RealSense camera view. The realtime filter also falls back to the
current capped PCD if filtering would erase a nonempty semantic layer, and async filter
outputs older than three frames do not replace the current raw frame. The
controller layer also falls back to capped current-frame PCD when filtering
would retain less than half of its capped points, prioritizing two-hand
visibility over aggressive cleanup; if voxel capping makes the controller
filter output less than half of the raw semantic controller points, it falls
back to raw current-frame controller PCD and lets the render cap handle display
density. The render cap uses deterministic coarse spatial bucket balancing so a
sparse separated controller region is not hidden by a denser region when the
display layer is capped. EdgeTAM still runs as a frame-by-frame live session
rather than an offline batch video path, but the live session retains only the
recent 64-frame state window by default so fake-live replay does not accumulate
the entire two-minute stream on the GPU. For Demo 3.2 / 3.3 local FFS, the
runtime serializes TensorRT depth execution and caches a small number of
color-aligned FFS depth frames by sequence so PCD rendering and TAPNext++ marker
lifting do not enter the same TensorRT runner concurrently. When the TAPNext++
overlay is enabled, Demo 3.x processes a lossless 5 FPS task stream and renders
only complete same-`seq` PCD/marker pairs. The PCD and tracker workers run in
parallel, the viewer holds the last complete pair while waiting, and fake-live
replay drains every offered sequence before shutdown. If any stage accumulates
more than the bounded backlog, the demo treats that as a fatal pipeline error
instead of silently dropping to latest-wins behavior.

For a table-Z before/after overlay sweep from a Demo 3.2/3.3 headless capture,
run. The helper reads `table_z_above_direction` from capture metadata and
defaults to `negative` for older captures:

```bash
conda run -n demo_2_max --no-capture-output \
  python scripts/harness/diagnostics/demo/render_demo32_headless_capture.py \
  --capture-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke \
  --output result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/video_unused.mp4 \
  --table-z-overlay-sweep \
  --table-z-overlay-output-dir result/single_demo_v3_2_ffs_masked_pcd/headless_smoke/table_z_overlay
```

Demo 3.2 can also render a 1x3 fake-live review panel with latest RGB on the
left, while the projected PCD and tracking columns stay on the same strict
same-`seq` processed pair.

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

Live FFS preview:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py
```

Use `--render-mode none` when you want a throughput probe that skips panel
assembly and `cv2.imshow()`:

```bash
conda run -n FFS-SAM-RS python cameras_viewer_FFS.py --render-mode none
```

## 2. Calibrate

```bash
python cameras_calibrate.py
```

The default target is the current lab Calib.io ChArUco board:
`calibio-12x9-30mm`. This writes `calibrate.pkl` and
`calibrate_metadata.json` in the repo root by default.

Useful options:

```bash
python cameras_calibrate.py --width 1280 --height 720 --fps 5
python cameras_calibrate.py --serials 239222300781
python cameras_calibrate.py --exposure 70 --gain 60
```

Rerun calibration after any physical camera-position change.

## 2a. Table Z0 Calibration

Place the current lab ChArUco board flat on the table surface, with the printed
board plane touching the tabletop. Then run:

```bash
conda run -n demo_2_max --no-capture-output python cameras_calibrate_table.py
```

This writes `table_calibrate.pkl`, `table_calibrate_metadata.json`, and
`table_calibrate_diagnostic.png` in the repo root when the strict one-shot
check passes. The table calibration is separate from `calibrate.pkl`. Demo 3.1,
Demo 3.2, and Demo 3.3 use repo-root `table_calibrate.pkl` by default and fail
fast when it is missing or invalid; pass `--table-calibrate <path>` only to use an
alternate table calibration. Recording and alignment commands still require an
explicit `--table-calibrate` when table-world output is requested.

Rerun table calibration after moving the camera, moving the table, changing the
camera mount, or changing the tabletop surface used as `Z=0`.

## 3. Record

Default RealSense RGB-D path:

```bash
python record_data.py --case_name my_case --capture_mode rgbd
```

Optional FFS raw capture path:

```bash
python record_data.py --case_name my_case --capture_mode stereo_ir --emitter on
```

Short non-interactive smoke capture:

```bash
python record_data.py --case_name smoke_case --capture_mode rgbd --max_frames 5 --disable-keyboard-listener
```

Raw recordings are written under `data_collect/<case_name>/`.

## 4. Realtime Native Aligned Export

```bash
python record_data_realtime_align.py --case_name native_rt_baseline
```

The output case is written under `data/different_types_real_time/<case_name>/`
and intentionally keeps only the formal downstream interface:

- `calibrate.pkl`
- `metadata.json`
- `color/0/<frame>.png`
- `depth/0/<frame>.npy`

Runtime stats are written outside the formal case under
`data/different_types_real_time/_logs/`.

## 5. Align

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --depth_backend realsense
```

Optional mp4 generation:

```bash
python data_process/record_data_align.py --case_name my_case --start 0 --end 120 --write_mp4
```

Optional FFS backend:

```bash
python data_process/record_data_align.py \
  --case_name my_case \
  --start 0 \
  --end 120 \
  --depth_backend ffs \
  --ffs_repo ../Fast-FoundationStereo \
  --ffs_model_path ../Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth \
  --write_ffs_float_m
```

`realsense` remains the default backend. `ffs` requires raw `ir_left` /
`ir_right` plus runtime geometry metadata. `both` remains an explicit
comparison path.

## 6. Compare Native vs FFS

Per-camera diagnostic panels:

```bash
python scripts/harness/diagnostics/depth/visual_compare_depth_panels.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --write_mp4 --use_float_ffs_depth_when_available
```

Cross-view reprojection comparison:

```bash
python scripts/harness/diagnostics/depth/visual_compare_reprojection.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --camera_pair 0,1 --write_mp4 --use_float_ffs_depth_when_available
```

Single-frame object-centric compare:

```bash
python scripts/harness/diagnostics/visualization/visual_compare_turntable.py --case_name my_case --aligned_root ./data --frame_idx 0
```

Professor-facing summary pack:

```bash
python scripts/harness/diagnostics/visualization/visual_make_professor_triptych.py --aligned_root ./data --realsense_case native_case --ffs_case ffs_case --frame_idx 0
```

## 7. Validation

```bash
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile exhaustive
```
