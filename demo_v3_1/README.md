# Demo 3.1: Single-Camera RealSense Masked PCD

Demo 3.1 keeps the Demo 3.1 naming lineage for the single-camera branch. It
uses one RealSense camera or the shared fake-live camera source, RealSense
depth, object/controller masks, masked point clouds, and TAPNext++ 3D marker
overlay.

Dry-run:

```bash
python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
```

Demo 3.1 uses repo-root `table_calibrate.pkl` by default. If that file or its
`table_calibrate_metadata.json` sidecar is missing or invalid, the wrapper fails
before live or fake-live execution. Pass `--table-calibrate <path>` only when
using an alternate single-camera table-world calibration. With table calibration
enabled, runtime PCD and lifted TAPNext++ marker output are in `table_world_z0`;
the tabletop is reported as `table_z_m = 0.0`. The current table-world
calibration treats points above the tabletop as negative Z
(`table_z_above_direction = negative`), so table-Z filtering uses signed
clearance from the table instead of assuming positive Z is up.

Fake-live replay:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --replay-fps 5
```

`--fake-live-case` is an alias for `--recording-case`. If no case is provided,
fake-live uses `data_collect/sloth_both_eval_2min_e45_g35_20260614_155543`.
Playback publishes `seq=0` first, waits for first-frame initialization, then
streams the remaining frames at 5 FPS by default and exits at EOF. Pass
`--replay-fps 0` to use the recording metadata FPS instead. Fake-live runs in
demo mode.

World-Z diagnostics are observe-only by default. The runtime reports
object/controller Z quantiles plus hand_a/hand_b stats when those masks are
available, with candidate counts for table bands at 5, 10, 20, and 30 mm.
Runtime deletion is opt-in:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --enable-table-z-filter \
  --table-z-filter-threshold-m 0.02 \
  --table-z-filter-classes both
```
