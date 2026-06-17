# Demo 3.3: Single-Camera FFS Masked PCD

Demo 3.3 is the single-camera FFS-depth masked point-cloud runtime reserved for
the Demo 3.3 lineage on this branch. Its live and fake-live camera contract
matches Demo 3.2: one RGB stream, one IR stereo pair, FFS depth, SAM3.1/HF
EdgeTAM masks, masked PCD, and TAPNext++ 3D marker overlay.

Dry-run:

```bash
python demo_v3_3/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Demo 3.3 uses repo-root `table_calibrate.pkl` by default. If that file or its
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
  python demo_v3_3/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --replay-fps 5
```

Fake-live defaults to
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543`. The first complete
recorded frame is used as runtime `seq=0`; later frames stream at 5 FPS by
default, or at metadata FPS when `--replay-fps 0` is passed. Fake-live runs in
demo mode.

World-Z diagnostics are always reported for table-calibrated PCD. The runtime
reports object/controller Z quantiles plus hand_a/hand_b stats when those masks
are available, with candidate counts for table bands at 5, 10, 20, and 30 mm.
Demo 3.3 `--demo-visual-mode pcd|tracking` enables runtime table-Z deletion by
default at 0 mm signed clearance; use `--disable-table-z-filter` for unfiltered
ablations, or pass a larger threshold explicitly:

```bash
--table-z-filter-threshold-m 0.01 --table-z-filter-classes both
```

Headless captures include `camera_to_world_c2w` in metadata and per-frame
`world_z_stats.jsonl`. Use
`scripts/harness/diagnostics/demo/render_demo32_headless_capture.py
--table-z-overlay-sweep` on the capture directory to render before/after/removed
RGB overlays for the default threshold sweep.
