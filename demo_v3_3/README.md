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
using an alternate single-camera table-world calibration.

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
