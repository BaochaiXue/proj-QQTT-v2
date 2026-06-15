# Demo 3.1: Single-Camera RealSense Masked PCD

Demo 3.1 keeps the Demo 3.1 naming lineage for the single-camera branch. It
uses one RealSense camera or the shared fake-live camera source, RealSense
depth, object/controller masks, masked point clouds, and TAPNext++ 3D marker
overlay.

Dry-run:

```bash
python demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
```

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
