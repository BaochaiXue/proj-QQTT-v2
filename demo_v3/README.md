# Demo 3: Single-Camera RealSense Masked PCD

Demo 3 is the single-camera RealSense-depth masked point-cloud runtime. It uses
one RealSense stream or the shared fake-live camera source, SAM3.1 first-frame
initialization, HF EdgeTAM online mask propagation, and TAPNext++ 3D marker
overlay through `qqtt.demo.single_demo_v3_runtime`.

Dry-run:

```bash
python demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
```

Fake-live replay:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3/realtime_single_camera_realsense_masked_pcd.py \
  --input-source fake-live \
  --mode demo
```

`fake-live` defaults to
`data_collect/sloth_both_eval_2min_e45_g35_20260614_155543`. The first complete
camera-0 frame becomes runtime `seq=0` for SAM3.1 initialization; after that,
frames are emitted at metadata or CLI replay FPS and the process exits cleanly
at the end of the recording. Fake-live runs in demo mode.
