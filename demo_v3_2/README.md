# Demo 3.2: Single-Camera FFS Masked PCD

Demo 3.2 is the single-camera FFS-depth masked point-cloud runtime. It uses one
camera or the shared fake-live source, runs Fast-FoundationStereo from the IR
stereo pair, propagates SAM3.1/HF EdgeTAM masks, and renders masked PCD plus
TAPNext++ 3D marker overlay.

Dry-run:

```bash
python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Fake-live replay:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
  --mode demo \
  --replay-fps 30
```

If the default TensorRT engine is not present in this checkout, pass the local
engine explicitly:

```bash
conda run -n demo_2_max --no-capture-output \
  python demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --input-source fake-live \
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
