# Single Demo 3.3: FFS Masked PCD

Single Demo 3.3 keeps the Demo 3.3 version label for the single-camera branch
while removing the live three-camera shape-prior warmup orchestration. The live
path is one RealSense camera with FFS depth and the single-camera masked PCD
delegate.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run:

```bash
python single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py \
  --dry-run \
  --camera-ids 0
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py \
  --camera-ids 0 \
  --depth-source ffs \
  --mode exp
```

`--shape-prior-warmup` is kept as a dry-run contract flag only. The single
entrypoint does not launch the previous three-camera warmup pipeline, does not
need `--camera-ids 0,1,2`, and does not require dual GPUs.
