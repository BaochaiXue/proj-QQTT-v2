# Single Demo 3.2: FFS Masked PCD

Single Demo 3.2 is the single-camera FFS branch of the copied Demo 3.2 surface.
It keeps local FFS depth available, but the FFS TensorRT batch size is one and
there is no strict batch=3 scheduler.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run:

```bash
python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --dry-run \
  --camera-ids 0
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --camera-ids 0 \
  --depth-source ffs \
  --mode exp
```

Removed from this single-camera surface:

- static batch=3 FFS contract
- three-view tracker bundle matching
- dual-GPU tracker worker requirement
- per-camera query balancing across three views
- three-camera calibration and fused-world rendering
