# Single Demo 3.2: FFS Masked PCD

Single Demo 3.2 is the single-camera FFS masked PCD demo. It runs one
RealSense camera and local FFS depth with TensorRT batch size one.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run:

```bash
python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --mode exp
```
