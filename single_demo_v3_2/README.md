# Single Demo 3.2: FFS Masked PCD

Single Demo 3.2 is the single-camera FFS masked PCD demo. It runs one
RealSense camera and local FFS depth with TensorRT batch size one, and exposes
the same TAPNext++ point-tracker overlay controls used by Single Demo 3.1.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.
The final Open3D view renders at most 5000 object points and 5000 controller
points by default; use `--render-max-points-per-layer 0` only for uncapped
display debugging.

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

Lower-VRAM GUI example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_2/realtime_single_camera_ffs_masked_pcd.py \
  --mode exp \
  --render-max-points-per-layer 5000
```
