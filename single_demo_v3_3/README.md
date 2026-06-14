# Single Demo 3.3: FFS Masked PCD

Single Demo 3.3 keeps the Demo 3.3 version label for the single-camera branch
with one RealSense camera, FFS depth, and the single-camera masked PCD delegate.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.
The final Open3D view renders at most 5000 object points and 5000 controller
points by default; use `--render-max-points-per-layer 0` only for uncapped
display debugging.

Dry-run:

```bash
python single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py --dry-run
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py \
  --mode exp
```

Lower-VRAM GUI example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_3/realtime_single_camera_ffs_masked_pcd.py \
  --mode exp \
  --render-max-points-per-layer 5000
```
