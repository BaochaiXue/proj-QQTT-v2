# Single Demo 3.1: RealSense Masked PCD

Single Demo 3.1 keeps the Demo 3.1 naming lineage for the single-camera
branch. The live path uses one RealSense camera, RealSense depth, and the
single-camera masked PCD delegate. The public default also enables the
TAPNext++ point-tracker overlay on top of the 3D point-cloud path, matching the
Demo 3.2 q4096 / GPU1 / all-tracks lift convention where it applies to a
single camera.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.
The final Open3D view renders at most 5000 object points and 5000 controller
points by default; use `--render-max-points-per-layer 0` only for uncapped
display debugging.

Dry-run:

```bash
python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py --dry-run
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --mode exp
```

Lower-VRAM GUI example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --mode exp \
  --pcd-max-points 20000 \
  --pcd-stride 2 \
  --render-max-points-per-layer 5000 \
  --enable-pcd-filter
```

RGB-D recording replay:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --input-source recording \
  --recording-case data_collect/sloth_hand_rgbd_2min_20260612_221051 \
  --mode demo \
  --render-mode pointcloud \
  --tracker-backend tapnextpp \
  --tracker-device cuda:1 \
  --replay-fps 30
```

In replay mode, the first numerically sorted camera-0 RGB-D frame becomes demo
`seq=0` for SAM3.1 initialization, then subsequent frames are emitted at the
requested replay FPS. Demo 3/3.1 recording replay is required to use the
masked point-cloud render path with controller-object tracking and TAPNext++
3D query markers.
