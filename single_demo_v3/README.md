# Single Demo 3: RealSense Masked PCD

Single Demo 3 is the single-camera branch RealSense masked PCD demo. It runs
one RealSense camera, uses RealSense depth, and launches the
single-camera HF EdgeTAM masked point-cloud demo.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run contract:

```bash
python single_demo_v3/realtime_single_camera_realsense_masked_pcd.py --dry-run
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3/realtime_single_camera_realsense_masked_pcd.py \
  --mode exp
```
