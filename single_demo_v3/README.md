# Single Demo 3: RealSense Masked PCD

Single Demo 3 is the single-camera branch version of the copied Demo 3
surface. It runs one RealSense camera, uses RealSense depth, and delegates live
execution to the existing single-camera HF EdgeTAM masked point-cloud demo.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run contract:

```bash
python single_demo_v3/realtime_single_camera_realsense_masked_pcd.py \
  --dry-run \
  --camera-ids 0
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3/realtime_single_camera_realsense_masked_pcd.py \
  --camera-ids 0 \
  --depth-source realsense \
  --mode exp
```

Removed from this single-camera surface:

- exactly-three-RealSense validation
- multi-camera synchronization and world fusion
- mandatory `calibrate.pkl` world transforms
- batch=3 FFS scheduling
- dual-GPU tracker sidecar requirements
- strict three-camera frame-bundle invariants
