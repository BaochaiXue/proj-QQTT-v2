# Single Demo 3.1: RealSense Masked PCD

Single Demo 3.1 keeps the Demo 3.1 naming lineage for the single-camera
branch, but removes the dual-4090 and batch-view tracker requirements. The live
path uses one RealSense camera, RealSense depth, and the single-camera masked
PCD delegate.

The current experiment defaults remain object `stuffed animal` and controller
`towel`. `--mode demo` switches the controller prompt to `human hand`.

Dry-run:

```bash
python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --dry-run \
  --camera-ids 0
```

Live example:

```bash
conda run --no-capture-output -n demo_2_max \
  python single_demo_v3_1/realtime_single_camera_realsense_masked_pcd.py \
  --camera-ids 0 \
  --depth-source realsense \
  --mode exp
```

This single-camera entrypoint does not require `--require-two-cuda`, tracker
batch views, cross-GPU tensor transfer, or three-camera calibration.
