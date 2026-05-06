# Demo 2.1 Three-View Fused Masked PCD

Demo 2.1 is the three-camera successor to Demo 2. It keeps the Demo 2 quality contract but changes the output from a single-camera masked PCD to fused semantic point clouds from `cam0/cam1/cam2`.

Official quality path:

```text
3x RealSense color + IR
FFS 20-30-48, valid_iters=4, 848x480 -> pad 864x480, TensorRT builderOpt5
HF EdgeTAM streaming, vision-reduce-overhead
masked PCD per camera
semantic fused PCD
```

Supported tracking modes:

```text
object-only
controller-object
```

Fusion policy:

```text
object:
  fuse cam0/cam1/cam2 object clouds
  postprocess with enhanced-pt

controller:
  fuse cam0/cam1/cam2 controller clouds
  postprocess with pt-filter
```

Object and controller are kept as separate fused semantic layers. Do not concatenate object and controller into a single cloud before filtering unless explicitly running a diagnostic, because enhanced cleanup can remove controller fingertips or contact patches.

Current CLI smoke:

```bash
python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run
```

WSLg/Open3D launch wrapper:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run
```
