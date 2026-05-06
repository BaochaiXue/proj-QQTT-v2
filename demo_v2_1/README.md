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
  voxel-cap before cleanup
  postprocess with enhanced-pt

controller:
  fuse cam0/cam1/cam2 controller clouds
  voxel-cap before cleanup
  postprocess with pt-filter
```

Object and controller are kept as separate fused semantic layers. Do not concatenate object and controller into a single cloud before filtering unless explicitly running a diagnostic, because enhanced cleanup can remove controller fingertips or contact patches.

Realtime filter policy:

```text
hot path:
  render raw/capped semantic PCD every frame/group

filter path:
  object     -> cap 20k by default -> enhanced-pt
  controller -> cap 20k by default -> pt-filter
  async/latest-wins is the target scheduler; sync filtering remains a diagnostic mode
```

Dry-run contract:

```bash
python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run
```

Headless live correctness run:

```bash
conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --track-mode controller-object \
  --depth-source ffs \
  --ffs-worker-mode shared \
  --ffs-schedule strict3-latest \
  --edgetam-worker-mode per-camera \
  --edgetam-model-topology replicated \
  --fusion-target-fps 10 \
  --render-mode none \
  --duration-s 30 \
  --debug \
  --profile-cuda-events
```

Pointcloud live run:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --track-mode controller-object \
  --depth-source ffs \
  --ffs-worker-mode shared \
  --ffs-schedule strict3-latest \
  --edgetam-worker-mode per-camera \
  --edgetam-model-topology replicated \
  --fusion-target-fps 10 \
  --render-mode pointcloud \
  --enable-pcd-filter \
  --pcd-filter-mode async \
  --filter-every-n 3 \
  --object-filter-cap 20000 \
  --controller-filter-cap 20000 \
  --duration-s 30 \
  --debug \
  --profile-cuda-events
```

Isolation runs:

```bash
# Capture grouping only
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --depth-source none --track-mode none --render-mode none --duration-s 20 --debug

# EdgeTAM only, three per-camera workers
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --depth-source none --track-mode controller-object --render-mode none --duration-s 30 --debug

# Shared FFS only
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --depth-source ffs --track-mode none --render-mode none --duration-s 30 --debug
```

Runtime architecture:

```text
CaptureGroupBuilder:
  emits one strict group_id for cam0/cam1/cam2 latest frames

SharedFfsWorker:
  owns one FFS/TensorRT runner
  sequentially computes cam0, cam1, cam2 depth for each group_id

EdgeTamCameraWorker:
  one worker per camera
  one HF EdgeTAM streaming session per camera
  obj_id=1 controller, obj_id=2 object

FusionWorker:
  joins same group_id depth + masks
  fuses object separately from controller
  object -> enhanced-pt
  controller -> pt-filter
```

WSLg/Open3D launch wrapper:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run
```
