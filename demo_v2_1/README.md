# Demo 2.1 Three-View Fused Masked PCD

Demo 2.1 is the three-camera successor to Demo 2. It keeps the Demo 2 quality contract but changes the output from a single-camera masked PCD to fused semantic point clouds from `cam0/cam1/cam2`.

Official quality path:

```text
3x RealSense color + IR
temporal-coherent CaptureGroup gating
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

Temporal grouping policy:

```text
No temporal-coherent CaptureGroup, no FFS.

professor-safe / visual-5fps defaults:
  capture-group-policy=timestamp-nearest
  max-capture-skew-ms=33.4
  max-frame-age-ms=150
  capture-buffer-size=4
  drop-skewed-groups=true
```

The CaptureGroupBuilder keeps a small per-camera frame buffer and emits only
the cam0/cam1/cam2 triplet with the nearest timestamps. If the selected triplet
exceeds the skew threshold, it is dropped before FFS. The shared FFS worker and
fusion worker both re-check the temporal skew contract.

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
python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run --preset professor-safe
```

Presets:

```text
professor-safe:
  848x480@30
  fusion-target-fps=2
  controller-object by default; pass --track-mode object-only when no hand/controller is visible
  render-mode=pointcloud by default
  GPU gate off by default
  temporal grouping uses timestamp-nearest, max skew 33.4 ms

visual-5fps:
  848x480@30
  fusion-target-fps=5
  controller-object by default; pass --track-mode object-only when no hand/controller is visible
  render-mode=pointcloud by default
  GPU gate off by default
  quality path unchanged: FFS depth + object enhanced-pt
  temporal grouping uses timestamp-nearest, max skew 33.4 ms

visual-5fps-single-owner:
  848x480@30
  fusion-target-fps=5
  render-mode=pointcloud by default
  gpu-pipeline-mode=single-owner
  single-owner-order=ffs-then-edgetam
  disables separate shared-FFS and per-camera EdgeTAM worker threads
  publishes depth and masks together as one CompleteInferenceGroup
  quality path unchanged: FFS depth + object enhanced-pt

climb-5:
  848x480@30
  fusion-target-fps=5
  render-mode=none by default

climb-10:
  848x480@30
  fusion-target-fps=10
  render-mode=none by default

diagnostics:
  starts from the same 848x480@30, serialized GPU gate surface
  combine with --depth-source none, --track-mode none, or --render-mode none
```

Current no-hand professor-safe object-only run:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```

Current no-hand 5 FPS visual candidate:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-cuda-events
```

No-GPU-gate profiling baseline:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps-no-gate \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_live_sam31_no_gate_profile_object_only_120s.json
```

This is a profiling baseline, not a default professor preset. It disables only
the global `GpuInferenceGate`. The shared FFS worker still owns a single FFS
runner/context and processes cam0/cam1/cam2 sequentially, while the quality
contract remains unchanged: FFS-derived depth, live SAM3.1 init, timestamp
grouping, and object `enhanced-pt`.

Single GPU-owner profiling candidate:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps-single-owner \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-h2d \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_single_owner_no_pin_object_only_120s.json
```

In this mode one worker owns the FFS TensorRT runner and all EdgeTAM sessions.
The worker processes one temporal-coherent `CaptureGroup` into a
`CompleteInferenceGroup`, then fusion consumes complete depth+masks directly.
This is designed to reduce partial same-group joins and GPU worker contention.
The `--static-device-buffers` and `--preallocate-pcd-buffers` flags are recorded
as memory-for-speed ablation hooks; they do not change the quality contract.

Profiling the 5 FPS candidate:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 20 \
  --profile-json-output docs/generated/demo2_1_visual5fps_profile_object_only.json
```

Formal Demo 2.1 initialization requirement:

```text
The professor-facing demo uses --init-mode sam31-first-frame.
SAM3.1 must segment the live first frame in the room, then HF EdgeTAM tracks from that mask.
The live path uses SAM3.1 image one-frame segmentation (`Sam3Processor.set_image` + text prompt), not video propagation.
The default mode is controller-object; current no-hand lab runs must explicitly pass --track-mode object-only.
```

If SAM3.1 object-only initialization fails in a no-hand run, Demo 2.1 fails fast. That is intentional: there is no saved-mask or native-depth fallback in the formal path.

The default controller prompt remains `hand`. Non-hand controller prompts are
allowed only as explicit experimental overrides, for example when using two
cloth pieces to stress-test the controller slot:

```bash
--track-mode controller-object --controller-prompt "cloth"
```

That override does not change the default professor-facing controller label.

Live SAM3.1 5 FPS profiling command:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset visual-5fps \
  --track-mode object-only \
  --init-mode sam31-first-frame \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug \
  --profile-pipeline \
  --profile-filter \
  --profile-visualization \
  --profile-gpu-gate \
  --profile-warmup-exclude-s 40 \
  --profile-json-output docs/generated/demo2_1_visual5fps_live_sam31_profile_object_only_120s.json
```

`saved-masks` is rejected by the formal Demo 2.1 runtime. Use it only in separate diagnostic scripts, not this live demo.

Controller-object run, only when a hand is visible:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset professor-safe \
  --track-mode controller-object \
  --controller-prompt "hand" \
  --object-prompt "stuffed animal" \
  --duration-s 120 \
  --debug
```

Isolation runs:

```bash
# Capture grouping only
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset diagnostics --depth-source none --track-mode none --render-mode none --duration-s 20 --debug

# EdgeTAM only, three per-camera workers
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset diagnostics --depth-source none --track-mode controller-object --render-mode none --duration-s 30 --debug

# Shared FFS only
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset diagnostics --depth-source ffs --track-mode none --render-mode none --duration-s 30 --debug

# Target climb, headless first
python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --preset climb-5 --track-mode object-only --render-mode none --duration-s 60 --debug
```

Runtime architecture:

```text
CaptureGroupBuilder:
  maintains small per-camera timestamp buffers
  emits one strict group_id only for a temporal-coherent cam0/cam1/cam2 triplet
  drops skewed groups before FFS

SharedFfsWorker:
  owns one FFS/TensorRT runner
  sequentially computes cam0, cam1, cam2 depth only for temporal-coherent group_id

GpuInferenceGate:
  serializes FFS and EdgeTAM GPU inference in professor-safe mode
  records per-stage wait time in debug and session summary

EdgeTamCameraWorker:
  one worker per camera
  one HF EdgeTAM streaming session per camera
  obj_id=1 controller, obj_id=2 object

FusionWorker:
  joins same group_id depth + masks
  re-checks temporal skew before fusing
  fuses object separately from controller
  object -> enhanced-pt
  controller -> pt-filter
```

WSLg/Open3D launch wrapper:

```bash
./demo_v2_1/run_wslg_open3d.sh conda run --no-capture-output -n demo_2_max \
  python demo_v2_1/realtime_three_view_masked_fused_pcd.py --dry-run
```
