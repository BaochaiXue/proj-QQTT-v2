# Demo 3.2 FFS LiteTracker

Demo 3.2 is its own FFS + LiteTracker live work. It reuses shared camera,
FFS, EdgeTAM, IPC, and marker helpers, but the public entrypoint runs
`qqtt.demo.demo32_runtime` instead of treating Demo 3.2 as a Demo 3.1 preset:

- depth is FFS TensorRT `builderOptimizationLevel=5`, static batch `3`
- FFS TensorRT depth and SAM3.1/HF EdgeTAM masks share GPU0
- LiteTracker remains isolated in the child tracker process on GPU1
- the intended order is capture -> FFS depth -> EdgeTAM masks -> LiteTracker
  batch=3 camera views -> render/diagnostics
- the tracker backend defaults to `litetracker` with experimental batch-view
  execution: `--tracking-backend-execution-mode batch-views` and
  `--cotracker-update-mode batch`
- LiteTracker ONNX-CUDA is retained as an explicit serial A/B profiling path,
  not the default live path; local live profiling showed ONNX serial slower
  than the batch-view PyTorch runtime
- Demo 3.2 uses `--tracker-batch-query-count-policy min-common` by default so
  all three camera views have the same query count for the batch tensor
- LiteTracker uses lazy query initialization by default: the child process is
  ready to receive inputs immediately, and query-dependent tracker state is
  initialized from the first valid RGB + mask packet
- warmup fails fast by default if SAM3.1 first-frame init does not produce both
  required masks (`object` and `controller`); use
  `--no-sam31-init-quick-fail-empty-masks` only for debug
- before that first LiteTracker packet, Demo 3.2 builds enhanced PT trackable
  masks from FFS depth plus object/controller masks; LiteTracker receives RGB
  plus `union_trackable_mask`, `object_trackable_mask`, and
  `controller_trackable_mask`
- object/controller enhanced PT defaults are object top-1 and controller top-2
  3D components, with the controller trackable pixels capped after enhanced PT
  filtering, default `--controller-trackable-max-points-per-camera 4999`
- `--mode demo` uses the SAM3.1 controller prompt `human hand` while the
  controller semantic remains a hand
- `--controller-mask-erode-px` shrinks the controller mask before building the
  tracking union, trackable masks, and anchor inputs; the implicit default is
  `1` in `--mode demo` (human-hand controller prompt) and `0` in `--mode exp`
- controller body points are voxel-downsampled before Open3D render with
  `--controller-render-voxel-m 0.003`; this is render-only and does not affect
  LiteTracker input or red tracking/control markers
- depth, intrinsics, and `c2w` remain in the main process for filtering,
  anchors, and marker validation; they are not sent to the LiteTracker child
- trackable masks are published to LiteTracker from both fused and async
  raw-fused paths
- rendered LiteTracker markers use the same exact target frame bundle by
  default: tracker result, FFS depth/lift inputs, and rendered PCD must share
  the same `group_id`; nearest-frame fallback is debug-only
- render waits for a LiteTracker result and 3D tracking control markers
- every visible LiteTracker point with valid depth is eligible for the 3D
  anchor layer by default (`--tracker-visualization-mode all-tracks-3d-lift`);
  Demo 3.2 does not apply surface-snap matching or semantic bbox rejection in
  this default path
- the all-tracks lift is still render-bounded by default:
  `--all-tracks-lift-max-points-per-camera 512` with
  `--all-tracks-lift-selection visible-spread`; pass
  `--all-tracks-lift-max-points-per-camera 0` only for explicit all-point
  quality/debug runs
- rendered profiles report all-tracks lift candidate, selected, rendered, cap,
  timing, and exact-depth-group fields so marker count cannot silently explode
- the Open3D warmup HUD is generated from the active runtime pipeline, so Demo
  3.2 reports LiteTracker query-init and 3D anchors instead of a hard-coded
  Demo 2.3 FFS/EdgeTAM-only status line
- object/controller semantics stay the current experiment default: object `stuffed animal`, controller `towel`

Dry-run:

```bash
conda run --no-capture-output -n demo_3_1_max \
  python demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py \
  --dry-run \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl
```

Rendered profiling:

```bash
QQTT_WSLG_OPEN3D_FAST_EXIT=1 conda run --no-capture-output -n demo_3_1_max \
  python demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py \
  --duration-s 60 \
  --camera-ids 0,1,2 \
  --mask-gpu 0 \
  --cotracker-gpu 1 \
  --require-two-cuda \
  --calibrate-path calibrate.pkl \
  --render-mode pointcloud \
  --render-micro-profile \
  --gpu-sampling \
  --gpu-sampling-device-indexes 0,1 \
  --tracker-visualization-mode all-tracks-3d-lift \
  --all-tracks-lift-max-points-per-camera 512 \
  --profile-json-output docs/generated/demo32_litetracker_ffs_rendered_60s_profile.json
```

LiteTracker external defaults are the current local validation paths:

- repo: `/home/xinjie/external/lite-tracker`
- weights: `/home/xinjie/external/weights/cotracker3/scaled_online.pth`

Override them with `--litetracker-repo-dir` and `--litetracker-weights` when
using another machine.
