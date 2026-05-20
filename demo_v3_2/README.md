# Demo 3.2 FFS LiteTracker

Demo 3.2 is its own FFS + LiteTracker live work. It reuses shared camera,
FFS, EdgeTAM, IPC, and marker helpers, but the public entrypoint runs
`qqtt.demo.demo32_runtime` instead of treating Demo 3.2 as a Demo 3.1 preset:

- depth is FFS TensorRT `builderOptimizationLevel=5`, static batch `3`
- FFS TensorRT depth and SAM3.1/HF EdgeTAM masks share GPU0
- LiteTracker remains isolated in the child tracker process on GPU1
- the intended order is capture -> FFS depth -> EdgeTAM masks -> LiteTracker serial -> render/diagnostics
- the tracker backend defaults to `litetracker` with serial execution
- LiteTracker uses lazy query initialization by default: the child process is
  ready to receive inputs immediately, and query-dependent tracker state is
  initialized from the first valid RGB + mask packet
- RGB plus object/controller/union masks are published to LiteTracker from
  both fused and async raw-fused paths
- render waits for a LiteTracker result and surface-snapped 3D control marker
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
  --profile-json-output docs/generated/demo32_litetracker_ffs_rendered_60s_profile.json
```

LiteTracker external defaults are the current local validation paths:

- repo: `/home/xinjie/external/lite-tracker`
- weights: `/home/xinjie/external/weights/cotracker3/scaled_online.pth`

Override them with `--litetracker-repo-dir` and `--litetracker-weights` when
using another machine.
