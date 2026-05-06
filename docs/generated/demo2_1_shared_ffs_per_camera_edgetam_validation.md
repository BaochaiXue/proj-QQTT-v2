# Demo 2.1 Shared FFS + Per-Camera EdgeTAM Validation

Date: 2026-05-06

## Contract

- `frame_by_frame_streaming=true`
- `offline_video_input_used=false`
- official depth source: `ffs`
- FFS checkpoint: `20-30-48`
- FFS valid iters: `4`
- FFS input shape: `480x864`
- FFS capture shape: `480x848`
- FFS padding policy: `pad_width_848_to_864`
- TensorRT builder optimization level: `5`
- EdgeTAM backend: HF `EdgeTAMVideo`
- EdgeTAM compile mode: `vision-reduce-overhead`
- object filter: `enhanced-pt`
- controller filter: `pt-filter`
- object/controller union before filter: `false`

## Runtime Design

FFS worker:

- one shared worker
- one runner/context owner
- cam0/cam1/cam2 depth generation is sequential

EdgeTAM:

- three per-camera workers
- one streaming session per camera
- `obj_id=1 controller`
- `obj_id=2 object`

Fusion:

- strict `group_id` matching
- object fused cloud uses `enhanced-pt`
- controller fused cloud uses `pt-filter`
- object and controller are never unioned before filtering

## Validation Commands

```bash
python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py

python demo_v2_1/realtime_three_view_masked_fused_pcd.py \
  --dry-run \
  --track-mode controller-object \
  --ffs-worker-mode shared \
  --edgetam-worker-mode per-camera

python -m unittest -v \
  tests.test_demo_v2_1_three_view_fused_pcd_smoke \
  tests.test_check_all_smoke

conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py
```

## Status

The live runtime code path is implemented with lazy hardware/model imports so deterministic checks can validate the CLI and fusion contract without cameras. Hardware profiling still needs to be run on the WSL-5090 rig.
