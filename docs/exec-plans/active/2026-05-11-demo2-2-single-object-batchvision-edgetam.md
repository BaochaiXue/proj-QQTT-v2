# 2026-05-11 Demo 2.2 Single-Object Batch-Vision EdgeTAM

## Goal

Add a Demo 2.2 integration surface for the validated single-object EdgeTAM
batch-vision backend:

```text
backend: hf_batch_vision_seq_session
track mode: object-only
object prompt: stuffed animal
compile mode: vision-reduce-overhead
batch vision encoder: enabled
depth path: local FFS TensorRT batch=3
```

This is the QQTT-facing version of the external EdgeTAM-HF-batched result. It
does not claim that the true `hf_batched_multisession` runtime is complete.

## Non-Goals

- Do not make `hf_batched_multisession` usable or default.
- Do not enable controller/hand by default; compiled hand still has low-IoU
  outliers in the external validation.
- Do not add remote FFS to Demo 2.2 in this slice. The current Demo 2.2 runtime
  is the local FFS fused-PCD path.
- Do not change production defaults for Demo 2.2 controller-object.

## Files To Touch

- `demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `demo_v2_2/runtime.py`
- `demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py`
- `tests/test_demo_v2_2_async_filtered_fused_pcd_smoke.py`
- `docs/generated/demo2_2_single_object_batchvision_edgetam.md`
- `docs/generated/README.md`

## Validation Plan

```bash
conda run --no-capture-output -n demo_2_max \
  python -m unittest -v tests.test_demo_v2_2_async_filtered_fused_pcd_smoke

conda run --no-capture-output -n demo_2_max \
  python demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py \
  --preset demo2.2-single-object-batchvision-edgetam \
  --dry-run

conda run --no-capture-output -n demo_2_max \
  python scripts/harness/check_all.py
```

## Hardware Outcome

The no-render local FFS hardware profile was run after the deterministic checks.
The first parallel-init attempt hit the known SAM3.1/torchvision import race, so
the formal profile was repeated with `--no-parallel-init`.

Warmup-excluded result:

```text
capture_group_fps: 13.80
raw_fusion_fps:    8.33
filter_output_fps: 8.33
render_fps:        0.00
EdgeTAM batch-vision total p50/p90/p95: 22.74 / 24.74 / 25.52 ms
GPU owner total p50/p90/p95:            112.95 / 126.05 / 135.92 ms
FFS cycle p50/p90/p95:                  56.82 / 61.81 / 71.24 ms
```

The single-object batch-vision EdgeTAM path works in the Demo 2.2 local FFS
pipeline. The current no-render fused-PCD output rate is about 8.33 FPS, not
the 15 FPS target; the limiter is upstream supply/local FFS plus single-owner
scheduling, not EdgeTAM batch vision.
