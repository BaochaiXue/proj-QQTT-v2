# 2026-05-06 Demo 2.1 Single GPU-Owner Pipeline

Status: implemented; deterministic validation passed.

## Goal

Add a Demo 2.1 single GPU-owner pipeline mode that keeps capture, temporal grouping, fusion/filtering, and rendering asynchronous, but moves all heavy GPU inference for one capture group into one owning worker thread:

```text
CaptureGroup -> FFS cam0/1/2 -> EdgeTAM cam0/1/2 -> CompleteInferenceGroup -> fusion/filter/render
```

## Non-Goals

- no FFS checkpoint / TensorRT engine changes
- no EdgeTAM compile-mode changes
- no saved-mask fallback
- no RealSense-depth fallback
- no semantic filter changes
- no object/controller union before filtering

## Files To Touch

- `demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `tests/test_demo_v2_1_three_view_fused_pcd_smoke.py`
- Demo 2.1 generated validation docs

## Implementation Plan

1. add `--gpu-pipeline-mode separate-workers|single-owner` and `--single-owner-order`
2. add `CompleteInferenceGroup` and a latest slot for complete inference outputs
3. add `visual-5fps-single-owner` preset and dry-run contract
4. implement `GPUOwnerPipelineWorker` path that owns FFS runner and EdgeTAM model/session state
5. route fusion worker to consume complete inference groups in single-owner mode
6. keep pin-memory H2D ablation and FFS staging flags available in both modes
7. record per-group `gpu_owner` profile fields

## Validation Plan

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Validation Result

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py tests/test_demo_v2_1_three_view_fused_pcd_smoke.py` passed.
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke` passed.
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py` passed.
