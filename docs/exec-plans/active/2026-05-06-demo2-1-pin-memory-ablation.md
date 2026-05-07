# 2026-05-06 Demo 2.1 Pin-Memory Transfer Ablation

Status: implemented; deterministic validation passed.

## Goal

Add a quality-preserving H2D transfer ablation to Demo 2.1 so we can compare pageable vs pinned CPU staging for EdgeTAM and FFS without changing the semantic depth/mask/filter pipeline.

## Non-Goals

- no FFS checkpoint / TensorRT engine changes
- no EdgeTAM compile-mode changes
- no saved-mask fallback
- no RealSense-depth fallback
- no semantic filter changes

## Files To Touch

- `demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `data_process/depth_backends/fast_foundation_stereo.py`
- `tests/test_demo_v2_1_three_view_fused_pcd_smoke.py`
- Demo 2.1 generated validation docs

## Implementation Plan

1. add pin-memory and H2D profiling CLI flags with defaults that preserve current behavior
2. keep FFS pinned staging as the default but add an explicit pageable staging mode for true baseline comparison
3. add an EdgeTAM pinned-pixel staging path that runs the processor on CPU and transfers via a reusable pinned ring buffer
4. record per-group H2D staging/enqueue/wait metrics in the existing profile JSON
5. add tests for CLI precedence, quality-contract preservation, pinned ring safety, and byte/value preservation

## Validation Plan

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py data_process/depth_backends/fast_foundation_stereo.py`
- `conda run --no-capture-output -n SAM21-max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`
