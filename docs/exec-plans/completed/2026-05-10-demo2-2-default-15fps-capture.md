# 2026-05-10 Demo 2.2 Default 15 FPS Capture

Status: completed.

## Goal

Make Demo 2.2 use `15 FPS` RealSense capture by default while keeping the
consumer/fusion target at `5 FPS`.

Also keep the wrapper default on the fastest current Demo 2.2 schedule:
single-owner async filter, not the staged-parallel probe.

## Changes

1. DONE: Update Demo 2.2 presets so their default `fps` is `15`.
2. DONE: Keep `fusion_target_fps=5.0`.
3. DONE: Make `demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py` default to
   `demo2.2-async-filter-5fps`.
4. DONE: Update deterministic tests and generated benchmark notes.

## Validation

- PASS: dry-run Demo 2.2 wrapper default confirmed `fps=15`,
  `fusion_target_fps=5.0`, `preset=demo2.2-async-filter-5fps`, and
  `gpu_pipeline=single-owner`.
- PASS: `conda run --no-capture-output -n demo_2_max python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py tests/test_demo_v2_2_async_filtered_fused_pcd_smoke.py`
- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
