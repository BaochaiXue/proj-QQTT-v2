# 2026-05-10 Demo 2.2 Decouple Capture Group Target

Status: completed.

## Goal

Stop using `fusion_target_fps=5.0` as the capture-group throttle for Demo 2.2.
The 5 FPS value should remain the profile/pass target, while capture grouping
should run at the camera input cadence by default.

## Plan

1. DONE: Add `--capture-group-target-fps`.
2. DONE: Use it in `_capture_group_worker` for the group builder interval.
3. DONE: Keep `fusion_target_fps` for target deficit and bottleneck classification.
4. DONE: For Demo 2.2 presets, default `capture_group_target_fps` to the resolved
   camera `fps` value, currently `15`.
5. DONE: Update contract/tests/docs.

## Validation

- PASS: dry-run Demo 2.2 default confirmed `fps=15`,
  `capture_group_target_fps=15.0`, `fusion_target_fps=5.0`, and
  `gpu_pipeline=single-owner`.
- PASS: `conda run --no-capture-output -n demo_2_max python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py tests/test_demo_v2_2_async_filtered_fused_pcd_smoke.py`
- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
