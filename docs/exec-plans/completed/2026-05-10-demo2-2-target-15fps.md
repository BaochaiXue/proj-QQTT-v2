# 2026-05-10 Demo 2.2 Target 15 FPS

Status: completed.

## Goal

Make the formal Demo 2.2 report/validation target `15 FPS` for the full local
3-camera object+controller tracking and fused filtered PCD path.

`capture_group_target_fps` should remain `15 FPS`; `fusion_target_fps` should no
longer mean "5 FPS pass target" for Demo 2.2.

## Plan

1. DONE: Set Demo 2.2 preset `fusion_target_fps` to `15.0`.
2. DONE: Keep Demo 2.2 `capture_group_target_fps` defaulting to camera `fps`, currently
   `15.0`.
3. DONE: Make Demo 2.2 pass threshold scale from target FPS instead of being hardcoded
   to `4.8`.
4. DONE: Update tests/docs to state that old 5 FPS reports are historical and that the
   current formal target is 15 FPS.

## Validation

- PASS: dry-run Demo 2.2 default confirmed `fps=15`,
  `capture_group_target_fps=15.0`, `fusion_target_fps=15.0`, and
  `gpu_pipeline=single-owner`.
- PASS: `conda run --no-capture-output -n demo_2_max python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py demo_v2_2/realtime_three_view_async_filtered_fused_pcd.py tests/test_demo_v2_2_async_filtered_fused_pcd_smoke.py`
- PASS: `conda run --no-capture-output -n demo_2_max python -m unittest tests.test_demo_v2_2_async_filtered_fused_pcd_smoke tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- PASS: `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
