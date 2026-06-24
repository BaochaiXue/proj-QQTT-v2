# Demo 3.2 Always-Visible FPS HUD

## Goal

Keep live measured FPS visible in Demo 3.2 across normal Open3D rendering,
strict-sync waiting, side-by-side panel windows, and saved side-by-side panel
video frames.

## Plan

1. Add a focused regression test for the side-by-side panel HUD FPS line.
2. Add FPS fields to the side-by-side panel HUD contract and populate them from
   runtime stage statistics.
3. Reuse one runtime FPS summary line for normal and waiting HUD text.
4. Document that Demo 3.2 panels always show measured FPS.
5. Run focused panel/runtime tests and the smoke validation profile.

## Validation

- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_side_by_side_panel.Demo32SideBySidePanelTest.test_hud_lines_report_stage_fps tests.test_single_demo_tapnextpp_overlay.SingleDemoTapNextOverlayTest.test_panel_hud_label_reports_stage_fps`
- PASS: `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_side_by_side_panel tests.test_single_demo_tapnextpp_overlay tests.test_demo32_headless_render_helper`
- PASS: `python -m py_compile qqtt/demo/demo32_side_by_side_panel.py qqtt/demo/realtime_masked_edgetam_pcd.py scripts/harness/diagnostics/demo/render_demo32_headless_capture.py tests/test_demo32_side_by_side_panel.py tests/test_single_demo_tapnextpp_overlay.py`
- PASS: `git diff --check`
- PASS: `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
