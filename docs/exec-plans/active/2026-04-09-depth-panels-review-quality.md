# 2026-04-09 Depth Panels Review Quality

## Goal

Upgrade the current per-camera native-vs-FFS depth diagnostic panels into a cleaner presentation-quality board without replacing the existing workflow or changing the current CLI structure.

## Non-Goals

- replace the workflow
- add a new compare product
- change repo scope
- redesign turntable / reprojection / calibration semantics

## Planned Changes

1. Add a `review_quality` preset to the current depth-panel CLI.
2. Improve panel board composition:
   - clearer title strip
   - explicit case/camera/frame context
   - larger ROI detail panels
   - consistent spacing and headers
3. Keep native/FFS depth ranges identical and add legends where useful.
4. Add optional edge overlay panels for RGB-vs-depth contour reading.
5. Add simple summary metrics per camera/frame to the board and summary JSON.
6. Add software-only tests for layout, preset handling, fixed scale propagation, and ROI overlays.

## Files To Modify

- `scripts/harness/visual_compare_depth_panels.py`
- `data_process/visualization/panel_compare.py`
- `data_process/visualization/depth_diagnostics.py`
- `data_process/visualization/layouts.py`
- `data_process/visualization/workflows/depth_panels.py`
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`
- tests for panel layout / preset / ROI overlays

## Validation Plan

- run focused depth-panel tests
- run `python scripts/harness/check_all.py`

## Acceptance Criteria

1. Panel boards are visually cleaner and easier to read.
2. Native vs FFS depth differences are easier to inspect.
3. ROI detail panels are larger and overlaid on overview panels.
4. Current command structure remains backward compatible.
5. Deterministic tests still pass.
