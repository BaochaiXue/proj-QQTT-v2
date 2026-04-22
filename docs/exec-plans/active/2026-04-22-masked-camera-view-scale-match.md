# 2026-04-22 Masked Camera-View Scale Match

## Goal

Tighten the static masked comparison workflow so it produces:

- one `1x3` masked RGB panel for the 3 original camera views
- one `2x3` masked point-cloud panel rendered with Open3D

The Open3D panels must use the original camera viewpoints with fixed camera extrinsics and a deterministic pinhole camera setup so the masked object occupies a similar image scale to the masked RGB view for the same camera.

## Non-Goals

- no change to recording or alignment outputs
- no change to the repo's core data product boundary
- no new segmentation runtime inside production record/align paths
- no decorative contour overlays or extra per-panel markup beyond bounded labels/contracts

## Files To Touch

- `data_process/visualization/workflows/masked_camera_view_compare.py`
- `data_process/visualization/triplet_video_compare.py`
- `data_process/visualization/layouts.py`
- `scripts/harness/visual_compare_masked_camera_views.py`
- `tests/test_masked_camera_view_compare_smoke.py`
- docs describing the workflow and output contract

## Implementation Plan

1. switch the masked camera-view point-cloud render path from loose `lookat/front/up/zoom` framing to original-camera pinhole rendering:
   - use per-camera `K_color`
   - use the real camera `c2w` / inverted extrinsic
   - keep rendering deterministic under the Open3D hidden-window path
2. add a `1x3` masked RGB board:
   - use the same 3 camera columns
   - apply the resolved union mask directly to RGB without extra contour styling
3. keep the `2x3` PCD board:
   - rows = `Native` and `FFS`
   - columns = camera `0 / 1 / 2`
   - ensure object scale is driven by the same original pinhole framing as the RGB row rather than an ad hoc shared zoom
4. update summary/test/doc output contracts and validate on deterministic smoke coverage plus repo checks

## Validation Plan

- `python scripts/harness/visual_compare_masked_camera_views.py --help`
- `python -m unittest -v tests.test_masked_camera_view_compare_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
