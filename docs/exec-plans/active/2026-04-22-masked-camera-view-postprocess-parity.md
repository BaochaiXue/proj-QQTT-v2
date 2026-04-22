# 2026-04-22 Masked Camera-View Postprocess Parity

## Goal

Extend the masked camera-view comparison workflow so the static `1x3` masked RGB board and `2x3` masked point-cloud board can be rendered after applying the same PhysTwin-like depth postprocess chain to both:

- `Native` depth
- `FFS` depth

The compare must keep the original camera pinhole viewpoint fixed per column so only the depth source and postprocess behavior change.

## Non-Goals

- no change to aligned case layout or canonical data outputs
- no new segmentation/runtime dependency in the record/align product surface
- no Meshlab/manual random view comparisons
- no decorative contour overlays or extra annotations on the masked RGB board

## Files To Touch

- `data_process/visualization/io_case.py`
- `data_process/visualization/workflows/masked_camera_view_compare.py`
- `scripts/harness/visual_compare_masked_camera_views.py`
- `tests/test_masked_camera_view_compare_smoke.py`
- docs describing the new postprocess-parity option

## Implementation Plan

1. extend aligned depth loading so the same software postprocess chain can be applied on the fly to realsense/native depth as well as FFS depth
2. thread explicit `native_depth_postprocess` and `ffs_native_like_postprocess` controls into the masked camera-view workflow and CLI
3. record both postprocess flags and provenance in the workflow summary
4. render static round1-3 outputs with:
   - `1x3` masked RGB board
   - `2x3` masked PCD board
   - postprocess enabled for both `Native` and `FFS`

## Validation Plan

- `python -m unittest -v tests.test_masked_camera_view_compare_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- render static round1-3 outputs under `data/static/`
