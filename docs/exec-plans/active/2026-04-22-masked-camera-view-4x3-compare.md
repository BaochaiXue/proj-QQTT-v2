# 2026-04-22 Masked Camera-View 4x3 Compare

## Goal

Extend the static masked camera-view comparison workflow so it can render a `4x3` fixed-view PCD board for round 1-3 comparisons:

- `Native`
- `Native + PS`
- `FFS`
- `FFS + PS`

The workflow must keep the existing `1x3` masked RGB reference board and preserve the current `2x3` contract unless both postprocess flags are enabled.

## Non-Goals

- no change to canonical aligned depth outputs
- no new CLI flags beyond the existing postprocess toggles
- no switch back to depth-frame postprocess semantics for this workflow

## Files To Touch

- `data_process/visualization/workflows/masked_camera_view_compare.py`
- `data_process/visualization/layouts.py` or existing generic matrix helper usage
- `scripts/harness/visual_compare_masked_camera_views.py`
- `tests/test_masked_camera_view_compare_smoke.py`
- layout smoke coverage if needed
- workflow / architecture docs

## Implementation Plan

1. keep raw union masks and postprocessed masks in parallel so the workflow can render both raw and `PS` rows
2. enter `4x3` compare mode only when both `native_depth_postprocess` and `ffs_native_like_postprocess` are enabled
3. reuse the existing generic matrix-board helper for the `4x3` board rather than adding a dedicated layout function
4. expand summary/debug artifacts to describe all four rendered variants
5. regenerate static round 1-3 outputs with the new board layout

## Validation Plan

- `python -m unittest -v tests.test_masked_camera_view_compare_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- re-render static round 1-3 `masked_camera_view_compare_postprocess_*`
