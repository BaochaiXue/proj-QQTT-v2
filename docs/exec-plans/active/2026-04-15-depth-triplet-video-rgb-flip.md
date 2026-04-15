# 2026-04-15 Depth Triplet Video RGB Flip

## Goal

Add a formal aligned-case triplet time-axis point-cloud video workflow for `native`, `ffs_raw`, and `ffs_postprocess` that renders with aligned RGB colors and fixes the current upside-down Open3D capture output.

## Non-Goals

- no change to existing depth-video / turntable / professor-facing workflow defaults
- no global renderer default changes
- no new color-mode matrix for the triplet video workflow

## Files To Touch

- new triplet video workflow + wrapper + CLI
- visualization exports / deterministic check wiring
- docs for workflow / architecture / harness map
- smoke coverage for output contract, RGB path, and vertical-flip behavior

## Implementation Plan

1. add a dedicated triplet time-axis workflow that reuses the aligned-case cloud loader for:
   - `native`
   - `ffs_raw`
   - `ffs_postprocess`
2. keep triplet video rendering local to this workflow and use Open3D hidden-window capture because offscreen rendering is unavailable on this machine
3. lock the workflow to aligned RGB colors and apply a vertical image flip before writing frames
4. write exactly 3 videos plus `summary.json`, with shared crop/view metadata and per-variant point-count stats
5. add smoke tests for:
   - three-video output contract
   - aligned auxiliary vs on-the-fly postprocess provenance
   - RGB color path and vertical flip
6. update docs / harness index / architecture guards and rerun deterministic checks

## Validation Plan

- `python scripts/harness/visual_compare_depth_triplet_video.py --help`
- `python -m unittest -v tests.test_triplet_video_compare_workflow_smoke`
- `python scripts/harness/check_all.py`
