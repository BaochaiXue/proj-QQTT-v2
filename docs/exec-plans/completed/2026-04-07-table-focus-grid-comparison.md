# 2026-04-07 Table-Focus Grid Comparison

## Goal

Upgrade the fused point-cloud comparison output so it can produce a single 2x3 video
that is easier to interpret:

- top row = Native
- bottom row = FFS
- 3 columns = the 3 real calibrated camera viewpoints
- camera views are refocused toward the tabletop

## Non-Goals

- do not replace the existing panel or reprojection diagnostics
- do not change the calibration schema
- do not require Open3D headless rendering

## Current Repo State Being Reused

- existing aligned-case comparison workflow
- existing `pointcloud_compare.py` renderer and fallback path
- existing calibration loader and depth decoding
- existing real aligned cases: `native_30_static` and `ffs_30_static`

## Architecture Changes

1. Add camera-pose-driven view configs derived from `calibrate.pkl` `c2w`.
2. Add a shared table-focus point estimator.
3. Add a new fused layout mode: `grid_2x3`.
4. Preserve old per-view outputs for compatibility.

## Validation Plan

Deterministic:

- add tests for:
  - camera-pose view config generation
  - table focus point estimation
  - `grid_2x3` output structure
- keep prior fused comparison tests passing
- run `python scripts/harness/check_all.py` in `qqtt-ffs-compat`

Practical:

- render a new `grid_2x3` RGB-colored tabletop-focused comparison using:
  - `data/native_30_static`
  - `data/ffs_30_static`

## Risks

- calibrated camera axes may not match a human-intuitive room frame
- the current fallback renderer is still point-based, not mesh-based
- Open3D offscreen remains unavailable on the current Windows machine

## Acceptance Criteria

- fused comparison supports `view_mode camera_poses_table_focus`
- fused comparison supports `layout_mode grid_2x3`
- the 2x3 view uses the 3 real camera poses
- the tabletop focus is visibly closer than the prior outputs
- old pair outputs still work
- deterministic tests pass

## Completion Checklist

- [x] add camera-pose view configs
- [x] add table-focus estimation
- [x] add `grid_2x3` layout output
- [x] keep prior outputs compatible
- [x] add tests
- [x] update docs
- [x] run deterministic validation
- [x] render a real tabletop-focused example

## Progress Log

- 2026-04-07: added `view_mode fixed|camera_poses_table_focus`
- 2026-04-07: added `focus_mode none|table`
- 2026-04-07: added `layout_mode pair|grid_2x3`
- 2026-04-07: added tests for camera-pose view configs, table focus estimation, and `grid_2x3` output
- 2026-04-07: rendered a real `grid_2x3` tabletop-focused comparison from `native_30_static` and `ffs_30_static`

## Completion Summary

The fused comparison workflow now supports a single 2x3, tabletop-focused comparison
panel using the real calibrated camera poses, while preserving the older per-view
output structure and deterministic fallback renderer.
