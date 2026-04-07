# 2026-04-07 Readable Tabletop Comparison

## Goal

Make the existing native-vs-FFS fused comparison video actually readable for tabletop inspection.

## Non-Goals

- do not replace the existing comparison workflow
- do not remove same-case or two-case fallback support
- do not require Open3D offscreen rendering

## Current Repo State Being Reused

- existing aligned-case comparison CLI and output layout
- existing `grid_2x3` support
- existing `camera_poses_table_focus` support
- existing fallback renderer and metadata pipeline

## Architecture Changes

1. Add world-space tabletop crop before framing.
2. Add geometry-aware view distance scaling and projection controls.
3. Densify the fallback renderer with splat-like rendering.
4. Strengthen 2x3 labels and headers.
5. Add a preset for the tabletop-focused comparison workflow.

## Validation Plan

Deterministic:

- add tests for:
  - table crop ROI selection
  - 2x3 label layout
  - projection modes
  - fallback splat density
- keep existing comparison tests passing
- run `python scripts/harness/check_all.py` in `qqtt-ffs-compat`

Practical:

- render a new tabletop-focused 2x3 comparison from:
  - `data/native_30_static`
  - `data/ffs_30_static`

## Risks

- the current geometry still uses pinhole intrinsics without explicit distortion coefficients
- fallback rendering is still an approximation of surfaces rather than a full mesh pipeline
- same-take comparison remains hardware-profile dependent

## Acceptance Criteria

- tabletop and tabletop objects occupy more of each panel
- native and FFS remain directly comparable under identical view settings
- row and column labels are readable
- fallback render no longer looks like isolated 1-pixel dots
- deterministic tests pass
- docs explain the new recommended workflow

## Completion Checklist

- [x] add world-space tabletop crop
- [x] add view distance and projection controls
- [x] densify fallback rendering
- [x] strengthen 2x3 labels
- [x] add tabletop comparison preset
- [x] add tests
- [x] update docs
- [x] run deterministic validation
- [x] render a real tabletop-focused 2x3 example

## Progress Log

- 2026-04-07: added `scene_crop_mode`, manual ROI bounds, and table auto-crop support
- 2026-04-07: added `view_distance_scale`, `projection_mode`, and orthographic support in fallback rendering
- 2026-04-07: upgraded fallback rendering to a denser splat-like path with point radius and supersampling controls
- 2026-04-07: strengthened the 2x3 composition with title, row labels, and column headers
- 2026-04-07: added the `tabletop_compare_2x3` preset and updated docs/examples
- 2026-04-07: rendered a new tabletop-focused comparison from `native_30_static` and `ffs_30_static`
