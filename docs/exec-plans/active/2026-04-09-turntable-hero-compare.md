# 2026-04-09 Turntable Hero Compare

## Goal

Add clean professor-facing hero static outputs to the existing single-frame turntable compare:

- `hero_compare_geom.png`
- `hero_compare_rgb.png`

Keep the current compare engine, orbit logic, and debug artifacts intact.

## Non-Goals

- add new render modes
- redesign source/support/mismatch diagnostics
- rewrite the turntable workflow
- change calibration semantics or compare scope

## Files To Modify

- `data_process/visualization/layouts.py`
- `data_process/visualization/turntable_compare.py`
- `scripts/harness/visual_compare_turntable.py` only if a stable surface change is required
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`
- turntable layout/workflow tests

## Before / After Validation

Before:
- main side-by-side outputs include too much footer/debug text for slide use
- no dedicated hero still exists as the first output

After:
- one run writes `hero_compare_geom.png`
- one run writes `hero_compare_rgb.png` when RGB output is enabled
- existing videos, gifs, sheets, and debug artifacts still write as before

## Acceptance Criteria

- hero outputs are 1x2 Native vs FFS only
- hero outputs reuse the same frame, ROI, angle, crop, projection, and scale
- hero outputs keep text minimal and overview small
- current command surface remains backward compatible
- deterministic tests and `python scripts/harness/check_all.py` pass
