# 2026-04-09 Source Attribution Full-360 Rerender

## Goal

Regenerate the teddy-bear professor-facing turntable compare with:

- strict source colors:
  - Cam0 red
  - Cam1 green
  - Cam2 blue
- a full 360 orbit
- slower playback for inspection
- less smoothing so depth mismatch and missing regions stay visible

## Non-Goals

- add a new visualization product
- change repo scope
- redesign the current compare workflow

## Planned Changes

1. Update source-attribution color mapping and matching docs/tests.
2. Keep current turntable workflow and artifact names.
3. Re-render a new output set using:
   - `full_360`
   - more orbit steps
   - lower fps
   - smaller splat radius / lower supersampling
   - object ROI focused by the existing `fullhead` manual image ROI

## Validation

- run source color tests
- run `python scripts/harness/check_all.py`
- verify new `orbit_compare_source.*` outputs exist
