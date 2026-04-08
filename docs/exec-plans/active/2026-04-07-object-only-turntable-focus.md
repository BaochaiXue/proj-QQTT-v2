# 2026-04-07 Object-Only Turntable Focus

## Goal

Fix the current professor-facing turntable output so the main panels focus on the
actual object pair instead of the tabletop:

- lock ROI onto the teddy-bear + box object cluster
- make the turntable orbit center object-centric in practice, not only in name
- render the main geom/rgb/support outputs from object-only clouds
- keep overview context, but stop letting the table dominate the main comparison

## Current Problem

- `auto_object_bbox` still leaves a table-scale ROI in the real case
- the current orbit radius is therefore too large
- the tabletop is still the dominant rendered surface
- the object appears too small, broken, and blurry in the main side-by-side videos

## Reuse

- keep the current turntable workflow and output structure
- keep the current camera coverage logic
- reuse the current fused loading, c2w transforms, and support rendering path

## Changes

1. Improve object ROI extraction so it isolates the object cluster more tightly.
2. Use object-center focus for turntable view generation.
3. Add object-only filtering for the main render clouds.
4. Keep overview and coverage annotations, but make the main compare object-first.

## Validation

- update synthetic object ROI tests
- add or update an object-only render smoke test
- rerun `python scripts/harness/check_all.py`
- rerender the real `native_30_static` vs `ffs_30_static` output
