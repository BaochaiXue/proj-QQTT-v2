# 2026-04-26 Native FFS Fused Object PCD

## Goal

Add an isolated experiment workflow for static round `1/2/3` frame `0` that
renders object-only masked point-cloud boards comparing:

- native RealSense depth
- original aligned FFS depth
- fused native/FFS depth

The fused depth should keep native pixels when native depth is valid and at
least `0.6m`; otherwise it should take the same-pixel FFS depth. The board
should match the existing professor-facing masked object PCD style: rows are
depth variants, columns are the three original camera views, object masks are
reused from existing SAM 3.1 sidecars, and displayed fused clouds pass the
PhysTwin-like radius-neighbor cleanup.

## Non-Goals

- no change to existing workflows or their default behavior
- no change to formal aligned-case `depth/` outputs
- no new FFS model run or confidence filtering
- no SAM/mask regeneration

## Files To Touch

- `data_process/visualization/workflows/native_ffs_fused_pcd_compare.py`
- `scripts/harness/visual_compare_native_ffs_fused_pcd.py`
- `tests/test_native_ffs_fused_pcd_compare_smoke.py`
- `scripts/harness/README.md`
- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- this exec plan

## Implementation Plan

1. Define default static round specs pairing existing native and FFS cases plus
   the existing static FFS SAM mask roots.
2. Load frame `0` native depth and original aligned FFS depth in meters for
   each camera.
3. Build fused depth in memory using `native_valid && native >= 0.6m` as the
   native-keep rule.
4. Apply the object mask, with optional default `1px` erosion, to native, FFS,
   and fused depth before point-cloud construction.
5. Build per-camera world clouds, fuse all 3 cameras per row, then apply the
   display-only PhysTwin-like radius-neighbor cleanup.
6. Render one `3x3` board per round from the three original camera pinhole
   views and write JSON summaries with fusion counts, point counts, mask
   provenance, and postprocess stats.
7. Add focused smoke tests with a fake renderer.
8. Add the new harness CLI to docs and deterministic check coverage.

## Validation

- `python -m unittest -v tests.test_native_ffs_fused_pcd_compare_smoke tests.test_check_all_smoke`
- `python scripts/harness/visual_compare_native_ffs_fused_pcd.py --help`
- `python scripts/harness/check_all.py`
- optional render:
  `python scripts/harness/visual_compare_native_ffs_fused_pcd.py --frame_idx 0`
