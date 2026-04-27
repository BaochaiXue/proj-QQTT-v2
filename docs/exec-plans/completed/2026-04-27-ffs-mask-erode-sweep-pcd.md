# 2026-04-27 FFS Mask Erode Sweep PCD

## Goal

Create a dedicated workflow for static round `1/2/3` frame `0` that renders
one `10x3` Open3D object-PCD board per round:

- row 1: native depth with the original object mask
- row 2: original FFS depth with the original object mask
- rows 3-10: original FFS depth with object mask eroded inward by `1..8px`
- columns: the original three FFS camera views

All displayed rows should use the same PhysTwin-like radius-neighbor
postprocess before rendering, and all outputs should live under one experiment
result folder.

## Non-Goals

- no FFS rerun or TensorRT path
- no confidence filtering
- no native/FFS depth fusion
- no formal aligned-case `depth/` changes
- no SAM/mask regeneration

## Files To Touch

- `data_process/visualization/experiments/ffs_mask_erode_sweep_pcd_compare.py`
- `data_process/visualization/layouts.py`
- `scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py`
- compatibility wrappers:
  - `data_process/visualization/workflows/ffs_mask_erode_sweep_pcd_compare.py`
  - `scripts/harness/visual_compare_ffs_mask_erode_sweep_pcd.py`
- `tests/test_ffs_mask_erode_sweep_pcd_compare_smoke.py`
- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `scripts/harness/README.md`
- this exec plan

## Implementation Plan

1. Reuse the static round specs and mask roots from the native/FFS object-PCD
   workflow.
2. Load aligned native depth and original FFS depth from existing aligned cases.
3. Load per-camera union object masks and generate eroded masks for `1..8px`
   using a `3x3` erosion kernel.
4. Build object-only native and FFS point clouds per row/camera.
5. Fuse the three camera clouds for each row, apply the PhysTwin-like
   radius-neighbor cleanup, render each row in the original FFS camera views,
   and compose one `10x3` board per round.
6. Use a wider configurable row-label band so the long mask-erode labels remain
   visible in the final board.
7. Write per-round summaries and one top-level manifest under a single result
   folder.
8. Add deterministic smoke tests with a fake renderer and register the CLI in
   the quick validation surface.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_mask_erode_sweep_pcd_compare_smoke tests.test_check_all_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_ffs_mask_erode_sweep_pcd.py --frame_idx 0`
