# 2026-04-27 FFS Mask Erode Multipage Sweep PCD

## Goal

Create a dedicated experiment workflow for static round `1/2/3` frame `0`
that renders three `10x3` Open3D object-PCD boards per round:

- page 1: native depth, original FFS depth, and FFS masks eroded `1..8px`
- page 2: FFS masks eroded `9..18px`
- page 3: FFS masks eroded `19..28px`
- columns: the original three FFS camera views

The displayed clouds must be object-mask-only and must apply the same
PhysTwin-like radius-neighbor cleanup before rendering. All pages and summaries
belong to one result folder.

## Contract Note

`native + original FFS + 9..18px` would be `12x3`, not `10x3`. To preserve the
requested `10x3` layout, baseline rows appear on page 1 only; pages 2 and 3 are
ten-row erode-only continuation pages. The summary records this page contract.

## Files To Touch

- `data_process/visualization/experiments/ffs_mask_erode_sweep_pcd_compare.py`
- `data_process/visualization/experiments/ffs_mask_erode_multipage_sweep_pcd_compare.py`
- `data_process/visualization/workflows/ffs_mask_erode_multipage_sweep_pcd_compare.py`
- `scripts/harness/experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py`
- `scripts/harness/check_all.py`
- `scripts/harness/check_experiment_boundaries.py`
- `scripts/harness/README.md`
- `tests/test_check_all_smoke.py`
- `tests/test_ffs_mask_erode_multipage_sweep_pcd_compare_smoke.py`
- this exec plan

## Implementation Plan

1. Extend the existing mask-erode PCD runner with optional board page specs,
   without changing the default one-page `1..8px` behavior.
2. Add default page specs for `1..8`, `9..18`, and `19..28` with explicit row
   labels and row-count contracts.
3. Add a new experiment workflow wrapper and CLI for the multipage experiment.
4. Write per-round page images plus per-round and top-level manifests under one
   output root.
5. Add deterministic smoke tests using the fake renderer.
6. Register the new experiment CLI in quick/full validation and documentation.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_mask_erode_multipage_sweep_pcd_compare_smoke tests.test_ffs_mask_erode_sweep_pcd_compare_smoke tests.test_check_all_smoke tests.test_experiment_boundary_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_ffs_mask_erode_multipage_sweep_pcd.py --frame_idx 0`
