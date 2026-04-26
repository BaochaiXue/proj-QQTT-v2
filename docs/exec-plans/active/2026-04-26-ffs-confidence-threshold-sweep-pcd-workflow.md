# 2026-04-26 FFS Confidence Threshold Sweep PCD Workflow

## Goal

Add a dedicated workflow for the static round `1/2/3` frame `0` confidence
experiment. For each threshold `0.01`, `0.05`, `0.10`, `0.15`, `0.20`,
`0.25`, and `0.50`, render a `6x3` Open3D point-cloud panel:

- rows: native depth, original FFS, FFS filtered by `margin`, `max_softmax`,
  `entropy`, and `variance`
- columns: original three FFS camera views
- displayed points: object mask only, mask eroded inward by `1px`
- displayed rows: after PhysTwin-like radius-neighbor postprocess

All threshold and round outputs should live under one experiment result folder.

## Non-Goals

- no TensorRT confidence path
- no change to formal aligned-case `depth/`
- no SAM/mask regeneration
- no new segmentation or tracking surface

## Files To Touch

- `data_process/visualization/workflows/ffs_confidence_threshold_sweep_pcd_compare.py`
- `scripts/harness/visual_compare_ffs_confidence_threshold_sweep_pcd.py`
- `tests/test_ffs_confidence_threshold_sweep_pcd_compare_smoke.py`
- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `scripts/harness/README.md`
- this exec plan

## Implementation Plan

1. Build a new sweep workflow instead of overloading the single-threshold
   comparison entrypoint.
2. Load the existing static round specs and SAM3.1 object masks.
3. Erode each per-camera union mask by `1px` using a `3x3` kernel, and record
   before/after mask pixel counts.
4. Run PyTorch FFS with classifier-logit confidence once per round/camera, then
   reuse the aligned depth and four aligned confidence maps across all thresholds.
5. For each threshold, build the six row variants in memory, fuse the three
   camera clouds per row, apply the PhysTwin-like radius-neighbor filter, and
   render the `6x3` board.
6. Write per-threshold summaries, per-round summaries, and one top-level
   manifest under a single result folder.
7. Add deterministic smoke tests using fake FFS and fake renderer paths.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_threshold_sweep_pcd_compare_smoke tests.test_check_all_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_ffs_confidence_threshold_sweep_pcd.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_ffs_confidence_threshold_sweep_pcd.py --frame_idx 0`
