# 2026-04-22 FFS Static Confidence PCD Panels

## Goal

Add a new static-round offline confidence visualization workflow that renders
masked `3x3` `RGB / fused masked FFS PCD / confidence` boards for static round
`1/2/3` frame `0`, while keeping the existing depth-row confidence workflow
unchanged.

## Non-Goals

- no change to the current `visualize_ffs_static_confidence_panels.py`
- no native-vs-FFS comparison in this workflow
- no SAM regeneration path; reuse the existing static stuffed-animal FFS masks
- no TensorRT confidence extraction path

## Files To Touch

- `data_process/visualization/workflows/ffs_confidence_pcd_panels.py`
- `scripts/harness/visualize_ffs_static_confidence_pcd_panels.py`
- `tests/test_ffs_confidence_pcd_panels_smoke.py`
- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- `scripts/harness/README.md`
- `AGENTS.md`
- this exec plan

## Implementation Plan

1. build a dedicated static-round workflow that:
   - resolves the three static FFS rounds
   - reruns PyTorch FFS with confidence on frame `0`
   - aligns depth and both confidence metrics to color
   - builds masked per-camera RGB panels
   - builds a fused masked FFS point cloud from the same rerun depth
2. reuse the existing original-camera pinhole render path for the PCD row so
   rendered object scale tracks the masked RGB row as closely as current repo
   utilities allow
3. render one `margin` board and one `max_softmax` board per round with the PCD
   row shared between the two boards
4. write per-round summaries plus a top-level manifest
5. add software-only tests with a fake confidence runner and fake Open3D render
   function
6. document the new CLI and add deterministic `--help` / smoke coverage

## Validation Plan

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_pcd_panels_smoke tests.test_check_all_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/visualize_ffs_static_confidence_pcd_panels.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/visualize_ffs_static_confidence_pcd_panels.py`

## Risks

- the PCD row depends on fused masked FFS depth rerun results rather than saved
  aligned depth files, so runtime is bounded by PyTorch FFS replay
- object-size matching to RGB is only as strong as the current original-camera
  pinhole Open3D path; there is no exact raster-level parity guarantee
