# 2026-04-26 FFS Confidence Filter PCD 6x3 Panels

## Goal

Render static round `1/2/3` frame `0` Open3D point-cloud boards comparing:

- native RealSense depth
- original aligned FFS depth
- FFS depth filtered by `margin`, `max_softmax`, `entropy`, and `variance`
  confidence at a configurable threshold, initially `0.10`

Each round should produce one `6x3` board with rows as depth variants and
columns as the three cameras.

Supplemental request: run a threshold sweep at `0.01`, `0.05`, `0.10`,
`0.15`, `0.20`, `0.25`, and `0.50`, and make every displayed point cloud pass
the same PhysTwin-like radius-neighbor cleanup used by the masked point-cloud
diagnostics.

## Non-Goals

- no TensorRT confidence path
- no change to the formal aligned-case `depth/` output
- no float depth artifacts written as formal depth
- no SAM/mask regeneration

## Files To Touch

- `data_process/visualization/workflows/ffs_confidence_filter_pcd_compare.py`
- `scripts/harness/visual_compare_ffs_confidence_filter_pcd.py`
- `tests/test_ffs_confidence_filter_pcd_compare_smoke.py`
- `scripts/harness/check_all.py`
- `tests/test_check_all_smoke.py`
- `scripts/harness/README.md`
- this exec plan

## Implementation Plan

1. Resolve the native and FFS static round pairs under `data/static`.
2. Load existing native and original FFS aligned uint16 depth for frame `0`.
3. Rerun PyTorch FFS once per camera with confidence, align depth and all four
   confidence maps to the color frame using the same nearest-depth projection.
4. Build filtered uint16 depth in memory at threshold `0.10`, then decode it
   back to meters only for point-cloud rendering.
5. Render each row/camera tile with the existing Open3D offscreen pinhole path.
6. Compose one `6x3` board per round and write per-round plus top-level JSON
   summaries.
7. Add software smoke tests with a fake runner and fake renderer, plus harness
   help coverage.
8. Add an optional display-only PhysTwin-like postprocess that removes fused
   object points with too few neighbors inside a radius before Open3D rendering.
9. Run the full threshold sweep into one experiment folder and summarize both
   confidence removal and radius-neighbor cleanup counts.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_filter_pcd_compare_smoke tests.test_check_all_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_ffs_confidence_filter_pcd.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_ffs_confidence_filter_pcd.py --frame_idx 0 --threshold 0.10`
- `conda run -n qqtt-ffs-compat python scripts/harness/visual_compare_ffs_confidence_filter_pcd.py --frame_idx 0 --threshold 0.10 --phystwin_like_postprocess --phystwin_radius_m 0.01 --phystwin_nb_points 40`
