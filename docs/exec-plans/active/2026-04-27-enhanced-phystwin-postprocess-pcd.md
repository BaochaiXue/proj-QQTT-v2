# 2026-04-27 Enhanced PhysTwin Postprocess PCD

## Goal

Create an experiment workflow for static round `1/2/3` frame `0` that renders a
`6x3` Open3D object-PCD board per round:

- native depth, object masked, no PCD postprocess
- native depth with PhysTwin-like radius-neighbor postprocess
- native depth with enhanced PhysTwin-like postprocess
- original FFS depth, object masked, no PCD postprocess
- original FFS depth with PhysTwin-like radius-neighbor postprocess
- original FFS depth with enhanced PhysTwin-like postprocess

The enhanced mode should first apply the current PhysTwin-like radius filter,
then run 3D voxel connected components on the fused object cloud, keep the main
component, remove disconnected smaller components, and report removed component
stats for tuning.

## Non-Goals

- no formal aligned-case `depth/` changes
- no FFS rerun or confidence/logits filtering
- no SAM/mask regeneration
- no native/FFS fusion row

## Files To Touch

- `data_process/visualization/experiments/ffs_confidence_filter_pcd_compare.py`
- `data_process/visualization/experiments/enhanced_phystwin_postprocess_pcd_compare.py`
- `data_process/visualization/workflows/enhanced_phystwin_postprocess_pcd_compare.py`
- `scripts/harness/experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py`
- `scripts/harness/check_all.py`
- `scripts/harness/check_experiment_boundaries.py`
- `scripts/harness/README.md`
- `tests/test_check_all_smoke.py`
- `tests/test_enhanced_phystwin_postprocess_pcd_compare_smoke.py`
- this exec plan

## Implementation Plan

1. Add an enhanced postprocess helper that wraps the existing radius-neighbor
   postprocess and then performs voxel connected-component filtering.
2. Record component stats: point count, bbox, centroid, kept/removed decision,
   and bbox gap to the main component.
3. Build a new experiment workflow that creates native and FFS object clouds,
   derives the six requested postprocess variants, renders them in the original
   FFS camera views, and writes one board per static round.
4. Add a harness experiment CLI, a workflow compatibility shim, smoke tests,
   and quick validation registration.

## Validation

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_enhanced_phystwin_postprocess_pcd_compare_smoke tests.test_check_all_smoke tests.test_experiment_boundary_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py --help`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/experiments/visual_compare_enhanced_phystwin_postprocess_pcd.py --frame_idx 0`
