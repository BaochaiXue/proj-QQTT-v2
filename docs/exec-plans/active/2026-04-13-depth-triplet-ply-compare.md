# 2026-04-13 Depth Triplet PLY Compare

## Goal

Add a single-frame aligned-case fused point-cloud workflow that writes `.ply` outputs for:

- `native`
- `ffs_raw`
- `ffs_postprocess`

The workflow must support both same-case and two-case compare modes, prefer aligned `depth_ffs_native_like_postprocess*` when present, and otherwise apply the same native-like depth postprocess on the fly before fusion.

## Non-Goals

- no Rerun output
- no professor-facing image board
- no semantic-world transform
- no object-only crop / ROI refinement
- no change to existing `visual_compare_rerun.py` remove-invisible semantics

## Files To Touch

- `data_process/visualization/io_case.py`
- `data_process/visualization/depth_diagnostics.py`
- new triplet compare workflow + wrapper + CLI
- workflow/docs/architecture inventory updates
- smoke tests and deterministic check wiring

## Implementation Plan

1. centralize aligned depth loading plus FFS native-like postprocess fallback in `io_case.py`
2. make depth-panel loading reuse that shared contract so 2D and 3D compare stay aligned
3. add a single-frame triplet PLY workflow that:
   - resolves same-case vs two-case inputs
   - loads `native`, `ffs_raw`, and `ffs_postprocess`
   - fuses all 3 cameras in raw calibration-world coordinates
   - writes exactly 3 fused PLYs plus `summary.json`
4. add a thin harness CLI with default output naming
5. add smoke coverage for:
   - aligned auxiliary postprocess preference
   - on-the-fly postprocess fallback
   - two-case and same-case triplet output contracts
6. update operator/docs/guard surfaces for the new compare entrypoint

## Validation Plan

- `python scripts/harness/visual_compare_depth_triplet_ply.py --help`
- `python -m unittest -v tests.test_io_case_ffs_native_like_loader_smoke`
- `python -m unittest -v tests.test_triplet_ply_compare_workflow_smoke`
- `python scripts/harness/check_all.py`
