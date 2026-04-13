## Goal

Rename the repo's optional FFS native-like depth postprocess surface so the public flag, auxiliary aligned streams, helper symbols, and summaries all use `ffs_native_like_postprocess` naming without changing behavior.

## Non-Goals

- no change to the native RealSense filter chain or parameter values
- no default change to canonical aligned `depth/` or `depth_ffs/`
- no rollout into reprojection / turntable / rerun / fused 3D compare yet

## Files To Touch

- shared RealSense depth postprocess helper and exports
- `data_process/record_data_align.py`
- depth-panel comparison CLI / workflow / depth loading
- tests, exec plan naming, and docs for the renamed optional mode

## Validation Plan

- helper-level postprocess tests under the renamed module path
- alignment smoke coverage for `--ffs_native_like_postprocess`
- depth panel smoke coverage for renamed auxiliary directories and summary fields
- `python scripts/harness/check_all.py`
