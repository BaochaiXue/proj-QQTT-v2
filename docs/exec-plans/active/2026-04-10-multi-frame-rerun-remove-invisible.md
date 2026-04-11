## Goal

Add a raw diagnostic workflow that replays aligned native and FFS point clouds over time in Rerun while also exporting fused per-frame full-scene PLYs for:

- `native`
- `ffs_remove_1`
- `ffs_remove_0`

## Non-Goals

- no new professor-facing render board
- no ROI cropping or semantic-world display transform
- no change to aligned-case on-disk schema
- no change to `env_install/env_install.sh`

## Files To Touch

- `data_process/depth_backends/fast_foundation_stereo.py`
- `data_process/depth_backends/__init__.py`
- `data_process/visualization/__init__.py`
- new rerun compare workflow + CLI
- new tests for remove-invisible semantics and rerun workflow
- docs covering workflow usage, architecture, env notes, and validation

## Implementation Plan

1. add a shared helper that reproduces upstream `remove_invisible` disparity invalidation semantics and reports counts
2. add a new visualization workflow that:
   - loads native aligned depth normally
   - reruns FFS from aligned `ir_left` / `ir_right`
   - derives `remove_0` and `remove_1` from the same disparity
   - reprojects to color, deprojects to world, fuses 3 cameras, logs to Rerun, and writes fused PLYs
3. add a thin CLI wrapper with explicit `rerun_output` and optional FFS config overrides
4. add deterministic tests with fake FFS runner and fake Rerun module
5. update docs and record a real validation run under `docs/generated/`

## Validation Plan

- targeted unit tests for remove-invisible masking and rerun workflow smoke coverage
- `python scripts/harness/visual_compare_rerun.py --help`
- `python scripts/harness/check_all.py`
- real `qqtt-ffs-compat` run against `native_30_static` / `ffs_30_static`
