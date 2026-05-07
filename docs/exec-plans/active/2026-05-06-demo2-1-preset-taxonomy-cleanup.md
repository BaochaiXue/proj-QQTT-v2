# Demo 2.1 Preset Taxonomy Cleanup

## Goal

Separate official Demo 2.1 runtime presets from performance/profiling presets.
`visual-5fps` accumulated too many meanings, so keep it as a compatibility alias
only and introduce clearer canonical preset names.

## New Canonical Presets

- `official-lowfps`: stable professor-facing low-FPS quality path
- `perf-5fps`: 5 FPS performance target with separate workers
- `perf-5fps-single-owner`: 5 FPS performance target with the single GPU-owner
  pipeline
- `diagnostics`, `climb-5`, `climb-10`: remain diagnostic/profiling presets

## Compatibility

Keep existing names as aliases:

- `professor-safe` -> `official-lowfps`
- `visual-5fps` -> `perf-5fps`
- `visual-5fps-no-gate` -> `perf-5fps`
- `visual-5fps-single-owner` -> `perf-5fps-single-owner`

## Validation

- `python -m py_compile demo_v2_1/realtime_three_view_masked_fused_pcd.py`
- `conda run --no-capture-output -n demo_2_max python -m unittest -v tests.test_demo_v2_1_three_view_fused_pcd_smoke`
- `conda run --no-capture-output -n demo_2_max python scripts/harness/check_all.py`
