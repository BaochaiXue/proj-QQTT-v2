# 2026-04-22 Floating Point Projection Masked 4x3

## Goal

Extend floating-point diagnostics so static round 1-3 can render masked `4x3` projection boards that place detected outliers back onto:

- Native RGB
- Native Depth
- FFS RGB
- FFS Depth

with one column per original camera view.

## Non-Goals

- no change to aligned depth outputs
- no change to camera-view pointcloud compare outputs
- no requirement to generate new masks automatically

## Files To Touch

- `data_process/visualization/floating_point_diagnostics.py`
- `scripts/harness/diagnose_floating_point_sources.py`
- `tests/test_diagnose_floating_point_sources_smoke.py`
- workflow docs

## Implementation Plan

1. add optional masked-mode inputs to floating-point diagnostics so existing SAM masks can suppress background before outlier detection and rendering
2. reuse viewer depth colormap defaults for masked depth overlays
3. keep existing per-source outputs, but add a combined `4x3` comparison board per frame plus a single-frame top-level board artifact
4. regenerate static round 1-3 frame-0000 outputs under the existing `floating_point_diagnostics_round*_frame0000` directories

## Validation Plan

- `python -m unittest -v tests.test_diagnose_floating_point_sources_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- visually inspect regenerated static round 1-3 boards
