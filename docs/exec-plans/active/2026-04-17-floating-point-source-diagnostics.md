# 2026-04-17 Floating Point Source Diagnostics

## Goal

Add a diagnostic-only harness that loads aligned native and FFS cases, finds PhysTwin-style radius outliers in fused full-scene point clouds, projects those outlier points back to their source images, and summarizes where they come from.

The workflow must:

- stay isolated from canonical alignment outputs
- support same-case and two-case compare modes
- emit per-source overlays plus structured JSON metrics
- classify each outlier into `occlusion`, `edge`, `dark`, or `other`

## Non-Goals

- no change to `data_process/record_data_align.py`
- no change to canonical aligned `depth/` semantics
- no dependency on raw IR-left / IR-right assets
- no object-only ROI crop in the first version
- no reuse of multi-camera support count as the primary outlier rule

## Files To Touch

- new `data_process/visualization/floating_point_diagnostics.py`
- new `scripts/harness/diagnose_floating_point_sources.py`
- `tests/visualization_test_utils.py`
- new floating-point unit and workflow smoke tests
- `scripts/harness/check_all.py`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`

## Implementation Plan

1. add a diagnostic helper module that:
   - loads per-camera aligned color/depth with source-pixel metadata
   - fuses per-camera world points while preserving point provenance
   - applies Open3D `remove_radius_outlier(radius=0.01, nb_points=40)`
   - classifies outliers from source-image and cross-view evidence
2. add a thin harness CLI that mirrors existing compare scripts and writes:
   - `native/frames/*.png`
   - `ffs/frames/*.png`
   - `native/per_frame_metrics.json`
   - `ffs/per_frame_metrics.json`
   - `summary.json`
   - optional `comparison.mp4`
3. extend visualization fixtures with an optional sparse outlier injection path for deterministic workflow smoke coverage
4. add deterministic tests for:
   - radius outlier selection
   - source-pixel provenance
   - cause-priority assignment
   - cross-view occlusion classification
   - end-to-end harness output contracts
5. update workflow and architecture docs for the new diagnostic-only entrypoint
6. wire the new script/tests into deterministic harness checks

## Validation Plan

- `python scripts/harness/diagnose_floating_point_sources.py --help`
- `python -m unittest -v tests.test_floating_point_diagnostics_smoke`
- `python -m unittest -v tests.test_diagnose_floating_point_sources_smoke`
- `python scripts/harness/check_all.py`
