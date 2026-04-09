# Visual Stack Refactor Validation

Date: 2026-04-09

Environment:

- repo: `C:\Users\zhang\proj-QQTT`
- conda env: `qqtt-ffs-compat`
- validation python: `C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe`

## Refactor Summary

The visualization cleanup kept the repo product boundary unchanged, but split the comparison stack into clearer lower-level modules:

- added `data_process/visualization/types.py`
- added `data_process/visualization/io_case.py`
- added `data_process/visualization/io_artifacts.py`
- added `data_process/visualization/roi.py`
- added `data_process/visualization/views.py`
- added `data_process/visualization/layouts.py`
- added `data_process/visualization/renderers/`
- added `data_process/visualization/workflows/`
- kept `pointcloud_compare.py` and `turntable_compare.py` as compatibility-facing entry modules while delegating shared logic out

## Largest Module Reduction

Before cleanup:

- `data_process/visualization/pointcloud_compare.py`: 1476 lines
- `data_process/visualization/turntable_compare.py`: 2629 lines

After cleanup:

- `data_process/visualization/pointcloud_compare.py`: 473 lines
- `data_process/visualization/turntable_compare.py`: 2201 lines

Notes:

- `pointcloud_compare.py` now primarily keeps the older fused-depth workflow orchestration plus compatibility exports.
- `turntable_compare.py` is still the largest visualization module, but orbit/output planning and board composition are no longer all colocated there.

## New Guardrails

Added:

- `scripts/harness/check_visual_architecture.py`

It currently checks:

- no visualization module imports `scripts.harness`
- low-level visualization modules do not import `data_process.visualization.workflows`
- renderer modules do not import CLI-only modules
- line-count thresholds for the largest compatibility modules and harness wrappers

## New Tests Added

- `tests/test_visual_import_graph_smoke.py`
- `tests/test_visual_types_contract_smoke.py`
- `tests/test_turntable_workflow_smoke.py`
- `tests/test_merge_diagnostics_workflow_smoke.py`
- `tests/test_artifact_writer_smoke.py`
- `tests/test_layout_builder_smoke.py`

## Commands Run

Architecture guard:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness/check_visual_architecture.py
```

Targeted new smoke tests:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_visual_types_contract_smoke tests.test_artifact_writer_smoke tests.test_layout_builder_smoke tests.test_merge_diagnostics_workflow_smoke tests.test_visual_import_graph_smoke
```

Workflow wrapper smoke:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_turntable_workflow_smoke
```

Compatibility spot checks:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_dual_triple_output_planning_smoke tests.test_visual_compare_turntable_smoke tests.test_turntable_board_layout_smoke
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe -m unittest -v tests.test_fused_cloud_render_config_smoke tests.test_projection_mode_smoke tests.test_grid_2x3_label_layout_smoke
```

Full deterministic validation:

```bash
C:\Users\zhang\miniconda3\envs\qqtt-ffs-compat\python.exe scripts/harness/check_all.py
```

## Results

- `scripts/harness/check_visual_architecture.py`: passed
- new architecture-level smoke tests: passed
- workflow wrapper smoke test: passed
- compatibility spot checks: passed
- `scripts/harness/check_all.py`: passed

## Compatibility Notes

Kept stable:

- current harness CLI commands
- current aligned-case comparison output naming
- legacy import paths under:
  - `data_process.visualization.pointcloud_compare`
  - `data_process.visualization.turntable_compare`

Changed internally:

- case IO, artifact writing, crop math, orbit math, and board composition are now split into dedicated modules
- render-output planning now has a typed `RenderOutputSpec` model via `workflows/merge_diagnostics.py`
- single-frame case selection now has an internal typed `CompareCaseSelection` model in the turntable workflow

## Remaining Cleanup Opportunities

1. `turntable_compare.py` is still large because overview/debug composition is still colocated with the main workflow.
2. `object_compare.py` still mixes pure masking logic with debug artifact generation.
3. `panel_compare.py` and `reprojection_compare.py` can eventually move under `data_process/visualization/workflows/` once compatibility shims are in place.
