## Current Pain Points

- display-frame semantics now exist, but the contract is still spread across `calibration_frame.py`, `semantic_world.py`, `turntable_compare.py`, and `stereo_audit.py`
- angle-selection logic is duplicated across `match_board.py` and `professor_triptych.py`
- summary and metrics JSON contracts overlap without a shared typed schema
- product outputs and debug outputs are not governed by one explicit artifact contract
- some workflow modules still own too much selection / artifact / summary logic
- docs point to current workflows, but do not cleanly explain shared contracts

## Non-Goals

- no new visualization product family
- no calibrate.pkl format or loader semantic changes
- no broad rendering redesign
- no repo-scope expansion

## Files To Inventory

- `scripts/harness/*.py` compare entrypoints
- `data_process/visualization/match_board.py`
- `data_process/visualization/professor_triptych.py`
- `data_process/visualization/turntable_compare.py`
- `data_process/visualization/stereo_audit.py`
- `data_process/visualization/semantic_world.py`
- `data_process/visualization/io_artifacts.py`
- `data_process/visualization/types.py`
- current docs and tests that mention workflow/output contracts

## Target Module Boundaries

- CLI wrappers: parse args, call one workflow, print output path
- display-frame logic: one focused contract helper path
- angle selection: one shared scoring/selection contract module
- artifact schemas: typed summaries and artifact-set helpers
- workflows: orchestration only, not ad hoc schema invention
- debug gating: centralized product-vs-debug output planning

## Contract Unification Plan

1. inventory current workflow/schema ownership
2. add explicit typed contracts for:
   - display frame
   - angle selection
   - product/debug artifact sets
   - ROI pass summaries
3. move duplicated angle-selection logic into one shared module
4. move output gating into one shared artifact helper contract
5. update workflows to reuse shared contracts instead of hand-rolling JSON blobs

## Migration / Compatibility Plan

- keep current CLI commands stable
- keep current top-level filenames stable where already user-facing
- keep old metrics/debug files available when workflows explicitly write them
- only clean schema assembly and gating, not user-visible product families

## Validation Plan

- add/update tests for:
  - display-frame contract
  - shared angle-selection contract
  - summary schema contract
  - product/debug artifact separation
- update docs/inventory/validation notes
- run `python scripts/harness/check_all.py`

## Acceptance Criteria

- display-frame semantics are centralized
- angle-selection logic is unified or clearly de-duplicated
- artifact/schema contracts are explicit and typed
- product/debug output separation is clearer
- workflows remain compatible
- docs/tests/checks pass
