# Visual Stack Cleanup Validation

Date: 2026-04-10

## What Was Wrong Before

- display-frame semantics existed but were not owned by one focused module
- `match_board.py` imported private scene/orbit helpers from `professor_triptych.py`
- angle-selection summaries were near-duplicates assembled separately in each workflow
- product-vs-debug artifact metadata was assembled ad hoc in each workflow summary
- docs described the workflow families, but not the shared contracts that now govern them

## What Is Now Centralized

### Display-frame contract

- owner:
  - `data_process/visualization/semantic_world.py`
  - `data_process/visualization/calibration_frame.py`
- shared contract:
  - `DisplayFrameContract`

### Shared scene/orbit compare helpers

- owner:
  - `data_process/visualization/compare_scene.py`
- reused by:
  - `match_board.py`
  - `professor_triptych.py`
  - `stereo_audit.py`

### Shared selection contracts

- owner:
  - `data_process/visualization/selection_contracts.py`
- shared contract types:
  - `AngleSelectionSummary`
  - `TruthPairSelectionSummary`

### Shared artifact contracts

- owner:
  - `data_process/visualization/io_artifacts.py`
  - `data_process/visualization/types.py`
- shared contract types:
  - `ProductArtifactSet`
  - `DebugArtifactSet`

## Product vs Debug Output Behavior

This cleanup pass preserved current user-facing workflow families, but made the product/debug split explicit in summaries:

- match board:
  - product = one board + one summary
  - debug = candidate JSON only when requested
- professor triptych:
  - product = three figures + one summary
  - debug = selection JSONs and optional deeper bundles only when requested
- stereo-order registration:
  - product = one board + one summary (+ optional closeup)
  - debug = overview/selection bundle only when requested

## Tests / Checks Run

Targeted:

- `python -m unittest -v tests.test_visual_types_contract_smoke`
- `python -m unittest -v tests.test_selection_contracts_smoke`
- `python -m unittest -v tests.test_match_board_output_contract_smoke`
- `python -m unittest -v tests.test_professor_triptych_output_contract_smoke`
- `python -m unittest -v tests.test_turntable_frame_contract_smoke`
- `python -m unittest -v tests.test_stereo_order_registration_workflow_smoke`
- `python scripts/harness/check_visual_architecture.py`

Full deterministic validation:

- `python scripts/harness/check_all.py`

Outcome:

- passed

## Result

- shared display-frame semantics are now owned by one focused module
- duplicated scene/orbit helper ownership between `match_board` and `professor_triptych` was reduced
- shared angle-selection contracts now exist and are exercised directly by tests
- product/debug artifact sets are now explicit in current summary outputs
- existing workflows remained compatible and deterministic checks passed
