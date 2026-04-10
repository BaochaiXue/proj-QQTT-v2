# 2026-04-09 Professor Three-Figure Object-Aware Selection

## Goal

Fix the existing professor three-figure package so the selected hero angle and truth-board camera pair are object-readable and object-informative, not just globally well-supported.

## Non-Goals

- add another visualization workflow
- add more top-level figures
- change compare engine semantics
- restore large default debug dumps

## Files To Modify

- `data_process/visualization/professor_triptych.py`
- professor-triptych tests
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`

## Validation

Before:
- hero angle can be table-dominant
- truth pair can be globally valid but weak for the object

After:
- hero scoring includes object visibility / fill / support / residual / silhouette terms
- truth pair scoring uses object-region warp/residual terms
- top-level output contract remains exactly 3 figures + `summary.json`
- `python scripts/harness/check_all.py` passes

## Acceptance Criteria

- object-aware hero-angle fields are recorded in `summary.json`
- object-aware truth-pair fields are recorded in `summary.json`
- debug-selection outputs stay gated
- no new top-level clutter is introduced
