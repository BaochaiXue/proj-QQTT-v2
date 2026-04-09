# 2026-04-09 Contract Hardening Pass

## Exact Issues Being Fixed

1. `AGENTS.md` / workflow-map inconsistency around whether aligned comparison visualization is in scope and which compare entrypoints are current.
2. `record_data.py` preflight / operator UX inconsistency around blocked vs warning-only modes and serial discovery timing.

## Non-Goals

- any `calibrate.pkl` format or loader semantic change
- visualization feature work
- architecture rewrite
- ROI/object-compare redesign
- repo scope expansion

## Files To Modify

- `AGENTS.md`
- `record_data.py`
- `qqtt/env/camera/preflight.py`
- `docs/WORKFLOWS.md`
- focused tests under `tests/`
- `scripts/harness/check_all.py`
- validation note under `docs/generated/`

## Before / After Validation Plan

Before:

- inspect current `AGENTS.md` inconsistency
- inspect current `record_data.py` preflight flow

After:

- run focused preflight tests
- run `python scripts/harness/check_all.py`
- record results in `docs/generated/contract_hardening_validation.md`

## Acceptance Criteria

1. `AGENTS.md` is internally consistent about compare visualization scope.
2. AGENTS file map points to the current comparison entrypoints.
3. record preflight decision states are explicit and testable.
4. operator-facing preflight summaries are clearer before recording starts.
5. tests are added or updated.
6. `python scripts/harness/check_all.py` passes.
7. `calibrate.pkl` semantics remain untouched.
