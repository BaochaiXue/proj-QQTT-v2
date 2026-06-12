# Delete Historical Contracts And Completed Plans

## Objective

Remove stale historical contract documents and completed execution-plan history
from the `single-camera` branch so current docs focus on active branch behavior.

## Scope

- Delete legacy Demo 3 contract markdown files under `docs/`.
- Delete old completed execution plans under `docs/exec-plans/completed/`.
- Update doc indexes and smoke tests that referenced those historical files.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo3_tracking_contract_smoke tests.test_check_all_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`

## Status

- 2026-06-12: Started on branch `single-camera` after `git pull --ff-only
  origin main` reported up to date.
- 2026-06-12: Completed. Kept this note under `active/` rather than
  recreating `completed/`, because this change intentionally removes completed
  plan history from the branch.
