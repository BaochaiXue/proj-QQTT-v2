# Single-Camera Branch Policy

## Objective

Make the repository-local agent and harness rules explicit that all single-camera
work belongs on the `single-camera` branch and must not be committed or pushed
directly to `main`.

## Scope

- Update `AGENTS.md` with a branch policy for single-camera changes.
- Update harness-facing documentation so future agents see the rule from the
  validation map.
- Add deterministic guard coverage so the rule is not silently removed.

## Non-Goals

- Change camera runtime defaults.
- Modify recording, calibration, alignment, demo, or visualization behavior.
- Rewrite historical three-camera documentation.

## Validation

- Run focused unit tests covering `AGENTS.md` and harness checks.
- Run the deterministic scope and harness-engineering guards.

## Status

- 2026-06-12: Started on branch `single-camera` after confirming
  `git pull --ff-only origin main` is up to date.
- 2026-06-12: Updated `AGENTS.md`, `scripts/harness/README.md`,
  `scripts/harness/check_scope.py`, `scripts/harness/check_harness_engineering.py`,
  and `tests/test_agents_scope_contract_smoke.py`.
- 2026-06-12: Validation passed:
  - `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_agents_scope_contract_smoke tests.test_demo23_harness_engineering_smoke`
  - `conda run -n demo_2_max --no-capture-output python -m scripts.harness.check_scope`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_engineering.py`
  - `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
