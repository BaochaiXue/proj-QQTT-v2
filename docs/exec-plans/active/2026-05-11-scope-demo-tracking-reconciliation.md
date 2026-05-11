# Scope Demo/Tracking Reconciliation

## Goal

Bring the written repo scope back in line with the current codebase:

- keep the formal data product boundary at recording, alignment, and aligned-case comparison
- explicitly include sanctioned live demos, remote FFS proxy services, and tracking diagnostics
- keep physics, shape-prior, reconstruction/rendering evaluation, and teleoperation outside this repo

## Scope

- Update root scope wording in `README.md`.
- Update detailed boundaries in `docs/SCOPE.md` and `docs/repo-scope.md`.
- Update `AGENTS.md` so future agent instructions match the current repository.
- Add a small scope-contract smoke assertion for the newly documented boundary.

## Validation

- `python scripts/harness/check_scope.py`
- `python -m unittest -v tests.test_agents_scope_contract_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
