# Results Retention Cleanup

## Goal

Clean result artifacts so operators do not confuse smoke, interrupted, obsolete,
and current results.

## Scope

- Inventory top-level result folders under `data/experiments/` and
  `data/ffs_benchmarks/`.
- Delete folders whose names explicitly identify smoke-test output.
- Rename obsolete but still useful result roots with an `archived_obsolete_`
  prefix instead of deleting them.
- Update docs and harness references for renamed roots.
- Avoid changing raw/aligned data products under `data/static/`,
  `data/dynamics/`, `data/different_types/`, or camera runtime code.

## Validation

- `git diff --check`
- `python scripts/harness/check_harness_catalog.py`
- `python scripts/harness/check_experiment_boundaries.py`
- `conda run -n FFS-SAM-RS python scripts/harness/check_all.py`
