# Result Root Unification

## Goal

Make `./result/` the only local result root for experiment, benchmark, and
visualization outputs.

## Scope

- Move the former `data/experiments/` contents under `result/`.
- Keep formal aligned data products under `data/` unchanged.
- Delete clearly named smoke/tmp/interrupted local result outputs.
- Move legacy top-level `result/` outputs into an obsolete archive subfolder so
  old names cannot be mistaken for current work.
- Update harness defaults and docs that still point new experiment output at the
  former experiments path.

## Retention Rules

- Current result roots stay at `result/<run_name>/`.
- Obsolete but potentially useful outputs move to `result/_archived_obsolete/`.
- Invalid-for-QQTT controls stay under `result/_archived_invalid_for_qqtt/`.
- Saved-pair offline FFS screens stay under
  `data/ffs_benchmarks/_archived_saved_pair_offline/` unless those benchmark
  logs are explicitly moved in a separate pass.

## Validation

- `find data -maxdepth 2 -type d -name experiments`
- `rg -n "data/experiments|./data/experiments" . --glob '!result/**'`
  should only find this plan, the move report, and historical completed exec plans.
- `git diff --check`
- `python scripts/harness/check_all.py`
