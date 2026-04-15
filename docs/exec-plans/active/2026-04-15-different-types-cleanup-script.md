# 2026-04-15 Different Types Cleanup Script

## Goal

Add a user-facing cleanup CLI that rewrites `data/different_types/<case_name>/` into the minimal downstream-facing structure with only:

- `color/0|1|2`
- `depth/0|1|2`
- `calibrate.pkl`
- `metadata.json`

## Non-Goals

- no metadata merging
- no backup/export-copy mode in v1
- no promise that cleaned cases remain compatible with repo-internal FFS comparison workflows

## Files To Touch

- new cleanup CLI under `scripts/harness/`
- deterministic smoke coverage
- workflow / harness / architecture docs
- `check_all.py`

## Implementation Plan

1. add a cleanup module/CLI that inspects one or more cases under `data/different_types`
2. validate required retained entries before mutation
3. default to `dry-run`, emitting a structured summary of kept/deleted/error items
4. delete all non-contract top-level entries plus non-`0|1|2` camera subdirs inside `color/` and `depth/`
5. keep `metadata.json` byte-for-byte and delete `metadata_ext.json`
6. add smoke tests for dry-run, execute, and selected-case-only behavior
7. wire `--help` and the new tests into deterministic checks

## Validation Plan

- `python scripts/harness/cleanup_different_types_cases.py --help`
- `python -m unittest -v tests.test_cleanup_different_types_cases_smoke`
- `python scripts/harness/check_all.py`
