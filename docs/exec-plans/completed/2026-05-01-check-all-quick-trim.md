# 2026-05-01 check_all Quick Trim

## Goal

Make the default `scripts/harness/check_all.py` quick profile faster by removing render-heavy visualization smoke tests from the default path while keeping them in `--full`.

## Plan

1. Keep quick help checks and deterministic scope / architecture / experiment-boundary guards.
2. Remove CLI visualization smoke tests that render temporary outputs from `QUICK_UNITTEST_BATCHES`.
3. Keep those visualization smoke tests covered by `FULL_UNITTEST_MODULES`.
4. Update `tests/test_check_all_smoke.py` so the quick/full contract is explicit.
5. Update harness docs to state that render-heavy visualization smoke tests run under `--full`.
6. Run focused tests and the quick check profile.

## Validation

- `python -m unittest -v tests.test_check_all_smoke`
- `python scripts/harness/check_all.py`

## Outcome

- Removed these render-heavy modules from the default quick unittest batch:
  - `tests.test_visual_compare_depth_panels_smoke`
  - `tests.test_visual_compare_reprojection_smoke`
  - `tests.test_visual_compare_turntable_smoke`
- Kept those modules in the `--full` unittest surface.
- Updated `tests/test_check_all_smoke.py` to assert quick excludes them and full includes them.
- Documented the quick/full split in `scripts/harness/README.md`.
- Validation passed:
  - `python -m unittest -v tests.test_check_all_smoke`
  - `conda run --no-capture-output -n SAM21-max python scripts/harness/check_all.py`

## Completion

Move this plan to `docs/exec-plans/completed/` after validation passes.
