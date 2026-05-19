# Concise Check-All Test Lists

## Goal

Make `scripts/harness/check_all.py` easier to maintain by removing repeated
quick/full unittest module lists and keeping the generated validation process
concise.

## Scope

- Refactor check-all list definitions only.
- Preserve quick/full command behavior and deterministic validation coverage.
- Keep the already-removed stale Demo 2.1 test out of the harness.

## Plan

1. Split unittest modules into shared quick coverage and full-only additions.
2. Generate full coverage from shared + full-only lists with de-duplication.
3. Update `tests/test_check_all_smoke.py` to validate generated commands without
   repeating the full module list.
4. Run targeted check-all smoke tests and quick harness validation.

## Results

- Added shared quick unittest/help lists and full-only additions in
  `scripts/harness/check_all.py`.
- Generated the full unittest module list from quick + full-only modules with
  stable de-duplication.
- Updated `tests/test_check_all_smoke.py` so the test asserts generated command
  shape and invariants instead of duplicating the whole quick module list.
- Confirmed the stale Demo 2.1 fused-PCD smoke test is absent from the generated
  check-all command surface.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_check_all_smoke` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_engineering.py` passed.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py` passed.
