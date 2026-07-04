# Test Suite Simplification

## Requirement

Problem:
The current tests have small but real maintenance drift: validation does not
include every top-level test module, the validation runner still has an empty
pytest path, and several Demo v5.1 tests repeat the same setup and assertions.

Required final behavior:
Keep existing behavioral coverage while making the validation manifest harder
to drift and reducing repeated test scaffolding in focused files.

Inputs:
Existing `unittest` modules under `tests/`, Demo v5.1 test helpers, and the
catalog-driven validation runner.

Outputs:
Updated validation runner/tests and slimmer local test helpers. No production
behavior change unless a focused test exposes a required bug fix.

State changes:
No data layout or runtime state changes are intended.

Invalid cases:
Missing validation module paths and unlisted top-level test modules must fail
in the validation manifest test.

Constraints:
Stay on `single-camera`. Do not revert existing uncommitted Demo v5.1 work.
Keep changes local to test organization and deterministic validation.

Unknowns:
`git pull --ff-only origin main` cannot fast-forward because `single-camera`
has diverged from `origin/main`; the branch is aligned with
`origin/single-camera`.

## Plan

- [x] Spawn read-only agents to inventory tests, validation profiles, and Demo
  v5.1 test duplication.
- [x] Remove empty pytest validation plumbing and include omitted top-level
  test modules in deterministic validation.
- [x] Add manifest coverage so new top-level tests cannot silently fall out of
  validation.
- [x] Consolidate repeated Demo v5.1 chunk-status and prewarm helper code
  without weakening assertions.
- [x] Run focused unit tests and smoke validation.

## Validation

- `PYTHONDONTWRITEBYTECODE=1 conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_validation_smoke_manifest tests.test_demo_v5_1_chunk_data tests.test_demo_v5_1_shape_prior_simplification`
  passed.
- `PYTHONDONTWRITEBYTECODE=1 conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
  passed.
