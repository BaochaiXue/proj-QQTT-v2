# Local Smoke Artifact Cleanup

## Goal

Remove local smoke-test outputs and generated caches so they do not look like
current benchmark results.

## Scope

- Delete local generated smoke reports under `docs/generated/`.
- Delete local smoke result roots under `result/`.
- Delete smoke-related Python bytecode under `tests/__pycache__/`.
- Do not delete `tests/test_*_smoke.py`; those are deterministic source tests
  and remain part of the harness contract.
- Do not rewrite historical docs that mention smoke tests unless they point at
  a deleted current artifact as a retained result.

## Validation

- Record deleted artifact names under `docs/generated/`.
- Run `git diff --check`.
- Run `python scripts/harness/check_all.py` in the repo validation env.

## Outcome

- Removed HF EdgeTAM streaming smoke generated docs.
- Removed HF EdgeTAM streaming smoke result roots.
- Removed smoke-named `.pyc` files under `tests/__pycache__/`.
- Preserved `tests/test_*_smoke.py` source tests.
- Added `docs/generated/local_ephemeral_artifact_cleanup_20260503.md`.
