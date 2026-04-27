# Harness Engineering Prune

## Goal

Reduce harness clutter without changing formal camera recording, alignment, or
aligned-case output contracts.

## Scope

- Remove local Python cache artifacts under harness and adjacent repo packages.
- Remove obsolete root-level experiment compatibility CLIs now superseded by
  `scripts/harness/experiments/`.
- Update harness boundary checks so experiment CLIs have one canonical location.
- Update harness docs to make the cleaned layout explicit.

## Non-Goals

- No changes to formal recording or alignment behavior.
- No changes to aligned case directory layout or metadata fields.
- No removal of current user-facing native-vs-FFS comparison CLIs.
- No external dependency or hardware validation work.

## Validation

- `python scripts/harness/check_experiment_boundaries.py`
- `python scripts/harness/check_all.py`

