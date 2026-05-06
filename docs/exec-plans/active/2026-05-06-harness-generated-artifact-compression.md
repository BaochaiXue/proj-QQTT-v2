# Harness Generated Artifact Compression

## Goal

Compress the growing `docs/generated/` harness engineering surface into a
small, operator-facing entrypoint without deleting historical validation
artifacts.

## Scope

- Add a compact generated-doc index for current harness engineering status.
- Add a machine-readable generated artifact inventory.
- Update `docs/generated/README.md` to point readers to the compact index.
- Preserve existing generated reports, JSON results, images, and logs.

## Non-goals

- Do not change live demo runtime behavior.
- Do not move or delete generated artifacts in this pass.
- Do not change formal recording/alignment code.
- Do not rewrite harness CLIs.

## Validation

- `python scripts/harness/check_all.py`

## Result

- Added `docs/generated/harness_engineering_compact_index.md`.
- Added `docs/generated/harness_engineering_artifact_inventory.json`.
- Compressed `docs/generated/README.md` into a short entrypoint.
- Linked the compact generated-artifact index from `scripts/harness/README.md`.
