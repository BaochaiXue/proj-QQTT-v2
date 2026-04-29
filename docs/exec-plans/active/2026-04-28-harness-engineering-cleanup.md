# 2026-04-28 Harness Engineering Cleanup

## Goal

Reduce local harness clutter and make `scripts/harness/README.md` easier to scan without changing runtime behavior.

## Scope

- Remove generated Python cache artifacts under `scripts/harness/`.
- Compress the harness README into grouped command categories and short operational rules.
- Preserve existing user work around still-object case registry and FFS default benchmark rules.

## Non-Goals

- Do not move or rename public harness CLIs.
- Do not change recording, alignment, FFS, RealSense, or visualization behavior.
- Do not rewrite broad visualization workflows.

## Validation

- Run focused registry tests if the registry remains referenced.
- Run the deterministic quick harness profile:
  - `python scripts/harness/check_all.py`
