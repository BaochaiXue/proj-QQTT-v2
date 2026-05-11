# 2026-05-11 Harness Engineering Files Cleanup

## Goal

Make harness engineering files easier to maintain by keeping the live script list in `scripts/harness/_catalog.py`, reducing `scripts/harness/README.md` back to a compact operator map, and removing generated cache artifacts.

## Scope

- Remove local `__pycache__/` directories under `scripts/harness/`.
- Compact `scripts/harness/README.md` so it references `_catalog.py` and `docs/generated/harness_engineering_compact_index.md` instead of duplicating long operational notes.
- Preserve all public harness CLI paths and runtime behavior.

## Non-Goals

- Do not delete or rename harness CLIs.
- Do not change camera, FFS, SAM, EdgeTAM, TensorRT, or demo behavior.
- Do not modify unrelated dirty files.

## Validation

- `python scripts/harness/check_harness_catalog.py`
- `python scripts/harness/check_all.py`
