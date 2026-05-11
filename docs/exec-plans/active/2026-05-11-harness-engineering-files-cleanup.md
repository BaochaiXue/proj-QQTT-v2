# 2026-05-11 Harness Engineering Files Cleanup

## Goal

Make harness engineering files easier to maintain and less misleading by
keeping the live script list in `scripts/harness/_catalog.py`, keeping generated
harness claims synchronized with the latest local reports, and making validation
commands point at the documented default environment.

## Scope

- Compact `scripts/harness/README.md` so it references `_catalog.py` and `docs/generated/harness_engineering_compact_index.md` instead of duplicating long operational notes.
- Refresh `docs/generated/harness_engineering_compact_index.md` and
  `docs/generated/harness_engineering_artifact_inventory.json` from current
  tracked generated artifacts.
- Replace stale camera-only / bare-Python validation wording that no longer
  matches the sanctioned demo/proxy/tracking scope or the `demo_2_max` default
  environment.
- Preserve all public harness CLI paths and runtime behavior.

## Non-Goals

- Do not delete or rename harness CLIs.
- Do not change camera, FFS, SAM, EdgeTAM, TensorRT, or demo behavior.
- Do not modify unrelated dirty files.

## Validation

- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_harness_catalog.py`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
- `git diff --check`
