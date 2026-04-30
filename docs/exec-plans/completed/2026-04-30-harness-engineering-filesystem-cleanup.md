# 2026-04-30 Harness Engineering Filesystem Cleanup

## Goal

Compress and organize the harness engineering surface so operators can quickly see what belongs in `scripts/harness/`, while deterministic checks prevent new uncategorized harness scripts from accumulating.

## Scope

- Add a small harness catalog that groups public CLIs and support modules by purpose.
- Refactor `scripts/harness/check_all.py` to read help-script coverage from that catalog instead of maintaining another manual list.
- Add a deterministic catalog guard that verifies harness Python files are categorized.
- Compress `scripts/harness/README.md` into a shorter operating summary that points to the catalog categories.
- Preserve existing public CLI paths and current runtime behavior.

## Non-Goals

- Do not move or rename operator-facing harness CLIs in this pass.
- Do not change recording, alignment, RealSense, FFS, SAM, or visualization behavior.
- Do not delete generated validation history under `docs/generated/`.
- Do not alter user-local executable-bit changes already present on WSLg helper scripts.

## Validation

- `python scripts/harness/check_harness_catalog.py`: passed.
- `python scripts/harness/check_experiment_boundaries.py`: passed.
- `python scripts/harness/check_visual_architecture.py`: passed.
- `python -m unittest -v tests.test_check_all_smoke`: passed.
- `python scripts/harness/check_all.py`: failed under base Python 3.13 because `cameras_viewer.py --help` imports `cv2`, which is not installed in base.
- `conda run -n FFS-SAM-RS python scripts/harness/check_all.py`: passed, including 68 quick unittest cases.

## Result

- Added `scripts/harness/_catalog.py` as the compact source of truth for harness categories and help coverage.
- Added `scripts/harness/check_harness_catalog.py` to fail on uncategorized public harness Python files.
- Updated `scripts/harness/check_all.py` to derive harness help commands from the catalog and to run the catalog guard in quick/full profiles.
- Compressed `scripts/harness/README.md` from a long manual script table into a concise filesystem contract, catalog summary, and primary entrypoint map.
- Updated `tests/test_check_all_smoke.py` for the additional quick-profile catalog guard.
