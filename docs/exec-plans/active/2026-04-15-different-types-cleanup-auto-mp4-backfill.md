# 2026-04-15 Different Types Cleanup Auto MP4 Backfill

## Summary
- Enhance `cleanup_different_types_cases.py` so execute mode auto-generates missing `color/0.mp4`, `1.mp4`, and `2.mp4` sidecars from aligned RGB PNG frames before final cleanup.
- Keep dry-run non-mutating while reporting which mp4 sidecars would be generated.
- Preserve current formal export contract otherwise.

## Changes
- Add ffmpeg-backed color video backfill helpers to the cleanup harness.
- Execute mode now:
  - validates the formal case contract
  - generates missing color mp4 sidecars when possible
  - then removes non-formal extras
- Dry-run mode reports planned mp4 generation without changing files.
- Update cleanup smoke tests to cover missing-mp4 backfill behavior.
- Update docs to state that cleanup can repair missing RGB sidecars.

## Validation
- `python -m unittest -v tests.test_cleanup_different_types_cases_smoke`
- `python scripts/harness/check_all.py`
