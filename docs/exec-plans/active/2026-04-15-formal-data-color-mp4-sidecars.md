# 2026-04-15 Formal Data Color MP4 Sidecars

## Summary
- Preserve downstream-required `color/0.mp4`, `color/1.mp4`, and `color/2.mp4` sidecars in `data/different_types/*`.
- Keep the formal export contract minimal otherwise: `color/`, `depth/`, `calibrate.pkl`, `metadata.json`, plus optional color mp4 sidecars.
- Backfill color mp4 sidecars for current formal cases after the cleanup-script fix.

## Changes
- Update `cleanup_different_types_cases.py` so it preserves `color/<camera>.mp4` files when present.
- Add regression coverage for preserving color mp4 sidecars during execute mode.
- Update docs to reflect the optional color mp4 sidecars in downstream-facing formal exports.
- Backfill `color/0.mp4`, `1.mp4`, `2.mp4` for existing formal cases with ffmpeg.

## Validation
- `python -m unittest -v tests.test_cleanup_different_types_cases_smoke`
- `python scripts/harness/check_all.py`
- Verify current formal cases contain `color/0.mp4`, `1.mp4`, `2.mp4`
