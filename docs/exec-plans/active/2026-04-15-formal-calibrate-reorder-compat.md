# 2026-04-15 Formal Calibrate Reorder Compatibility

## Summary
- Restore old-repo downstream compatibility for `data/different_types/*` by writing `calibrate.pkl` in case camera order, not raw calibration-reference order.
- Keep richer serial-remap semantics for repo-internal aligned cases outside `data/different_types/`.
- Backfill current formal cases so old downstream consumers that do `c2ws = pickle.load(...); c2ws[cam_idx]` read the correct camera extrinsics.

## Changes
- Update `record_data_align.py` to rewrite `calibrate.pkl` into `serial_numbers` order when exporting directly to `data/different_types/<case_name>/`.
- Add regression tests covering formal-export reorder and non-formal direct-copy behavior.
- Update docs to state that formal exports intentionally normalize `calibrate.pkl` to case camera order for downstream compatibility.
- Rewrite current formal cases' `calibrate.pkl` files using their raw-case serial mapping.

## Validation
- `python -m unittest -v tests.test_record_data_align_smoke`
- `python scripts/harness/check_all.py`
- Inspect current formal cases' `calibrate.pkl` against their raw-case serial mapping
