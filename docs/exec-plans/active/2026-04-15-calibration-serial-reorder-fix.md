# 2026-04-15 Calibration Serial Reorder Fix

## Summary
- Fix aligned-case calibration loading so `calibrate.pkl` is reordered by `metadata["calibration_reference_serials"]` whenever that metadata is present and differs from `metadata["serial_numbers"]`.
- Preserve current direct behavior only when serial order already matches or when no calibration reference serials are available.
- Re-export the current formal `frame 0` fused point cloud after the fix.

## Changes
- Update `data_process/visualization/calibration_io.py` reorder semantics for same-length serial lists with differing order.
- Add regression tests covering same-length reorder and mapping-mode inference.
- Update architecture docs to clarify that any aligned case with differing serial order must remap through `calibration_reference_serials`, not only subset captures.

## Validation
- `python -m unittest -v tests.test_calibrate_loader_smoke`
- `python -m unittest -v tests.test_calibration_contract_hardening`
- `python scripts/harness/check_all.py`
- Re-render `result/ply_compare_triplet_sloth_base_motion_native_vs_sloth_base_motion_ffs_frame_0000`
