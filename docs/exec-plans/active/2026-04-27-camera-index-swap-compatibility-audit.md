# 2026-04-27 Camera Index Swap Compatibility Audit

## Goal

Audit and harden camera index / serial-number handling so swapping physical
RealSense devices does not silently pair the wrong color/depth stream,
intrinsics, or calibration transform with a logical camera index.

## Non-Goals

- no change to the repo's 3-camera D400 scope
- no redefinition of existing aligned case layout under `color/<idx>/` and
  `depth/<idx>/`
- no hardware validation in CI

## Planned Actions

1. Search for hard-coded `0,1,2`, `range(3)`, `num-cam=3`, and serial-order
   assumptions across runtime, alignment, visualization, and tests.
2. Classify each finding as:
   - safe fixed 3-camera contract
   - test-only fixture
   - risky camera-order assumption
3. Fix risky code by deriving camera ids from metadata / serial lists and by
   validating calibration coverage against serial numbers before writing or
   rendering outputs.
4. Add deterministic tests for swapped serial order and mismatched calibration
   order.
5. Update docs with the compatibility contract.
6. Run targeted tests plus `python scripts/harness/check_all.py`.

## Validation

- `conda activate qqtt-ffs-compat`
- `python -m unittest -v tests.test_aligned_metadata_loader_smoke tests.test_record_data_align_smoke tests.test_calibrate_loader_smoke`
- `python scripts/harness/check_all.py`

## Outcome

- Hardened `MultiRealsense` and `CameraSystem` so logical camera iteration
  follows `serial_numbers` explicitly, and duplicate requested serials fail
  before capture starts.
- Added aligned metadata validation for duplicate serials, per-camera list
  length mismatches, and incomplete `calibration_reference_serials`.
- Documented the camera identity contract and the difference between USB-port
  swaps and physical rig-position swaps.
- Validation passed:
  - `python -m unittest -v tests.test_aligned_metadata_loader_smoke tests.test_calibrate_loader_smoke tests.test_record_data_align_smoke`
  - `python scripts/harness/check_all.py`

## Physical Swap Follow-Up

- Added `calibrate_metadata.json` sidecar generation so new `calibrate.pkl`
  files carry their calibration reference serial order.
- `cameras_calibrate.py` accepts `--serials` when the operator wants an
  explicit logical calibration order.
- `record_data.py` and `record_data_realtime_align.py` now prefer the
  sidecar's serial order when writing / normalizing calibration references.
- Missing sidecars keep the legacy fallback but warn operators to recalibrate
  after any physical camera-position swap.
- `cameras_viewer.py` and `cameras_viewer_FFS.py` now use sorted serial order
  by default and accept `--serials` for explicit preview order, matching the
  recording/calibration logical camera contract.
- Validation passed:
  - `python -m unittest -v tests.test_calibration_metadata_smoke tests.test_record_data_preflight_message_smoke tests.test_record_data_realtime_align_smoke tests.test_recording_metadata_schema_v2 tests.test_multi_realsense_order_smoke tests.test_aligned_metadata_loader_smoke tests.test_check_all_smoke`
  - `python scripts/harness/check_all.py`
