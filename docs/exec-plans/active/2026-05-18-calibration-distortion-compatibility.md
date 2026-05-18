# Calibration Distortion Compatibility Hardening

## Goal

Harden the current Calib.io ChArUco calibration path without changing the `calibrate.pkl`
compatibility contract. `calibrate.pkl` must remain a list of camera-to-world `4x4`
transforms ordered by calibration serials so existing QQTT and future PhysTwin readers
continue to work.

## Plan

- Preserve RealSense stream distortion metadata for color and IR streams.
- Use color distortion coefficients during ChArUco pose estimation and reprojection checks.
- Record calibration world-frame convention, distortion usage, and corner counts in
  `calibrate_metadata.json`.
- Add an explicit calibration world-frame option for the native OpenCV board frame and the
  Robopil Rx180 converted frame.
- Add a converter for Robopil/yfang `cam_params.pkl` dictionaries into QQTT-compatible
  `calibrate.pkl` plus sidecar metadata.
- Update docs and deterministic tests around the compatibility contract.

## Validation

- `python cameras_calibrate.py --help`
- `python scripts/convert_robopil_cam_params_to_qqtt_calibrate.py --help`
- `python -m py_compile cameras_calibrate.py qqtt/env/camera/camera_system.py qqtt/env/camera/realsense/single_realsense.py scripts/convert_robopil_cam_params_to_qqtt_calibrate.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_calibration_board_profiles tests.test_calibration_metadata_smoke tests.test_recording_metadata_schema_v2 tests.test_robopil_calibration_converter`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
