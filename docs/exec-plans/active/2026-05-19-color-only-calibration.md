# Color-Only Calibration

## Goal

Run ChArUco calibration without opening RealSense depth streams. Calibration
uses RGB images, color intrinsics, and color distortion coefficients only, so
depth should not constrain calibration stream profiles or consume USB bandwidth.

## Plan

- Add a color-only `CameraSystem` capture mode.
- Make `cameras_calibrate.py` instantiate `CameraSystem` with that mode.
- Document that calibration does not require depth.
- Add tests for the calibration entrypoint contract.

## Validation

- `python -m py_compile cameras_calibrate.py qqtt/env/camera/camera_system.py`
- targeted unit tests for calibration CLI/capture-mode contract
- hardware ChArUco preflight and calibration retry
