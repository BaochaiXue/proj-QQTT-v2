# Balanced RGB Camera Controls

## Goal

Make calibration and recording use the same per-serial RealSense color controls
so the three current lab D455 RGB streams have comparable brightness by default.

## Plan

- Move the current lab rig exposure overrides into shared camera defaults.
- Add gain override support in `CameraSystem` without changing existing callers.
- Wire calibration, raw recording, realtime aligned recording, and viewer CLIs to
  the shared defaults.
- Record the validated per-serial settings in docs and tests.

## Validation

- `python -m py_compile cameras_calibrate.py record_data.py record_data_realtime_align.py cameras_viewer.py qqtt/env/camera/defaults.py qqtt/env/camera/camera_system.py`
- targeted unit tests for color-control defaults and parsers
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
