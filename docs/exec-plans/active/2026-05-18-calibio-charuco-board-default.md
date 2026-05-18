# Calib.io ChArUco Board Default

## Goal

Make `cameras_calibrate.py` default to the new Calib.io ChArUco target used by the lab reference script: `12x9`, 30 mm checker size, 22 mm marker size, and ArUco `DICT_5X5_250`. Keep the previous `4x5 / 50mm / 37mm / DICT_4X4_50` board available but clearly deprecated.

## Plan

- Add explicit ChArUco board profile definitions under `qqtt/env/camera/`.
- Expose calibration-board CLI selection and low-level override flags in `cameras_calibrate.py`.
- Pass the selected board profile into `CameraSystem.calibrate()` and record board metadata in `calibrate_metadata.json`.
- Update calibration docs and deterministic smoke tests.

## Validation

- `python cameras_calibrate.py --help`
- `python -m py_compile cameras_calibrate.py qqtt/env/camera/camera_system.py qqtt/env/camera/calibration_boards.py tests/test_calibration_board_profiles.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_calibration_board_profiles`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
