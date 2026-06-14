# Record Data Startup Timeout

## Problem
`record_data.py` can appear to hang before recording starts when a RealSense
worker never reaches its first frame. The current startup path waits on
`ready_event` without a timeout and can also mark the worker ready while exiting
through `finally`, causing the parent process to block later while waiting for
metadata.

## Plan
1. Add startup timeout handling to `SingleRealsense.start_wait()`.
2. Preserve a worker startup error message for the parent process.
3. Add timeout-aware startup orchestration to `MultiRealsense`.
4. Expose `--camera-start-timeout-s` from `record_data.py` and pass it through
   `CameraSystem`.
5. Add deterministic tests for startup timeout cleanup and error propagation.
6. Run targeted tests and the quick harness.

## Validation
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_realsense_recovery_smoke tests.test_multi_realsense_order_smoke tests.test_record_data_preflight_message_smoke tests.test_camera_system_partial_stall_smoke`
- `conda run -n demo_2_max --no-capture-output python record_data.py --help | rg -n "camera-start-timeout|capture_mode|max_frames" -C 2`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
