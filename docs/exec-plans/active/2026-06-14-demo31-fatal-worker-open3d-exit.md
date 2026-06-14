# Demo 3.1 Fatal Worker Exit

## Goal
Prevent the Open3D viewer from appearing frozen after a fatal background worker
failure, especially CUDA OOM during EdgeTAM segmentation.

## Plan
1. Add a small fatal-error state to the Demo 3.1 runtime.
2. Route capture, segmentation, tracker, and PCD fatal failures through that
   state instead of only printing and setting the stop event.
3. Wake the Open3D main thread when a fatal error is recorded, update the HUD,
   and quit the GUI cleanly.
4. Add deterministic tests for fatal-state recording and no-op duplicate fatal
   handling.
5. Run the targeted tests and `scripts/harness/check_all.py`.

## Validation
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_tapnextpp_overlay`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_recorded_rgbd_replay_source`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
