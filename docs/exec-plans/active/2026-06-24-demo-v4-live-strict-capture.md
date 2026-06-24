# Demo v4 Live Strict Capture

## Goal

Allow Demo v4 to launch Demo 3.2 strict PhysTwin-like headless capture from a
live RealSense input, while keeping panel/live preview restrictions unchanged.

## Steps

- [x] Add failing tests for Demo 3.2 wrapper live strict validation/delegate.
- [x] Add failing tests for lower-level masked Demo 3.2 live headless strict
  validation.
- [x] Add failing tests for Demo v4 live command generation and mocked live launch.
- [x] Relax strict/headless validation to accept live or fake-live, but continue
  rejecting recording and panel modes.
- [x] Make Demo v4 command generation pass `--track-mode controller-object` and
  `--tracker-backend tapnextpp` explicitly.
- [x] Run targeted unittest suites and smoke validation.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_single_demo_v3_runtime tests.test_realtime_single_camera_pointcloud_smoke tests.test_demo_v4_futurephystwin_chunks`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- `git diff --check`
