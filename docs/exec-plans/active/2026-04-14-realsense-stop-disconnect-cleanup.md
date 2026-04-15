# 2026-04-14 RealSense Stop Disconnect Cleanup

## Goal

Make `record_data.py` shut down cleanly when a RealSense worker sees a disconnect during or immediately after stop, instead of emitting a secondary traceback from inactive-pipeline recovery code.

## Non-Goals

- no change to successful recording data layout
- no change to alignment or visualization behavior
- no attempt to hide genuine mid-record disconnect failures

## Files To Touch

- `qqtt/env/camera/realsense/single_realsense.py`
- deterministic smoke coverage for the stop-time disconnect path
- `docs/WORKFLOWS.md`

## Implementation Plan

1. extract a small helper for `wait_for_frames()` runtime-error handling
2. if `stop_event` is already set, exit the worker loop instead of trying to recover
3. during genuine recovery, guard `get_active_profile()`, `hardware_reset()`, and `pipeline.stop()` so a failed lookup does not trigger a second exception
4. stop the pipeline safely during worker teardown
5. add a smoke test covering the stop-requested disconnect path

## Validation Plan

- `python -m unittest -v tests.test_single_realsense_recovery_smoke`
- `python -m unittest -v tests.test_camera_system_partial_stall_smoke`
- `python scripts/harness/check_all.py`
