# 2026-04-14 Record Partial Stall Guard

## Goal

Make short bounded recordings fail quickly when one camera stops progressing but other cameras keep streaming, instead of hanging until manually interrupted.

## Non-Goals

- no change to successful recording semantics
- no change to calibration or alignment logic
- no attempt to auto-recover a missing/stalled camera

## Files To Touch

- `qqtt/env/camera/camera_system.py`
- deterministic smoke coverage for partial-stall behavior
- workflow / hardware validation docs

## Implementation Plan

1. keep the existing global no-progress timeout
2. add per-camera last-progress tracking during `record(max_frames=...)`
3. raise a targeted runtime error when any camera stays below the requested frame target past the stall timeout while others continue progressing
4. include lagging camera indexes / serials in the error message
5. add a fake-realsense smoke test that reproduces one stalled camera and verifies fast failure

## Validation Plan

- `python -m unittest -v tests.test_camera_system_partial_stall_smoke`
- `python scripts/harness/check_all.py`
