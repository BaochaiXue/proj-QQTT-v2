# 2026-04-22 Cameras Viewer Threaded Capture

## Goal

Refactor `cameras_viewer.py` so native RGB-D preview uses one capture thread per
camera and a main-thread latest-frame renderer, matching the freshness-first
execution model already used by the FFS viewer.

## Non-Goals

- no change to camera startup profile fallback order
- no change to native depth colormap semantics or panel layout
- no new recording/export path
- no change to FFS viewer worker topology

## Files To Touch

- `cameras_viewer.py`
- `tests/test_cameras_viewer_fps_smoke.py`
- `docs/WORKFLOWS.md`
- this exec plan

## Implementation Plan

1. add native-viewer camera state helpers with a lock, latest-frame buffers, and
   per-camera capture sequence tracking
2. move `wait_for_frames -> align -> numpy materialization` into one background
   thread per active camera
3. keep the main thread focused on:
   - snapshotting the latest frame pair
   - updating measured display-side fresh-frame stats
   - panel rendering and window refresh
4. add smoke coverage for the new camera-state/render helpers
5. run targeted tests plus the full deterministic check harness

## Validation Plan

- `python -m unittest -v tests.test_cameras_viewer_fps_smoke`
- `python cameras_viewer.py --help`
- `python scripts/harness/check_all.py`

## Risks

- the overlay still reports display-side fresh-frame delivery, so capture and
  display can diverge when the renderer stays slower than the camera threads
- `pyrealsense2.align` is now used from the per-camera capture threads, so this
  pass depends on that path staying thread-safe for one pipeline per thread
