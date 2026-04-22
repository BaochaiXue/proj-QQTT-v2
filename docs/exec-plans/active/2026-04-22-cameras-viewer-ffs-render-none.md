# 2026-04-22 Cameras Viewer FFS Render None

## Goal

Add a true `--render-mode none` path to `cameras_viewer_FFS.py` so the viewer can:

- skip panel assembly and `cv2.imshow()`
- skip IR-to-color depth reprojection in worker processes
- keep capture and worker-side FPS statistics for throughput profiling

## Scope

- add the new CLI and runtime mode to the live FFS viewer only
- preserve existing default behavior for panel rendering
- update smoke tests and operator docs

## Non-Goals

- no change to native `cameras_viewer.py`
- no change to aligned-case generation
- no change to model inference contracts

## Validation

- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
