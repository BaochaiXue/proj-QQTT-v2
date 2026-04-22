# 2026-04-22 Cameras Viewer Display-Fit Cache

## Goal

Remove the native viewer's per-frame screen-size probe so `cameras_viewer.py`
stops paying the `tkinter` startup/destruction cost on every display refresh.

## Non-Goals

- no change to camera stream selection or fallback startup profiles
- no change to preview panel layout or depth colormap semantics
- no threading refactor for the native viewer in this pass
- no change to FFS worker topology or recording/alignment behavior

## Files To Touch

- `cameras_viewer.py`
- `tests/test_cameras_viewer_fps_smoke.py`
- `docs/WORKFLOWS.md`
- this exec plan

## Implementation Plan

1. split the screen probe into a dedicated helper and cache its result
2. add a pure helper that computes the display target size from grid dimensions
   plus the cached screen bounds
3. resolve the native viewer's display target size once after camera startup and
   reuse it inside the render loop
4. add smoke coverage for the cache and deterministic display-size math
5. update preview docs to note that the screen fit is resolved once at startup

## Validation Plan

- `python -m unittest -v tests.test_cameras_viewer_fps_smoke`
- `python scripts/harness/check_all.py`

## Risks

- startup-time screen bounds will no longer auto-refresh if the operator moves
  the viewer to a different monitor mid-session
- the fixed display target size assumes the active camera count stays constant,
  which matches the current viewer lifecycle
