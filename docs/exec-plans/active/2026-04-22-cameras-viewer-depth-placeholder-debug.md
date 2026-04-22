# 2026-04-22 Cameras Viewer Depth Placeholder Debug Mode

## Goal

Add a native viewer debug mode where the depth half of each panel is replaced by
an unrendered black placeholder that only reports the received depth FPS.

## Non-Goals

- no change to the default native viewer behavior
- no change to camera stream selection or fallback startup profiles
- no change to FFS viewer behavior
- no additional recording/export workflow

## Files To Touch

- `cameras_viewer.py`
- `tests/test_cameras_viewer_fps_smoke.py`
- `docs/WORKFLOWS.md`
- this exec plan

## Implementation Plan

1. add a switchable native viewer depth render mode with a default colormap path
2. implement a black placeholder bottom panel that renders only depth-FPS text
3. keep the existing top-panel RGB and overlay labels unchanged
4. add smoke coverage for the placeholder formatting and render-path selection
5. run targeted tests plus the full deterministic check harness

## Validation Plan

- `python -m unittest -v tests.test_cameras_viewer_fps_smoke`
- `python cameras_viewer.py --help`
- `python scripts/harness/check_all.py`

## Risks

- the placeholder mode reports the viewer's fresh-frame rate for the latest
  received RGB-D pair, not a separate sensor-side depth counter
- operators may mistake the placeholder mode for a broken depth renderer unless
  the CLI flag is used intentionally and documented
