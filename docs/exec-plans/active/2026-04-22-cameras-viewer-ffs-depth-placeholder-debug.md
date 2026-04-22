# 2026-04-22 Cameras Viewer FFS Depth Placeholder Debug Mode

## Goal

Add an FFS viewer debug mode where the lower half of each panel is replaced by a
black placeholder box that reports the received FFS FPS instead of rendering the
latest aligned FFS depth colormap.

## Non-Goals

- no change to the default FFS viewer behavior
- no change to FFS worker topology, queue policy, or backend selection
- no change to native viewer behavior in this pass
- no new recording or export path

## Files To Touch

- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- `docs/WORKFLOWS.md`
- this exec plan

## Implementation Plan

1. add a switchable lower-panel render mode for the FFS viewer
2. implement a black placeholder bottom panel that reports `ffs fps`
3. keep the existing RGB top panel and overlay labels unchanged
4. add smoke coverage for placeholder formatting, render-path selection, and CLI parsing
5. run targeted tests plus the full deterministic check harness

## Validation Plan

- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python cameras_viewer_FFS.py --help`
- `python scripts/harness/check_all.py`

## Risks

- operators may confuse the placeholder mode with a broken FFS depth renderer if
  they miss the CLI flag
- placeholder mode still reports viewer-side fresh-result FPS rather than a
  deeper worker-internal counter
