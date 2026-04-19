# 2026-04-19 Cameras Viewer FPS Overlay

## Goal

Add an in-panel FPS overlay to `cameras_viewer.py` so each camera panel shows both:

- the negotiated stream fps that actually started
- a measured per-camera fps based on recent valid color+depth deliveries

## Non-Goals

- no global viewer-loop fps
- no change to startup profile fallback order
- no change to depth colormap or panel tiling

## Files To Touch

- `cameras_viewer.py`
- new viewer-focused unit tests
- `scripts/harness/check_all.py`
- viewer workflow / hardware-validation docs

## Implementation Plan

1. add pure helper functions for:
   - pruning a 1-second rolling timestamp window
   - computing measured fps from that window
   - formatting a deterministic two-line overlay label
2. extend per-camera runtime state to keep recent valid-frame timestamps and the latest rendered data needed to rebuild the panel label every loop
3. update panel rendering so each panel shows:
   - line 1: serial / usb / negotiated resolution and fps
   - line 2: configured vs measured fps
4. add targeted tests for the rolling-fps and label-format helpers
5. update deterministic checks and minimal docs

## Validation Plan

- `python cameras_viewer.py --help`
- `python -m unittest -v tests.test_cameras_viewer_fps_smoke`
- `python scripts/harness/check_all.py`
