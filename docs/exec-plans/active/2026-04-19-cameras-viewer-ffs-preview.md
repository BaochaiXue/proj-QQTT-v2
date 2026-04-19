# 2026-04-19 Cameras Viewer FFS Preview

## Goal

Turn `cameras_viewer_FFS.py` into a real live preview that shows RGB on top and color-aligned Fast-FoundationStereo depth on bottom for the active D455 cameras.

## Non-Goals

- no strict cross-camera synchronization barrier
- no frame-complete backlog processing
- no recording or export path
- no native depth display in this viewer

## Files To Touch

- `cameras_viewer_FFS.py`
- new FFS viewer smoke tests
- deterministic check wiring
- preview / hardware-validation docs

## Implementation Plan

1. replace native depth capture with `color + ir_left + ir_right`
2. keep direct RealSense startup / fallback behavior from the existing viewer
3. add one capture thread and one FFS worker process per camera
4. use latest-only queue behavior between capture and FFS
5. reproject FFS IR-left depth into the color frame before display
6. render per-camera overlays with negotiated profile, capture fps, and FFS fps
7. add software-only tests for queue policy, label formatting, rolling fps, and reprojection

## Validation Plan

- `python cameras_viewer_FFS.py --help`
- `python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `python scripts/harness/check_all.py`
