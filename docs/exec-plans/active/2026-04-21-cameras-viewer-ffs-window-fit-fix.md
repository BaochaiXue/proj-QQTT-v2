# 2026-04-21 Cameras Viewer FFS Window Fit Fix

## Goal

Fix the live `cameras_viewer_FFS.py` display path so the Qt/OpenCV preview does not
collapse the multi-camera grid into a tiny thumbnail in the corner of a large window.

## Non-Goals

- no change to capture topology
- no change to FFS inference behavior
- no change to viewer CLI surface
- no TensorRT backend changes

## Files To Touch

- `cameras_viewer_FFS.py`
- `tests/test_cameras_viewer_ffs_smoke.py`
- this exec plan

## Implementation Plan

1. confirm the current FFS viewer uses a different grid-fit path than the working native viewer
2. replace the unstable window-rect bootstrap logic with the same screen-bounded fit helper
   already used by `cameras_viewer.py`
3. add a regression-oriented smoke test so the FFS viewer keeps using the stable fit path
4. run targeted tests and the deterministic repo checks

## Validation Plan

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_cameras_viewer_ffs_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`

## Risks

- the FFS viewer will no longer dynamically fit to a user-resized window every frame
- the fix should prioritize correct initial visibility over live resize responsiveness

## Completion Checklist

- [x] isolate the unstable window-fit path
- [x] patch the viewer
- [x] add regression coverage
- [x] run targeted tests
- [x] run deterministic checks
