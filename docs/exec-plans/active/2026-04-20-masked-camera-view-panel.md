# 2026-04-20 Masked Camera-View Panel

## Goal

Add an independent single-frame masked point-cloud compare workflow that renders:

- rows = `Native`, `FFS`
- columns = the 3 original calibrated camera viewpoints

Each panel must use the same fixed camera extrinsic for both rows of that column so
`Native` and `FFS` can be judged under the exact same original camera view.

## Non-Goals

- no change to the existing `2x2` masked oblique workflow
- no PhysTwin dependency
- no mutation of aligned case `depth/` outputs
- no live-viewer integration
- no fallback renderer path for this workflow

## Files To Touch

- new `data_process/visualization/workflows/masked_camera_view_compare.py`
- new `scripts/harness/visual_compare_masked_camera_views.py`
- `scripts/harness/check_all.py`
- `tests/visualization_test_utils.py` only if new fixtures are needed
- new smoke test for the camera-view masked workflow
- workflow / architecture / harness docs

## Implementation Plan

1. Reuse the existing masked point-cloud loading and SAM sidecar resolution path.
2. Add a workflow that:
   - loads masked `Native` and `FFS` per-camera clouds
   - computes one shared object crop from the masked union
   - derives 3 strict view configs from real `c2w` camera extrinsics:
     - exact camera position
     - exact forward
     - exact up
   - renders a `2x3` board with Open3D hidden-window rendering only
3. Add a CLI that reuses the existing SAM sidecar resolution / generation logic.
4. Add deterministic tests for:
   - exact camera-pose view derivation
   - `2x3` output contract
   - shared crop / fixed per-column view reuse across `Native` and `FFS`
5. Wire the CLI and tests into deterministic checks.
6. Render one real static-case result.

## Validation Plan

- `python scripts/harness/visual_compare_masked_camera_views.py --help`
- `python -m unittest -v tests.test_masked_camera_view_compare_smoke`
- `python scripts/harness/check_all.py`
- real render on `static/native_30_static_round3_20260414` vs `static/ffs_30_static_round3_20260414`
