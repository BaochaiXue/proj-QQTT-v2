# Demo v2 Async PCD Filter

## Goal

Update Demo v2 masked EdgeTAM PCD filtering so the live hot path does not run expensive full-size point-cloud cleanup synchronously.

## Scope

- Add a shared fast PCD filter utility module under `demo_v2/`.
- Add voxel cap and voxel-density approximate filter helpers.
- Add an async latest-wins filter worker.
- Wire `demo_v2/realtime_masked_edgetam_pcd.py` with:
  - object filter default: `enhanced-pt`
  - controller/hand filter default: `pt-filter`
  - per-filter voxel cap before expensive cleanup
  - async/sync/none filter scheduling
  - `filter-every-n`
  - filter telemetry in HUD/debug output.
- Update README and smoke coverage.

## Non-Goals

- Do not change FFS TensorRT engine defaults.
- Do not move point-cloud filtering onto GPU.
- Do not make filter output part of a formal tracking/evaluation result.

## Validation

- `python -m py_compile demo_v2/pcd_filter_fast.py demo_v2/realtime_masked_edgetam_pcd.py`
- `python -m unittest -v tests.test_realtime_single_camera_pointcloud_smoke`
- `python scripts/harness/check_harness_catalog.py`

## Outcome

- Added `demo_v2/pcd_filter_fast.py` with voxel cap, voxel-density approximate filtering, a latest-wins async worker, and a soft budget cap controller.
- Wired `demo_v2/realtime_masked_edgetam_pcd.py` with async/sync/none filter scheduling, object `enhanced-pt`, controller `pt-filter`, per-object voxel caps, `filter-every-n`, and HUD/debug telemetry.
- Added Demo 2.1 fused-layer cap-before-cleanup support so fused object/controller layers do not feed full-size clouds directly into postprocess.
- Updated Demo v2 / Demo v2.1 README examples and smoke coverage.
- Validation passed in `FFS-SAM-RS`; base Python still cannot import repo OpenCV paths because it lacks `cv2`.
