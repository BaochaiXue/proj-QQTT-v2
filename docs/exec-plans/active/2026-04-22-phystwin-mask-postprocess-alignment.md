# 2026-04-22 PhysTwin Mask Postprocess Alignment

## Goal

Modify the masked camera-view comparison workflow so its optional postprocess path matches PhysTwin `data_process/data_process_mask.py` semantics:

- start from existing per-camera object masks
- build masked object-only point clouds across cameras
- run Open3D `remove_radius_outlier(nb_points=40, radius=0.01)` on the fused masked cloud
- map rejected 3D points back to the originating 2D mask pixels
- re-render masked RGB and masked PCD boards using the refined masks under fixed original camera viewpoints

## Non-Goals

- no change to canonical aligned case layout or record/align outputs
- no change to the existing aligned `depth_ffs_native_like_postprocess*` auxiliary streams
- no attempt to import PhysTwin as a runtime dependency
- no Meshlab/manual viewpoint path

## Files To Touch

- `data_process/visualization/io_case.py`
- `data_process/visualization/workflows/masked_pointcloud_compare.py`
- `data_process/visualization/workflows/masked_camera_view_compare.py`
- `scripts/harness/visual_compare_masked_camera_views.py`
- `tests/test_masked_camera_view_compare_smoke.py`
- docs describing masked camera-view postprocess semantics

## Implementation Plan

1. extend per-camera cloud loading so each point keeps aligned `source_pixel_uv`
2. add a PhysTwin-aligned mask refinement helper that:
   - filters fused masked clouds with `remove_radius_outlier(nb_points=40, radius=0.01)`
   - clears rejected source pixels from the per-camera masks
   - reports per-camera and fused refinement metrics
3. switch masked camera-view compare postprocess flags to use the new mask-refinement helper for both `Native` and `FFS`
4. update workflow summaries, CLI wording, tests, and docs to reflect the new semantics

## Validation Plan

- `python -m unittest -v tests.test_masked_camera_view_compare_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- re-render static round1-3 masked camera-view compare outputs
