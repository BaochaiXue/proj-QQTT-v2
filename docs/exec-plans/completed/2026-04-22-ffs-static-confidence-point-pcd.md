# 2026-04-22 FFS Static Confidence Point-PCD Binding

## Goal

Update the static-round confidence PCD workflow so classifier-logit-derived
confidence is attached to each reconstructed 3D point and the third row renders
the same fused masked FFS point cloud colored by confidence instead of showing a
separate 2D confidence raster.

## Non-Goals

- no change to the existing `visualize_ffs_static_confidence_panels.py`
- no new confidence definition beyond the current `margin` and `max_softmax`
- no confidence-based filtering or support-count fusion in this change
- no change to the static-round case list, mask cache policy, or frame index

## Files To Touch

- `data_process/visualization/workflows/ffs_confidence_pcd_panels.py`
- `tests/test_ffs_confidence_pcd_panels_smoke.py`
- `docs/WORKFLOWS.md`
- `docs/ARCHITECTURE.md`
- `scripts/harness/visualize_ffs_static_confidence_pcd_panels.py`
- this exec plan

## Implementation Plan

1. sample color-aligned confidence at each point's `source_pixel_uv` after
   `depth_to_camera_points(...)`
2. store per-point confidence alongside each camera cloud and carry it into the
   fused masked cloud
3. render row 2 with the fused masked RGB-colored PCD and render row 3 with the
   exact same fused masked points, crop, and view configuration but with colors
   derived from the selected confidence metric
4. update workflow summaries and debug outputs so the row-3 render contract is
   explicit
5. update smoke tests to verify row-3 render calls use the fused masked PCD with
   confidence-derived colors
6. refresh workflow docs and rerun the static confidence PCD generation command

## Validation Plan

- `conda run -n qqtt-ffs-compat python -m unittest -v tests.test_ffs_confidence_pcd_panels_smoke tests.test_check_all_smoke`
- `conda run -n qqtt-ffs-compat python scripts/harness/check_all.py`
- `conda run -n qqtt-ffs-compat python scripts/harness/visualize_ffs_static_confidence_pcd_panels.py`

## Risks

- pointwise confidence is still only a proxy derived from FFS logits and is not
  a calibrated correctness probability
- sampling confidence after color-plane alignment can still smear values when
  multiple IR pixels collapse onto the same color pixel
