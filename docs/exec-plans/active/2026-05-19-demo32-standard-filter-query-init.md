# Demo 3.2 Standard-Filter Query Init

## Goal

Make Demo 3.2 initialize LiteTracker queries from geometry-filtered trackable masks instead of raw semantic masks.

## Code Reading Result

- The current fused/render filter outputs only `xyz/rgb`.
- `CameraLayerCloud`, `FusedLayerCloud`, `FilterOutput`, and `ObjectVolumeFilterOutput` do not preserve source pixels.
- Low-level helpers do expose survivor information:
  - `phystwin_volume_sample_indices_fast()` returns object survivor indices.
  - `apply_enhanced_phystwin_like_postprocess_with_trace()` returns kept masks.
  - `_detect_radius_outlier_indices()` is used by the controller pt-filter path.

## Plan

1. Add a Demo 3.2 trackable-mask builder that backprojects per-camera semantic pixels with source `yx`.
2. Run the same standard object/controller filter semantics in a source-preserving way.
3. Map survivor indices back into `object_trackable_mask`, `controller_trackable_mask`, and `union_trackable_mask`.
4. Before the tracking union and trackable-filter path, optionally erode the
   controller mask by `--controller-mask-erode-px`; the implicit default is
   `1` in `--mode demo` and `0` in `--mode exp`.
5. Apply `controller_trackable_max_points_per_camera` after filtering.
6. Feed LiteTracker only RGB + trackable masks. Do not send depth/intrinsics/c2w to the tracker child.
7. Keep the existing render/data filter path; the query-init filter selects tracker eligibility, while the later filter still controls rendered/final PCD size and cleanup.
8. Apply controller render-only voxel downsampling before Open3D display using
   `--controller-render-voxel-m`; this reduces the controller body PCD only and
   must not change LiteTracker query inputs or red tracking/control markers.

## Validation

- Add unit tests for invalid depth rejection and controller cap-after-filter.
- Update Demo 3.2 contract tests for standard-filter query-init fields.
- Run focused tests and `scripts/harness/check_all.py`.

## Follow-up: All Tracking Points As Anchors

The current live visualization should treat every visible, depth-valid
LiteTracker point as a 3D anchor/control marker for Demo 3.2. This is a
deliberate departure from the previous PhysTwin-style surface snap safety gate:

- add a Demo 3.2 default visualization mode that directly lifts all visible
  tracking points with depth/intrinsics/c2w
- do not apply surface-snap matching
- do not apply semantic bbox rejection
- do not apply semantic scope mask rejection during the lift
- render every lifted tracking point as a red 3D sphere marker
- keep depth validity/in-bounds checks, because a 2D point with no depth cannot
  be displayed in the 3D PCD

## Follow-up: Dynamic Warmup HUD

The Open3D warmup HUD must describe the actual runtime pipeline instead of a
hard-coded Demo 2.3 text. Demo 3.2 should mention LiteTracker query-init and 3D
anchors while Demo 2.3 should continue to describe capture, FFS, and EdgeTAM.
