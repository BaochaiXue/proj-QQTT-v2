# Demo 3.1 Tracking Control Point Markers

## Goal

Make Demo 3.1 visualize PhysTwin-style 3D tracking control points anchored to
the current fused semantic surface instead of appending raw 2D-to-3D lifted
CoTracker overlay points into the controller point cloud.

## Reference

FuturePhysTwin stores final controller control points in
`track_data["controller_points"]`, samples sparse valid controller points, and
visualizes each one as a 3D handle over its reconstructed geometry. Demo 3.1
should keep dense online tracking internally, but its live 3D marker must be
snapped to the current fused PCD surface so tracker/camera/depth mistakes cannot
create floating detached overlay blobs.

## Plan

- Add `--tracker-visualization-mode none|3d-surface-markers|2d-debug|legacy-3d-lift`,
  defaulting to `3d-surface-markers`.
- Build same-group surface anchor snapshots keyed by camera and semantic label
  with source pixel `yx` and existing fused surface `points_world`.
- In the default path, reject non-exact render/anchor matches and snap visible
  controller tracking points to the nearest same-camera controller anchor within
  `--tracker-3d-snap-radius-px`.
- Render accepted controls as small PhysTwin-style red 3D sphere markers in the controller
  layer, default 16 controls per camera and 6mm marker radius.
- Keep direct `lift_tracks_yx_to_world` only for `legacy-3d-lift` debug mode.
- Add profile fields for surface cache hit, accepted/rejected marker counts,
  pixel snap error, marker radius, and legacy-lift usage.
- Add focused tests and run quick deterministic checks.

## Validation

- Focused unittest coverage for Demo 3.1 overlay/control-point rendering.
- Demo 3.1 dry-run contract includes control-point marker fields.
- Surface marker tests prove legacy lift is not called and out-of-radius tracks
  are rejected.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.
