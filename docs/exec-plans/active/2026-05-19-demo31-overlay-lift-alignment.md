# Demo 3.1 overlay lift alignment

## Goal

Fix and instrument the Demo 3.1 CoTracker overlay lift path so 2D controller tracks are
converted to world points through the same projection-grid convention used by the semantic
PCD fusion path.

## Scope

- Keep `overlay_display_scope=controller`.
- Keep Demo 3.1 GPU0/GPU1 ownership and CPU-only IPC unchanged.
- Do not change CoTracker query semantics or the 4096/view batch default.
- Update focused tests for the render overlay path.

## Plan

1. Route `lift_tracks_yx_to_world()` through a ray-grid backprojection helper matching the
   semantic PCD path.
2. Add Demo 3.1 overlay diagnostics for per-camera lifted point counts and centroids.
3. Add an overlay-by-camera debug color option so live runs can identify which camera
   contributes a displaced overlay cluster.
4. Make the live lift mask match `overlay_display_scope`: controller overlays must
   lift only through current controller masks, not the broader union mask.
5. Run focused unit tests and a Demo 3.1 dry-run.
