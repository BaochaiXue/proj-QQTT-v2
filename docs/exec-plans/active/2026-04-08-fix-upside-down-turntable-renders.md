# Fix Upside-Down Turntable Renders

## Goal

Correct the shared screen-space projection convention so fused point-cloud renders and overview overlays are no longer vertically inverted.

## Root Cause

`look_at_view_matrix()` constructs a view basis with `true_up` pointing upward in view space, but `_project_view_coordinates()` currently maps positive view-space `y` to larger image-row indices. That flips the render vertically.

## Plan

1. Change the perspective and orthographic `v` formulas in `_project_view_coordinates()` to subtract from `height * 0.5`.
2. Audit image-flip helpers so the convention fix is not followed by an accidental second vertical flip.
3. Add regression tests for point projection and a tiny upright synthetic render.
4. Re-render the latest professor-facing geom/rgb/support turntable outputs and verify the overview inset matches the corrected orientation.
5. Run deterministic checks.
