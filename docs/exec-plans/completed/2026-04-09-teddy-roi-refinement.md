# 2026-04-09 Teddy ROI Refinement

## Goal

Fix the latest professor-facing teddy-bear compare by correcting the ROI / component / pixel-mask refinement path rather than continuing to tune orbit or renderer parameters.

## Current Failure Mode

- pass1 `auto_object_bbox` still starts from fused world-space points, so sparse teddy head / ear regions can be under-represented before the final compare is built
- current `union` selection is still anchored to the top component instead of using a transitive graph-style closure
- per-camera pixel masks exist, but world ROI still dominates too early instead of being refined and expanded by pixel-derived object evidence

## Plan

1. Add an automatic pass1 -> pass2 object ROI refinement path for `auto_object_bbox`.
2. Project coarse world ROI into each real camera to derive automatic per-camera 2D boxes and masks when manual JSON is absent.
3. Rebuild the refined world ROI from the union of pixel-derived per-camera object points.
4. Replace the top-component anchored union heuristic with a graph-union mode and make it the professor-facing default.
5. Emit explicit pass1/pass2 ROI artifacts, projected per-camera bbox artifacts, and richer compare debug metrics.
6. Add targeted smoke tests for projected-bbox refinement, graph-union protrusion preservation, pixel-mask-driven world ROI expansion, and refinement debug metrics.
7. Re-render the teddy compare on the latest case and rerun `scripts/harness/check_all.py`.
