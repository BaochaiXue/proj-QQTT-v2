# 2026-04-08 Object-First Teddy Compare

## Goal

Fix the professor-facing teddy-bear compare by preserving small protrusions such as the head and ears before fusion/rendering.

## Current Failure Mode

- per-camera `max_points_per_camera` sampling still happens before the object is fully isolated
- `auto_object_bbox` still compresses toward the densest torso core
- sparse head/ear regions are lost before the final orbit render

## Plan

1. Build an object-first dense loading/splitting path for turntable compare.
2. Strengthen ROI selection with a union-oriented object component mode.
3. Preserve dense object points, sample only sparse context afterward.
4. Emit debug artifacts:
   - per-camera object mask overlays
   - per-camera object clouds
   - fused object-only clouds
   - fused object+context clouds
   - `compare_debug_metrics.json`
5. Add smoke tests for union bbox, object-first sampling, and debug artifact writing.
6. Re-render the teddy compare with head-inclusive ROI and rerun deterministic checks.
