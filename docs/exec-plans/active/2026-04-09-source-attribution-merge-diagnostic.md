# 2026-04-09 Source Attribution Merge Diagnostic

## Goal

Expose 3-view merge quality honestly in the professor-facing compare by preserving per-point camera provenance and adding source-attribution and mismatch render modes.

## Current Failure Mode

- fused object/context clouds lose source identity as first-class render data
- current geom/rgb/support outputs do not show which camera contributed which surface
- support count shows overlap quantity but not source identity or double-surface fringing

## Plan

1. Preserve `source_camera_idx` and optional serial metadata through object/context/fused cloud construction.
2. Add source-attribution render paths:
   - semi-transparent overlay by source camera
   - split per-camera source contribution view
3. Add a mismatch/residual render and summary metrics.
4. Automatically emit source/mismatch videos, gifs, keyframe sheets, and legend/debug artifacts.
5. Add targeted smoke tests for source-id propagation, source overlay/split rendering, legend generation, and mismatch residual rendering.
6. Update workflow/docs and rerun `scripts/harness/check_all.py`.
