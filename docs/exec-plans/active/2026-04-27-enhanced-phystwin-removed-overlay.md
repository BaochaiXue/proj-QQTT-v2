# Enhanced PhysTwin Removed Overlay

Add a diagnostic visualization that projects points removed by the enhanced
PhysTwin-like postprocess back onto the source FFS RGB/depth views.

## Plan

1. Add a trace-capable enhanced postprocess helper that preserves the existing
   filtered point-cloud API while exposing kept, radius-removed, and
   component-removed masks over the original fused point order.
2. Add a static round workflow that builds FFS object-masked clouds with source
   camera and pixel metadata, runs the trace helper, and renders one `4x3`
   board per round:
   - row 1: RGB with object mask overlay
   - row 2: fused PCD with removed points highlighted
   - row 3: object-masked FFS depth with removed-point overlay
   - row 4: RGB with removed-point overlay
3. Add an experiment CLI under `scripts/harness/experiments/` with defaults for
   static round 1-3 frame 0 and `enhanced_component_voxel_size_m=0.01`.
4. Register the workflow shim, docs, quick/full check-all help command, and
   smoke tests.
5. Validate with targeted unittests, CLI help, experiment boundary check, and
   `python scripts/harness/check_all.py`.

## Notes

- Highlight scope defaults to all removed points: radius outliers plus
  component-filter removals.
- Removed points are drawn back onto their source camera pixels only; no
  cross-camera reprojection is performed.
- This is diagnostic visualization only and does not modify aligned case data.
