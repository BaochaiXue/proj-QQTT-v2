# Enhanced PhysTwin Removed Overlay

Add a diagnostic visualization that projects points removed by the enhanced
PhysTwin-like postprocess back onto the source FFS RGB/depth views.

## Plan

1. Add a trace-capable enhanced postprocess helper that preserves the existing
   filtered point-cloud API while exposing kept, radius-removed, and
   component-removed masks over the original fused point order.
2. Add a static round workflow that builds FFS object-masked clouds with source
   camera and pixel metadata, runs the trace helper, and renders one `5x3`
   board per round:
   - row 1: RGB with object mask overlay
   - row 2: object-masked native depth
   - row 3: fused PCD with removed points highlighted by source camera
   - row 4: object-masked FFS depth with removed-point overlay by source camera
   - row 5: RGB with removed-point overlay by source camera
3. Add an experiment CLI under `scripts/harness/experiments/` with defaults for
   static round 1-3 frame 0 and `enhanced_component_voxel_size_m=0.01`.
4. Register the workflow shim, docs, quick/full check-all help command, and
   smoke tests.
5. Validate with targeted unittests, CLI help, experiment boundary check, and
   `python scripts/harness/check_all.py`.

## Notes

- Highlight scope defaults to all removed points: radius outliers plus
  component-filter removals.
- Removed-point color encodes source camera: Cam0 magenta, Cam1 cyan, Cam2
  amber.
- Removed points are drawn back onto their source camera pixels only; no
  cross-camera reprojection is performed.
- This is diagnostic visualization only and does not modify aligned case data.
