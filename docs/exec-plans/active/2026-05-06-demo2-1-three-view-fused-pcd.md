# Demo 2.1 Three-View Fused Masked PCD

## Goal

Clone the Demo 2 realtime masked PCD surface into a Demo 2.1 namespace for a three-camera fused point-cloud demo. The new path should support:

- object-only tracking
- controller + object tracking
- three-view masked point-cloud fusion
- FFS-derived depth as the official quality depth source
- HF EdgeTAM streaming with `vision-reduce-overhead`

## Constraints

- Do not change Demo 2 local professor baseline defaults.
- Keep object and controller as separate semantic fused layers.
- Do not run enhanced PhysTwin cleanup over a combined object+controller cloud.
- Default fused-layer postprocess:
  - object: `enhanced-pt`
  - controller: `pt-filter`
- Native RealSense depth remains fallback/debug only, not official Demo 2.1 quality output.

## Implementation Slice

1. Add `demo_v2_1/` with a dedicated CLI surface and README. Done.
2. Add helper functions for semantic layer selection, postprocess policy, and per-label three-camera fusion. Done.
3. Add a smoke test to lock the object/controller policy and label-preserving fusion contract. Done.
4. Register the new demo help command in deterministic validation. Done.
5. Add live runtime workers:
   - `CaptureGroup` builder for cam0/cam1/cam2.
   - one shared FFS worker with one runner owner.
   - three per-camera EdgeTAM streaming workers.
   - strict `group_id` fusion.
   - latest-only Open3D fused render. Done.

## Follow-Up Slice

Run and profile the full hardware live loop:

- 3-camera RealSense capture
- per-camera HF EdgeTAM session
- shared-worker FFS depth
- calibration transform into the shared world frame
- fused Open3D render of object/controller layers
