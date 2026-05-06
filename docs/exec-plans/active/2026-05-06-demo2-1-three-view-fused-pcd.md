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

## Professor-Safe Runtime Slice

The first hardware smoke proved the three-camera object-only live skeleton at
`848x480@30` and `fusion-target-fps=2`. The next slice hardens the professor
demo entrypoint without changing Demo 2.0:

1. Add staged presets:
   - `professor-safe`: low-FPS, FFS-quality, controller-object default
   - `climb-5`: headless profiling at target 5 FPS
   - `climb-10`: headless profiling at target 10 FPS
   - `diagnostics`: explicit isolation surface
2. Add one shared GPU inference gate for FFS and EdgeTAM workers.
3. Keep FFS strict-latest with one shared runner/context owner.
4. Keep three per-camera EdgeTAM streaming sessions.
5. Record GPU gate waits and fusion completeness in debug/summary.
6. Keep object and controller separated:
   - object fused cloud -> enhanced-pt
   - controller fused cloud -> pt-filter
   - never union object/controller before filtering
