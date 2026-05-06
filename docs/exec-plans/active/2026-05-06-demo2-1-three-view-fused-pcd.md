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
   - `professor-safe`: low-FPS, FFS-quality default controller-object demo; current no-hand runs explicitly pass `--track-mode object-only`
   - `visual-5fps`: WSLg/Open3D 5 FPS candidate using the unchanged FFS/enhanced-PT quality path and `gpu_gate_max_concurrent=2`
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

## Five FPS Visualization Slice

Goal: reach a 5 FPS WSLg/Open3D fused PCD visualization without downgrading the official Demo 2.1 quality contract:

- depth stays FFS-derived
- object stays separate from controller
- object quality path keeps `enhanced-pt`
- controller quality path keeps `pt-filter`
- no fallback to native RealSense depth

Working hypotheses:

1. Render is not the primary blocker because `render_fps` tracks `fusion_fps`.
2. Serialized GPU gate is too conservative for climb-5; `max_concurrent=2` may recover enough throughput while keeping quality.
3. Synchronous enhanced object filtering costs ~35-45 ms and causes occasional ~200 ms spikes; if this is the blocker, move filtering to a latest-wins async quality path instead of disabling it.
4. Point count caps may be tuned only after visual quality is checked; do not treat lower point caps as the first-line quality-preserving solution.

Experiment matrix:

1. `climb-5 pointcloud`: baseline visualization.
2. `climb-5 render none`: isolate Open3D.
3. `climb-5 object-postprocess none`: diagnostic upper bound only, not a final quality mode.
4. `climb-5 gpu-gate max_concurrent=2`: quality-preserving candidate.
5. If needed, implement async enhanced filter so render uses the latest enhanced cloud without blocking fusion.

Result:

- `climb-5 pointcloud` with serialized gate reached median `render_fps=2.48`.
- `climb-5 render none` with `gpu_gate_max_concurrent=2` reached median `fusion_fps=5.00`.
- `climb-5 pointcloud` with `gpu_gate_max_concurrent=2` reached median `render_fps=4.85` and p90 `render_fps=5.19` while keeping FFS depth and `enhanced-pt`.
- Therefore the quality-preserving candidate is promoted to the `visual-5fps` preset. Remaining work is reducing occasional filter spikes and mask/point-count drops, not changing the main quality contract.

## visual-5fps Profiling Slice

Add profiling that is off by default and only records detailed per-group data when explicitly requested:

- `--profile-pipeline`
- `--profile-filter`
- `--profile-filter-detail`
- `--profile-visualization`
- `--profile-gpu-gate`
- `--profile-json-output`
- `--profile-warmup-exclude-s`

The profile records one `group_id` timeline across capture, EdgeTAM, FFS, fusion/filter, and render. The summary reports full-run and warmup-excluded FPS, medians, p90/p95/max timings, target deficit, bottleneck class, and top slowest object enhanced-PT groups. This lets us decide whether the remaining gap to stable 5 FPS is filter spike, GPU gate wait, FFS cycle, EdgeTAM per-camera model time, fusion timeout, or Open3D render.

Follow-up:

- A 120s `visual-5fps` profile with `warmup_exclude_s=40` was attempted, but the current live scene failed SAM3.1 first-frame initialization on `cam0`, so it produced zero complete fused groups and is not a valid FPS comparison.
- The FFS GPU gate scope was narrowed so the gate wraps only per-camera FFS TensorRT inference. FFS color alignment now runs outside the gate, preserving quality while reducing avoidable EdgeTAM/FFS serialization. An invalid 60s smoke after this change still showed lower FFS cycle timing, but it cannot be used as a throughput result until initialization is stable.
- Formal Demo 2.1 must use `--init-mode sam31-first-frame`; `controller-object` is the default supported mode, and current no-controller runs explicitly pass `--track-mode object-only`.
- SAM3.1 live initialization is fail-fast by default. If object-only SAM3.1 cannot initialize the object in the live scene, the run is invalid and should fail rather than falling back to saved masks.
- The next valid 5 FPS ablation should keep live SAM3.1 initialization, pass `--track-mode object-only` while no controller is visible, and compare gate2 vs gate3 and any fair/deadline gate policy only after SAM3.1 succeeds.

## Live SAM3.1 One-Frame Init Slice

The first formal live profile exposed that the old first-frame path still reused
the generic SAM3.1 video helper: write one temporary frame, start a video
session, add a text prompt, run one-frame propagation, read mask files back, then
delete the temporary session. That was semantically one-frame segmentation but
operationally still video-session plumbing.

Change the live init path to a dedicated one-frame image path:

- use `build_sam3_image_model` and `Sam3Processor`
- call `set_image(live_rgb_frame)` once
- call `set_text_prompt(...)` for object/controller prompts
- return masks directly in memory
- do not create a temporary video case
- do not call `propagate_in_video`
- keep the existing offline `run_case_segmentation` helper unchanged for
  generated-mask workflows

Formal Demo 2.1 remains fail-fast and live-only: if the one-frame SAM3.1 path
does not produce the requested object/controller mask, the run fails with no
saved-mask fallback.

## Temporal-Coherent CaptureGroup Gate Slice

Dynamic three-view fused PCD must not run FFS on arbitrary latest frames. The
CaptureGroupBuilder now enforces the temporal contract before downstream GPU
work:

- keep a small per-camera ring buffer
- select the cam0/cam1/cam2 triplet with nearest timestamps
- default `max_capture_skew_ms=33.4` for `848x480@30`
- drop skewed groups before shared FFS
- make shared FFS and fusion re-check the skew contract
- record timestamp source, skew, per-camera offsets, skew drops, and no-candidate drops

Default presets use:

```text
capture_group_policy=timestamp-nearest
max_capture_skew_ms=33.4
max_frame_age_ms=150
capture_buffer_size=4
drop_skewed_groups=true
```

This may reduce FPS when cameras drift, but it prevents wasting FFS on depth
that cannot be fused and prevents dynamic ghosting from cross-camera temporal
mismatch.
