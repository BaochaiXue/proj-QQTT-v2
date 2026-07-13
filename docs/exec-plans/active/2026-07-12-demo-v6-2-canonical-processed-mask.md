# Demo v6.2 canonical processed mask

## Requirement

Move the fixed PhysTwin radius-outlier rule ahead of every formal realtime
consumer. Demo v6.2 must build one processed mask for each formal 5 FPS frame,
then make tracker classification/validity, runtime PCD construction, and the
prepared PhysTwin product consume that same mask and dense world-space RGB-D
grid.

Match `data_process_origin` mask semantics where they apply to the sanctioned
single-camera realtime pipeline:

- use the full depth image without a point cap;
- keep only pixels with `0.2 < depth < 1.5` metres;
- filter object and combined controller point sets independently with the
  fixed radius rule (`0.01 m`, at least `40` neighbors);
- clear rejected 3D points from their source 2D masks;
- build every downstream formal product from those processed masks.

## Final behavior

- PT processing runs once per formal 5 FPS sampled frame, not on every 30 FPS
  camera capture.
- A required camera-to-world calibration is applied before PT processing;
  missing or invalid calibration fails immediately.
- Empty processed object or controller masks fail immediately. There is no raw
  mask fallback.
- Object/controller overlap retains origin semantics: the classes are filtered
  independently. The canonical stage documents the resulting tracker identity
  ambiguity instead of inventing a priority rule.
- `hand_a` and `hand_b` remain separate EdgeTAM identities, while formal
  controller geometry is their union as required by `design_spec.md`.
- Tracker visibility, temporary-invalid state, frozen controller identity, and
  controller recovery continue to follow `design_spec.md`.
- Table-Z filtering, PCD-mask erosion, and their configuration/telemetry/output
  fields are removed as legacy behavior absent from `data_process_origin`.
- Raw semantic masks may feed a visual diagnostic but are not a second formal
  product contract.

## Implementation

1. Introduce an immutable canonical processed-frame packet containing the
   cleaned mask packet, metric depth, and dense world-space RGB-D grid.
2. Replace the two-way raw-mask fanout with one ordered mask-refinement stage;
   fan its canonical result to runtime PCD and tracker consumers.
3. Build runtime class PCDs by indexing the shared dense grid with processed
   masks, without rebuilding depth geometry.
4. Make tracker query identity and per-frame mask/depth gates consume the
   processed mask and shared geometry while preserving `design_spec.md` state
   rules.
5. Make prepared PhysTwin serialization validate and package the canonical
   mask/grid instead of running a second PT pass.
6. Remove table-Z filter and mask-erosion options, helpers, metadata, artifacts,
   and tests.
7. Update Demo v6.2 pipeline documentation with the 30 FPS capture / 5 FPS
   formal PT boundary and the single-source mask invariant.

## Validation

- Unit-test origin-compatible depth boundaries and fixed PT mask clearing.
- Assert that tracker, runtime PCD, and prepared output use the same processed
  masks and sequence.
- Assert empty required classes and invalid/missing c2w fail at the canonical
  boundary.
- Search for removed table-Z filter and mask-erosion options/fields.
- Compile Demo v6.2 and run focused tests.
- Run the repository smoke validation profile and record any external blocker.

## Results

- The formal camera flow is now
  `raw mask -> canonical processed frame -> tracker/runtime PCD/prepared product`.
  `apply_radius_outlier_to_mask_frame` has one runtime call site.
- Table-Z filtering, PCD-mask erosion, the former dual raw-mask queues, sparse
  runtime backprojection helpers, and their options/metadata/artifacts were
  removed. The tracker-only geometry helpers now live in
  `mdp_tracker_geometry.py`.
- The canonical `_build_processed_frame_result` boundary requires a finite
  camera-to-world transform, applies
  strict `0.2 < depth < 1.5` validity and the fixed PT rule, and rejects empty
  object/controller results. Shape prior and prepared serialization consume the
  already-processed mask/dense grid without another PT pass.
- Tracker full-frame product visibility now uses the existing
  `design_spec.md` processed-mask/depth observation gate; raw model visibility
  remains diagnostic.
- Existing 848x480 prepared frames measured approximately 150 ms cold / 73--84
  ms steady for the complete canonical stage, including approximately 50--52 ms
  steady PT time. This stays below the 200 ms formal 5 FPS frame budget on the
  local workstation sample.
- `python -m compileall -q demo_v6_2 tests`: passed.
- Focused Ruff undefined/unused-name checks: passed (the deliberate
  `mdp_constants.py` re-export module remains outside the F401 check).
- `python -m pytest -q tests`: 19 passed, 3 subtests passed.
- Harness catalog, Demo v5 compatibility, experiment-boundary, and visual
  architecture guards: passed.
- Smoke and exhaustive profile runners were attempted. Both remain blocked by
  pre-existing deleted `demo_v5_1` paths: smoke first fails importing
  `demo_v5_1` from `render_demo32_headless_capture.py`; exhaustive first fails
  opening `demo_v5_1/main.py`. The scope guard independently reports the same
  five missing Demo v5.1 shape-prior carveout files.
- `PIPELINE.md`, `readme.md`, `design_spec.md`, and the A4 `PIPELINE.pdf` were
  synchronized with the final data flow.
