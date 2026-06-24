# PhysTwin-Like Quality-Consistent Optimization Audit - 2026-06-24

## Scope

This audit identifies what currently blocks speed and quality optimizations for
the Demo 3.2 / Demo v4 PhysTwin-like product path.

“Consistent with PhysTwin” means compatible product semantics and qualitative
training behavior:

- first-frame object/controller query identity;
- per-frame semantic visibility correction;
- dense world-space RGB-D lift indexed by image pixels;
- data-process-style mask validity and radius-outlier filtering;
- object per-transition motion validity;
- controller whole-window visibility plus global motion rejection;
- controller FPS to 30 points;
- object 5 mm first-frame grid sampling;
- `track_process_data.pkl`, `final_data.pkl`, masks, tracking, PCD, metadata,
  and `READY` case publication compatible with FuturePhysTwin.

It does not mean bitwise reproduction of PhysTwin official model outputs,
random sampling order, or CoTracker numerics. TAPNext++, EdgeTAM, RealSense/FFS,
and SAM3D single-view remain the chosen local stack.

## Current Evidence

- Contract: `docs/PHYSTWIN_LIKE_CONTRACT.md` defines TAPNext++ as the tracker,
  EdgeTAM as the mask provider, RealSense/FFS as depth, and PhysTwin as the data
  contract target.
- Demo v4 docs record strict data-process compatibility rules, atomic `READY`
  publication, chunk telemetry, and FuturePhysTwin acceptance runs.
- The latest validated single-GPU fake-realtime run recorded:
  - 7 chunks, 25 frames per chunk, native RealSense depth;
  - publish intervals near the 5 s source window:
    `[4.722, 4.969, 4.904, 5.135, 4.874, 5.049]`;
  - materialization latency per chunk around `3.888-4.298 s`;
  - `backlog_chunks=0` for all chunks;
  - validation chunks loaded in FuturePhysTwin and completed CMA plus
    `train_warp.py`.
- Shape prior remains the dominant startup cost when computed inline. A verified
  full SAM3D profile recorded about:
  - `image_upscale_ms=20508.7`;
  - `sam3d_inference_ms=10881.6`;
  - `sampling_ms=29954.4`;
  - `shape_prior_total_ms=78569.7`;
  - `time_to_shape_prior_ready_ms=96647.7`.

## Non-Negotiable Quality Boundaries

These must not be weakened for speed unless the product mode is explicitly
renamed away from strict/PhysTwin-like final data:

1. Query identity is first-frame semantic identity, not residual PCD identity.
   Optimizations cannot initialize strict product queries from filtered PCD,
   table-Z, depth-valid pixels only, or once-false marker state.
2. Controller finalization is window-final. The `np.prod(controller_visibility)`
   rule and global motion rejection mean final controller points cannot be
   published causally before the chunk/window closes.
3. Radius-outlier and motion-filter constants are product semantics:
   1 cm neighborhood, 40 mask outlier neighbors, 5 motion neighbors, and
   5 mm motion-similarity threshold.
4. Shape-prior final data must keep SAM3D single-view semantics: surface and
   interior fields, 700/1000 target counts, observed-object priority,
   5 mm sampling, and preserved table/z frame.
5. FuturePhysTwin consumers must only see complete chunks. The staging +
   `READY` + atomic rename contract is a correctness feature, not overhead to
   remove.

## Optimization Blockers

### 1. Window-Final Controller Semantics Create A Latency Floor

Controller points require whole-window visibility and whole-window motion
consistency. For 5 s chunks at 5 FPS, the earliest strict controller output is
after the 25th frame plus finalization/materialization time. Shorter chunks
reduce latency but also reduce motion context and can reduce controller
survival quality. With default FuturePhysTwin training using frame 17 in the
current validation commands, chunks shorter than 18 frames are also not
equivalent validation targets.

Safe optimization direction:

- keep `--chunk-seconds` tunable;
- add explicit quality gates for controller survival count, object point count,
  and train-frame availability before recommending chunk durations below 5 s;
- label sub-window outputs as provisional, not final.

### 2. Demo v4 Recomputes Dense PCD From Depth Per Chunk

`demo_v4/headless_chunk_bridge.py` currently loads each RGB/depth frame and
calls `strict.dense_world_pcd_grid()` again for the chunk. Demo 3.2 headless
already writes a `pcd_path`, but that file currently contains filtered object
and controller point lists, not the dense `[1,H,W,3]` world-space grid required
for pixel-indexed track lifting. Therefore the bridge cannot safely reuse the
existing `pcd_path` without changing artifact semantics.

Safe optimization direction:

- extend Demo 3.2 headless capture to optionally write a dense world PCD grid
  path, or a compact dense xyz/depth cache with exact pixel indexing;
- prove equality against regenerated `dense_world_pcd_grid()` for sampled
  frames before the bridge consumes the cache;
- keep existing filtered PCD artifacts for visualization separate from strict
  dense product artifacts.

### 3. Per-Frame Mask Refinement Is CPU/Serial

The bridge applies depth validity and 3D radius-outlier filtering per frame and
per class. This mirrors `data_process_sam3d/data_process_mask.py`, but it is
done serially in Python around each chunk. Current materialization is still
within the 5 s window, but at roughly 3.9-4.3 s it leaves limited headroom for
live camera jitter, FFS depth, larger masks, or larger chunk rates.

Safe optimization direction:

- parallelize independent frame preprocessing inside each chunk;
- precompute depth-valid/radius-filtered processed masks as frames arrive;
- keep the same inlier/outlier predicate and add golden tests comparing mask
  pixels against the current implementation on real captured chunks.

### 4. Motion Filtering Is CPU KDTree Per Transition

`qqtt/demo/phystwin_strict_product.py` uses SciPy `cKDTree` and Python loops
over valid points for each transition and class. This is structurally close to
the PhysTwin/Open3D reference but becomes a scaling limit when query count,
frame count, or camera count increases.

Safe optimization direction:

- add timing telemetry around object motion filter, controller motion filter,
  and controller FPS separately;
- implement a vectorized or GPU radius-neighbor candidate path behind an
  equivalence test;
- compare boolean `object_motions_valid`, `controller_mask`, and selected
  controller coverage on synthetic cases and real chunks.

### 5. Shape Prior Is Quality-Critical And Expensive

The SAM3D path dominates startup when not reused. The expensive parts are
upscale, SAM3D model/inference, and surface/interior sampling. Skipping
upscale, reducing target counts, or returning partial samples makes the final
data less aligned with the DataProcess/SAM3D contract.

Safe optimization direction:

- keep a preloaded remote SAM3D worker;
- submit from first mask-depth-table snapshot, not first tracker pair;
- cache shape-prior outputs by source case/object snapshot hash when running
  repeated experiments;
- optimize sampling internals while preserving 700/1000 targets, 5 mm grid,
  observed-object priority, max-distance cap, and metadata.

### 6. Stage Timing Is Too Coarse For Confident Optimization

Chunk manifests currently expose materialization latency and backlog, which is
good for runtime cadence. They do not yet split materialization into:

- RGB/depth/trajectory/mask load;
- dense PCD lift;
- depth-valid mask filtering;
- radius-outlier filtering;
- track input build;
- object motion filter;
- controller motion filter;
- controller FPS;
- final_data sampling;
- filesystem write/validation/rename.

Without this split, speed work can only guess which stage is dominant for a new
camera/depth/backend setting.

Safe optimization direction:

- add per-stage chunk finalizer telemetry to each chunk manifest;
- keep the existing aggregate `materialize_latency_ms` and `backlog_chunks`;
- gate future speed changes on before/after stage timing and unchanged quality
  checks.

### 7. Real-Camera Quality Is Not Yet Proved At The Same Level

The strongest acceptance evidence is fake-live/saved-case based. This proves
schema, semantics, cadence, and FuturePhysTwin trainability for known captured
data. It does not yet prove that live D455 exposure, depth holes, table
calibration, SAM3D snapshot selection, and strict chunk finalization remain
quality-consistent during a fresh live capture.

Safe optimization direction:

- run a live-camera strict Demo v4 capture once hardware is available;
- validate second-last/fifth-last chunks with FuturePhysTwin loader, CMA, and
  `train_warp.py`;
- compare live chunk metrics against the fake-live acceptance envelope:
  object count, controller candidate count, zero placeholders, shape point
  counts, materialization latency, and backlog.

## Prioritized Next Work

### P0 - Add Measurement And Guardrails

1. Add stage-level finalizer timing to `headless_chunk_bridge.py` and chunk
   manifests.
2. Add strict quality summary fields per chunk:
   object count, controller candidate count before FPS, final controller count,
   first-frame zero/nonfinite counts, shape surface/interior counts,
   train-frame-safe flag, and mask radius filtering enabled flag.
3. Add a validation script that checks these fields over all READY chunks and
   fails if any quality invariant is violated.

### P1 - Remove Duplicate Dense PCD Work Safely

1. Add optional dense world PCD grid output to Demo 3.2 headless capture.
2. Teach Demo v4 bridge to consume that dense grid only when metadata proves it
   is unfiltered, table/world-calibrated, and shape-compatible.
3. Add equality/regression tests comparing regenerated dense PCD and cached
   dense PCD on fixture frames.

### P2 - Parallelize Finalization Without Changing Semantics

1. Preprocess independent frames in parallel: RGB/depth load, dense lift,
   depth-valid masks, and radius-outlier masks.
2. Keep motion filtering and FPS as the window-closing barrier.
3. Add before/after cadence tests proving publish intervals stay below the
   source window and backlog remains bounded.

### P3 - Optimize Shape Prior With Contract Locks

1. Cache SAM3D result artifacts for repeated operator runs.
2. Profile and optimize surface/interior sampling while preserving target
   counts and distance/grid semantics.
3. Keep “partial shape prior” out of strict final_data unless explicitly
   labeled as non-strict/provisional.

### P4 - Live Hardware Acceptance

1. Run Demo v4 with `--input-source live` and default 5 s chunks.
2. Validate READY chunks with FuturePhysTwin loader/CMA/train.
3. Record the live metrics next to the fake-live acceptance report.

## Bottom Line

The current implementation is close enough to support speed work, but the
optimization boundary is narrow:

- Do optimize scheduling, caching, parallelism, dense-grid reuse, and
  instrumentation.
- Do not optimize by changing strict query source, dropping depth/mask
  validity gates, reducing shape-prior targets, skipping whole-window
  controller finalization, or publishing incomplete chunks.

The first practical step is not another heuristic change. It is stage-level
finalizer telemetry plus quality invariant summaries, because that will tell us
which optimization is safe and worth doing for the next live run.
