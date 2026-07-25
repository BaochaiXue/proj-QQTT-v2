# Demo v6.2 Warm-up Critical-Path Overlap (SAM3.1 preload + align prerender)

## Requirement

Problem:
After the canonical-mesh cache landed, the cache-hit warm-up (~31s) still
serializes three avoidable costs onto its critical path: (1) the SAM3.1
checkpoint load starts only after frame 0 arrives; (2) the align stage runs
SuperPoint on the unchanged reference crop once per candidate view (32x); and
(3) the align candidate renders + their SuperPoint features are computed after
GO even though, on a cache hit, their only inputs (cached mesh + frame-0
geometry) are known seconds earlier.

Required final behavior (all three are scheduling changes only — identical
weights, identical inference paths, identical outputs):

- The seg worker starts a `Sam31PreloadThread` at worker start (when any
  tracking prompt is enabled); the load overlaps the EdgeTAM load and the
  frame-0 wait. `run_sam31_first_frame_mask_bundle` consumes the preloaded
  model via the runtime cache (`reuse_model=True`); whether the model survives
  the call is still decided by `requires_sam31_reuse` (mesh-cache miss keeps
  it for the second segmentation, hit releases it). A preload failure
  re-raises on the seg worker at the exact point the lazy build would have
  raised.
- `image_pair_matching` reads the reference image and runs its SuperPoint
  exactly once, passing precomputed features for both sides through
  `Matching` (which skips SuperPoint whenever keypoints are supplied).
- On a mesh-cache hit, when frame-0 geometry (width/height/fx) becomes known
  the seg worker sends a `PRERENDER {json}` stdin directive to the waiting
  align worker (new `PrewarmWorkerPool.send_directive`, lock-serialized with
  `pop_and_go`). The worker renders the candidate views (192 poses from the
  num_samples=8/num_ups=4 parameterization) from the cached
  mesh and extracts their SuperPoint features before GO. At GO it verifies
  mesh sha256 + width/height/fov against the run's real inputs and reuses the
  prerender only on an exact match; any mismatch or prerender failure falls
  back to the cold render. The hint can therefore never change the output.

Invalid cases:
A `PRERENDER` line to a stage without a directive handler is a protocol error
(unchanged fail-fast). A dead/popped worker makes `send_directive` return
False (hint dropped, cold path). Prerender exceptions inside the align worker
log and clear the state instead of killing the pre-GO worker.

Constraints:
No change to matching numerics, pose selection order, PnP/ARAP, sampling, or
any output byte. demo_v6_1 stays frozen.

## Touched files

- `demo_v6_2/perception/sam31_image_segmentation.py` — `preload_sam31_image_runtime`
- `demo_v6_2/mdp/warmup.py` — `Sam31PreloadThread`; bundle uses `reuse_model=True`
  (survival still keyed on `reuse_sam31_runtime`)
- `demo_v6_2/mdp/segmentation.py` — preload start/join around frame-0;
  `_notify_frame0_geometry` (k_color preference mirrors the frame-0 request)
- `demo_v6_2/utils/stage_prewarm.py` — `PRERENDER ` directive in `wait_for_go`
  via optional `on_directive`
- `demo_v6_2/shape_prior/timing.py` — `StageProfileRun.wait_for_go(on_directive=)`
- `demo_v6_2/shape_prior/warmup.py` — `PrewarmWorkerPool.send_directive`,
  `ShapePriorLocalClient.send_align_prerender`,
  `ShapePriorWarmupManager.notify_frame0_geometry`
- `demo_v6_2/shape_prior/match_pairs.py` — hoisted reference SuperPoint,
  `extract_superpoint_features` / `prepare_candidate_features`,
  optional `candidate_features` input
- `demo_v6_2/shape_prior/align.py` — `_prerender_candidates` /
  `_verify_prerender` / `pose_selection_render_superglue(prerender=)`,
  `prerender_ms` timing field (pre-GO, excluded from the compute total);
  `get_matching_ray_registration` batches all per-vertex occlusion rays into
  one `intersects_location` call (trimesh pure-python backend resolves rays
  independently: per-ray rtree candidates + per-ray closest-hit argmin under
  `multiple_hits=False`, verified in trimesh 4.12.2 source, so every
  visibility decision is identical to the one-ray-at-a-time loop)

## Follow-up: sample stage joins the prewarm pool (2026-07-19, approved)

The sample stage was the last cold subprocess on the warm-up critical path
(~3.3s, dominated by its import chain — torch/pytorch3d arrive via
utils.align_util just for as_mesh). It now supports `--wait-signal`
(imports up front, then block for GO; no models to load) and is part of
`PREWARM_STAGES`; `_prewarm_stages` on a cache hit returns (align, sample).
`request_shape_prior` runs it through `_run_stage_maybe_prewarmed` like the
other subprocess stages. Note the stage's sampling is nondeterministic
run-to-run by design (unseeded trimesh sample_surface/volume_mesh — verified
pre-existing on HEAD), so verification is structural + timing, not byte
comparison: the diff only moves process start earlier, compute untouched.

Verified 2026-07-19 (fake-live, cache hit, downstream disabled, max-chunks 2,
exit 0, 2 chunks + points.npz): sample execution_mode=prewarmed, module
import 1.32s paid inside go_wait 17.7s; parent critical-path sample entry
3.26s -> 2.14s. The remaining 2.14s is real compute (volume_mesh interior
sampling 1.71s), so the import tax is gone and further gains would need
algorithm changes (out of scope). Whole timeline this run: run_start ->
warmup_ready 22.6s (vs 34.5s in the previous session's run; part of that
delta is warm OS page cache, attribute only ~1.1s to sample prewarm).
Cache tests updated: hit prewarms (align, sample); miss/disabled prewarm all
four; request harness fakes sample via _run_stage_maybe_prewarmed. Suite 78
passed + 15 subtests.

## Verification

- `tests/test_demo_v6_2_warmup_prewarm.py` (17 tests): directive protocol,
  pool send_directive against live/popped/dead workers, client payload +
  cache gating, manager forwarding/error-swallow, preload thread semantics.
  Full demo_v6_2 suite green (78 passed + 15 subtests).
- GPU A/B on a fixed real case (outputs/shape_prior_case copy): HEAD align
  (old) vs working-tree align cold vs working-tree align with a real
  PRERENDER directive; `best_match.pkl` + `final_mesh.glb` compared
  byte-for-byte, `best_match.png` compared as pixels; plus a run-to-run
  determinism pair. See scratchpad align_ab/.
- Bounded fake-live product run on the mesh-cache-hit path with warm-up
  timing comparison.

## Status

- 2026-07-19: implemented + unit-verified (17 new tests; full v6_2 suite 78
  passed + 15 subtests).
- 2026-07-19 GPU A/B (fixed real case copied from outputs/shape_prior_case;
  HEAD worktree = old code): all 9 comparisons identical — new code
  deterministic run-to-run; ref-vs-new cold and ref-vs-new+PRERENDER both
  byte-identical on best_match.pkl + final_mesh.glb and pixel-identical on
  best_match.png. Directive path confirmed engaged ("prerendered 192 pose
  candidates pre-GO" / "reusing 192 prerendered pose candidates").
  Align post-GO compute: 15.07s (old cold) -> 14.74s (new cold, ref-SuperPoint
  hoist) -> 6.70s (new + prerender: render_candidates 7.42s -> 0, superglue
  3.73s -> 2.61s); prerender itself 8.11s, absorbed pre-GO.
- 2026-07-19 bounded fake-live product run (cache hit, downstream disabled,
  max-chunks 2, scratch base): exit 0, 2 chunks + shape_prior/points.npz, full
  chain engaged (hint at frame-0 -> "prerendered 192 pose candidates pre-GO"
  -> "reusing"). Timeline run_start -> warmup_ready 34.5s. Evidence:
  initial_sam31.model_load_ms = 0.0 with sam31_preload preload_ms 9.48s /
  join_wait_ms 5.30s (load now overlaps EdgeTAM init 4.18s + frame wait; net
  ~-4.2s on this run, more on live where frame 0 arrives later); align parent
  critical-path entry 5.67s (was ~13.7s), prerender 8.19s absorbed inside
  go_wait 20.6s. Remaining largest on-path items: pre-submit segment ~23s
  (EdgeTAM runtime import 3.2s + init, tracker init, frame-0 plumbing) and
  cold sample stage 3.26s (prewarm candidate for a future pass, not yet
  approved).
- 2026-07-19 ARAP occlusion-ray batching (approved follow-up): outputs still
  byte/pixel-identical to the HEAD reference on the same fixed case;
  arap_ray_ms 1032.9 -> 408.2 (align cold total 15.07s -> 14.32s). Decision:
  do NOT install embree — trimesh would auto-prefer it and its float32 hits
  could flip near-threshold visibility decisions (breaking byte parity) for
  at most ~0.2s further gain now that the Python per-ray call overhead is
  gone. Remaining arap_ray cost is mostly per-vertex query_ball_point /
  Python bookkeeping (possible identical-result batching, unscoped).
