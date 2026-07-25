# Demo v6.2 Origin-Unified Structure Sampling (full PhysTwin downsample parity)

## Requirement

Problem (audited 2026-07-23 against data_process_origin):
Origin samples object + surface + interior through ONE shared 5mm voxel
occupancy (`data_process_sample.py:47-119`: np.unique value order -> z>0->0
clamp -> min_bound over RAW candidates + clamped frame 0 -> object claims
first, then surface, then interior). demo_v6_2 instead ran TWO independent
passes: the warm-up voxel-filtered the priors against the throwaway dense
object-mask PCD (D), and chunk 0 voxel-sampled only the tracked object (O)
with a fresh occupancy — cross-category duplicates possible, visible-face
priors systematically over-deleted by D, min_bound anchored on already
filtered priors, no z clamp, unique order re-sorted. Controller FPS(30) was
already parity and stays untouched.

Required final behavior (rule parity — numeric parity is impossible by
construction: O is the online TAPNext query set, mesh sampling is unseeded):

- Warm-up publishes RAW candidate pools only: the sample stage
  (`shape_prior/sample.py`) samples 1024 surface + 10000 interior points from
  the aligned mesh and writes `case/shape/candidates.npz`
  (schema `shape_prior_candidates_v1`); no dense-D voxel pass, no
  final_data.pkl. The camera's ShapePriorResult / capture shape npz carry the
  candidates (`shape_prior_structure_points_role = raw_candidates`); the
  formal gate still opens at candidates-ready (hold-still unchanged).
- Chunk-0 identity freeze runs the unified origin sampler
  (`tracking.sample_origin_unified_structure`): float64 end to end, exact
  `np.unique` on frame 0 (value order kept), z>0->0 clamp on the
  published-trajectory copy (motion gating stays on raw tracks upstream),
  min_bound over raw S + raw I + clamped frame 0, ONE shared occupied set
  claimed object -> surface -> interior. Object columns AND final
  surface/interior freeze together; `process_window` publishes the clamped
  object trajectories every chunk (origin clamps the whole video).
- `points.npz` keeps its schema and its downstream-trigger role but is now
  written (atomically) by the chunk session right after chunk 0 commits, with
  the FINAL frozen points. Two states are therefore split:
  capture candidates ready (heavy GPU stages done) vs points.npz (final
  unified sampling done -> phystwin_shen may start, ~1 chunk later than
  before). `main.py` trigger polling is unchanged
  (`before_poll` + `on_chunk_written` -> `_ensure_phystwin_shen_running`).
- Explicit `--surface-points-npy/--interior-points-npy` overrides are treated
  as candidates and go through the unified sampling.
- Config/CLI cleanup: the camera process lost `--shape-prior-points-npz` and
  `--volume-sample-size-m` (the voxel size now lives only in the orchestrator
  CLI -> `TrackingRuntime`); the sample stage lost `--shape_prior` and
  `--volume_sample_size`; `ShapePriorLocalClient` lost `points_npz` /
  `volume_sample_size_m`; `POINTS_NPZ` constant deleted.

Invalid cases:
Chunk 0 without candidates (shape prior required) still fails via the
existing `_wait_for_shape_candidates` timeout/terminal-status path. A failed
chunk-0 materialize leaves no points.npz, so downstream never launches on a
dead run (stricter than before).

## Touched files

- `demo_v6_2/tracking.py` — `OriginUnifiedSample`,
  `sample_origin_unified_structure`, `_claim_unoccupied_voxel_indices`
  (replaces `_volume_sample_indices`); freeze stores final S/I + min_bound;
  `frozen_structure_points()`; publish-side z clamp
- `demo_v6_2/shape_prior/sample.py` — candidates-only stage
  (`sample_shape_prior_candidates`, `write/load_shape_prior_candidates`);
  dense-D pass and dead no-prior path deleted
- `demo_v6_2/shape_prior/warmup.py` — sample command/require_file ->
  candidates; finalize loads candidates, no points.npz write; client params
  trimmed
- `demo_v6_2/shape_prior/case.py` — `write_shape_prior_points_npz` atomic
  (tmp + replace); still the single writer used by the chunk session
- `demo_v6_2/streaming/capture_meta.py` — `_wait_for_shape_candidates`
- `demo_v6_2/streaming/session.py` — chunk-0 candidates wait; freezes final
  S/I from the runtime; writes points.npz after chunk-0 commit; later chunks
  skip the shape wait entirely
- `demo_v6_2/streaming/window_builder.py` — ChunkDataWindow carries the
  frozen finals
- `demo_v6_2/mdp/cli.py`, `demo_v6_2/mdp/runtime.py`,
  `demo_v6_2/main_subprocess.py`, `demo_v6_2/main.py` — CLI/wiring cleanup,
  session points_npz param, trigger comment updated
- Tests: `VolumeSampleTests` rewritten against a literal origin reference
  implementation (unique order, clamp, raw min_bound anchor kept by a
  candidate later rejected by occupancy, priority fixtures); cache tests'
  fake sample stage writes candidates.npz

## Verification

- Suite 95 passed + 15 subtests.
- Literal-reference unit test: unified sampler output identical to a verbatim
  reimplementation of origin L47-L119 on randomized fixtures.
- Bounded fake-live product run: exit 0, candidates.npz at warm-up,
  points.npz appears only after chunk 0, downstream trigger order correct,
  final chunk S/I == points.npz contents; legacy-vs-unified count comparison
  against the 2026-07-22 legacy run on the same recorded case.

## Status

- 2026-07-23: implemented; suite 95 passed + 15 subtests. Fake-live run
  exit 0, artifact order correct (candidates at warm-up, points.npz only
  after chunk 0, == chunk final_data S/I). Legacy-vs-unified on the recorded
  sloth case: object 2058 -> 2049, surface 567 -> 734 (dense-D over-deletion
  recovered), interior 1097 -> 1056 (now correctly displaced by
  object/surface) — direction matches the audit's predictions.
- 2026-07-23 adversarial review (13 agents): fixed — (1) MAJOR: candidate
  float64->float32->float64 round trip through ShapePriorResult/headless
  writer shifted floor bins; the capture channel now stays float64
  end-to-end. (2) others/view_shape_prior_outputs.py and
  lbs_shape_prior_from_chunks.py read the removed warmup-case final_data.pkl
  -> repointed to track_process_data.pkl + shape_prior/points.npz.
  (3) ASAP augment republished unclamped ARAP-filled object estimates ->
  fill now clamps z>0 like the direct publish. (4) The dead
  capture-finished fast-fail conjunct in _wait_for_shape_candidates ->
  one-confirming-poll fast fail.
  NOTE: point sets / spring topology intentionally differ from the legacy
  split (the parity goal); validate phystwin_shen tuning on a real case.
