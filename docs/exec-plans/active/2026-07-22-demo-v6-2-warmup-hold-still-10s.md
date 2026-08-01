# Demo v6.2 Warm-up Hold-Still ≤10s (preload-before-camera + frame-0 readiness barrier)

## Requirement

Problem:
After the 2026-07-19 critical-path overlap pass, the live cache-hit warm-up
still costs ~21s of operator hold-still, because frame 0 is the FIRST camera
frame after `pipeline.start` while every model loads afterwards on the worker
threads: capture publishes frame 0 at ~t+0.4s, then the seg thread pays
EdgeTAM init (~5.2s) ∥ SAM3.1 preload (~9.6s), the frame-0 chain (SAM3.1
bundle 0.84s incl. 0.33s inline release, EdgeTAM frame-0 forward 4.08s of
one-time compile tax), and the shape-prior request (6.31s incl. ~1.3s of
worker exit tails + 0.38s serial case write). None of the model loads needs
anything from the camera.

Required final behavior (scheduling/dedup changes only — identical weights,
identical inference paths, identical outputs):

- A1 — all camera-free model loads start at `run()` entry on parallel daemon
  threads (`mdp/preload.py::PerceptionPreloader`): EdgeTAM runtime
  (`load_edgetam_runtime`, moved verbatim from `SegmentationStage._init_hf_model`),
  SAM3.1 (`Sam31PreloadThread`, start site moved from the seg worker to the
  preloader), TAPNext++ (A4, below). On `--depth-source ffs` the preloader
  starts only after the `FfsDepthEngine` constructor, so the FFS loader's
  global `torch.compile` disposition still precedes the EdgeTAM compile wrap
  (identical numerics to the old inline order). Each stage worker joins its
  own leg; a leg failure re-raises at the join — the same worker and
  fatal-latch route as the old inline load.
- A2 — live frame-0 readiness barrier: the live capture worker designates
  frame 0 only when every preload leg is done
  (`wait_frame0_consumers_ready`), pumping the warm-up preview meanwhile
  (display-only, camera keeps streaming, operator frames the scene). The
  hold-still window therefore opens at the barrier, not at camera open.
  Approved semantic note (user, 2026-07-22): live has no canonical frame-0
  identity; the late-designated frame 0 is captured AFTER the sensor's
  exposure/AWB settle, unlike today's first-frame-ever anchor.
  Fake-live is UNCHANGED (no barrier): the recording's frame 0 stays the
  anchor and the replay/preview clock is untouched (parity harness).
- A3 — the align PRERENDER hint fires from camera metadata right after
  `prepare_source` (`runtime._notify_frame0_geometry_from_camera`,
  width/height from the session, fx from the runtime k_color — the values
  frame-0 packets carry), replacing the seg-side `_notify_frame0_geometry`
  at frame-0 pickup (single send site, no double-send). Under the barrier
  the ~6-8s pre-render is absorbed ~10s before GO.
- A4 — the TAPNext++ adapter builds AND loads its 2.5GB checkpoint on the
  preload leg (`tracker.build_tracker_adapter` + `adapter.warmup()`); the
  tracker worker joins instead of building inline; the old lazy first-packet
  load is gone.
- C1 — deferred worker reap: align/sample prewarmed workers return to the
  parent at their COMPLETED profile snapshot (all stage outputs precede it;
  `PrewarmWorkerPool.pop_and_go_nowait` + `PendingStageReap.wait_snapshot`),
  and their CUDA-context exit tails (~0.93s + ~0.35s) run under the next
  stage / result_finalize. READY is still gated on every reaped exit code
  (same `CalledProcessError` surface) via the `worker_reap` barrier entry
  before the request returns. upscale/generate keep the synchronous
  `pop_and_go` (VRAM budget: their contexts must be gone before the next
  stage loads weights). Known bounded leak: a request that fails between GO
  and the reap barrier can leave ≤2 already-exiting workers unreaped until
  process exit (failure path tears the run down anyway).
- C2 — the cache-hit SAM3.1 release (~0.33s gc/empty_cache) is deferred out
  of `run_sam31_first_frame_mask_bundle` (`defer_release=True`) to a daemon
  thread started strictly AFTER the frame-0 EdgeTAM forward (never overlaps
  the deferred-compile CUDA-graph capture). Exception paths still release
  inline; the cache-miss trim is unchanged.
- C3 — `write_shape_prior_case` writes its nine independent artifacts on a
  ThreadPoolExecutor (all inputs validated/derived before any write,
  join-all + re-raise before return; PNG/zlib/pickle release the GIL).
  Same bytes per file; only the failure path changes (validation now fully
  precedes writes).
- B1 (byte-A/B gated, `DEFAULT_EDGETAM_PRECOMPILE_FIRST_FORWARD`) — a
  throwaway prompted EdgeTAM forward on a scratch session (real frame
  geometry, same prompt ids/order/autocast/inference-mode as frame 0) runs
  on the seg thread between the EdgeTAM join and the SAM3.1 join, pre-paying
  the ~4s 'vision-reduce-overhead' compile+capture so the real frame-0
  forward replays it. GATE: frame-0 mask artifacts byte-identical to the
  lazy path AND ≥1s measured saving, else the constant stays False
  (user directive 2026-07-22).

Invalid cases:
A preload-leg failure surfaces on its consuming worker exactly where the old
inline load raised (fatal latch → stop_event → barrier loop exits). A dead
prewarmed worker before GO still fail-fasts in `pop_and_go_nowait`; nonzero
exits surface as `CalledProcessError` at `wait_snapshot` or the reap barrier.

Constraints:
No change to any model weights, inference path, mask numerics, align/sample
compute, chunk contents, or the fake-live replay semantics. demo_v6_1 frozen.

## Touched files

- `demo_v6_2/mdp/preload.py` — NEW: `EdgetamRuntime`, `load_edgetam_runtime`,
  `_PreloadLeg`, `PerceptionPreloader`
- `demo_v6_2/mdp/runtime.py` — preloader wiring + start order (post-FFS),
  `_notify_frame0_geometry_from_camera` after `prepare_source`
- `demo_v6_2/mdp/segmentation.py` — `_prepare_warmup` joins preload legs
  (SAM3.1 join hoisted BEFORE the frame-0 wait), `_init_hf_model` and
  seg-side `_notify_frame0_geometry` removed, `_precompile_first_forward`
  (B1), `_release_sam31_after_frame0` (C2)
- `demo_v6_2/mdp/capture.py` — live frame-0 readiness barrier + preview pump
- `demo_v6_2/mdp/tracker.py` — `build_tracker_adapter` module function with
  eager `warmup()`; stage joins the preloaded adapter
- `demo_v6_2/mdp/warmup.py` — `Sam31PreloadThread.wait_done`,
  `run_sam31_first_frame_mask_bundle(defer_release=)`
- `demo_v6_2/mdp/constants.py` — `DEFAULT_EDGETAM_PRECOMPILE_FIRST_FORWARD`
- `demo_v6_2/shape_prior/warmup.py` — `pop_and_go_nowait`/`PendingStageReap`,
  deferred reap for align/sample, `worker_reap` critical-path entry +
  `shape_prior_worker_reap_ms`
- `demo_v6_2/shape_prior/case.py` — parallel artifact writes
- `tests/test_demo_v6_2_warmup_prewarm.py` — PerceptionPreloader legs/barrier,
  live frame-0 barrier + display-only pump, PendingStageReap snapshot/exit,
  deferred SAM3.1 release, camera-open prerender hint
- `tests/test_demo_v6_2_shape_prior_cache.py` — fake stage runner accepts
  `defer_reap`

## Follow-up round 2 (2026-07-23, approved: "如果他说的对你就做")

- Multi-pass precompile hypothesis TESTED AND REJECTED: three scratch
  passes reach steady state within the scratch session ([4989, 65, 19]ms)
  but the REAL session's frame-0 forward still pays ~2.33s (frame-1 33ms) —
  the residual is per-SESSION (state/graph tied to the session), not
  per-process, so extra dummy passes buy nothing. B1 stays single-pass;
  masks stayed byte-identical throughout the experiment. Eliminating the
  remaining ~2.3s would need warming the REAL session (e.g. a resettable
  session or per-session graph reuse inside EdgeTAM) — out of scope.
- FFS `_disable_torch_compile` global monkeypatch replaced by the scoped
  `torch_compile_disabled` context manager (ffs_foundation_loader) wrapping
  only the FFS repo imports — on ffs runs EdgeTAM's compile wrap was
  silently a no-op before ("applied=[...]" logs the target list, not whether
  torch.compile was real). Behavior change on ffs runs only (EdgeTAM now
  really compiles there, matching the native path); the runtime start-order
  comment is no longer load-bearing.
- Startup-tax cleanup: `as_mesh` split into light `utils/mesh_utils.py`
  (sample stage + mesh-cache validation + the camera import chain no longer
  pull torch/PyTorch3D/matplotlib via align_util); FFS/numba imports lazy in
  runtime.py + session.py (TYPE_CHECKING); `@njit(cache=True, nogil=True)`;
  `[startup] camera subprocess imports took X.XXs` stamp in
  main_data_processing.py (previously invisible pre-run() tax).
- best_match.png (matplotlib) off the formal path: align gains
  `--render-route-visualizations` (default off), wired from the camera's
  previously-dead `--shape-prior-skip-route-visualizations` flag through
  `ShapePriorLocalClient(render_route_visualizations=)`. best_match.pkl is
  kept (align's own resume mechanism).
- Assessed but deferred (need their own pass + GPU A/B): GeometryStage
  depth/PCD ∥ segmentation; SuperGlue micro-batch / FRAME_PREMATCH;
  SAM3.1-mask-sourced shape prior (semantic change, needs sign-off);
  align+sample worker merge; ARAP bookkeeping vectorization; TAPNext dummy
  forward; hold-still UI state split + warm-up frame buffering.

## Verification

- Full suite: 90 passed + 15 subtests (12 new tests).
- Bounded fake-live product run (cache hit, downstream disabled,
  max-chunks 2, scratch base): exit 0, chunks + points.npz, prerender chain
  engaged from the camera-open hint site.
- B1 A/B on the fixed fake-live case: OFF vs OFF (run-to-run determinism
  reference) vs ON; frame-0 mask artifacts under shape_prior_case compared
  byte-for-byte; timing delta from the perception profile.
- Live validation pending real hardware/operator: barrier print
  `[warmup] frame-0 readiness barrier opened after X.XXs; hold still now`
  marks the hold-still start; hold-still = that print → WARMUP_FINISHED.

## Status

- 2026-07-22: implemented A1-A4, C1-C3 + B1; suite 90 passed + 15 subtests
  (12 new). The barrier also waits `mark_seg_frame0_ready` (seg joins +
  precompile done) so a slow precompile can never eat into hold-still.
- 2026-07-22 fake-live A/B (cache hit, downstream disabled, max-chunks 2,
  scratch base; all runs exit 0, 2 chunks, prerender chain engaged from the
  camera-open hint):
  - Determinism reference: two B1-OFF runs byte-identical on all six frame-0
    case artifacts (color/masks/processed_masks/track_process_data/calibrate).
  - B1 gate PASSED: B1-ON artifacts byte-identical to B1-OFF; frame-0
    EdgeTAM forward 3.80s -> 2.30s (steady frames 21-22ms unchanged);
    precompile itself 4.6-4.7s, absorbed pre-frame0 on live by the barrier.
    Constant flipped to True.
  - warmup_total_ms (fake-live, includes the load block by construction —
    no barrier there): 21.34s (2026-07-19 reference) -> 19.34s (B1 off);
    20.05-20.25s with B1 on (the precompile is NOT hidden on fake-live —
    accepted dev-mode cost, live is the product).
  - Request breakdown after C1/C3: 6.31s -> 5.30s (case_write 383->297ms,
    align 4.72->3.66s snapshot-gated, sample 1.21->0.97s, residual
    worker_reap 368ms).
  - First run after cold OS/inductor caches showed frame-0 forward 13.2s and
    slower loads (29.2s total) — cache-warmth variance, not a code path
    difference; warm reruns confirm.
- Projected LIVE hold-still (barrier print -> WARMUP_FINISHED), from
  measured components: frame designation ~0.1 + SAM3.1 bundle 0.2-0.5 +
  EdgeTAM frame-0 forward 2.3 + depth/PCD ~0.5 + shape-prior request ~5.3 +
  gate <=0.2 ≈ **8.6-9.2s** (was ~21s). Camera-open -> barrier ≈ max(load
  legs) ≈ 9.5s of operator framing time with live preview. PENDING: real
  live run (needs operator; watch the barrier print and warmup_total).
- 2026-07-22 adversarial review (18-agent workflow): three real defects
  found and fixed —
  (1) MAJOR: the C2 deferral must be gated on the B1 precompile having run;
  with the constant off, frame-1 is the cudagraph RECORD forward and the
  release thread's synchronize/empty_cache invalidates the capture
  (empirically reproduced). `_sam31_release_deferred` now requires
  `DEFAULT_EDGETAM_PRECOMPILE_FIRST_FORWARD` + a tracked identity.
  (2) MAJOR: the deferred-reap barrier now runs BEFORE result_finalize —
  points.npz is the downstream launch trigger and must never exist when a
  worker exit code can still fail the request (finalize is ~1.4ms, so the
  overlap loss is nil).
  (3) MAJOR: `_run_stage_maybe_prewarmed` unlinks the stage profile before
  GO on the deferred path — a GO racing a slow worker import could
  otherwise satisfy `wait_snapshot` with a stale COMPLETED file from a
  previous run.
  Minor fixes: case/mask mkdir hoisted before the parallel writers
  (processed_masks.pkl opened the dir without creating it); `worker_runner`
  catches BaseException (a SystemExit re-raised from a preload-leg join
  would otherwise bypass the fatal latch and hang the barrier/pump loops);
  snapshot poll 2ms -> 10ms. Accepted-as-is: daemon preloads keep loading
  during a startup failure (process exit reaps them); a SAM3.1 preload
  failure surfaces only after the ~4.6s precompile (failure path only);
  ≤2 zombie workers on a mid-request failure (process exits shortly after).

## Round 3 (2026-07-25): GPT exact-serial list — fact-check + implemented subset

Fact-check verdicts on the round-3 external list:
- CONFIRMED: per-candidate `.cpu().numpy()` syncs in `image_pair_matching`
  (192x3 D2H + full CPU materialization, only the winner consumed); formal
  align call inherited `cache=True` (192 stale-readable `matches_i.npz`
  stat/read hazard, no manifest validation); matplotlib imported at module
  top of both `match_pairs.py` and `models/utils.py`; SAM3.1 preload leg
  imports outside `HEAVY_IMPORT_LOCK` + tapnet `sys.path` insert still
  thread-side (residual race window); ARAP `deform_ARAP_ray_registration`
  O(N^2) `list.index`/`not in` bookkeeping; FFS engine ctor + numba warmup
  still serial before `preload.start()`.
- Implemented this round (all byte-gated):
  1. `pin_tapnet_import_path()` (mdp/tracker.py) — main-thread sys.path pin
     in `PerceptionPreloader.__init__` before any leg starts; leg-side
     inserts are now read-only no-ops. Structural fix for the tapnet import
     flake; `HEAVY_IMPORT_LOCK` retained as defense in depth.
  2. Formal align call passes `viz=False, cache=False, save=False`
     explicitly — formal path can never read a stale `matches_i.npz`;
     `best_match.pkl` (align's own resume point) unchanged.
  3. GPU-resident matching loop (`gpu_resident = not (viz or viz_best or
     save or cache)`): per-candidate matches stay on device, one
     `torch.stack(counts).cpu()` sync, winner-only materialization. Same
     forwards, same order, same B=1 — proven BYTE-IDENTICAL vs legacy loop
     (3 seeds x 8 cands + 192-cand set, winner index + all four arrays).
     Synthetic timing: 2.55s -> 2.51s (192 cands) — the loop is
     compute-bound, not sync-bound; the external 0.2-0.8s estimate did not
     materialize on this GPU.
  4. matplotlib fully lazy on the formal matching path (viz-branch imports
     in match_pairs.py; `_load_pyplot()` in models/utils.py) — align worker
     no longer imports matplotlib at all when route viz is off.
  5. ARAP `index_position` dict replaces list scans — identical append
     order / first-target-wins / in-place clamp semantics.
- NOT adopted (deferred, each needs its own GPU A/B round): B=1 CUDA
  Graph / dual-stream matching, keypoint-count-bucketed micro-batch
  (changes batch kernels — strictest gate), EdgeTAM compile-policy A/B
  matrix (none/model-default vs vision-reduce-overhead; potentially
  removes the 4.6-12.2s precompile AND the ~2.3s session tax — biggest
  candidate win, full-video mask byte gate required), GeometryStage
  (depth/PCD parallel to seg), FFS engine deserialization as a preload
  leg, SAM3.1-mask-sourced shape prior (semantic change, owner sign-off).

## Round 4 (2026-07-26): EdgeTAM compile-policy A/B (deterministic offline gate)

Owner: FFS is NOT the default going forward — ffs-only items dropped from
the queue. Instrument: scratchpad/compile_ab/edgetam_compile_ab.py mirrors
the demo seg path exactly (same runtime loader/model id/dtype/autocast/
prompt order/post-process/threshold/session pruning) over a FIXED 150-frame
list (recording steps 902+6k) with FIXED frame-0 prompts (fixtures from the
baseline run's frame-0 masks, controller split into two hands by connected
components). One mode per fresh process; per-frame per-object mask sha256.

Results:
- Determinism control PASSED: vision-reduce-overhead run twice -> all
  150x3 hashes IDENTICAL (the harness is a valid byte gate; EdgeTAM
  streaming is run-to-run deterministic on this GPU).
- `none` (eager): session frame-0 tax 445ms (vs 2329/2171ms compiled!),
  steady p50 25.1ms / p95 27.7ms (vs 19.4/22.2 compiled; 5 FPS budget is
  200ms), zero compile cost. BUT masks DIFFER from the compiled mode:
  frame 0 identical, then 447/450 (frame,object) pairs differ (low-order
  numeric differences compound through the streaming memory). Under the
  zero-quality-change bar this CANNOT be adopted unilaterally — owner
  decision pending, IoU quantification in progress.
- `model-default`: NOT RUNNABLE — full-model torch.compile graph-breaks in
  transformers' EdgeTAM streaming (`get_frame` in _prepare_vision_features,
  dynamo error). External claim falsified.
- 12s precompile mystery: in a quiet process the same precompile costs
  10.8s cold / 4.0s WARM. The demo pays ~12.1s on consecutive runs and
  writes fresh inductor cache entries each time -> the demo side misses the
  FX graph cache every run. Fixing that miss (not changing modes) is the
  no-quality-risk win: ~8s off the pre-barrier path. Cache-miss-reason
  probe (TORCH_LOGS=+torch._inductor.codecache) in progress.

### Round-4 resolutions (2026-07-26 evening)

- IoU between `vision-reduce-overhead` and `none` masks over the 150-frame
  gate: mean 0.987-0.991, min 0.94-0.97; object-mask xor mean 222 px on a
  ~21.5k-px mask (~1% boundary pixels). Boundary noise, not a quality
  regression in either direction — but byte-level output change, so the
  mode switch is an OWNER DECISION (gain if switched: frame-0 session tax
  2.3s -> 0.45s, precompile 4.5-15s -> 0, steady p50 19 -> 25ms vs the
  200ms/5FPS budget; every downstream product re-baselines).
- 12s precompile mystery RESOLVED: not load contention — a full demo run
  with TORCH_LOGS=+torch._inductor.codecache showed 2 fx-cache HITS, 0
  misses, precompile 4.53s WITH sam31/tapnext loading in parallel. The two
  12.1s runs were fx-cache misses right after the 3-pass-precompile
  experiment churned the compiled-graph set. Warm steady state is ~4.0-4.9s.
- Hardening landed: `TORCHINDUCTOR_CACHE_DIR` pinned to
  `~/.cache/qqtt_torchinductor` (main_data_processing.py, setdefault so an
  operator override wins). The torch default under /tmp is wiped every
  reboot, re-imposing the ~10-15s cold compile on the first run after boot.
  Verified: cold first run 15.05s -> warm second run 4.85s; frame-0 masks
  byte-identical; cache location cannot affect numerics.
- BASELINE SHIFT (not a regression): owner regenerated the shape-prior
  mesh cache entry (schema_v1/sloth object.glb + manifest, 2026-07-25
  18:04) and committed be79eb0/1bf8bd0 (rounds 2-3 work). All align byte
  baselines from before that (best_match.pkl / final_mesh.glb, e.g.
  run_p02final/run_r3) are STALE — SAM3D generate is nondeterministic, so
  a new mesh means new (equally valid) align products. Today's three runs
  (cacheprobe / cachedir1 / cachedir2) are byte-identical to EACH OTHER;
  run_cachedir2 is the new golden baseline for future byte gates.
