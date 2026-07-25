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
