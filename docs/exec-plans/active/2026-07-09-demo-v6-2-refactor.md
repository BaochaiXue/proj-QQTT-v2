# Demo v6.2 large-scale refactor & cleanup

Goal: refactor+clean `demo_v6_2` (a byte copy of `demo_v6_1`) so that every file
is <1000 lines, every "big step" is a clearly-named module/function, legacy /
over-defensive / dead code is removed, module docstrings answer the 23 pipeline
questions below, and a live "what is the pipeline doing / warmup failed" status
is visualized — without changing valid prepared-frame product behavior. Inputs
that depend on removed legacy paths now fail explicitly.

`demo_v6_1` is the FROZEN reference (never edit). `demo_v6_2` is the target.

## Verification protocol (run after every step)

1. **Bounded fake-live product validation**: run the realtime fake-live path to
   a fixed chunk target and validate chunk/track/filter/schema/ASAP/manifest/archive
   products. The completed-capture conversion gate was retired with its deprecated
   runtime mode on 2026-07-12.
2. **Import smoke**: all core `demo_v6_2` modules import; zero `demo_v6_1` /
   `services` / `data_process` leakage (`tests/test_demo_v6_2_standalone_ffs.py`).
3. **Unit tests**: `tests/test_demo_v6_2_downstream.py` + standalone (15 tests).
4. **Camera/threading half** (not byte-deterministic): `--dry-run` contract
   equality + a bounded fake-live smoke run producing well-formed chunks/schema.

## Status

- [x] Phase 0 — self-contained rewire (58 imports + 7 launch/path refs
  demo_v6_1→demo_v6_2; identity strings kept for exact parity). PARITY OK, no leak.
- [x] Phase 0b — ported tests to `test_demo_v6_2_*` (+ no-v6_1-import guard),
  wired into harness `run.py`. 15 pass.
- [x] Cleanup C1 — deleted dead leaf modules `utils/{io,visualizer,side_by_side_panel}.py`
  (876 lines, 0 importers). PARITY OK.
- [x] Phase A1 — split `chunk_data_stream.py` (1743) → focused modules
  (chunk_jsonl_tail, chunk_warmup_trim, chunk_capture_meta, chunk_window_builder,
  chunk_materialize) + thin entrypoint shell, all <1000 ln. Extract-and-re-export,
  PARITY OK. Removed dead `WarmupTrimResult.warmup_row` + compute loop (~56 ln), PARITY OK.
- [x] Review cleanup P3 — removed `_chunk_data_window_from_rows` and its duplicate
  mask/trajectory helpers; capture rows now require `prepared_phystwin_frame_path`
  and fail immediately when it is absent. Removed the unused orchestrator mask
  reprocess flags, the OpenCV flood-fill fallback, and the two zero-call SAM3.1
  tuple wrappers. Canonical prepared-frame filtering remains fixed at the existing
  `True / 0.01 m / 40 neighbors` behavior. Added 8 cleanup regression tests;
  smoke passes 185 tests, and regenerated chunk 0 is byte-identical to the frozen
  output (`sha256 04b3c1ba...d9009`).
- [x] End-to-end camera-half verification — bounded fake-live smoke of demo_v6_2
  (orchestrator → camera/tracker/warmup subprocess → SAM3D shape-prior → chunk
  materialization) exits 0, 2 chunks + points.npz + final_mesh.glb, well-formed.
  Confirms the Phase-0 subprocess-launch rewire works on the live path (offline
  parity does not cover it).
- [x] Phase C — split `main.py` (1663→6: main_config/cli/options/layout/subprocess+thin)
  PARITY OK+11 tests; split `visualize_track.py` (2365→7 viz_* modules) import/CLI OK.
- [x] Phase B — split+prune `fast_foundation_stereo.py` (1603→5, deleted dead torch +
  single-engine runners + apply_remove_invisible_mask ~450 ln); split
  `main_data_processing.py` (6773→thin+14 modules incl. 6-mixin decomposition of
  MainDataProcessingDemo, 74 methods preserved). **Every runtime file now <1000 lines**
  (max phystwin_strict_product.py 968). Verified: offline PARITY OK + fake-live smoke
  (exit 0, 2 chunks + points.npz + final_mesh.glb, zero runtime errors).
- [x] Duplication pass 1 — shared helpers: `_radius_outlier_filter_stage`
  (pcd_postprocess), `_reclaim_cuda_memory` (main_warmup release/trim),
  `_normalize_mask_stack` (sam31), `utils/jsonl_io.read_jsonl_rows` (viz delegates;
  phystwin_strict correctly NOT merged — its inline loop raises on malformed lines vs
  the helper skipping them). PARITY OK + 15 tests.
- [ ] Phase A2-A5 — remaining product-schema cleanup
  (chunk_data_output/payload schema fallback ladders [careful, parity-gated],
  tracking re-validation, asap [KEEP silent-freeze]).
- [ ] Duplication pass 2 — bigger items: query schema derived 3×/chunk; REPO_ROOT
  (mostly essential — bootstrap needs inline). Each fake-live/smoke-gated.
- [x] Phase D — live pipeline-status visualization (Q23). New `pipeline_status.py`
  (best-effort append-only `pipeline_status.jsonl`, no-op on None base, never
  raises); wired into the orchestrator (run_start / chunk_committed /
  downstream_start / run_finished|fatal), the camera process (capture_start /
  shape_prior / warmup_ready / fatal_error via `_record_fatal_worker_error`), and
  the shape-prior submit. Renderer `viz_panels.draw_pipeline_status` (status band,
  red on fatal) wired into `viz_playback.run_side_by_side`. Verified: fake-live run
  populated the full event sequence; `tests/test_demo_v6_2_pipeline_status.py`
  (6 tests: roundtrip, none-base no-op, torn-line tolerance, never-raises, band +
  fatal-red). Offline PARITY still OK (new sidecar file only).
- [x] Phase E — `demo_v6_2/PIPELINE.md` (all 23 questions answered in Chinese
  against the current module structure) + question-answering module docstrings on the "big step"
  modules (mdp_capture_source Q1-6, mdp_demo_segwarmup Q7-14, chunk_materialize
  Q15-21, online_frame_archive Q20-22; tracking + pipeline_status already carried
  theirs). main.py docstring corrected to demo_v6_2.
  DEFERRED (optional): emitted-identity rename (`demo_version`/`case_prefix`/
  `schema_version` still read `demo_v6_1`) — kept so online output stays
  byte-identical to the frozen reference; needs a comparator carve-out + a
  downstream check before flipping. Code structure is fully self-contained v6_2.

- [x] Live-camera verification (2026-07-09 evening) — first live run failed at
  startup: `resolve_serial` picks the sorted-first D400 device; with 4 cameras
  connected that is the D405 (no 848x480 BGR8 color sensor → "Could not find
  requested sensor type!"). Pre-existing behavior (identical in frozen v6_1), not
  a refactor regression (confirmed by per-device minimal repro: D405 fails,
  D455 starts). Fix: `--camera-serial` passthrough (main_cli → main_subprocess →
  mdp_cli `--serial` → `_start_realsense_pipeline`), default None keeps old
  selection; 3 new tests. Also closed a Phase-D gap: camera startup errors now
  emit a `fatal_error` status event (previously only worker-thread failures did).
  Verified live on the table-calibration camera (`--camera-serial 239222300740`):
  capture starts (metadata.json written), frame-0 SAM3.1 runs, and with nobody at
  the table the run fail-fasts exactly at the design_spec two-hand gate
  ("SAM3.1 did not produce two separable controller masks for 'hand'"), with the
  red fatal_error visible in pipeline_status.jsonl. That failure is the EXPECTED
  no-operator outcome; the live path is healthy.
- [x] Camera-serials config (2026-07-09 evening, user request) — replaced the
  single-value `--camera-serial` default with a config list
  `camera.camera_serials: ["239222300740"]` (multi-camera-extensible schema).
  `main_options.resolve_camera_serials` enforces the single-camera invariant:
  any count != 1 fails fast with "single-camera runtime requires exactly one
  serial" (wired into validate_runtime_args, the subprocess command builder —
  which now always forwards `--serial <resolved>` —, the dry-run contract, and
  the run summary). CLI `--camera-serial` is now repeatable (append) and
  overrides the config list. 6 serial tests (27 total green), exact-message CLI
  check, offline PARITY OK, fake-live smoke green. demo_v6_2 only.
- [x] Real-time online_data color/depth streaming (2026-07-09 night, user request) —
  `OnlineFrameArchive.stream_frame` writes each frame's `color/0/{k}.png` +
  `depth/0/{k}.npy` the moment the stream bridge accepts the capture row (frame
  cadence), instead of waiting for chunk materialization. `archive_chunk` now
  VERIFIES already-streamed frames (seq identity) instead of rewriting; the
  committed output remains byte-identical to the safe fixture.
  `metadata.json` `frame_num` stays commit-gated (strict
  readers unchanged; live consumers watch the dirs). `discard_streamed_tail`
  removes never-committed tail frames at stream end. Controlled one-chunk
  fake-live proof:
  frames 0..34 landed over 6.60 s at 5 FPS; the chunk committed 5.04 s after
  frame 34, and metadata then advanced to `frame_num=35`.
- [x] Trainer finish policy reconciliation (2026-07-09 night) — local config
  intentionally sets `train.stop_when_finished: false`, so Trainer runs exactly
  `iterations` total steps instead of stopping on manifest `finished`. Default
  config and explicit-command tests now assert the same policy; the override
  test proves `true` still passes through when selected.
- [x] Warm-up live RGB input preview (2026-07-09 night, user request) — new
  `mdp_warmup_preview.WarmupRgbPreview`: an OpenCV window in the camera process
  showing the live RGB input during warm-up in EVERY downstream.mode (reads
  `input_preview_slot` in memory, zero disk IO; NOT the tracking-chunk
  visualizer, whose per-mode policy is unchanged). Closes at warm-up end (the
  WARMUP_FINISHED banner site; frame-0-seed site when shape-prior warm-up is
  disabled) and IMMEDIATELY on warm-up failure/cancel/early exit (render loop
  watches `stop_event`; `stop()` also joins-closes). Best-effort GUI: display
  failures log one line and disable, never touching capture. Flags
  `--warmup-rgb-preview/--no-warmup-rgb-preview` (default on) at both
  orchestrator and camera CLIs, forwarded by build_main_data_processing_command.
  4 new WarmupRgbPreviewTests (fake-cv2 injection: shows frames, closes on
  normal end, closes on stop_event, disabled never opens; flag forwarding).
- [x] PhysTwin supervisor terminal output (2026-07-10, user request) — Demo
  now drains the supervisor's combined stdout/stderr in a dedicated relay
  thread, writes every line to both the parent terminal with a
  `[phystwin_shen]` prefix and the existing
  `phystwin_shen/online_full_pipeline.log`, and sets `PYTHONUNBUFFERED=1` for
  immediate Python-stage output. Relay failure terminates the supervisor and
  surfaces a launch error; normal/forced cleanup waits for the relay to drain.
  Regression test proves stdout and stderr arrive before a blocked supervisor
  exits and remain in the log.
- [x] PhysTwin Boba runtime ownership (2026-07-10, user request) — added
  Stage 1/2 `max_online_chunks`, `cma_popsize`, `zero_order_backend`, and
  `sim_force_mode` to Demo's authoritative YAML and explicit supervisor
  command. Defaults preserve the updated external pipeline behavior
  (`2/4/boba/gather` for Stage 1 and `10/4/boba/gather` for Stage 2); invalid
  limits/backend/mode fail before camera launch, and command tests assert all
  eight flags.
- [x] PhysTwin single combined viewer (2026-07-10, user request) — changed the
  authoritative defaults to `cma_viewer.source=all` and
  `train_viewer.enabled=false`, so one HTML process displays Stage 1, optional
  Stage 2, and train rows. Validation rejects any runtime mapping that enables
  both viewer sections; launch tests now prove only the combined viewer port is
  reclaimed, and the real-wrapper dry run emits exactly one viewer command.
- NOTE (parity fixture): the P3 session curated `outputs_v6_1/capture/frames.jsonl`
  (549→539 rows; removed the 10 startup rows lacking prepared frames, src
  2376..2430, mtime 16:55). Golden refs generated before that are stale — always
  regenerate the golden ref from frozen demo_v6_1 against the CURRENT fixture
  before diffing (done: PARITY OK on the current fixture; no code regression).

## Final state (2026-07-09)

Every runtime `.py` in demo_v6_2 is <1000 lines (was one 6773-line file). The
runtime path is verified by a bounded fake-live smoke (exit 0, well-formed,
status stream populated). 44 scoped
tests green (29 demo_v6_2 including cleanup, standalone-no-v6_1 guard, and
pipeline-status; 15 demo_v6_1 reference intact); the repository smoke profile
passes all 185 tests. Nothing committed (working tree, `single-camera` branch).
Remaining optional: duplication pass 2 (query-schema 3x), product-schema cleanup
A2-A5, and viewer renderer display check.

## 2026-07-12 follow-up — config predicates out of the mixins + typing contract

Motivation: the six `mdp_demo_*` mixins shared 50 cross-file method-call edges
through `self`; about a quarter were pure config predicates that only read
`args` (or one startup-frozen attribute), and the rest were invisible to type
checkers because every mixin alone is an incomplete class.

Step 1 — predicates become module-level pure functions (call sites rewritten
mechanically, methods deleted from `_LifecycleMixin`; no behavior change):

- `mdp_cli.py` (home of the existing `tracker_enabled`-style accessors) gains
  `lossless_enabled(args)`, `lossless_input_fps(args)`, `shape_prior_profile(manager)`,
  `shape_prior_profile_payload(manager, args)`, `write_shape_prior_profile_json(manager, args, profile=None)`.
- `mdp_constants.py` gains `table_world_enabled(table_c2w)` and
  `pcd_coordinate_frame(table_c2w)` next to the frame-kind constants.
- `mdp_demo_segwarmup.py` metadata now uses `lossless_enabled(...)` for the four
  keys that previously inlined `tracker_enabled(args) and args.pcd_mode == "masked"`
  (identical expression, single name).

Step 2 — `mdp_demo_contract.py` declares `_DemoRuntimeContract`, an
annotation-only base all six mixins inherit: 71 shared attributes (everything
constructed in `_LifecycleMixin.__init__` that ≥2 mixins touch, grouped by
subsystem) plus `NotImplementedError` stubs for the remaining cross-mixin
methods, each labeled with its implementing mixin. Zero runtime effect: type
imports sit under `TYPE_CHECKING`, the assembled MRO puts every stub behind its
real implementation (asserted programmatically), and `__init__` chains exactly
as before.

Verified: py_compile + import smoke; MRO/stub-override assertion; predicate
truth-table equivalence vs the original expressions; scoped v6_2 tests green;
all three harness guards green; bounded fake-live structural run: run_finished
(max_chunks_reached), 2 chunks committed, shape_prior/points.npz, zero
tracebacks, and every predicate-derived metadata field spot-checked
(tracker_sync_policy=strict_same_seq_lossless_5fps, lossless_input_fps=5.0,
backlog=15, coordinate_frame=pcd_coordinate_frame=table_world_z0,
tracker_strict_same_seq_render=true). The runtime-filter cap field recorded by
the original validation was removed with the runtime PCD filter on 2026-07-12.

Two PRE-EXISTING environment landmines hit while verifying (not caused by the
refactor; both will bite any future automation/headless launch):

1. SAM3D `notebook/inference.py` line 5 does
   `os.environ["CUDA_HOME"] = os.environ["CONDA_PREFIX"]` at import. With
   base-conda `CONDA_PREFIX` (any non-interactive shell) torch regenerates
   `~/.cache/torch_extensions/py312_cu130/nvdiffrast_plugin/build.ninja`
   against the header-less `miniforge3/bin/nvcc` → `cuda_runtime.h: No such
   file`. Same root cause as the 2026-07-02 v5.1 verify failures. demo_2_max's
   conda CUDA also fails here (headers under targets/, conda-gcc sysroot
   clash); a standard layout works: `CONDA_PREFIX=/usr/local/cuda`.
2. The 2026-07-10 `.venv` rebuild (torch 2.10→2.11.0, nvdiffrast 0.3.3):
   torch 2.11's `_import_module_from_library` no longer registers the JIT
   module in `sys.modules`, so nvdiffrast ops.py's follow-up
   `importlib.import_module("nvdiffrast_plugin")` raises ModuleNotFoundError
   even after a successful build — generate's mesh postprocess is broken in
   the current .venv REGARDLESS of env vars. Workaround used for the verify
   run and available for operators until nvdiffrast is patched/pinned:
   `PYTHONPATH=~/.cache/torch_extensions/py312_cu130/nvdiffrast_plugin`
   (the .so imports directly by name). Also note `.venv` (uv py3.12+cu130)
   and conda `demo_2_max` (py3.12+cu130) SHARE that torch-extensions dir and
   rewrite each other's build.ninja; setting a per-env `TORCH_EXTENSIONS_DIR`
   would end this class of breakage.

## Target module layout (every file <1000 lines; each is one "big step")

Oversized sources and their decomposition (validated against the actual code by
the mapping pass):

- `main_data_processing.py` (6773) → `mdp_packets.py`, `mdp_cli.py`,
  `mdp_capture_source.py` (RealSense + replay), `headless_capture_writer.py`,
  `mdp_pipeline_plumbing.py` (OrderedPacketQueue/SameSeqPairer/StageStats),
  `mdp_pcd_depth.py`, `mdp_segmentation.py`, `mdp_warmup_orchestration.py`,
  `mdp_tracker_pcd_workers.py`, `mdp_demo.py` (MainDataProcessingDemo lifecycle),
  thin `main_data_processing.py` (__main__).
- `chunk_data_stream.py` (1743) → `chunk_jsonl_tail.py`, `chunk_warmup_trim.py`,
  `chunk_capture_meta.py`, `chunk_window_builder.py`, `chunk_materialize.py`, thin
  `chunk_data_stream.py` (one prepared-frame-only realtime entry point).
- `visualize_track.py` (2365) → `viz_camera_model.py`, `viz_input_timeline.py`,
  `viz_panels.py` (+ new `_draw_pipeline_status`), `viz_renderers.py`,
  `viz_playback.py`, `viz_video_export.py`, thin `visualize_track.py` CLI.
- `utils/fast_foundation_stereo.py` (1603) → `ffs_disparity_products.py`,
  `ffs_tensorrt_infra.py`, `ffs_foundation_loader.py`,
  `ffs_runner_two_stage.py` (the ONLY live runner, built at main_data_processing
  depth_source=='ffs'); DELETE the dead `FastFoundationStereoRunner` (torch) and
  `FastFoundationStereoSingleEngineTensorRTRunner` + `apply_remove_invisible_mask`.
- `main.py` (1663) → `main_cli.py`, `main_options.py`, `main_layout.py`,
  `main_subprocess.py`, thin `main.py`.

## Refactor catalog (highlights)

Dead code removed includes the legacy FFS runners,
`apply_remove_invisible_mask`, and `main_warmup`
`run_sam31_first_frame_masks`/`resolve_initial_masks` (0 callers). Remaining
catalog: `shape_prior_align.py` `VIS` blocks (~90 ln, const False);
`main_data_processing._run_deferred_shape_prior_after_teardown` (no-op);
`chunk_data_stream` `WarmupTrimResult.warmup_row` + its dead compute loop;
9/11 `utils/depth_geometry` fns + 10/12 `utils/ffs_defaults` consts (unused);
`utils/render.py` unused Open3D layer classes; `env/check_demo_v6_1_env.py`.

Remaining over-defensive code to simplify (only where provably safe under parity): `getattr(args,
'flag', default)` on own argparse Namespace (67× in main_data_processing);
schema fallback ladders in `chunk_data_output._static_mapping_vectors` /
`chunk_data_payload._final_data_payload` (rebuild ids from weaker sources the
realtime path always provides); `tracking._check_frozen_identity` re-validation.
**KEEP** `asap._deform_frame` silent-freeze fallback — it is a DELIBERATE
downstream-contract behavior mandated by design_spec_v6_1.md.

Duplication to collapse: REPO_ROOT resolution (6 copies, inconsistent depth) →
one helper; JSONL reader (4 copies); query schema derived up to 3×/chunk;
tracking alias arrays; rainbow color helpers.

Legacy paths still pending separate proof: `ffs_depth_path`/legacy-npz fallbacks.
The chunk stream itself is prepared-frame-only; it no longer reconstructs a
window from RGB/depth/mask/trajectory sidecars.

Scope note: `demo_v6_2/others/` (experimental scripts + committed image dataset)
is not imported by the runtime; quarantine/exclude, do NOT delete unilaterally
(committed data, hard to reverse) — confirm with user.

## The 23 pipeline questions — answers (file:line into demo_v6_2)

1. **Camera start** — orchestrator `main.py` spawns `main_data_processing.py` as a
   subprocess (`main.py:1498`); that process starts capture in
   `MainDataProcessingDemo.run` → `_start_realsense_pipeline` (live,
   `main_data_processing.py:2518`) or `RecordedRgbdFrameSource` (fake-live,
   `:396`).
2. **Camera thread creation** — capture runs on daemon `threading.Thread`s started
   in the demo's thread setup (`_capture_worker` `:4422`, `_capture_recording_worker`
   `:4234`; masked-EdgeTAM workers `:4186`). Threads collected in `self._threads`
   (`:3599`).
3. **Thread/process mechanism** — multi-process (orchestrator + camera + shape-prior
   subprocesses + downstream), and within the camera process Python `threading`
   with custom `OrderedPacketQueue` (seq-ordered) + `LatestSlot` (latest-wins) +
   `SameSeqPairer`; `stop_event` coordinates shutdown.
4. **RealSense RGB/depth FPS** — a single fps applied to every enabled stream:
   `int(args.fps)` in `_start_realsense_pipeline` (`:2527-2532`), fed from
   `--camera-fps` (default **30**, choices {5,15,30,60}; `config/default.yaml:45`).
   Output/chunk cadence is separate: `--replay-fps` default **5** samples the
   latest frame every 1/replay_fps.
5. **Per-frame read** — capture worker pulls RealSense frames (or replays recorded
   RGB-D refs), wraps them as `FramePacket` (`:312`), pushes into
   `lossless_frame_queue`/`capture_slot`; consumers read via `get_latest_after`.
6. **frame_id / timestamp origin** — `seq` (packet sequence) is the frame id;
   `source_timestamp_s`/`source_frame_index` come from recording metadata /
   capture (`:499-547`), persisted on prepared frames (`:1185-1188`). Published
   timeline time = `frame_index × 1/fps` (design_spec.md:8); source_* kept as
   provenance.
7. **Warm-up = first single frame** (not a segment) — `prepare_segmentation_warmup`
   grabs exactly one frame via `_wait_for_first_frame` (`main_warmup.py:493`) and
   runs SAM3.1 on frame 0 for the frozen identity masks; EdgeTAM seeded on frame 0
   with `add_prompt=True` (`main_data_processing.py:4742`).
8. **How it knows it's the first frame** — sentinel-seq read: `_wait_for_first_frame`
   calls `capture_slot.get_latest_after(-1)` (seq>-1 ⇒ the very first frame), or the
   head of `lossless_frame_queue` (`:5616-5625`); the seg worker treats that packet
   as frame 0 and seeds EdgeTAM once.
9. **Leftover frames during warm-up** — consumed but NOT chunked: later frames run
   through EdgeTAM with `add_prompt=False` (`:4759-4786`) so trackers advance and
   the left preview updates, but `frames.jsonl` writes only the warmup frame-0 row
   until the formal gate lifts (`:6245`; design_spec.md:7).
10. **Most time-consuming warm-up step** — the SAM3D shape-prior chain (5 stages in
    `ShapePriorLocalClient.request_shape_prior`, `shape_prior_warmup.py:574-688`):
    upscale (`:591`), SAM3.1 segment (`:609`), **SAM3D generate (`:614`)** — the
    dominant cost — align/SuperGlue (`:621`), sample (`:627`); each timed as
    `shape_prior_*_ms`. No fixed duration is checked in; `asap_ms`/stage timings
    are telemetry.
11. **Post-warmup states** — (a) `InitialMaskBundle` frozen frame-0 masks
    (`main_warmup.py:356`); (b) `SegmentationWarmupState` (hf_stream, model,
    processor, first_frame, masks; `:511`); (c) a seeded EdgeTAM session for
    hand_a/object/hand_b; (d) on disk the shape-prior case dir + `points.npz`
    (surface/interior) + `final_mesh.glb` (`shape_prior_warmup.py:265`).
12. **State validation** — fail-fast shape/existence checks: masks match frame
    shape (`main_warmup.py:382`), `_union_masks` rejects empty/mismatched
    (`:86-96`), `split_controller_hand_instances` requires two hands (else raise);
    shape-prior rejects empty object mask / no valid depth points
    (`shape_prior_warmup.py:239,329`) and a point-count gate. No geometric quality
    check.
13. **Warm-up error handling** — isolated then fail-fast: stage subprocesses run
    `check=True` (`shape_prior_warmup.py:172`) → non-zero raises; seg-warmup
    exceptions route to `_record_fatal_worker_error` which sets `stop_event` and
    tears the process down to a non-zero exit (`main_data_processing.py:4782`,
    `3988-4001`). Shape-prior terminal failure lifts the chunk gate so the bridge
    reports loudly instead of hanging (design_spec.md:10-11).
14. **Formal timeline start** — one frame after warmup frame 0: warmup frame-0 row
    holds output-frame-0's anchor slot; the first frame processed after
    shape_prior=READY (gate lifts, `_formal_chunk_rows_gated` `:976`) becomes
    output frame 1, stitched directly after frame 0 (`:6241-6252`; design_spec.md:8).
15. **Chunk stitching** — rows/prepared-frames accumulate into a buffer until it
    reaches `chunk_size`, then the window closes; per-frame arrays are stacked on a
    new leading frame axis and lifted to world PCD
    (`chunk_data_stream.py:1706-1721`, window build `:955`, stack `:899`).
16. **Chunk size config** — `ChunkDataWriter.chunk_size` (validated >0,
    `chunk_data_output.py:237`), supplied as `chunk_frame_count` to both entry
    points; resolved in `main.py:637` `resolve_chunk_frame_count` =
    `round(replay_fps × chunk_seconds)` (default 5×7=35) unless `--chunk-frame-count`.
17. **Time vs frame-count** — strictly by **frame/row count**: window closes when
    `len(buffer) == chunk_size` (`chunk_data_stream.py:1709`, comment `:1703`).
18. **Post-stitch tracking** — a session-lived `tracking.TrackingRuntime` (created
    once, `chunk_data_stream.py:1588`) so chunk-0 identity/anchors/neighbor-table
    freeze; per window `build_window_observations` (`tracking.py:59`) →
    `process_window` runs the design_spec state machine (frozen labels, per-frame
    temporary_invalid, local-rigid anchor recovery).
19. **Post-track filtering** — motion-consistency gate `motion_consistency`
    (`tracking.py:196`), a verbatim port of
    `data_process_origin/data_process_track.filter_motion` (0.01 m radius, ≥5
    neighbors incl. self, 0.005 m similarity, 50% agreement). Depth-validity and
    3D radius-outlier mask refinement run once when the camera writes each
    prepared frame (`phystwin_strict_product.prepare_phystwin_frame`).
20. **Training-side schema** — per-window pickle `online_data/chunks/chunk_{id:06d}.pkl`
    (`chunk_data_output.py:164`): meta (case_name, chunk_id, start/end_frame,
    source_frame_indices, source_timestamps_s) + TIME_KEYS sliced to the window
    (`data_keys.py`): required object_points/colors/visibilities/motions_valid
    (T,N,3 / T,N,3 / T,N / T,N) + controller_points (T,M,3), optional
    asap_surface/interior_points + recovery masks; plus the RGB-D archive
    `online_data/color/0/{k}.png`, `depth/0/{k}.npy` (uint16 mm), `calibrate.pkl`,
    `metadata.json`, `enhance_metadata.json`, and `data/final_data.pkl` aggregate.
21. **Manifest update timing** — fsync-ordered so a reader that sees a new manifest
    is guaranteed the data is durable: on each commit — atomic_pickle_dump chunk
    (`chunk_data_output.py:308`) → fsync archive frame → rewrite manifest/metadata
    (`:411`); manifest `status` goes `recording` → `finished`/`failed` (`:353`).
22. **Training-side start** — when `shape_prior/points.npz` appears, Demo starts
    one phystwin_shen `scripts/run_online_full_pipeline.py` supervisor. The
    wrapper starts one combined `cma_viewer.source=all` viewer and runs Stage 1
    → optional Stage 2 → train; the dedicated train viewer is disabled. Its
    readers poll `online_data/manifest.json` from the first
    committed chunk forward; trainer finish behavior comes from the local
    `train.stop_when_finished` value passed explicitly by Demo.
23. **Real-time viz today** — a separate `visualize_track.py` subprocess
    (`main.py:810`) shows side-by-side live input RGB vs `final_data` output chunks
    (render modes rgb-overlay / sam3d-final-data). NEW: a live pipeline-status
    overlay (`viz_panels._draw_pipeline_status`) fed by a `<base_path>/pipeline_status.jsonl`
    that the orchestrator + camera + shape-prior stages append to (stage,
    frame, warmup progress/fail, chunk count, timings).

## Live pipeline-status visualization (Phase D design)

Status source: append-only `<base_path>/pipeline_status.jsonl`, written by the
orchestrator (`main.py`: run start, warmup gate, chunk commits, phystwin
supervisor launch),
the camera process (`main_data_processing.py`: capturing, frame-0 seed, warmup
stage, fatal error), and shape-prior stages (per-stage start/done/fail). Each
line: `{t, source, stage, detail, frame_index?, ok?}`. Renderer: the visualizer
tails the file and draws a status band via `viz_panels._draw_pipeline_status`
(current stage, warmup ✓/✗ with the failing stage + message, chunks published,
last stage timings). Fail display: on a fatal warmup/shape-prior line the band
turns red and pins the error so the operator sees exactly what broke.

## 2026-07-12 follow-up — remove non-lossless masked PCD

Requirement: masked PCD is a strict same-sequence product and therefore
requires the TAPNext++ tracker. A masked-PCD request without a tracker must
fail during argument validation instead of starting the legacy latest-frame
`_pcd_worker`.

Plan:

- [x] Make TAPNext++ the camera-runtime CLI default and reject
  `pcd_mode=masked` with no tracker or `track_mode=none`.
- [x] Delete `_pcd_worker`, its unconsumed `pcd_slot`, and the corresponding
  mixin-contract/thread-selection branches.
- [x] Update the pipeline documentation and add fail-fast/default regression
  tests.
- [x] Run scoped static/tests plus the repository smoke profile.

Validation result: targeted runtime/archive suite passed 7 tests; Python
compilation, scoped Ruff undefined-name/unused-import checks, all harness
guards, CLI help probes, and the repository smoke profile passed.
