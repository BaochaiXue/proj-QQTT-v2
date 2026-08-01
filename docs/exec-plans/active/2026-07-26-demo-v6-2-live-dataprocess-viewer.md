# Demo v6.2 realtime data-process viewer (2026-07-26)

Owner ask: in one formal run, BOTH visualizations at once — the
PhysTwin_shen combined viewer (Stage 1 / optional Stage 2 / train / physics)
AND a Demo-6.2-own data-process viewer that no longer reads
`online_data/chunks/chunk_*.pkl` or waits for whole chunks, but subscribes
to each frame's finished products: processed object/controller masks,
current object/controller PCD, TAPNext++ tracks, shape-prior state, and
latency/FPS/point-count diagnostics. Constraints: coexist with
PhysTwin_shen (no more either/or), show the latest completed frame
continuously, never block camera/EdgeTAM/TAPNext++/PCD/chunk pipeline, drop
display frames when behind but never pipeline data, minimal VRAM/CPU/disk,
no GPU-0 usage (ideally no extra CUDA context at all), strictly read-only
on formal products.

## Design (implemented)

- **Not a downstream consumer.** `downstream.mode` mutual exclusivity is
  untouched (`phystwin_shen` keeps its GPU-0 combined viewer). The new
  viewer lives INSIDE the camera subprocess (GPU-1 namespace) as a daemon
  thread — `demo_v6_2/mdp/live_viewer.py::LiveDataProcessViewer`.
- **Publisher tap:** `FormalProductStage._publish_strict_pair` puts the
  shape-prior-enriched `PairedBuildResult` into a new latest-wins
  `LatestSlot` (`runtime.live_viz_slot`) — one O(1) lock+ref-swap per
  frame, no copies, no waiting. Every frame is published, including rows
  the formal timeline gate withholds during warm-up, so the viewer is live
  before WARMUP_FINISHED.
- **Viewer loop** mirrors `WarmupRgbPreview` discipline: poll
  `get_latest_after(last_seq)` (stale display frames drop via the slot),
  render only on a new seq, `waitKey(33)` paces/pumps; any GUI failure
  prints one line and exits the thread; `stop()`/`stop_event` close it.
- **Single GUI owner — `CvGuiLoop` (`mdp/gui_loop.py`):** the first design
  (sequential handoff: viewer defers until the preview thread exits) FAILED
  in the real run — this cv2 build is Qt HighGUI, and a second thread's
  `namedWindow` after the first GUI thread exits hangs forever (empirically
  verified: >10s block, zero errors; exactly the silent no-window symptom).
  The shipped design runs ONE persistent daemon GUI thread for the whole
  process: preview and viewer compose frames on their own threads (numpy +
  raster `putText`) and `submit`; only the loop thread calls
  `namedWindow/imshow/waitKey/destroyWindow`. Latest-wins per window;
  `close_window` tombstones a name so racing submits can never resurrect
  it; lazy start on first submit (headless runs never init HighGUI);
  windowless phases pace on the close event because `waitKey` returns
  immediately with zero windows (busy-spin found by adversarial review).
  The viewer window opens on the first published pair, DURING warm-up,
  alongside the preview window.
- **Rendering is CUDA-free, torch-free, disk-free:** numpy + cv2 on copies.
  Left panel RGB + processed-mask tints (object green, hand A red, hand B
  orange; controller fallback when hands absent) + visible TAPNext++ tracks
  painted with frozen query-rainbow colors (vectorized dot painting, no
  per-point cv2 calls). Right panel: object/controller PCD + shape-prior
  points reprojected via `CameraIntrinsics` (+ `inv(table_c2w)` when the
  packet frame is `table_world_z0`; camera-frame packets project directly).
  HUD: seq, end-to-end latency (same formula as `headless_writer`), stage
  FPS from `StageStatsBoard.fps_snapshot()`, display-drop count from the
  slot, point/track counts, `mask_ms/pcd_ms/tracker model/lift`,
  `shape_prior_status`, timeline open/warming-up. Composite downscaled by
  0.75 for screen fit.
- **Read-only guarantees** verified against the ownership audit: every
  array reachable from a published pair is written once per frame (or
  frozen, for query tables); the renderer draws exclusively on
  `.copy()`/fresh buffers, and a purity test asserts inputs are
  byte-unchanged after rendering.
- **Config/CLI:** `live_viewer.enabled: true` (default.yaml) →
  `DEFAULT_LIVE_DATAPROCESS_VIEWER` → `--live-dataprocess-viewer /
  --no-live-dataprocess-viewer` on both the orchestrator and camera CLIs,
  forwarded like the warm-up preview flag. Auto-off when the mode has no
  lossless pair publish.

## Files

- NEW `demo_v6_2/mdp/live_viewer.py` (renderer + compose thread)
- NEW `demo_v6_2/mdp/gui_loop.py` (`CvGuiLoop`, the single GUI thread)
- `demo_v6_2/mdp/formal_products.py` (ctor `live_viz_slot`, tap in
  `_publish_strict_pair`)
- `demo_v6_2/mdp/runtime.py` (gui loop + slot + viewer construction, start
  in `run()`, close/shutdown in `stop()`)
- `demo_v6_2/mdp/warmup_preview.py` (composes + submits through the loop)
- `demo_v6_2/config/default.yaml` (`live_viewer:` section),
  `orchestration/main_config.py`, `main_cli.py`, `mdp/cli.py`,
  `main_subprocess.py` (flag chain)
- NEW `tests/test_demo_v6_2_live_viewer.py` (9 tests: render purity,
  world-frame projection, latest-wins drop, GUI deferral handoff,
  disabled/broken-GUI, formal-stage tap, config chain)
- `demo_v6_2/PIPELINE.md` new section

## Verification

- 107 tests + 15 subtests green (12 viewer/gui-loop tests: render purity,
  world-frame projection, latest-wins, tombstoning, backend-failure
  disable, windowless no-busy-spin, formal-stage tap, config chain).
- Adversarial review (12-agent workflow) confirmed 4 findings, all fixed:
  (1) MAJOR CvGuiLoop busy-spin after last window closes (waitKey returns
  ~0.1us with zero windows — now paced on the close event); (2) HUD
  timeline label contradicted the real gate semantics for status=failed
  (now the inverse of the gate predicate); (3) stale tests from the
  abandoned handoff design (rewritten + CvGuiLoop coverage added);
  (4) stale docs/help describing the is_finished handoff (rewritten).
  Refuted: waitKey NULL-guiReceiver crash claim; GUI-failure propagation
  claim; slot-memory claim.
- Final fake-live run (exit 0, 2 chunks, run_finished ok): preview window
  opened via the gui loop, viewer submitted its FIRST frame at seq=0
  (during warm-up), viewer window opened alongside the preview, preview
  window closed alone at warm-up end; frame-0 masks, best_match.pkl and
  final_mesh.glb all BYTE-IDENTICAL to the no-viewer baseline
  (run_cachedir2) — the observer changed nothing.
- Known unrelated flake: the EdgeTAM precompile intermittently misses the
  fx cache (this run 20.5s vs 5.0s warm; fresh cache entries written with
  zero compiled-code changes). Pre-barrier cost only; tracked in the
  compile-policy investigation (next probe: run with
  TORCH_LOGS=+torch._inductor.codecache to catch a miss's reason).
