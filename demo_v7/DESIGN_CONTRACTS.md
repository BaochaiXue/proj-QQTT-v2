# demo_v7 implementation contracts (for parallel implementation)

Rules for every module:
- **Never edit anything under `demo_v6_2/` or `qqtt/`** — import (or
  subclass) only. PhysTwin_shen is call-only via
  `demo_v6_2.phystwin_shen_launch`.
- All command/event/channel names come from `demo_v7/ipc/protocol.py`.
- Service side must never block its pipeline threads on the GUI: frame
  sends are best-effort latest-wins; a dead GUI connection = drop frames,
  keep running; control socket disconnect in FORMAL = keep running,
  finish cleanly on next connect or shutdown command.
- Style: match demo_v6_2 (type hints, docstrings explaining constraints,
  no needless comments), files < 1000 lines.

## ipc/channel.py

```python
class ControlServer:  # service side
    def __init__(self, socket_path: Path, *, on_command: Callable[[dict], dict | None]): ...
    # accept ONE client at a time (re-accept after disconnect); each JSON
    # line -> on_command(obj) on a dedicated reader thread; returned dict
    # (the ack) is sent immediately; send_event(dict) is thread-safe and
    # never blocks the caller >50ms (per-connection outbox queue + writer
    # thread; drop events if the outbox exceeds 1000).
    def send_event(self, event: dict) -> None: ...
    def close(self) -> None: ...

class ControlClient:  # GUI side
    def __init__(self, socket_path: Path, *, on_event: Callable[[dict], None]): ...
    def send_command(self, cmd: dict) -> None: ...   # fire-and-forget; acks arrive as events
    def close(self) -> None: ...

class FrameStreamServer:  # service side
    def __init__(self, socket_path: Path): ...
    def publish(self, channel: str, jpeg_bytes: bytes, *, width: int, height: int) -> None: ...
    # latest-wins per channel: writer thread sends the newest pending frame
    # per channel; publish() never blocks (swap under lock).
    def close(self) -> None: ...

class FrameStreamClient:  # GUI side
    def __init__(self, socket_path: Path, *, on_frame: Callable[[FrameHeader, bytes], None]): ...
    def close(self) -> None: ...
```
Wire format per `protocol.py`. Include unit tests (socketpair or real UDS in
tmpdir): round-trip commands/events, frame latest-wins under slow reader,
clean close. No Qt imports here.

## service/frame0_pipeline.py — frame-0 derived computation (no tracking)

Pure orchestration of existing demo_v6_2 functions on ONE captured frame:

```python
@dataclass(frozen=True)
class Frame0Bundle:
    color_bgr: np.ndarray; depth_u16: np.ndarray
    intrinsics: Any; depth_scale_m_per_unit: float
    object_mask: np.ndarray; hand_a_mask: np.ndarray; hand_b_mask: np.ndarray

def compute_sam31_masks(color_bgr, *, device, args) -> tuple[object_mask, hand_a, hand_b]:
    # EXACTLY the v6.2 frame-0 seeding source: the SAM3.1 initial-mask
    # bundle used by SegmentationStage._prepare_warmup — find it in
    # demo_v6_2/mdp/warmup.py (InitialMaskBundle + its producer; the
    # "two separable controller masks" gate must fire the same way).

def build_frame0_processed(bundle: Frame0Bundle, *, args, session) -> MaskedPcdPacket-like:
    # Reuse demo_v6_2.phystwin_strict_product: dense_world_pcd_grid,
    # apply_depth_validity_to_mask_frame, apply_radius_outlier_to_mask_frame
    # — same call pattern as FormalProductStage._build_processed_frame_result
    # (demo_v6_2/mdp/formal_products.py:209-330), single frame.

def submit_shape_prior(manager: ShapePriorWarmupManager, processed, *, args) -> None:
    # Build the same ShapePriorFrame0Request the v6.2 publisher builds
    # (demo_v6_2/mdp/shape_prior_flow.py maybe_start_from_pcd_result ->
    # shape_prior/warmup.py maybe_submit:1102) and submit it.

def save_review_artifacts(run_dir: Path, bundle, ...) -> dict[str, dict[str, str]]:
    # frame0 rgb/depth-preview pngs; mask pngs + 50%-alpha overlay png.
    # Returns {ARTIFACT_KIND_*: {name: abs_path}} for EVT_ARTIFACTS.
```

## service/staged_runtime.py + service/camera_service.py

`camera_service.py` = subprocess entry (mirrors demo_v6_2/main_data_processing.py
bootstrap incl. TORCHINDUCTOR_CACHE_DIR + repo sys.path + import stamp).
Args: `--socket-dir` plus the demo_v6_2 camera flags it forwards to
`demo_v6_2.mdp.cli.build_parser()` (reuse that parser via parse_known_args
on the v6.2 subset; v7 flags parsed separately).

`StagedRuntime` — the state machine (states in protocol.py):

- STARTING: CameraSession + prepare_source + initialize_table_calibration
  (demo_v6_2/mdp/session.py:53,81); PerceptionPreloader started
  (demo_v6_2/mdp/preload.py — reuse legs incl. sam31; tracker leg too, it
  is needed at FORMAL); ShapePriorWarmupManager created + prewarm()
  (demo_v6_2/shape_prior/warmup.py:469) + notify_frame0_geometry(:1074)
  once camera geometry known. Acquisition thread starts: reuse the same
  per-frame read/align/copy code CaptureStage uses (find the lowest-level
  per-frame functions in demo_v6_2/mdp/capture.py and drive them from a v7
  loop — fake-live replay must reuse the v6.2 replayer for identical
  pacing/frame selection; pause/resume hooks added V7-SIDE, not in v6.2).
  Frames -> FrameStreamServer (CH_RGB + CH_DEPTH colormap via cv2).
- PREVIEW: acquisition + preview streaming only. EdgeTAM/tracker NOT
  consuming (models preloading in background is fine).
- capture_frame0: freeze the latest aligned frame (color+depth+intrinsics)
  as the candidate; fake-live: pause replay. -> FRAME0_PENDING.
- confirm_frame0 -> WARMUP: run frame0_pipeline on a worker thread:
  compute_sam31_masks -> save_review_artifacts(masks) -> EVT_ARTIFACTS;
  build_frame0_processed -> submit_shape_prior; poll manager.profile()
  (:1208) for stage progress -> EVT_PROGRESS; on ready ->
  EVT_ARTIFACTS(shape_prior + alignment: pull artifact paths from the
  shape-prior case dir the same way demo_v6_2 lays them out) -> REVIEW.
  Preview keeps streaming (real camera; fake stays paused on frame-0).
- begin_reposition -> REPOSITION: acquisition frames additionally
  composited with 50% alpha of the saved masks -> CH_OVERLAY. Fake-live:
  resume replay here (assume the recording's own layout matches).
- start_formal -> FORMAL: construct the SAME lossless stage set as
  demo_v6_2/mdp/runtime.py _start_threads lossless branch (LosslessPipeline,
  SegmentationStage, FormalProductStage, TrackerStage, HeadlessCaptureWriter,
  ShapePriorPublisher, FormalTimelineGate, StageStatsBoard, live_viz_slot).
  Subclass SegmentationStage overriding ONLY its warmup/frame-0 seeding so
  the first formal frame is prompted with the SAVED frame-0 SAM3.1 masks
  (same _run_segmentation_frame(add_prompt=True) path, no SAM3.1 rerun,
  no readiness barrier); everything downstream identical. A v7 thread
  polls live_viz_slot and renders demo_v6_2.mdp.live_viewer.render_pair_frame
  -> JPEG -> CH_COMPOSITE; EVT_FORMAL_STATS periodically from
  StageStatsBoard.fps_snapshot(). Disable v6.2's own windows (warmup
  preview + live viewer args off).
- stop_formal / replay exhausted: close queues like runtime.stop(),
  finalize writer (finalize_headless_capture as in runtime), ->
  FINISHED (+ EVT_REPLAY_EXHAUSTED first when fake ran out).
- Fatal anywhere: FatalErrorLatch-style -> EVT_ERROR + STATE_FATAL.

## gui/ (+ app.py)

PySide6 Widgets, dark-neutral default styling, Chinese labels matching
README's 界面流程. `app.py`: parse args (--source, --fake-live-case,
--base-path), show SourceSelect dialog if unspecified, construct
`OrchestratorSession` (from orchestration/session.py), then MainWindow.

- MainWindow: QStackedWidget center (Capture/Warmup/Review/Reposition/
  Formal/Finished screens) + persistent top-right dock with two
  ImageView widgets (CH_RGB, CH_DEPTH). All IPC callbacks hop to the Qt
  main thread via signals. Buttons send protocol commands through the
  session; state events drive screen switching.
- widgets.ImageView: setFrame(jpeg_bytes) decodes (cv2) -> QImage ->
  QPixmap, keeps aspect; latest-wins (pending frame replaced, repaint
  coalesced with a 30Hz QTimer).
- widgets.ProgressTimeline: ordered stage rows with status glyphs + ms.
- Review screen: tabs Masks / Shape Prior / Gaussian; images from
  EVT_ARTIFACTS paths (turntable contact-sheet stills, no video widget).
- Formal screen: big ImageView on CH_COMPOSITE with [复合|RGB|深度]
  switcher (switch = which channel feeds the big view; dock keeps living),
  stats row from EVT_FORMAL_STATS, stop button; Finished screen offers
  [回到开始](tears down session, returns to source select) and shows the
  run dir. Replay-exhausted popup per README.

## orchestration/session.py + config/default.yaml + tests/drive_fake_live.py

`OrchestratorSession` (no Qt imports):
- prepares the run output dir exactly like demo_v6_2/main.py (reuse
  prepare_realtime_output_for_new_run + pipeline_status run_start).
- builds the camera-service command by REUSING
  demo_v6_2.main_subprocess.build_main_data_processing_command(args,...)
  then swapping argv[1] target to demo_v7/service/camera_service.py and
  appending --socket-dir; env: CUDA_VISIBLE_DEVICES from the same v6.2
  config resolution; spawn like main.py:263-267.
- connects ControlClient + FrameStreamClient once sockets appear; exposes
  send_command/on_event/on_frame plus threadsafe shutdown.
- runs ChunkStreamSession on a background thread once FORMAL starts,
  wired identically to demo_v6_2/main.py:279-299 (incl. points_npz and
  before_poll=phystwin trigger via demo_v6_2 main helpers or an equivalent
  v7 copy of `_ensure_phystwin_shen_running` semantics; downstream mode
  from config, disabled must work for tests).
- config/default.yaml: v7-only keys (source default, fake_live_case
  default, socket_dir template, preview caps override); pipeline knobs
  are NOT duplicated — the v6.2 orchestrator args come from
  demo_v6_2.main_cli.build_parser() defaults.

`tests/drive_fake_live.py`: headless end-to-end — start OrchestratorSession
(downstream disabled, fake-live case), no GUI; script the control channel:
wait PREVIEW -> capture -> confirm -> wait REVIEW (assert masks + shape
prior artifacts exist) -> begin_reposition -> start_formal -> wait 2 chunks
committed (poll online_data/chunks) -> stop -> assert clean FINISHED +
points.npz. Prints PASS/FAIL; exit code accordingly.

## gaussian backends (post-contract addition, 2026-08-14)

`gaussian_backend` run option (source-select combo / CLI / config):
- `triposplat` (default): independent generative model in a persistent
  worker subprocess; canonical splats are registered onto the world
  (chamfer chain) then upgraded by the observation-metric self-align
  (demo 7 default alignment).
- `mesh_surface` (trial; only offered when the mesh backend is trellis2):
  NO second generative model. Splats are deterministically sampled on the
  aligned world mesh (`shape/matching/final_mesh.glb`) with face_id +
  barycentric anchors (`gaussian_anchors.npz`, self-verifying topology
  hash) — splat centers sit ON the mesh by construction, so registration,
  ICP, ARAP residual transfer, floater pruning and self-align are all
  structurally absent. Live: bones deform the MESH VERTICES (same
  rest-anchored LBS + bone hygiene), splats replay from their triangles —
  they can never drift off the mesh. The mesh is the single geometry
  truth; splat world accuracy therefore equals mesh alignment accuracy by
  design.
- `none`: feature off.

Cross-option rule (enforced in orchestration/session.py, mirrored as a
fail-soft degrade in camera_service.py, gated in the GUI combo):
`mesh_surface` requires `shape_prior_backend == trellis2`.
