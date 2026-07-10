"""MainDataProcessingDemo lifecycle mixin (init/run/stop/threads)."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_capture_source import RecordedRgbdFrameSource, _start_realsense_pipeline
from demo_v6_2.mdp_cli import _is_replay_input_source, active_object_id_labels, controller_pcd_mask_erode_pixels, depth_backend_label, headless_capture_enabled, headless_capture_saved_pcd_source, object_pcd_mask_erode_pixels, pcd_filter_enabled, runtime_metadata_identity, tracker_enabled, tracker_marker_gate, tracker_marker_retirement_policy, tracker_query_source, tracker_retire_filtered_markers
from demo_v6_2.mdp_headless_writer import HeadlessCaptureWriter
from demo_v6_2.mdp_packets import FatalWorkerError
from demo_v6_2.mdp_pipeline_plumbing import OrderedPacketQueue, SameSeqPairer, StageStats
from demo_v6_2.mdp_warmup_preview import WarmupRgbPreview
from demo_v6_2.pipeline_status import (
    STAGE_CAPTURE_START,
    STAGE_FATAL,
    PipelineStatusWriter,
)
from demo_v6_2.utils.atomic_io import atomic_json_dump


class _LifecycleMixin:
    """MainDataProcessingDemo lifecycle mixin (init/run/stop/threads)."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize MainDataProcessingDemo."""
        self.args = args
        self.width, self.height = parse_profile(DEFAULT_PROFILE)
        self.lossless_max_backlog_frames = max(
            1,
            int(round(self._lossless_input_fps() * float(args.lossless_max_backlog_seconds))),
        )
        self.runtime: RealtimeCameraRuntime | None = None
        self.ray_x: np.ndarray | None = None
        self.ray_y: np.ndarray | None = None
        self.input_preview_slot: LatestSlot[FramePacket] = LatestSlot()
        # Dedicated monotonic seq for EVERY put into input_preview_slot (frame 0,
        # the warm-up preview pump, and resumed live output alike). The slot's
        # sole consumer (WarmupRgbPreview) accepts only strictly-increasing seq,
        # so preview publishes must never regress — output_seq restarting at 1
        # after warm-up would otherwise be rejected behind the pump's seq.
        self._input_preview_publish_seq = 0
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.mask_slot: LatestSlot[MaskPacket] = LatestSlot()
        self.depth_profile_slot: LatestSlot[DepthProfilePacket] = LatestSlot()
        # Latest non-strict PCD packet; consumed only by the headless debug worker.
        self.pcd_slot: LatestSlot[MaskedPcdPacket] = LatestSlot()
        self.tracker_marker_slot: LatestSlot[TrackerMarkerPacket] = LatestSlot()
        self.paired_render_slot: LatestSlot[PairedRenderPacket] = LatestSlot()
        self.lossless_frame_queue: OrderedPacketQueue[FramePacket] = OrderedPacketQueue(
            name="frame",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_pcd_mask_queue: OrderedPacketQueue[MaskPacket] = OrderedPacketQueue(
            name="mask-pcd",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_tracker_mask_queue: OrderedPacketQueue[MaskPacket] = OrderedPacketQueue(
            name="mask-tracker",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.lossless_pair_output_queue: OrderedPacketQueue[PairedBuildResult] = OrderedPacketQueue(
            name="pair-output",
            max_backlog_frames=self.lossless_max_backlog_frames,
        )
        self.same_seq_pairer = SameSeqPairer(max_backlog_frames=self.lossless_max_backlog_frames)
        self._lossless_pairer_lock = threading.Lock()
        self._lossless_publish_condition = threading.Condition()
        self._lossless_next_publish_seq = 0
        self._startup_hold_s = 0.0
        self.stop_event = threading.Event()
        self._lossless_capture_done = threading.Event()
        self._lossless_processing_done = threading.Event()
        self._lossless_first_pair_published = threading.Event()
        self._lossless_pipeline_active = False
        self._threads: list[threading.Thread] = []
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.depth_stats = StageStats()
        self.pcd_stats = StageStats()
        self.tracker_stats = StageStats()
        self.filter_submit_stats = StageStats()
        self.filter_output_stats = StageStats()
        self.filter_worker: Any | None = None
        self._filter_submit_skip_count = 0
        self._last_filter_output_seq_recorded = -1
        controller_filter_min_cap = int(5000)
        if self._lossless_enabled():
            controller_filter_min_cap = min(controller_filter_min_cap, DEFAULT_LOSSLESS_CONTROLLER_FILTER_MIN_CAP)
        self.object_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(12.0)) * 0.5,
            min_cap=int(5000),
            max_cap=max(int(5000), int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000),
            init_cap=int(args.object_filter_cap) if int(args.object_filter_cap) > 0 else 200_000,
        )
        self.controller_filter_budget = FilterBudgetController(
            target_ms=max(0.0, float(12.0)) * 0.5,
            min_cap=int(controller_filter_min_cap),
            max_cap=max(int(controller_filter_min_cap), int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000),
            init_cap=int(args.controller_filter_cap) if int(args.controller_filter_cap) > 0 else 200_000,
        )
        self.ffs_runner: object | None = None
        self._local_ffs_lock = threading.Lock()
        self._local_ffs_depth_cache: OrderedDict[int, tuple[np.ndarray, float, float]] = OrderedDict()
        self.ir_to_color_aligner: FfsIrToColorAligner | None = None
        self._ir_to_color_aligner_key: tuple[
            tuple[int, int],
            tuple[int, int],
            tuple[float, ...],
            tuple[float, ...],
            tuple[float, ...],
        ] | None = None
        self.recording_source: RecordedRgbdFrameSource | None = None
        self.headless_capture_writer: HeadlessCaptureWriter | None = None
        # Live pipeline-status stream (design question 23), shared with the
        # orchestrator + shape-prior stages under
        # <base_path>/pipeline_status.jsonl. base_path is the parent of the
        # headless capture dir; a None capture dir yields a no-op writer.
        self._status = PipelineStatusWriter(
            Path(args.headless_capture_dir).parent
            if args.headless_capture_dir is not None
            else None,
            "camera",
        )
        self.shape_prior_manager = self._create_shape_prior_manager()
        # Live RGB input preview shown ONLY during warm-up, in every downstream
        # mode; closes at warm-up end and immediately on failure/cancel/early
        # exit (stop_event + stop()). Not the tracking-chunk visualizer.
        self.warmup_rgb_preview = WarmupRgbPreview(
            input_preview_slot=self.input_preview_slot,
            stop_event=self.stop_event,
            enabled=bool(args.warmup_rgb_preview),
        )
        self._shape_prior_written = False
        self._formal_timeline_gated_frames = 0
        self._formal_timeline_metadata_written = False
        self._warmup_anchor_row_written = False
        self._formal_timeline_gate_started_s: float | None = None
        self._formal_timeline_gate_expired = False
        self._warmup_runtime_start_perf_s: float | None = None
        self._warmup_perception_profile: dict[str, Any] = {}
        self.table_c2w: np.ndarray | None = None
        self.table_calibration_path: Path | None = None
        self._first_frame_segmented = threading.Event()
        self._lossless_offered_frames = 0
        self._lossless_segmented_frames = 0
        self._lossless_pcd_results = 0
        self._lossless_tracker_results = 0
        self._lossless_pairs_emitted = 0
        self._tracker_query_points_yx: np.ndarray | None = None
        self._tracker_query_rgb_u8: np.ndarray | None = None
        self._tracker_query_is_object: np.ndarray | None = None
        self._tracker_query_is_controller: np.ndarray | None = None
        self._tracker_query_target_id: np.ndarray | None = None
        self._tracker_query_controller_instance_id: np.ndarray | None = None
        self._tracker_consistent_visible: np.ndarray | None = None
        self._tracker_query_alive_mask: np.ndarray | None = None
        self._tracker_query_initial_seq: int | None = None
        self._fatal_error_lock = threading.Lock()
        self._fatal_error: FatalWorkerError | None = None

    @property
    def intrinsics(self) -> CameraIntrinsics:
        """Return the intrinsics."""
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        return self.runtime.intrinsics

    @property
    def serial(self) -> str:
        """Return the serial."""
        if self.runtime is None:
            return "<not-started>"
        return self.runtime.serial

    def _table_world_enabled(self) -> bool:
        """Return whether table world is enabled."""
        return self.table_c2w is not None

    def _pcd_coordinate_frame(self) -> str:
        """Return the PCD coordinate frame."""
        return TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else COORDINATE_FRAME

    def _create_shape_prior_manager(self) -> shape_prior_warmup.ShapePriorWarmupManager:
        """Create the shape-prior warmup manager for the runtime."""
        enabled = bool(getattr(self.args, "shape_prior_warmup", False))
        client = None
        if enabled:
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=Path(self.args.shape_prior_case_root),
                cuda_visible_devices=str(
                    getattr(
                        self.args,
                        "shape_prior_warmup_cuda_visible_devices",
                        shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
                    )
                ),
                object_name=str("stuffed animal"),
                controller_name=str(self.args.shape_prior_controller_name),
                points_npz=Path(self.args.shape_prior_points_npz),
                sam3d_root=getattr(self.args, "shape_prior_sam3d_root", None),
                sam3d_config=getattr(self.args, "shape_prior_config", None),
                sam31_device=str(self.args.device),
                reuse_sam31_model=True,
            )
            if bool(getattr(self.args, "shape_prior_prewarm_stage_workers", False)):
                client.prewarm()
        return shape_prior_warmup.ShapePriorWarmupManager(
            enabled=enabled,
            client=client,
        )

    def _shape_prior_profile(self) -> dict[str, Any]:
        """Return the shape prior profile."""
        manager = getattr(self, "shape_prior_manager", None)
        if manager is None:
            return shape_prior_warmup.default_profile(enabled=False)
        return manager.profile()

    def _shape_prior_profile_payload(self) -> dict[str, Any]:
        """Return the shape prior profile payload."""
        profile = self._shape_prior_profile()
        payload = dict(profile)
        if payload.get("input_source") is None:
            payload["input_source"] = str(getattr(self.args, "input_source", ""))
        if payload.get("depth_backend") is None:
            payload["depth_backend"] = depth_backend_label(self.args)
        if payload.get("depth_source_internal") is None:
            payload["depth_source_internal"] = str(getattr(self.args, "depth_source", ""))
        return payload

    def _write_shape_prior_profile_json(self, profile: dict[str, Any] | None = None) -> None:
        """Write shape prior profile JSON."""
        path = getattr(self.args, "shape_prior_profile_json", None)
        if path is None:
            return
        output_path = Path(path)
        payload = self._shape_prior_profile_payload() if profile is None else dict(profile)
        atomic_json_dump(payload, output_path)

    def _initialize_table_calibration(self) -> None:
        """Initialize table calibration."""
        if self.args.table_calibrate is None:
            return
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        path = Path(self.args.table_calibrate)
        try:
            transforms = load_table_calibration_transforms(path, serial_numbers=[str(self.runtime.serial)])
        except TableCalibrationLoadError as exc:
            raise RuntimeError(f"Invalid table calibration for active camera {self.runtime.serial}: {exc}") from exc
        self.table_c2w = np.ascontiguousarray(transforms[0], dtype=np.float32)
        self.table_calibration_path = path
        print(
            "[table-calibrate] "
            f"path={path} serial={self.runtime.serial} pcd_coordinate_frame={TABLE_WORLD_FRAME_KIND}",
            flush=True,
        )

    def _lossless_enabled(self) -> bool:
        """Return whether lossless is enabled."""
        return bool(tracker_enabled(self.args) and self.args.pcd_mode == "masked")

    def _lossless_input_fps(self) -> float:
        """Return the lossless input FPS."""
        return float(getattr(self.args, "lossless_input_fps", DEFAULT_LOSSLESS_INPUT_FPS))

    def _reset_lossless_state(self) -> None:
        """Reset lossless state."""
        self.lossless_frame_queue.reset()
        self.lossless_pcd_mask_queue.reset()
        self.lossless_tracker_mask_queue.reset()
        self.lossless_pair_output_queue.reset()
        self.same_seq_pairer.reset()
        with self._lossless_publish_condition:
            self._lossless_next_publish_seq = 0
            self._lossless_publish_condition.notify_all()
        self._lossless_capture_done.clear()
        self._lossless_processing_done.clear()
        self._lossless_first_pair_published.clear()
        self._first_frame_segmented.clear()
        self._lossless_pipeline_active = True
        self._lossless_offered_frames = 0
        self._lossless_segmented_frames = 0
        self._lossless_pcd_results = 0
        self._lossless_tracker_results = 0
        self._lossless_pairs_emitted = 0

    def _close_lossless_queues(self) -> None:
        """Close lossless queues."""
        self.lossless_frame_queue.close()
        self.lossless_pcd_mask_queue.close()
        self.lossless_tracker_mask_queue.close()
        self.lossless_pair_output_queue.close()
        self._lossless_pipeline_active = False

    def _wait_for_lossless_startup_pair(
        self,
        on_wait_tick: Callable[[], None] | None = None,
    ) -> bool:
        """Wait until frame 0 has complete PCD and tracking results."""
        if not self._lossless_enabled() or self.args.track_mode == "none":
            return True
        while not self.stop_event.is_set():
            if self._lossless_first_pair_published.wait(timeout=0.01):
                return True
            if on_wait_tick is not None:
                on_wait_tick()
        return False

    def _build_headless_capture_metadata(self) -> dict[str, Any]:
        """Build headless capture metadata."""
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        shape_profile = self._shape_prior_profile_payload()
        replay_fps = None
        recording_fps = None
        frame_count = None
        recording_case = None
        if self.recording_source is not None:
            replay_fps = float(self.recording_source.effective_fps)
            recording_fps = float(self.recording_source.recording_fps)
            frame_count = int(self.recording_source.frame_count)
            recording_case = _repo_relative_path_text(self.recording_source.case_path)
        frame_selection_policy = (
            FAKE_LIVE_FRAME_SELECTION_POLICY if str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE else None
        )
        return {
            **runtime_metadata_identity(self.args),
            "input_source": str(self.args.input_source),
            "recording_case": recording_case,
            "replay_fps": replay_fps,
            "recording_fps": recording_fps,
            "fake_live_frame_selection_policy": frame_selection_policy,
            "recording_frame_count": frame_count,
            "color_exposure": (
                None
                if getattr(self.args, "color_exposure", None) is None
                else float(getattr(self.args, "color_exposure"))
            ),
            "color_gain": (
                None
                if getattr(self.args, "color_gain", None) is None
                else float(getattr(self.args, "color_gain"))
            ),
            "depth_source": str(self.args.depth_source),
            "depth_source_internal": str(self.args.depth_source),
            "depth_units": "meters",
            "depth_coordinate_frame": COORDINATE_FRAME,
            "depth_alignment_target": "color",
            "track_mode": str(self.args.track_mode),
            "edgetam_tracking_identities": list(active_object_id_labels(self.args).values()),
            "demo_visual_mode": str(self.args.demo_visual_mode),
            "tracker_backend": str(self.args.tracker_backend),
            "tracking_product_backend": str(
                normalize_tracking_product_backend(getattr(self.args, "tracking_product_backend", DEFAULT_TRACKING_PRODUCT_BACKEND))
            ),
            "headless_prepared_only": bool(getattr(self.args, "headless_prepared_only", False)),
            "write_input_rgb_timeline": bool(getattr(self.args, "write_input_rgb_timeline", False)),
            "phystwin_strict_output_dir": (
                None
                if getattr(self.args, "phystwin_strict_output_dir", None) is None
                else _repo_relative_path_text(self.args.phystwin_strict_output_dir)
            ),
            "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
            "mask_backend": "edgetam",
            "depth_backend": depth_backend_label(self.args),
            "shape_prior_enabled": bool(shape_profile.get("shape_prior_enabled", False)),
            "shape_prior_status": str(
                shape_profile.get(
                    "shape_prior_status",
                    shape_prior_warmup.STATUS_DISABLED,
                )
            ),
            "shape_prior_timeout_ms": int(
                getattr(
                    self.args,
                    "shape_prior_timeout_ms",
                    shape_prior_warmup.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
                )
            ),
            "shape_prior_warmup_cuda_visible_devices": str(
                getattr(
                    self.args,
                    "shape_prior_warmup_cuda_visible_devices",
                    shape_prior_warmup.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
                )
            ),
            "shape_prior_controller_name": str(
                getattr(self.args, "shape_prior_controller_name", "")
            ),
            "shape_prior_case_root": _repo_relative_path_text(
                getattr(self.args, "shape_prior_case_root", None)
            ),
            "shape_prior_points_npz": _repo_relative_path_text(
                getattr(self.args, "shape_prior_points_npz", None)
            ),
            "shape_prior_skip_route_visualizations": bool(
                getattr(self.args, "shape_prior_skip_route_visualizations", True)
            ),
            "shape_prior_source_seq": shape_profile.get("shape_prior_source_seq"),
            "shape_prior_source_time_s": shape_profile.get("shape_prior_source_time_s"),
            "shape_prior_depth_backend": depth_backend_label(self.args),
            "shape_prior_depth_source_internal": str(self.args.depth_source),
            "execution_mode": PHYSTWIN_STRICT_EXECUTION_MODE,
            "tracker_query_count": int(DEFAULT_TRACKER_QUERY_COUNT),
            "tracker_query_source": tracker_query_source(self.args) if tracker_enabled(self.args) else None,
            "tracker_marker_gate": tracker_marker_gate(self.args) if tracker_enabled(self.args) else None,
            "tracker_retire_filtered_markers": (
                tracker_retire_filtered_markers(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_marker_retirement_policy": (
                tracker_marker_retirement_policy(self.args) if tracker_enabled(self.args) else None
            ),
            "tracker_display_scope": str(DEFAULT_TRACKER_DISPLAY_SCOPE),
            "tracker_sync_policy": (
                "strict_same_seq_lossless_5fps" if self._lossless_enabled() else "none"
            ),
            "lossless_input_fps": float(self._lossless_input_fps()) if self._lossless_enabled() else None,
            "lossless_max_backlog_frames": int(self.lossless_max_backlog_frames) if self._lossless_enabled() else None,
            "pcd_filter_enabled": pcd_filter_enabled(self.args),
            "pcd_filter_mode": str(self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE),
            "pcd_filter_preset": getattr(self.args, "pcd_filter_preset", None),
            "saved_pcd_source": (
                headless_capture_saved_pcd_source(self.args) if headless_capture_enabled(self.args) else None
            ),
            "object_filter": str(self.args.object_filter),
            "controller_filter": str(self.args.controller_filter),
            "object_filter_keep_components": int(self.args.object_filter_keep_components),
            "controller_filter_keep_components": int(self.args.controller_filter_keep_components),
            "filter_radius_m": float(self.args.filter_radius_m),
            "filter_nb_points": int(self.args.filter_nb_points),
            "filter_min_cap": int(5000),
            "lossless_controller_filter_min_cap": (
                int(self.controller_filter_budget.min_cap) if self._lossless_enabled() else None
            ),
            "enhanced_component_voxel_size_m": float(self.args.enhanced_component_voxel_size_m),
            "pcd_max_points": int(60000),
            "pcd_stride": int(1),
            "pcd_mask_erode_pixels": int(DEFAULT_PCD_MASK_ERODE_PIXELS),
            "object_pcd_mask_erode_pixels": object_pcd_mask_erode_pixels(self.args),
            "controller_pcd_mask_erode_pixels": controller_pcd_mask_erode_pixels(self.args),
            "depth_min_m": float(0.2),
            "depth_max_m": float(1.5),
            "serial": str(self.runtime.serial),
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_frame": self._pcd_coordinate_frame(),
            "pcd_coordinate_frame": self._pcd_coordinate_frame(),
            "camera_coordinate_frame": COORDINATE_FRAME,
            "table_calibration_path": _repo_relative_path_text(self.table_calibration_path),
            "table_world_frame_kind": TABLE_WORLD_FRAME_KIND if self._table_world_enabled() else None,
            "table_z_m": TABLE_Z_M if self._table_world_enabled() else None,
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "camera_to_world_c2w": (
                None
                if self.table_c2w is None
                else np.asarray(self.table_c2w, dtype=np.float32).reshape(4, 4).tolist()
            ),
            "world_z_diagnostic_thresholds_m": [
                float(value) for value in DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M
            ],
            "table_z_filter_enabled": bool(self.args.enable_table_z_filter),
            "table_z_filter_threshold_m": float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
            "table_z_filter_classes": str(TABLE_Z_FILTER_CLASS_BOTH),
            "intrinsics": {
                "fx": float(self.runtime.intrinsics.fx),
                "fy": float(self.runtime.intrinsics.fy),
                "cx": float(self.runtime.intrinsics.cx),
                "cy": float(self.runtime.intrinsics.cy),
            },
            "k_color": np.asarray(self.runtime.k_color, dtype=np.float32).tolist(),
        }

    def _fatal_error_snapshot(self) -> FatalWorkerError | None:
        """Return the fatal error snapshot."""
        with self._fatal_error_lock:
            return self._fatal_error

    def _record_fatal_worker_error(self, stage: str, exc: BaseException) -> FatalWorkerError:
        """Record fatal worker error."""
        fatal = FatalWorkerError(stage=str(stage), exc_type=type(exc).__name__, message=str(exc))
        should_notify = False
        with self._fatal_error_lock:
            if self._fatal_error is None:
                self._fatal_error = fatal
                should_notify = True
            else:
                fatal = self._fatal_error
        if should_notify:
            print(f"[FATAL] {fatal.log_message()}", flush=True)
            # Surface the failure (e.g. warm-up / shape-prior errors) on the live
            # status band so the operator sees exactly what broke.
            self._status.emit(
                STAGE_FATAL,
                f"{fatal.stage}: {fatal.message}",
                ok=False,
                exc_type=fatal.exc_type,
            )
            self.stop_event.set()
        return fatal

    def run(self) -> int:
        """Run MainDataProcessingDemo."""
        self._warmup_runtime_start_perf_s = time.perf_counter()
        self._status.emit(STAGE_CAPTURE_START, f"input={self.args.input_source}")
        # Warm-up live RGB preview: opens with capture in every downstream
        # mode; closed at the warm-up-finished banner, in stop(), and by
        # stop_event on fatal errors.
        self.warmup_rgb_preview.start()
        main_warmup.prepare_runtime_services_and_source(
            self,
            pcd_filter_enabled=pcd_filter_enabled,
            is_replay_input_source=_is_replay_input_source,
            recording_source_cls=RecordedRgbdFrameSource,
            start_realsense_pipeline=_start_realsense_pipeline,
            fake_live_input_source=INPUT_SOURCE_FAKE_LIVE,
            fake_live_frame_selection_policy=FAKE_LIVE_FRAME_SELECTION_POLICY,
        )
        try:
            main_warmup.prepare_runtime_projection_and_capture(
                self,
                headless_capture_enabled=headless_capture_enabled,
                headless_capture_writer_cls=HeadlessCaptureWriter,
            )
            self._run_headless()
            self._finalize_headless_tracking_product()
        finally:
            self.stop()
        return 2 if self._fatal_error_snapshot() is not None else 0

    def _finalize_headless_tracking_product(self) -> None:
        """Finalize headless tracking product."""
        if self._fatal_error_snapshot() is not None:
            return
        if self.headless_capture_writer is None:
            raise RuntimeError("phystwin-strict-tracking requires an initialized headless capture writer")
        output_dir = (
            Path(self.args.phystwin_strict_output_dir)
            if getattr(self.args, "phystwin_strict_output_dir", None) is not None
            else self.headless_capture_writer.output_dir / "phystwin_like"
        )
        print(f"[phystwin-strict] finalizing output_dir={output_dir}", flush=True)
        manifest = finalize_headless_capture(self.headless_capture_writer.output_dir, output_dir=output_dir)
        self.headless_capture_writer.update_metadata(
            {
                "phystwin_strict_output_dir": _repo_relative_path_text(output_dir),
                "phystwin_strict_manifest": _repo_relative_path_text(output_dir / "manifest.json"),
                "phystwin_strict_frame_count": int(manifest.get("frame_count", 0)),
                "phystwin_strict_query_count": int(manifest.get("query_count", 0)),
            }
        )
        print(
            "[phystwin-strict] "
            f"frames={manifest.get('frame_count')} queries={manifest.get('query_count')} "
            f"manifest={output_dir / 'manifest.json'}",
            flush=True,
        )

    def stop(self) -> None:
        """Stop MainDataProcessingDemo."""
        self.stop_event.set()
        # Warm-up failure, cancellation, or early exit must close the live RGB
        # preview immediately (the render loop also watches stop_event).
        self.warmup_rgb_preview.close()
        self._close_lossless_queues()
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads.clear()
        if self.runtime is not None:
            try:
                self.runtime.pipeline.stop()
            except Exception:
                pass
            self.runtime = None
        self.recording_source = None
        self._run_deferred_shape_prior_after_teardown()
        self._write_shape_prior_profile_json()
        if (
            self.headless_capture_writer is not None
            and self._formal_timeline_gated_frames > 0
            and not self._formal_timeline_metadata_written
        ):
            # The run ended while formal rows were still gated on the shape
            # prior: frames.jsonl holds only the warmup row and can never be
            # chunked. Mark the capture and route the failure through the
            # existing fatal-error path so the process exits nonzero.
            error_message = (
                "run ended while formal chunk rows were still gated on "
                f"the shape prior ({self._formal_timeline_gated_frames} frames "
                "withheld); the capture has no formal timeline and cannot be "
                "chunked."
            )
            self.headless_capture_writer.update_metadata(
                {
                    "formal_timeline_incomplete": True,
                    "formal_timeline_gated_frame_count": int(
                        self._formal_timeline_gated_frames
                    ),
                }
            )
            self._record_fatal_worker_error(
                "formal chunk timeline",
                RuntimeError(error_message),
            )
        self.headless_capture_writer = None
        if self.filter_worker is not None:
            self.filter_worker.stop()
            self.filter_worker = None
        with self._local_ffs_lock:
            self._local_ffs_depth_cache.clear()

    def _create_ffs_runner(self) -> object:
        """Create the configured FFS runner."""
        try:
            from demo_v6_2.utils.fast_foundation_stereo import (
                FastFoundationStereoTensorRTRunner,
            )

            return FastFoundationStereoTensorRTRunner(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=None if self.args.ffs_trt_root is None else Path(self.args.ffs_trt_root),
            )
        except Exception as exc:
            raise RuntimeError(f"failed to start FFS TensorRT runner: {type(exc).__name__}: {exc}") from exc

    def _get_ir_to_color_aligner(
        self,
        *,
        depth_shape: tuple[int, int],
        color_shape: tuple[int, int],
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
    ) -> FfsIrToColorAligner:
        """Return the get IR to color aligner."""
        k_ir = np.asarray(k_ir_left, dtype=np.float32).reshape(3, 3)
        transform = np.asarray(t_ir_left_to_color, dtype=np.float32).reshape(4, 4)
        k_col = np.asarray(k_color, dtype=np.float32).reshape(3, 3)
        key = (
            (int(depth_shape[0]), int(depth_shape[1])),
            (int(color_shape[0]), int(color_shape[1])),
            tuple(float(v) for v in k_ir.ravel()),
            tuple(float(v) for v in transform.ravel()),
            tuple(float(v) for v in k_col.ravel()),
        )
        if self._ir_to_color_aligner_key != key or self.ir_to_color_aligner is None:
            self.ir_to_color_aligner = FfsIrToColorAligner(
                k_ir_left=k_ir,
                t_ir_left_to_color=transform,
                k_color=k_col,
                ir_shape=depth_shape,
                color_shape=color_shape,
            )
            self._ir_to_color_aligner_key = key
        return self.ir_to_color_aligner

    def _start_threads(self) -> None:
        """Start threads."""
        if self._lossless_enabled():
            self._reset_lossless_state()
        workers: list[tuple[str, Callable[[], None]]] = [("capture", self._capture_worker)]
        if self.args.track_mode != "none":
            workers.append(("seg", self._seg_worker))
        if self._lossless_enabled():
            workers.append(("pcd", self._lossless_pcd_worker))
            workers.append(("tracker", self._lossless_tracker_worker))
            workers.append(("pair-output", self._lossless_pair_output_worker))
        elif tracker_enabled(self.args):
            workers.append(("tracker", self._tracker_worker))
        if self.args.pcd_mode == "masked":
            if not tracker_enabled(self.args):
                workers.append(("pcd", self._pcd_worker))
        elif self.args.depth_source == "ffs":
            workers.append(("depth", self._depth_profile_worker))

        def worker_runner(worker_name: str, worker_target: Callable[[], None]) -> Callable[[], None]:
            """Return the worker runner."""
            def run_worker() -> None:
                """Run worker."""
                try:
                    worker_target()
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self._record_fatal_worker_error(f"{worker_name} worker", exc)

            return run_worker

        for name, target in workers:
            thread = threading.Thread(target=worker_runner(name, target), name=f"masked-edgetam-{name}", daemon=True)
            thread.start()
            self._threads.append(thread)

    def _run_headless(self) -> None:
        """Run headless."""
        self._start_threads()
        try:
            while not self.stop_event.is_set():
                if self._lossless_enabled():
                    if self._lossless_processing_done.is_set():
                        self.stop_event.set()
                        break
                    time.sleep(0.05)
                    continue
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_event.set()


__all__ = ["_LifecycleMixin"]
