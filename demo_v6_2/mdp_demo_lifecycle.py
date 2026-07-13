"""MainDataProcessingDemo lifecycle mixin (init/run/stop/threads)."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_capture_source import (
    RecordedRgbdFrameSource,
    _start_realsense_pipeline,
)
from demo_v6_2.mdp_cli import (
    _is_replay_input_source,
    active_object_id_labels,
    depth_backend_label,
    headless_capture_enabled,
    lossless_enabled,
    lossless_input_fps,
    runtime_metadata_identity,
    shape_prior_profile_payload,
    tracker_enabled,
    write_shape_prior_profile_json,
)
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_headless_writer import HeadlessCaptureWriter
from demo_v6_2.utils.ffs_align import FfsDepthEngine
from demo_v6_2.mdp_packets import FramePacket, MaskPacket
from demo_v6_2.mdp_pipeline_plumbing import (
    FatalErrorLatch,
    LosslessPipeline,
    StageStats,
)
from demo_v6_2.visualization.mdp_warmup_preview import WarmupRgbPreview
from demo_v6_2.pipeline_status import STAGE_CAPTURE_START, PipelineStatusWriter


class _LifecycleMixin(_DemoRuntimeContract):
    """MainDataProcessingDemo lifecycle mixin (init/run/stop/threads)."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Initialize MainDataProcessingDemo."""
        self.args = args
        self.width, self.height = parse_profile(DEFAULT_PROFILE)
        self.runtime: RealtimeCameraRuntime | None = None
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
        self.tracker_marker_slot: LatestSlot[TrackerMarkerPacket] = LatestSlot()
        self.lossless = LosslessPipeline(
            max_backlog_frames=max(
                1,
                int(
                    round(
                        lossless_input_fps(args)
                        * float(args.lossless_max_backlog_seconds)
                    )
                ),
            )
        )
        self._startup_hold_s = 0.0
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self.capture_stats = StageStats()
        self.seg_stats = StageStats()
        self.depth_stats = StageStats()
        self.pcd_stats = StageStats()
        self.tracker_stats = StageStats()
        self.depth_engine: FfsDepthEngine | None = None
        self.recording_source: RecordedRgbdFrameSource | None = None
        self.headless_capture_writer: HeadlessCaptureWriter | None = None
        # Live pipeline-status stream (design question 25), shared with the
        # orchestrator + shape-prior stages under
        # <base_path>/pipeline_status.jsonl. base_path is the parent of the
        # headless capture dir; a None capture dir yields a no-op writer.
        self._status = PipelineStatusWriter(
            Path(args.headless_capture_dir).parent
            if args.headless_capture_dir is not None
            else None,
            "camera",
        )
        self.fatal = FatalErrorLatch(status=self._status, stop_event=self.stop_event)
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
        self._tracker_query_points_yx: np.ndarray | None = None
        self._tracker_query_rgb_u8: np.ndarray | None = None
        self._tracker_query_is_object: np.ndarray | None = None
        self._tracker_query_is_controller: np.ndarray | None = None
        self._tracker_query_target_id: np.ndarray | None = None
        self._tracker_query_controller_instance_id: np.ndarray | None = None
        self._tracker_consistent_visible: np.ndarray | None = None

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

    def _initialize_table_calibration(self) -> None:
        """Initialize table calibration."""
        if self.args.table_calibrate is None:
            raise RuntimeError(
                "formal runtime requires camera-to-world table calibration"
            )
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        path = Path(self.args.table_calibrate)
        try:
            transforms = load_table_calibration_transforms(
                path, serial_numbers=[str(self.runtime.serial)]
            )
        except TableCalibrationLoadError as exc:
            raise RuntimeError(
                f"Invalid table calibration for active camera {self.runtime.serial}: {exc}"
            ) from exc
        self.table_c2w = np.ascontiguousarray(transforms[0], dtype=np.float32)
        self.table_calibration_path = path
        print(
            "[table-calibrate] "
            f"path={path} serial={self.runtime.serial} "
            f"pcd_coordinate_frame={TABLE_WORLD_FRAME_KIND}",
            flush=True,
        )

    def _wait_for_lossless_startup_pair(
        self,
        on_wait_tick: Callable[[], None] | None = None,
    ) -> bool:
        """Wait until frame 0 has complete PCD and tracking results."""
        if not lossless_enabled(self.args) or self.args.track_mode == "none":
            return True
        while not self.stop_event.is_set():
            if self.lossless.first_pair_published.wait(timeout=0.01):
                return True
            if on_wait_tick is not None:
                on_wait_tick()
        return False

    def _build_headless_capture_metadata(self) -> dict[str, Any]:
        """Build headless capture metadata."""
        if self.runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        shape_profile = shape_prior_profile_payload(self.shape_prior_manager, self.args)
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
            FAKE_LIVE_FRAME_SELECTION_POLICY
            if str(self.args.input_source) == INPUT_SOURCE_FAKE_LIVE
            else None
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
            "edgetam_tracking_identities": list(
                active_object_id_labels(self.args).values()
            ),
            "demo_visual_mode": str(self.args.demo_visual_mode),
            "tracker_backend": str(self.args.tracker_backend),
            "tracking_product_backend": str(
                normalize_tracking_product_backend(
                    getattr(
                        self.args,
                        "tracking_product_backend",
                        DEFAULT_TRACKING_PRODUCT_BACKEND,
                    )
                )
            ),
            "headless_prepared_only": bool(
                getattr(self.args, "headless_prepared_only", False)
            ),
            "write_input_rgb_timeline": bool(
                getattr(self.args, "write_input_rgb_timeline", False)
            ),
            "phystwin_strict_output_dir": (
                None
                if getattr(self.args, "phystwin_strict_output_dir", None) is None
                else _repo_relative_path_text(self.args.phystwin_strict_output_dir)
            ),
            "compatibility_target": COMPATIBILITY_TARGET_PHYSTWIN,
            "mask_backend": "edgetam",
            "depth_backend": depth_backend_label(self.args),
            "shape_prior_enabled": bool(
                shape_profile.get("shape_prior_enabled", False)
            ),
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
            "tracker_query_source": (
                TRACKER_QUERY_SOURCE_UNION_MASK if tracker_enabled(self.args) else None
            ),
            "tracker_marker_gate": (
                TRACKER_MARKER_GATE_TARGET_MASK_DEPTH
                if tracker_enabled(self.args)
                else None
            ),
            "tracker_display_scope": str(DEFAULT_TRACKER_DISPLAY_SCOPE),
            "tracker_sync_policy": (
                "strict_same_seq_lossless_5fps"
                if lossless_enabled(self.args)
                else "none"
            ),
            "lossless_input_fps": (
                float(lossless_input_fps(self.args))
                if lossless_enabled(self.args)
                else None
            ),
            "lossless_max_backlog_frames": (
                int(self.lossless.max_backlog_frames)
                if lossless_enabled(self.args)
                else None
            ),
            "saved_pcd_source": (
                HEADLESS_CAPTURE_SAVED_PCD_SOURCE
                if headless_capture_enabled(self.args)
                else None
            ),
            "formal_mask_source": "origin_style_processed_masks",
            "formal_processing_fps": float(lossless_input_fps(self.args)),
            "depth_min_m": float(PHYSTWIN_DEPTH_MIN_M),
            "depth_max_m": float(PHYSTWIN_DEPTH_MAX_M),
            "mask_radius_outlier_radius_m": float(PHYSTWIN_RADIUS_OUTLIER_RADIUS_M),
            "mask_radius_outlier_nb_points": int(PHYSTWIN_RADIUS_OUTLIER_NB_POINTS),
            "serial": str(self.runtime.serial),
            "width": int(self.width),
            "height": int(self.height),
            "coordinate_frame": pcd_coordinate_frame(self.table_c2w),
            "pcd_coordinate_frame": pcd_coordinate_frame(self.table_c2w),
            "camera_coordinate_frame": COORDINATE_FRAME,
            "table_calibration_path": _repo_relative_path_text(
                self.table_calibration_path
            ),
            "table_world_frame_kind": (
                TABLE_WORLD_FRAME_KIND if table_world_enabled(self.table_c2w) else None
            ),
            "table_z_m": TABLE_Z_M if table_world_enabled(self.table_c2w) else None,
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "camera_to_world_c2w": (
                None
                if self.table_c2w is None
                else np.asarray(self.table_c2w, dtype=np.float32).reshape(4, 4).tolist()
            ),
            "intrinsics": {
                "fx": float(self.runtime.intrinsics.fx),
                "fy": float(self.runtime.intrinsics.fy),
                "cx": float(self.runtime.intrinsics.cx),
                "cy": float(self.runtime.intrinsics.cy),
            },
            "k_color": np.asarray(self.runtime.k_color, dtype=np.float32).tolist(),
        }

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
            is_replay_input_source=_is_replay_input_source,
            recording_source_cls=RecordedRgbdFrameSource,
            start_realsense_pipeline=_start_realsense_pipeline,
            fake_live_input_source=INPUT_SOURCE_FAKE_LIVE,
            fake_live_frame_selection_policy=FAKE_LIVE_FRAME_SELECTION_POLICY,
        )
        try:
            main_warmup.prepare_runtime_calibration_and_capture(
                self,
                headless_capture_enabled=headless_capture_enabled,
                headless_capture_writer_cls=HeadlessCaptureWriter,
            )
            self._run_headless()
            self._finalize_headless_tracking_product()
        finally:
            self.stop()
        return 2 if self.fatal.snapshot() is not None else 0

    def _finalize_headless_tracking_product(self) -> None:
        """Finalize headless tracking product."""
        if self.fatal.snapshot() is not None:
            return
        if self.headless_capture_writer is None:
            raise RuntimeError(
                "phystwin-strict-tracking requires an initialized headless capture writer"
            )
        output_dir = (
            Path(self.args.phystwin_strict_output_dir)
            if getattr(self.args, "phystwin_strict_output_dir", None) is not None
            else self.headless_capture_writer.output_dir / "phystwin_like"
        )
        print(f"[phystwin-strict] finalizing output_dir={output_dir}", flush=True)
        manifest = finalize_headless_capture(
            self.headless_capture_writer.output_dir, output_dir=output_dir
        )
        self.headless_capture_writer.update_metadata(
            {
                "phystwin_strict_output_dir": _repo_relative_path_text(output_dir),
                "phystwin_strict_manifest": _repo_relative_path_text(
                    output_dir / "manifest.json"
                ),
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
        self.lossless.close_queues()
        for thread in list(self._threads):
            if thread.is_alive():
                thread.join(timeout=1.0)
        self._threads.clear()
        if self.runtime is not None and self.runtime.pipeline is not None:
            try:
                self.runtime.pipeline.stop()
            except Exception:
                pass
        self.runtime = None
        self.recording_source = None
        self._run_deferred_shape_prior_after_teardown()
        write_shape_prior_profile_json(self.shape_prior_manager, self.args)
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
            self.fatal.record(
                "formal chunk timeline",
                RuntimeError(error_message),
            )
        self.headless_capture_writer = None
        self.depth_engine = None

    def _start_threads(self) -> None:
        """Start threads."""
        if lossless_enabled(self.args):
            self.lossless.reset()
            self._first_frame_segmented.clear()
        workers: list[tuple[str, Callable[[], None]]] = [
            ("capture", self._capture_worker)
        ]
        if self.args.track_mode != "none":
            workers.append(("seg", self._seg_worker))
        if lossless_enabled(self.args):
            workers.append(("processed-frame", self._lossless_processed_frame_worker))
            workers.append(("tracker", self._lossless_tracker_worker))
            workers.append(("pair-output", self._lossless_pair_output_worker))
        elif tracker_enabled(self.args):
            workers.append(("tracker", self._tracker_worker))
        if self.args.pcd_mode == "none" and self.args.depth_source == "ffs":
            workers.append(("depth", self._depth_profile_worker))

        def worker_runner(
            worker_name: str, worker_target: Callable[[], None]
        ) -> Callable[[], None]:
            """Return the worker runner."""

            def run_worker() -> None:
                """Run worker."""
                try:
                    worker_target()
                except Exception as exc:
                    if not self.stop_event.is_set():
                        self.fatal.record(f"{worker_name} worker", exc)

            return run_worker

        for name, target in workers:
            thread = threading.Thread(
                target=worker_runner(name, target),
                name=f"masked-edgetam-{name}",
                daemon=True,
            )
            thread.start()
            self._threads.append(thread)

    def _run_headless(self) -> None:
        """Run headless."""
        self._start_threads()
        try:
            while not self.stop_event.is_set():
                if lossless_enabled(self.args):
                    if self.lossless.processing_done.is_set():
                        self.stop_event.set()
                        break
                    time.sleep(0.05)
                    continue
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_event.set()


__all__ = ["_LifecycleMixin"]
