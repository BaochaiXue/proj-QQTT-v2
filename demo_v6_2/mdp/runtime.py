"""Composition root of the camera subprocess: build, wire, run, stop.

``MainDataProcessingDemo`` constructs the shared services (session, lossless
pipeline, fatal latch, formal-timeline gate, stage stats, shape-prior
manager) and the four pipeline stages, wires them together explicitly, and
owns the run loop, worker threads, and teardown ordering. All pipeline logic
lives in the stage classes; this class only composes them.
"""

from __future__ import annotations

import argparse
import json
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from demo_v6_2.shape_prior import warmup as shape_prior_warmup
from demo_v6_2.mdp.capture import CaptureStage
from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.constants import (
    DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES,
    HEADLESS_CAPTURE_SAVED_PCD_SOURCE,
    pcd_coordinate_frame,
)
from demo_v6_2.mdp.formal_products import FormalProductStage
from demo_v6_2.mdp.headless_writer import HeadlessCaptureWriter
from demo_v6_2.mdp.packets import FramePacket, MaskPacket
from demo_v6_2.mdp.plumbing import (
    FatalErrorLatch,
    FormalTimelineGate,
    LosslessPipeline,
    StageStatsBoard,
)
from demo_v6_2.mdp.segmentation import SegmentationStage
from demo_v6_2.mdp.session import CameraSession
from demo_v6_2.mdp.shape_prior_flow import ShapePriorPublisher
from demo_v6_2.mdp.tracker import TrackerStage
from demo_v6_2.mdp.warmup_preview import WarmupRgbPreview
from demo_v6_2.phystwin_strict_product import finalize_headless_capture
from demo_v6_2.pipeline_status import STAGE_CAPTURE_START, PipelineStatusWriter
from demo_v6_2.utils.concurrency import LatestSlot
from demo_v6_2.utils.ffs_align import FfsDepthEngine, warm_up_numba_ffs_align
from demo_v6_2.utils.render import apply_wslg_open3d_env_defaults


class MainDataProcessingDemo:
    """Camera -> segmentation -> processed frame -> tracker/PCD -> products."""

    def __init__(self, args: argparse.Namespace) -> None:
        """Build the shared services and wire the pipeline stages."""
        self.args = args
        self.mode = RunMode.from_args(args)
        self.session = CameraSession()
        self.input_preview_slot: LatestSlot[FramePacket] = LatestSlot()
        self.capture_slot: LatestSlot[FramePacket] = LatestSlot()
        self.mask_slot: LatestSlot[MaskPacket] = LatestSlot()
        self.lossless = LosslessPipeline(
            max_backlog_frames=max(
                1,
                int(
                    round(
                        self.mode.lossless_input_fps
                        * float(args.lossless_max_backlog_seconds)
                    )
                ),
            )
        )
        self.stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self.stage_stats = StageStatsBoard()
        # Live pipeline-status stream (design question 25), shared with the
        # orchestrator + shape-prior stages under
        # <base_path>/pipeline_status.jsonl. base_path is the parent of the
        # headless capture dir (validation guarantees it is set).
        self._status = PipelineStatusWriter(
            Path(args.headless_capture_dir).parent, "camera"
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
        manager = self.shape_prior_manager
        self.timeline_gate = FormalTimelineGate(
            shape_prior_status=lambda: str(
                manager.profile().get(
                    "shape_prior_status", shape_prior_warmup.STATUS_DISABLED
                )
            ),
            timeout_ms=int(args.shape_prior_timeout_ms),
        )
        self._first_frame_segmented = threading.Event()

        self.capture = CaptureStage(
            args=args,
            mode=self.mode,
            session=self.session,
            lossless=self.lossless,
            capture_slot=self.capture_slot,
            input_preview_slot=self.input_preview_slot,
            stage_stats=self.stage_stats,
            first_frame_segmented=self._first_frame_segmented,
            stop_event=self.stop_event,
            fatal=self.fatal,
        )
        self.segmentation = SegmentationStage(
            args=args,
            mode=self.mode,
            session=self.session,
            lossless=self.lossless,
            capture_slot=self.capture_slot,
            mask_slot=self.mask_slot,
            stage_stats=self.stage_stats,
            shape_prior_manager=self.shape_prior_manager,
            warmup_rgb_preview=self.warmup_rgb_preview,
            first_frame_segmented=self._first_frame_segmented,
            stop_event=self.stop_event,
            fatal=self.fatal,
        )
        self.shape_prior = ShapePriorPublisher(
            args=args,
            mode=self.mode,
            manager=self.shape_prior_manager,
            session=self.session,
            timeline_gate=self.timeline_gate,
            status=self._status,
            warmup_rgb_preview=self.warmup_rgb_preview,
            segmentation=self.segmentation,
        )
        self.tracker = TrackerStage(
            args=args,
            session=self.session,
            lossless=self.lossless,
            mask_slot=self.mask_slot,
            stage_stats=self.stage_stats,
            timeline_gate=self.timeline_gate,
            stop_event=self.stop_event,
            fatal=self.fatal,
        )
        self.formal = FormalProductStage(
            args=args,
            session=self.session,
            lossless=self.lossless,
            stage_stats=self.stage_stats,
            timeline_gate=self.timeline_gate,
            shape_prior=self.shape_prior,
            capture=self.capture,
            stop_event=self.stop_event,
            fatal=self.fatal,
        )

    def _create_shape_prior_manager(
        self,
    ) -> shape_prior_warmup.ShapePriorWarmupManager:
        """Create the shape-prior warmup manager for the runtime."""
        enabled = bool(self.args.shape_prior_warmup)
        client = None
        if enabled:
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=Path(self.args.shape_prior_case_root),
                cuda_visible_devices=str(
                    self.args.shape_prior_warmup_cuda_visible_devices
                ),
                object_name=str("stuffed animal"),
                controller_name=str(self.args.shape_prior_controller_name),
                points_npz=Path(self.args.shape_prior_points_npz),
                sam3d_root=self.args.shape_prior_sam3d_root,
                sam3d_config=self.args.shape_prior_config,
                sam31_device=str(self.args.device),
                reuse_sam31_model=True,
                volume_sample_size_m=float(self.args.volume_sample_size_m),
            )
            if bool(self.args.shape_prior_prewarm_stage_workers):
                client.prewarm()
        return shape_prior_warmup.ShapePriorWarmupManager(
            enabled=enabled,
            client=client,
            input_source=str(self.args.input_source),
            depth_backend_label=self.mode.depth_backend_label,
            depth_source=str(self.args.depth_source),
            profile_json=self.args.shape_prior_profile_json,
        )

    def _build_headless_capture_metadata(self) -> dict[str, Any]:
        """Build headless capture metadata."""
        session = self.session
        if session.camera_runtime is None:
            raise RuntimeError("camera runtime is not initialized")
        shape_profile = self.shape_prior_manager.profile_payload()
        replay_fps = None
        frame_count = None
        if session.recording_source is not None:
            replay_fps = float(session.recording_source.effective_fps)
            frame_count = int(session.recording_source.frame_count)
        return {
            "input_source": str(self.args.input_source),
            "replay_fps": replay_fps,
            "recording_frame_count": frame_count,
            "depth_source": str(self.args.depth_source),
            "depth_source_internal": str(self.args.depth_source),
            "depth_backend": self.mode.depth_backend_label,
            "headless_prepared_only": bool(self.args.headless_prepared_only),
            "write_input_rgb_timeline": bool(self.args.write_input_rgb_timeline),
            "shape_prior_status": str(
                shape_profile.get(
                    "shape_prior_status",
                    shape_prior_warmup.STATUS_DISABLED,
                )
            ),
            "shape_prior_error": shape_profile.get("shape_prior_error"),
            "lossless_input_fps": (
                float(self.mode.lossless_input_fps)
                if self.mode.lossless_enabled
                else None
            ),
            "saved_pcd_source": HEADLESS_CAPTURE_SAVED_PCD_SOURCE,
            "serial": str(session.camera_runtime.serial),
            "width": int(session.width),
            "height": int(session.height),
            "pcd_coordinate_frame": pcd_coordinate_frame(session.table_c2w),
            "camera_to_world_c2w": (
                None
                if session.table_c2w is None
                else np.asarray(session.table_c2w, dtype=np.float32)
                .reshape(4, 4)
                .tolist()
            ),
            "intrinsics": {
                "fx": float(session.camera_runtime.intrinsics.fx),
                "fy": float(session.camera_runtime.intrinsics.fy),
                "cx": float(session.camera_runtime.intrinsics.cx),
                "cy": float(session.camera_runtime.intrinsics.cy),
            },
            "k_color": np.asarray(
                session.camera_runtime.k_color, dtype=np.float32
            ).tolist(),
        }

    def run(self) -> int:
        """Run MainDataProcessingDemo."""
        self.segmentation.warmup_runtime_start_perf_s = time.perf_counter()
        self._status.emit(STAGE_CAPTURE_START, f"input={self.args.input_source}")
        # Warm-up live RGB preview: opens with capture in every downstream
        # mode; closed at the warm-up-finished banner, in stop(), and by
        # stop_event on fatal errors.
        self.warmup_rgb_preview.start()
        apply_wslg_open3d_env_defaults()
        if self.args.depth_source == "ffs":
            self.session.depth_engine = FfsDepthEngine(
                ffs_repo=Path(self.args.ffs_repo),
                model_dir=Path(self.args.ffs_trt_model_dir),
                trt_root=(
                    None
                    if self.args.ffs_trt_root is None
                    else Path(self.args.ffs_trt_root)
                ),
                cache_frames=DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES,
            )
            warm_up_numba_ffs_align()
        self.session.prepare_source(self.args, self.mode)
        try:
            self.session.initialize_table_calibration(self.args)
            self.session.headless_capture_writer = HeadlessCaptureWriter(
                self.args.headless_capture_dir,
                metadata=self._build_headless_capture_metadata(),
            )
            print(
                "[headless-capture] "
                f"dir={self.session.headless_capture_writer.output_dir}",
                flush=True,
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
        writer = self.session.headless_capture_writer
        if writer is None:
            raise RuntimeError(
                "formal runtime requires an initialized headless capture writer"
            )
        output_dir = Path(self.args.phystwin_strict_output_dir)
        print(f"[phystwin-strict] finalizing output_dir={output_dir}", flush=True)
        manifest = finalize_headless_capture(writer.output_dir, output_dir=output_dir)
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
        self.session.release_camera()
        self.shape_prior_manager.write_profile_json()
        incomplete_error = self.timeline_gate.incomplete_run_error()
        writer = self.session.headless_capture_writer
        if writer is not None and incomplete_error is not None:
            # The run ended while formal rows were still gated on the shape
            # prior. Route the failure through the fatal-error path; no reader
            # consumed the former duplicate capture-metadata diagnostics.
            self.fatal.record(
                "formal chunk timeline",
                RuntimeError(incomplete_error),
            )
        self.session.headless_capture_writer = None
        self.session.depth_engine = None

    def _start_threads(self) -> None:
        """Start threads."""
        if self.mode.lossless_enabled:
            self.lossless.reset()
            self._first_frame_segmented.clear()
        workers: list[tuple[str, Callable[[], None]]] = [
            ("capture", self.capture.run)
        ]
        if self.args.track_mode != "none":
            workers.append(("seg", self.segmentation.run))
        if self.mode.lossless_enabled:
            workers.append(("processed-frame", self.formal.processed_frame_worker))
            workers.append(("tracker", self.tracker.run_lossless))
            workers.append(("pair-output", self.formal.pair_output_worker))
        elif self.mode.tracker_enabled:
            workers.append(("tracker", self.tracker.run_latest))

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
        last_telemetry_s = time.perf_counter()
        try:
            while not self.stop_event.is_set():
                if self.mode.lossless_enabled:
                    if self.lossless.processing_done.is_set():
                        self.stop_event.set()
                        break
                    now_s = time.perf_counter()
                    if now_s - last_telemetry_s >= 5.0:
                        last_telemetry_s = now_s
                        print(
                            "[queue-telemetry] "
                            + json.dumps(self.lossless.telemetry(), sort_keys=True),
                            flush=True,
                        )
                    time.sleep(0.05)
                    continue
                time.sleep(0.05)
        except KeyboardInterrupt:
            self.stop_event.set()


__all__ = ["MainDataProcessingDemo"]
