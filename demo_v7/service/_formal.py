"""Formal stage-set construction for the demo_v7 camera service (private).

Builds the SAME lossless stage set as ``demo_v6_2/mdp/runtime.py``'s
``_start_threads`` lossless branch — LosslessPipeline, SegmentationStage,
FormalProductStage, TrackerStage, HeadlessCaptureWriter, ShapePriorPublisher,
FormalTimelineGate, StageStatsBoard, live_viz_slot — with exactly one
substitution: ``SavedMaskSegmentationStage`` seeds the fresh EdgeTAM session
from the SAVED frame-0 SAM3.1 masks instead of rerunning SAM3.1. Everything
downstream of frame-0 seeding is the untouched v6.2 code.
"""

from __future__ import annotations

import argparse
import threading
import time
from types import SimpleNamespace
from typing import Any

import numpy as np

from demo_v6_2.shape_prior import warmup as shape_prior_warmup
from demo_v6_2.mdp import warmup as mdp_warmup
from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.constants import (
    HEADLESS_CAPTURE_SAVED_PCD_SOURCE,
    pcd_coordinate_frame,
)
from demo_v6_2.mdp.formal_products import FormalProductStage
from demo_v6_2.mdp.gui_loop import _NullGuiLoop
from demo_v6_2.mdp.headless_writer import HeadlessCaptureWriter
from demo_v6_2.mdp.packets import FramePacket, MaskPacket
from demo_v6_2.mdp.plumbing import (
    FatalErrorLatch,
    FormalTimelineGate,
    LosslessPipeline,
    StageStatsBoard,
)
from demo_v6_2.mdp.preload import PerceptionPreloader
from demo_v6_2.mdp.segmentation import SegmentationStage, SegmentationWarmupState
from demo_v6_2.mdp.session import CameraSession
from demo_v6_2.mdp.shape_prior_flow import ShapePriorPublisher
from demo_v6_2.mdp.tracker import TrackerStage
from demo_v6_2.mdp.warmup_preview import WarmupRgbPreview
from demo_v6_2.pipeline_status import PipelineStatusWriter
from demo_v6_2.utils.concurrency import LatestSlot


class CaptureHold:
    """Duck-typed CaptureStage stand-in for FormalProductStage.

    The product stage reads only ``startup_hold_s`` (stamped when the formal
    frame-0 gate opens); the v7 acquisition loop owns everything else.
    """

    def __init__(self) -> None:
        """Initialize CaptureHold."""
        self.startup_hold_s = 0.0


class SavedMaskSegmentationStage(SegmentationStage):
    """v6.2 SegmentationStage seeded from the SAVED frame-0 SAM3.1 masks.

    Overrides ONLY ``_prepare_warmup``: no SAM3.1 rerun and no readiness
    barrier — the masks captured during the v7 WARMUP state prompt a brand
    new EdgeTAM session through the identical
    ``_run_segmentation_frame(add_prompt=True)`` path, so everything
    downstream of frame-0 seeding is byte-for-byte the v6.2 code.
    """

    def __init__(
        self,
        *,
        saved_masks: mdp_warmup.InitialMaskBundle,
        skip_precompile: bool = False,
        **kwargs: Any,
    ) -> None:
        """Initialize with the frozen frame-0 mask bundle."""
        super().__init__(**kwargs)
        self._saved_masks = saved_masks
        self._skip_precompile = bool(skip_precompile)

    def _prepare_warmup(self) -> SegmentationWarmupState:
        """Join EdgeTAM, wait for the formal frame 0, seed with saved masks."""
        prepare_start_s = time.perf_counter()
        edgetam = self.preload.join_edgetam()
        self.warmup_perception_profile["edgetam_runtime_init"] = dict(
            edgetam.timing_ms
        )
        model_ready_s = time.perf_counter()
        # Same pre-pay as v6.2 so the formal frame-0 forward replays compiled
        # graphs instead of paying the first-forward compile inline. When the
        # runtime already ran run_early_precompile during PREVIEW this is a
        # skip — paying the graph compile here would stall the lossless
        # producer gate for ~10s of dropped frames.
        if not self._skip_precompile:
            self._precompile_first_forward(edgetam)
        self.preload.mark_seg_frame0_ready()
        frame_wait_start_s = time.perf_counter()
        first_frame = self._wait_for_first_frame()
        frame_wait_end_s = time.perf_counter()
        self.warmup_perception_profile["segmentation_warmup"] = {
            "edgetam_join_wait_ms": (model_ready_s - prepare_start_s) * 1000.0,
            "frame_wait_ms": (frame_wait_end_s - frame_wait_start_s) * 1000.0,
            "total_ms": (frame_wait_end_s - prepare_start_s) * 1000.0,
            "frame0_available": first_frame is not None,
            "frame0_seeding": "saved_frame0_sam31_masks",
        }
        initial_masks: mdp_warmup.InitialMaskBundle | None = None
        if first_frame is not None:
            expected_shape = tuple(first_frame.color_bgr.shape[:2])
            masks = self._saved_masks
            if (
                masks.object_mask.shape != expected_shape
                or masks.controller_mask.shape != expected_shape
            ):
                raise RuntimeError(
                    "saved frame-0 SAM3.1 masks do not match the formal frame shape"
                )
            initial_masks = masks
        return SegmentationWarmupState(
            hf_stream=edgetam.hf_stream,
            torch_module=edgetam.torch_module,
            dtype=edgetam.dtype,
            model=edgetam.model,
            processor=edgetam.processor,
            first_frame=first_frame,
            initial_masks=initial_masks,
        )


class FormalPipeline:
    """The v6.2 lossless stage set constructed at start_formal."""

    def __init__(
        self,
        *,
        lossless: LosslessPipeline,
        capture_slot: LatestSlot[FramePacket],
        mask_slot: LatestSlot[MaskPacket],
        live_viz_slot: LatestSlot,
        stage_stats: StageStatsBoard,
        timeline_gate: FormalTimelineGate,
        first_frame_segmented: threading.Event,
        capture_hold: CaptureHold,
        seg: SavedMaskSegmentationStage,
        tracker: TrackerStage,
        product: FormalProductStage,
        shape_prior: ShapePriorPublisher,
        warmup_preview: WarmupRgbPreview,
    ) -> None:
        """Initialize FormalPipeline."""
        self.lossless = lossless
        self.capture_slot = capture_slot
        self.mask_slot = mask_slot
        self.live_viz_slot = live_viz_slot
        self.stage_stats = stage_stats
        self.timeline_gate = timeline_gate
        self.first_frame_segmented = first_frame_segmented
        self.capture_hold = capture_hold
        self.seg = seg
        self.tracker = tracker
        self.product = product
        self.shape_prior = shape_prior
        self.warmup_preview = warmup_preview
        self.threads: list[threading.Thread] = []


def build_headless_capture_metadata(
    *,
    args: argparse.Namespace,
    mode: RunMode,
    session: CameraSession,
    shape_prior_manager: shape_prior_warmup.ShapePriorWarmupManager,
) -> dict[str, Any]:
    """Build headless capture metadata (mirrors v6.2 runtime.py)."""
    if session.camera_runtime is None:
        raise RuntimeError("camera runtime is not initialized")
    shape_profile = shape_prior_manager.profile_payload()
    replay_fps = None
    frame_count = None
    if session.recording_source is not None:
        replay_fps = float(session.recording_source.effective_fps)
        frame_count = int(session.recording_source.frame_count)
    return {
        "input_source": str(args.input_source),
        "replay_fps": replay_fps,
        "recording_frame_count": frame_count,
        "depth_source": str(args.depth_source),
        "depth_source_internal": str(args.depth_source),
        "depth_backend": mode.depth_backend_label,
        "headless_prepared_only": bool(args.headless_prepared_only),
        "write_input_rgb_timeline": bool(args.write_input_rgb_timeline),
        "shape_prior_status": str(
            shape_profile.get(
                "shape_prior_status", shape_prior_warmup.STATUS_DISABLED
            )
        ),
        "shape_prior_error": shape_profile.get("shape_prior_error"),
        "lossless_input_fps": (
            float(mode.lossless_input_fps) if mode.lossless_enabled else None
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


def run_early_precompile(
    edgetam: Any,
    *,
    args: argparse.Namespace,
    mode: RunMode,
    session: CameraSession,
) -> float:
    """Run the v6.2 scratch-session EdgeTAM precompile ahead of formal start.

    v6.2 pays this before its readiness barrier; v7 pays it during PREVIEW so
    ``start_formal`` only carries the per-session first-forward tax and the
    lossless producer gate never holds frames behind a ~10s inline compile.
    Calls the REAL ``SegmentationStage._precompile_first_forward`` on a
    minimal shim (the method touches only args/mode/session dims/profile
    dict and ``_autocast_context``) — no compile math is duplicated here.
    """
    shim = SimpleNamespace(
        args=args, mode=mode, session=session, warmup_perception_profile={}
    )
    shim._autocast_context = SegmentationStage._autocast_context.__get__(shim)
    SegmentationStage._precompile_first_forward(shim, edgetam)
    return float(
        shim.warmup_perception_profile.get("edgetam_precompile_forward_ms", 0.0)
    )


def build_formal_pipeline(
    *,
    args: argparse.Namespace,
    mode: RunMode,
    session: CameraSession,
    preload: PerceptionPreloader,
    shape_prior_manager: shape_prior_warmup.ShapePriorWarmupManager,
    status: PipelineStatusWriter,
    saved_masks: mdp_warmup.InitialMaskBundle,
    warmup_runtime_start_perf_s: float,
    stop_event: threading.Event,
    fatal: FatalErrorLatch,
    skip_precompile: bool = False,
) -> FormalPipeline:
    """Construct the same lossless stage set as v6.2 ``_start_threads``.

    Side effect: creates the HeadlessCaptureWriter and installs it on the
    session (the v6.2 stages read ``session.headless_capture_writer``).
    """
    lossless = LosslessPipeline(
        max_backlog_frames=max(
            1,
            int(
                round(
                    mode.lossless_input_fps
                    * float(args.lossless_max_backlog_seconds)
                )
            ),
        )
    )
    capture_slot: LatestSlot[FramePacket] = LatestSlot()
    mask_slot: LatestSlot[MaskPacket] = LatestSlot()
    live_viz_slot: LatestSlot = LatestSlot()
    stage_stats = StageStatsBoard()
    first_frame_segmented = threading.Event()
    timeline_gate = FormalTimelineGate(
        shape_prior_status=lambda: str(
            shape_prior_manager.profile().get(
                "shape_prior_status", shape_prior_warmup.STATUS_DISABLED
            )
        ),
        timeout_ms=int(args.shape_prior_timeout_ms),
    )
    # v7 owns all windows: the v6.2 preview object exists only because the
    # seg stage and ShapePriorPublisher call close() on it.
    warmup_preview = WarmupRgbPreview(
        input_preview_slot=LatestSlot(),
        gui=_NullGuiLoop(),
        stop_event=stop_event,
        enabled=False,
    )
    writer = HeadlessCaptureWriter(
        args.headless_capture_dir,
        metadata=build_headless_capture_metadata(
            args=args,
            mode=mode,
            session=session,
            shape_prior_manager=shape_prior_manager,
        ),
    )
    session.headless_capture_writer = writer
    print(f"[headless-capture] dir={writer.output_dir}", flush=True)
    seg = SavedMaskSegmentationStage(
        saved_masks=saved_masks,
        skip_precompile=skip_precompile,
        args=args,
        mode=mode,
        session=session,
        lossless=lossless,
        capture_slot=capture_slot,
        mask_slot=mask_slot,
        stage_stats=stage_stats,
        shape_prior_manager=shape_prior_manager,
        warmup_rgb_preview=warmup_preview,
        preload=preload,
        first_frame_segmented=first_frame_segmented,
        stop_event=stop_event,
        fatal=fatal,
    )
    seg.warmup_runtime_start_perf_s = float(warmup_runtime_start_perf_s)
    shape_prior_pub = ShapePriorPublisher(
        args=args,
        mode=mode,
        manager=shape_prior_manager,
        session=session,
        timeline_gate=timeline_gate,
        status=status,
        warmup_rgb_preview=warmup_preview,
        segmentation=seg,
    )
    tracker = TrackerStage(
        args=args,
        session=session,
        lossless=lossless,
        mask_slot=mask_slot,
        stage_stats=stage_stats,
        timeline_gate=timeline_gate,
        preload=preload,
        stop_event=stop_event,
        fatal=fatal,
    )
    capture_hold = CaptureHold()
    product = FormalProductStage(
        args=args,
        session=session,
        lossless=lossless,
        stage_stats=stage_stats,
        timeline_gate=timeline_gate,
        shape_prior=shape_prior_pub,
        capture=capture_hold,
        stop_event=stop_event,
        fatal=fatal,
        live_viz_slot=live_viz_slot,
    )
    return FormalPipeline(
        lossless=lossless,
        capture_slot=capture_slot,
        mask_slot=mask_slot,
        live_viz_slot=live_viz_slot,
        stage_stats=stage_stats,
        timeline_gate=timeline_gate,
        first_frame_segmented=first_frame_segmented,
        capture_hold=capture_hold,
        seg=seg,
        tracker=tracker,
        product=product,
        shape_prior=shape_prior_pub,
        warmup_preview=warmup_preview,
    )


__all__ = [
    "CaptureHold",
    "FormalPipeline",
    "SavedMaskSegmentationStage",
    "build_formal_pipeline",
    "build_headless_capture_metadata",
]
