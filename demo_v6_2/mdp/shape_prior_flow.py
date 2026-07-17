"""Shape-prior flow through the runtime: frame-0 submission, packet
enrichment, and the ready-result headless write that opens the formal
timeline.

Called by ``FormalProductStage`` on every published pair; owns the
WARMUP-FINISHED transition (banner, status event, preview close).
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from typing import TYPE_CHECKING

import numpy as np

from demo_v6_2.shape_prior import case as shape_prior_case, warmup as shape_prior_warmup
from demo_v6_2.mdp.cli import RunMode
from demo_v6_2.mdp.packets import MaskedPcdPacket, PcdBuildResult
from demo_v6_2.pipeline_status import (
    STAGE_SHAPE_PRIOR,
    STAGE_WARMUP_READY,
    PipelineStatusWriter,
)

if TYPE_CHECKING:
    from demo_v6_2.mdp.plumbing import FormalTimelineGate
    from demo_v6_2.mdp.segmentation import SegmentationStage
    from demo_v6_2.mdp.session import CameraSession
    from demo_v6_2.mdp.warmup_preview import WarmupRgbPreview

WARMUP_FINISHED_BANNER = (
    "\n#############################\nWarmup finished\n#############################"
)


class ShapePriorPublisher:
    """Submit frame 0, attach prior state to packets, and open the timeline."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        mode: RunMode,
        manager: shape_prior_warmup.ShapePriorWarmupManager,
        session: CameraSession,
        timeline_gate: FormalTimelineGate,
        status: PipelineStatusWriter,
        warmup_rgb_preview: WarmupRgbPreview,
        segmentation: SegmentationStage,
    ) -> None:
        """Initialize ShapePriorPublisher."""
        self.args = args
        self.mode = mode
        self.shape_prior_manager = manager
        self.session = session
        self.timeline_gate = timeline_gate
        self._status = status
        self.warmup_rgb_preview = warmup_rgb_preview
        self._segmentation = segmentation

    def _frame0_request_from_pcd_result(
        self,
        result: PcdBuildResult,
    ) -> shape_prior_case.ShapePriorFrame0Request | None:
        """Return the shape prior frame0 request from PCD result."""
        if not bool(self.args.shape_prior_warmup):
            return None
        if self.session.table_c2w is None:
            raise RuntimeError(
                "shape-prior frame 0 requires camera-to-world calibration"
            )
        processed_frame = result.processed_frame
        mask_packet = processed_frame.mask_packet
        if int(result.pcd_packet.seq) != int(mask_packet.seq):
            raise RuntimeError("shape-prior PCD/mask sequence mismatch")
        k_color = mask_packet.k_color
        if k_color is None and self.session.camera_runtime is not None:
            k_color = np.asarray(
                self.session.camera_runtime.k_color, dtype=np.float32
            )
        if k_color is None:
            raise RuntimeError("shape-prior frame 0 requires color intrinsics")
        return shape_prior_case.ShapePriorFrame0Request(
            seq=int(mask_packet.seq),
            source_timestamp_s=mask_packet.source_timestamp_s,
            input_source=str(self.args.input_source),
            depth_backend=self.mode.depth_backend_label,
            depth_source_internal=str(self.args.depth_source),
            rgb_u8=mask_packet.color_bgr[:, :, ::-1],
            object_mask=mask_packet.object_mask,
            controller_mask=mask_packet.controller_mask,
            depth_color_m=processed_frame.depth_m,
            depth_valid_mask=processed_frame.depth_valid_mask,
            points_world_m=processed_frame.pcd_points[0],
            k_color=k_color,
            camera_to_world_c2w=self.session.table_c2w,
            warmup_runtime_start_perf_s=(
                self._segmentation.warmup_runtime_start_perf_s
            ),
            frame_receive_perf_s=float(mask_packet.receive_perf_s),
            frame_mask_ready_perf_s=float(mask_packet.process_done_perf_s),
            frame_pcd_ready_perf_s=float(result.pcd_packet.process_done_perf_s),
            frame0_pipeline_timing_ms={
                key: float(value)
                for key, value in asdict(result.pcd_packet.timing).items()
            },
            frame0_perception_profile=dict(
                self._segmentation.warmup_perception_profile
            ),
        )

    def maybe_start_from_pcd_result(self, result: PcdBuildResult) -> None:
        """Submit the frame-0 shape-prior request once the PCD result allows it."""
        frame0_request = self._frame0_request_from_pcd_result(result)
        if frame0_request is None:
            return
        if self.shape_prior_manager.maybe_submit(frame0_request):
            self.shape_prior_manager.write_profile_json()
            self._status.emit(
                STAGE_SHAPE_PRIOR, "frame-0 submitted; generating shape prior"
            )

    def packet_with_state(self, packet: MaskedPcdPacket) -> MaskedPcdPacket:
        """Return the packet with shape-prior points/status/profile attached."""
        profile = self.shape_prior_manager.profile()
        result = self.shape_prior_manager.ready_result()
        if result is not None:
            return replace(
                packet,
                shape_prior_points_m=np.ascontiguousarray(
                    result.points_m, dtype=np.float32
                ).reshape(-1, 3),
                shape_prior_colors_rgb_u8=np.ascontiguousarray(
                    result.colors_rgb_u8, dtype=np.uint8
                ).reshape(-1, 3),
                shape_prior_status=shape_prior_warmup.STATUS_READY,
                shape_prior_profile=profile,
            )
        return replace(
            packet,
            shape_prior_points_m=np.empty((0, 3), dtype=np.float32),
            shape_prior_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            shape_prior_status=str(
                profile.get("shape_prior_status", shape_prior_warmup.STATUS_DISABLED)
            ),
            shape_prior_profile=profile,
        )

    def maybe_write_headless_result(self) -> None:
        """Write the shape-prior result/profile to the headless capture once ready."""
        result = self.shape_prior_manager.ready_result()
        writer = self.session.headless_capture_writer
        if (
            writer is not None
            and result is not None
            and not self.timeline_gate.shape_prior_result_written
        ):
            writer.write_shape_prior_result(result)
            self.timeline_gate.shape_prior_result_written = True
            self.shape_prior_manager.mark_gate_open()
            profile = self.shape_prior_manager.profile_payload()
            self.shape_prior_manager.write_profile_json(profile)
            self._status.emit(
                STAGE_WARMUP_READY, "shape prior ready; formal timeline open"
            )
            print(WARMUP_FINISHED_BANNER, flush=True)
            # Warm-up is over: close the live RGB input preview (its
            # failure/cancel paths close via stop_event/stop() instead).
            self.warmup_rgb_preview.close()
            return
        profile = self.shape_prior_manager.profile_payload()
        if writer is not None:
            writer.update_metadata(
                {
                    "shape_prior_status": str(
                        profile.get(
                            "shape_prior_status",
                            shape_prior_warmup.STATUS_DISABLED,
                        )
                    ),
                    "shape_prior_error": profile.get("shape_prior_error"),
                }
            )
        self.shape_prior_manager.write_profile_json(profile)


__all__ = ["ShapePriorPublisher", "WARMUP_FINISHED_BANNER"]
