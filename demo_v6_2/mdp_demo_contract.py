"""Typing contract shared by the MainDataProcessingDemo mixins.

``MainDataProcessingDemo`` is assembled from six mixins that all operate on one
shared ``self`` (state constructed once in ``_LifecycleMixin.__init__``). Each
mixin file alone is an incomplete class, so without a declared contract every
cross-mixin attribute read or method call is invisible to type checkers and
IDEs. ``_DemoRuntimeContract`` is that contract:

- Attribute annotations below declare every piece of shared state that at
  least two mixins touch. They are annotations only — no values, no runtime
  effect; the assignments stay in ``_LifecycleMixin.__init__``.
- Method stubs declare every method one mixin calls but another implements.
  Exactly one mixin overrides each stub in the assembled class, so the
  ``NotImplementedError`` bodies are unreachable there; they only fire if a
  mixin is used outside ``MainDataProcessingDemo`` without its peers.

When adding a new cross-mixin attribute or call, declare it here in the same
section as its owner. State used by a single mixin should NOT be added.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import argparse
    import threading
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from demo_v6_2 import shape_prior_warmup
    from demo_v6_2.mdp_capture_source import RecordedRgbdFrameSource
    from demo_v6_2.mdp_headless_writer import HeadlessCaptureWriter
    from demo_v6_2.mdp_packets import (
        DepthProfilePacket,
        FramePacket,
        MaskedPcdPacket,
        MaskPacket,
        PcdBuildResult,
        RealtimeCameraRuntime,
        TrackerMarkerPacket,
    )
    from demo_v6_2.mdp_pipeline_plumbing import (
        FatalErrorLatch,
        LosslessPipeline,
        StageStats,
    )
    from demo_v6_2.visualization.mdp_warmup_preview import WarmupRgbPreview
    from demo_v6_2.pipeline_status import PipelineStatusWriter
    from demo_v6_2.utils.concurrency import LatestSlot
    from demo_v6_2.utils.ffs_align import FfsDepthEngine


class _DemoRuntimeContract:
    """Shared attributes + cross-mixin method signatures of the demo runtime."""

    # ------------------------------------------------------------------
    # Shared state, constructed in _LifecycleMixin.__init__
    # ------------------------------------------------------------------
    # Configuration / camera / calibration
    args: argparse.Namespace
    runtime: RealtimeCameraRuntime | None
    recording_source: RecordedRgbdFrameSource | None
    table_c2w: np.ndarray | None
    table_calibration_path: Path | None

    # Latest-value slots between stages
    input_preview_slot: LatestSlot[FramePacket]
    capture_slot: LatestSlot[FramePacket]
    mask_slot: LatestSlot[MaskPacket]
    depth_profile_slot: LatestSlot[DepthProfilePacket]
    tracker_marker_slot: LatestSlot[TrackerMarkerPacket]
    _input_preview_publish_seq: int

    # Strict same-seq lossless pipeline (queues, pairer, ordered publish)
    lossless: LosslessPipeline

    # Run control / stage telemetry
    stop_event: threading.Event
    fatal: FatalErrorLatch
    _first_frame_segmented: threading.Event
    _startup_hold_s: float
    capture_stats: StageStats
    seg_stats: StageStats
    depth_stats: StageStats
    pcd_stats: StageStats
    tracker_stats: StageStats
    _status: PipelineStatusWriter

    # FFS depth (constructed in main_warmup when --depth-source ffs)
    depth_engine: FfsDepthEngine | None

    # Headless capture / warm-up / shape prior
    headless_capture_writer: HeadlessCaptureWriter | None
    shape_prior_manager: shape_prior_warmup.ShapePriorWarmupManager
    warmup_rgb_preview: WarmupRgbPreview
    _shape_prior_written: bool
    _formal_timeline_gated_frames: int
    _formal_timeline_metadata_written: bool
    _warmup_anchor_row_written: bool
    _formal_timeline_gate_started_s: float | None
    _formal_timeline_gate_expired: bool
    _warmup_runtime_start_perf_s: float | None
    _warmup_perception_profile: dict[str, Any]

    # Frozen tracker query state (chunk-0 selection)
    _tracker_query_points_yx: np.ndarray | None
    _tracker_query_rgb_u8: np.ndarray | None
    _tracker_query_is_object: np.ndarray | None
    _tracker_query_is_controller: np.ndarray | None
    _tracker_query_target_id: np.ndarray | None
    _tracker_query_controller_instance_id: np.ndarray | None
    _tracker_consistent_visible: np.ndarray | None

    # ------------------------------------------------------------------
    # Implemented by _LifecycleMixin
    # ------------------------------------------------------------------
    def _wait_for_lossless_startup_pair(
        self,
        on_wait_tick: Callable[[], None] | None = None,
    ) -> bool:
        """Wait until frame 0 has complete PCD and tracking results."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _CaptureMixin
    # ------------------------------------------------------------------
    def _capture_worker(self) -> None:
        """Capture worker loop (live or fake-live replay)."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _SegWarmupMixin
    # ------------------------------------------------------------------
    def _seg_worker(self) -> None:
        """EdgeTAM segmentation worker loop."""
        raise NotImplementedError

    def _headless_product_rows_gated(self) -> bool:
        """True while post-warmup frames must stay out of the chunk timeline."""
        raise NotImplementedError

    def _maybe_start_shape_prior_from_pcd_result(
        self,
        result: PcdBuildResult,
        *,
        from_strict_pair: bool = False,
    ) -> bool:
        """Submit the frame-0 shape-prior request once the PCD result allows it."""
        raise NotImplementedError

    def _maybe_write_shape_prior_headless_result(self) -> None:
        """Write the shape-prior result/profile to the headless capture once ready."""
        raise NotImplementedError

    def _packet_with_shape_prior_state(
        self, packet: MaskedPcdPacket
    ) -> MaskedPcdPacket:
        """Return the packet with shape-prior points/status/profile attached."""
        raise NotImplementedError

    def _run_deferred_shape_prior_after_teardown(self) -> None:
        """Run a deferred shape-prior submission after runtime teardown."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _TrackerMixin
    # ------------------------------------------------------------------
    def _tracker_worker(self) -> None:
        """Latest-frame tracker worker loop (non-lossless)."""
        raise NotImplementedError

    def _lossless_tracker_worker(self) -> None:
        """Strict same-seq tracker worker loop."""
        raise NotImplementedError

    def _build_tracker_adapter(self) -> Any:
        """Build the point-tracker backend adapter."""
        raise NotImplementedError

    def _build_tracker_marker_packet(
        self,
        mask_packet: MaskPacket,
        adapter: Any,
        *,
        depth_for_lift: np.ndarray,
        depth_scale_m_per_unit: float,
    ) -> TrackerMarkerPacket | None:
        """Track one mask packet and build its marker packet."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _PcdMixin
    # ------------------------------------------------------------------
    def _lossless_processed_frame_worker(self) -> None:
        """Build canonical processed frames before tracker/PCD consumption."""
        raise NotImplementedError

    def _build_processed_frame_result(
        self,
        mask_packet: MaskPacket,
    ) -> PcdBuildResult:
        """Build one canonical processed frame and its runtime PCD."""
        raise NotImplementedError

    def _write_headless_pcd_result(
        self,
        result: PcdBuildResult,
        tracker_packet: TrackerMarkerPacket | None = None,
        *,
        gated: bool | None = None,
    ) -> None:
        """Write one PCD result to the headless capture."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _PairPublishMixin
    # ------------------------------------------------------------------
    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish raw masks to diagnostics and the canonical formal queue."""
        raise NotImplementedError

    def _lossless_pair_output_worker(self) -> None:
        """Ordered lossless pair publishing worker loop."""
        raise NotImplementedError

    def _depth_profile_worker(self) -> None:
        """FFS depth profiling worker loop (pcd_mode=none)."""
        raise NotImplementedError


__all__ = ["_DemoRuntimeContract"]
