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
    from collections import OrderedDict
    from collections.abc import Callable
    from pathlib import Path

    import numpy as np

    from demo_v6_2 import shape_prior_warmup
    from demo_v6_2.mdp_capture_source import RecordedRgbdFrameSource
    from demo_v6_2.mdp_headless_writer import HeadlessCaptureWriter
    from demo_v6_2.mdp_packets import (
        DepthProfilePacket,
        FatalWorkerError,
        FramePacket,
        MaskedPcdPacket,
        MaskPacket,
        PairedBuildResult,
        PcdBuildResult,
        RealtimeCameraRuntime,
        TrackerMarkerPacket,
    )
    from demo_v6_2.mdp_pipeline_plumbing import OrderedPacketQueue, SameSeqPairer, StageStats
    from demo_v6_2.mdp_warmup_preview import WarmupRgbPreview
    from demo_v6_2.pipeline_status import PipelineStatusWriter
    from demo_v6_2.utils.concurrency import LatestSlot
    from demo_v6_2.utils.ffs_align import FfsIrToColorAligner
    from demo_v6_2.utils.pcd_filter import FilterBudgetController, FilterInput, FilterOutput


class _DemoRuntimeContract:
    """Shared attributes + cross-mixin method signatures of the demo runtime."""

    # ------------------------------------------------------------------
    # Shared state, constructed in _LifecycleMixin.__init__
    # ------------------------------------------------------------------
    # Configuration / camera / calibration
    args: argparse.Namespace
    runtime: RealtimeCameraRuntime | None
    recording_source: RecordedRgbdFrameSource | None
    ray_x: np.ndarray | None
    ray_y: np.ndarray | None
    table_c2w: np.ndarray | None
    table_calibration_path: Path | None

    # Latest-value slots between stages
    input_preview_slot: LatestSlot[FramePacket]
    capture_slot: LatestSlot[FramePacket]
    mask_slot: LatestSlot[MaskPacket]
    depth_profile_slot: LatestSlot[DepthProfilePacket]
    tracker_marker_slot: LatestSlot[TrackerMarkerPacket]
    _input_preview_publish_seq: int

    # Lossless (strict same-seq) pipeline plumbing
    lossless_max_backlog_frames: int
    lossless_frame_queue: OrderedPacketQueue[FramePacket]
    lossless_pcd_mask_queue: OrderedPacketQueue[MaskPacket]
    lossless_tracker_mask_queue: OrderedPacketQueue[MaskPacket]
    lossless_pair_output_queue: OrderedPacketQueue[PairedBuildResult]
    same_seq_pairer: SameSeqPairer
    _lossless_pairer_lock: threading.Lock
    _lossless_publish_condition: threading.Condition
    _lossless_next_publish_seq: int
    _lossless_capture_done: threading.Event
    _lossless_processing_done: threading.Event
    _lossless_first_pair_published: threading.Event
    _lossless_offered_frames: int
    _lossless_segmented_frames: int
    _lossless_pcd_results: int
    _lossless_tracker_results: int
    _lossless_pairs_emitted: int

    # Run control / stage telemetry
    stop_event: threading.Event
    _first_frame_segmented: threading.Event
    _startup_hold_s: float
    capture_stats: StageStats
    seg_stats: StageStats
    depth_stats: StageStats
    pcd_stats: StageStats
    tracker_stats: StageStats
    filter_submit_stats: StageStats
    filter_output_stats: StageStats
    _status: PipelineStatusWriter

    # PCD filter workers / budgets
    filter_worker: Any | None
    _filter_submit_skip_count: int
    _last_filter_output_seq_recorded: int
    object_filter_budget: FilterBudgetController
    controller_filter_budget: FilterBudgetController

    # FFS depth
    ffs_runner: object | None
    _local_ffs_lock: threading.Lock
    _local_ffs_depth_cache: OrderedDict[int, tuple[np.ndarray, float, float]]

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
    _tracker_query_alive_mask: np.ndarray | None
    _tracker_query_initial_seq: int | None

    # ------------------------------------------------------------------
    # Implemented by _LifecycleMixin
    # ------------------------------------------------------------------
    def _record_fatal_worker_error(self, stage: str, exc: BaseException) -> FatalWorkerError:
        """Record the first fatal worker error and set stop_event."""
        raise NotImplementedError

    def _wait_for_lossless_startup_pair(
        self,
        on_wait_tick: Callable[[], None] | None = None,
    ) -> bool:
        """Wait until frame 0 has complete PCD and tracking results."""
        raise NotImplementedError

    def _get_ir_to_color_aligner(
        self,
        *,
        depth_shape: tuple[int, int],
        color_shape: tuple[int, int],
        k_ir_left: np.ndarray,
        t_ir_left_to_color: np.ndarray,
        k_color: np.ndarray,
    ) -> FfsIrToColorAligner:
        """Return the cached IR-to-color depth aligner for these calibrations."""
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

    def _packet_with_shape_prior_state(self, packet: MaskedPcdPacket) -> MaskedPcdPacket:
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
    ) -> TrackerMarkerPacket | None:
        """Track one mask packet and build its marker packet."""
        raise NotImplementedError

    # ------------------------------------------------------------------
    # Implemented by _PcdMixin
    # ------------------------------------------------------------------
    def _lossless_pcd_worker(self) -> None:
        """Strict same-seq PCD worker loop."""
        raise NotImplementedError

    def _build_pcd_packet_from_mask(
        self,
        mask_packet: MaskPacket,
        *,
        rng: np.random.Generator,
        require_filter_seq: bool = False,
    ) -> PcdBuildResult:
        """Build a masked point-cloud packet from a mask/depth pair."""
        raise NotImplementedError

    def _make_filter_input(
        self,
        *,
        seq: int,
        object_xyz: np.ndarray,
        object_colors: np.ndarray,
        object_yx: np.ndarray | None = None,
        controller_xyz: np.ndarray,
        controller_colors: np.ndarray,
        controller_yx: np.ndarray | None = None,
    ) -> FilterInput:
        """Create a capped PCD-filter input for the current budgets."""
        raise NotImplementedError

    def _filter_pcd_input(self, item: FilterInput) -> FilterOutput:
        """Run the object/controller PCD filters on one input."""
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
    def _publish_pairer_outputs(self, pairs: list[PairedBuildResult]) -> None:
        """Enqueue completed same-seq pairs for ordered publishing."""
        raise NotImplementedError

    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish a mask packet to the slot and, when lossless, both queues."""
        raise NotImplementedError

    def _maybe_finish_lossless_processing(self) -> None:
        """Close the pair-output queue once the same-seq pairer drained."""
        raise NotImplementedError

    def _lossless_pair_output_worker(self) -> None:
        """Ordered lossless pair publishing worker loop."""
        raise NotImplementedError

    def _depth_profile_worker(self) -> None:
        """FFS depth profiling worker loop (pcd_mode=none)."""
        raise NotImplementedError

    def _compute_external_ffs_depth_color_m(
        self,
        packet: MaskPacket | FramePacket,
    ) -> tuple[np.ndarray, float, float, float, float, float, float]:
        """Compute color-aligned FFS depth plus timing/transfer telemetry."""
        raise NotImplementedError


__all__ = ["_DemoRuntimeContract"]
