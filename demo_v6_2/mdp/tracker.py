"""Tracker stage: TAPNext++ point tracking over segmented frames."""

from __future__ import annotations

import argparse
import threading
import time
from typing import TYPE_CHECKING, Any

import numpy as np

from qqtt.tracking.backends.point_tracker_adapter import (
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
)
from qqtt.tracking.sampling import sample_phystwin_dense

from demo_v6_2.mdp.constants import (
    DEFAULT_TAPNET_REPO_DIR,
    DEFAULT_TAPNEXTPP_CHECKPOINT,
    DEFAULT_TRACKER_DISPLAY_SCOPE,
    DEFAULT_TRACKER_QUERY_COUNT,
    DEFAULT_TRACKER_SEED,
    OBJECT_ID,
    QUERY_CONTROLLER_INSTANCE_HAND_A,
    QUERY_CONTROLLER_INSTANCE_HAND_B,
    TRACKER_QUERY_SOURCE_UNION_MASK,
    pcd_coordinate_frame,
)
from demo_v6_2.mdp.packets import MaskPacket, TrackerMarkerPacket, TrackerQuerySet
from demo_v6_2.mdp.tracker_geometry import (
    _classify_query_targets_yx,
    _latest_tracker_arrays,
    _mask_packet_hand_a_mask,
    _mask_packet_hand_b_mask,
    _select_visible_spread_indices,
    _tracker_lift_valid_mask,
    _tracker_per_target_visibility,
    _tracker_union_mask,
)
from demo_v6_2.mdp.plumbing import (
    FatalErrorLatch,
    FormalTimelineGate,
    LosslessPipeline,
    LosslessPipelineError,
    StageStatsBoard,
)
from demo_v6_2.utils.concurrency import HEAVY_IMPORT_LOCK, LatestSlot
from demo_v6_2.phystwin_strict_product import (
    PHYSTWIN_DEPTH_MAX_M,
    PHYSTWIN_DEPTH_MIN_M,
)
from demo_v6_2.utils.concurrency import elapsed_ms as _elapsed_ms
from demo_v6_2.utils.projection import lift_tracks_yx_to_world
from demo_v6_2.utils.query_rainbow import query_rainbow_colors_from_points_yx_rgb_u8


if TYPE_CHECKING:
    from demo_v6_2.mdp.preload import PerceptionPreloader
    from demo_v6_2.mdp.session import CameraSession


def build_tracker_adapter(args: argparse.Namespace) -> Any:
    """Build the TAPNext++ adapter and eagerly load its checkpoint.

    Runs on the perception preload leg (before the camera opens): the 2.5GB
    checkpoint load that used to happen lazily on the first mask packet is
    part of the frame-0 readiness barrier instead. Same weights, same
    inference path — only the load time moves.
    """
    config = PointTrackerAdapterConfig(
        backend=str(args.tracker_backend),
        device=str(args.tracker_device),
        tapnet_repo_dir=str(DEFAULT_TAPNET_REPO_DIR),
        tapnextpp_checkpoint=str(DEFAULT_TAPNEXTPP_CHECKPOINT),
        tapnextpp_image_size=str("256,256"),
        tapnextpp_autocast_dtype=str("fp16"),
        tapnextpp_compile=bool(False),
        tapnextpp_fast_postprocess=bool(True),
    )
    # availability() prepends the tapnet repo to sys.path and imports it —
    # serialized against the other preload legs' import phases; the heavy
    # checkpoint load below stays parallel.
    with HEAVY_IMPORT_LOCK:
        adapter = build_point_tracker_adapter_factory(config)(0)
        availability = adapter.availability()
    if not availability.available:
        raise RuntimeError(availability.reason)
    warmup_info = adapter.warmup()
    print(
        "[tapnextpp-tracker] "
        f"preloaded checkpoint model_load_ms="
        f"{float(warmup_info.get('model_load_ms', 0.0)):.1f}",
        flush=True,
    )
    return adapter


class TrackerStage:
    """Track the frozen chunk-0 query set and emit marker packets."""

    def __init__(
        self,
        *,
        args: argparse.Namespace,
        session: CameraSession,
        lossless: LosslessPipeline,
        mask_slot: LatestSlot[MaskPacket],
        stage_stats: StageStatsBoard,
        timeline_gate: FormalTimelineGate,
        preload: PerceptionPreloader,
        stop_event: threading.Event,
        fatal: FatalErrorLatch,
    ) -> None:
        """Initialize TrackerStage."""
        self.args = args
        self.session = session
        self.lossless = lossless
        self.mask_slot = mask_slot
        self.stage_stats = stage_stats
        self.timeline_gate = timeline_gate
        self.preload = preload
        self.stop_event = stop_event
        self.fatal = fatal
        self._tracker_queries: TrackerQuerySet | None = None

    def _ensure_tracker_queries(
        self, mask_packet: MaskPacket, adapter: Any
    ) -> TrackerQuerySet | None:
        """Freeze the chunk-0 query selection on the first trackable frame."""
        if self._tracker_queries is not None:
            return self._tracker_queries
        object_query_mask = np.asarray(mask_packet.object_mask, dtype=bool)
        controller_query_mask = np.asarray(mask_packet.controller_mask, dtype=bool)
        union_mask = _tracker_union_mask(mask_packet)
        object_pixels = int(np.count_nonzero(object_query_mask))
        controller_pixels = int(np.count_nonzero(controller_query_mask))
        union_pixels = int(np.count_nonzero(union_mask))
        requested = int(DEFAULT_TRACKER_QUERY_COUNT)
        if object_pixels <= 0 or controller_pixels <= 0 or union_pixels <= 0:
            return None
        query_points = sample_phystwin_dense(
            union_mask,
            seed=int(DEFAULT_TRACKER_SEED),
            camera_idx=0,
            torch_device="cpu",
        )
        if requested > 0 and len(query_points) > requested:
            query_points = np.ascontiguousarray(
                query_points[:requested], dtype=np.float32
            )
        if len(query_points) == 0:
            return None
        hand_a_query_mask = (
            _mask_packet_hand_a_mask(mask_packet) & controller_query_mask
        )
        hand_b_query_mask = (
            _mask_packet_hand_b_mask(mask_packet) & controller_query_mask
        )
        (
            query_is_object,
            query_is_controller,
            query_target_id,
            query_controller_instance_id,
        ) = _classify_query_targets_yx(
            query_points,
            object_mask=object_query_mask,
            hand_a_mask=hand_a_query_mask,
            hand_b_mask=hand_b_query_mask,
            controller_mask=controller_query_mask,
        )
        adapter.initialize([], query_points)
        self._tracker_queries = TrackerQuerySet(
            points_yx=np.ascontiguousarray(query_points, dtype=np.float32),
            rgb_u8=query_rainbow_colors_from_points_yx_rgb_u8(query_points),
            is_object=np.ascontiguousarray(query_is_object, dtype=bool),
            is_controller=np.ascontiguousarray(query_is_controller, dtype=bool),
            target_id=np.ascontiguousarray(query_target_id, dtype=np.int64),
            controller_instance_id=np.ascontiguousarray(
                query_controller_instance_id, dtype=np.int64
            ),
            consistent_visible=np.ones((len(query_points),), dtype=bool),
        )
        hand_a_query_count = int(
            np.count_nonzero(
                query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A
            )
        )
        hand_b_query_count = int(
            np.count_nonzero(
                query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B
            )
        )
        print(
            "[tapnextpp-tracker] "
            f"initialized query_count={len(query_points)} "
            f"requested={requested or 'phystwin_dense'} "
            f"union_pixels={union_pixels} object_pixels={object_pixels} "
            f"controller_pixels={controller_pixels} "
            f"hand_a_queries={hand_a_query_count} "
            f"hand_b_queries={hand_b_query_count} "
            f"query_source={TRACKER_QUERY_SOURCE_UNION_MASK} "
            f"display_scope={DEFAULT_TRACKER_DISPLAY_SCOPE} "
            f"device={self.args.tracker_device}",
            flush=True,
        )
        return self._tracker_queries

    def _tracker_depth_for_lift(
        self, mask_packet: MaskPacket
    ) -> tuple[np.ndarray, float]:
        """Return the tracker depth for lift."""
        if mask_packet.depth_u16 is not None:
            return mask_packet.depth_u16, float(mask_packet.depth_scale_m_per_unit)
        if mask_packet.depth_source == "ffs":
            if self.session.depth_engine is None:
                raise RuntimeError("FFS depth engine is not initialized")
            depth_m, _ffs_ms, _align_ms = self.session.depth_engine.compute_color_depth(
                mask_packet
            )
            return np.ascontiguousarray(depth_m, dtype=np.float32), 1.0
        raise RuntimeError("tracker lift requires RGB-D depth")

    def _tracker_lift_mask(self, mask_packet: MaskPacket) -> np.ndarray | None:
        """Return the tracker lift mask."""
        return np.ascontiguousarray(_tracker_union_mask(mask_packet))

    def _build_tracker_marker_packet(
        self,
        mask_packet: MaskPacket,
        adapter: Any,
        *,
        depth_for_lift: np.ndarray,
        depth_scale_m_per_unit: float,
    ) -> TrackerMarkerPacket | None:
        """Build tracker marker packet."""
        queries = self._ensure_tracker_queries(mask_packet, adapter)
        if queries is None:
            return None
        query_points = queries.points_yx
        started_s = time.perf_counter()
        rgb = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
        result = adapter.update(rgb)
        tracks_latest, visibility_latest = _latest_tracker_arrays(result)
        query_is_object = queries.is_object
        query_is_controller = queries.is_controller
        query_target_id = queries.target_id
        query_controller_instance_id = queries.controller_instance_id
        common_count = min(
            int(len(tracks_latest)),
            int(len(visibility_latest)),
            int(len(query_is_object)),
            int(len(query_is_controller)),
            int(len(query_target_id)),
            int(len(query_controller_instance_id)),
        )
        tracks_latest = tracks_latest[:common_count]
        visibility_latest = visibility_latest[:common_count]
        query_is_object = query_is_object[:common_count]
        query_is_controller = query_is_controller[:common_count]
        query_target_id = query_target_id[:common_count]
        query_controller_instance_id = query_controller_instance_id[:common_count]
        target_visibility = _tracker_per_target_visibility(
            tracks_latest,
            visibility_latest,
            mask_packet=mask_packet,
            query_target_id=query_target_id,
        )
        display_visibility = target_visibility
        lift_mask = self._tracker_lift_mask(mask_packet)
        selected = _select_visible_spread_indices(
            tracks_latest,
            display_visibility,
            max_points=int(self.args.tracker_overlay_max_points),
        )
        selected_tracks = tracks_latest[selected]
        selected_visibility = display_visibility[selected]
        selected_query_is_object = query_is_object[selected]
        selected_query_is_controller = query_is_controller[selected]
        selected_query_target_id = query_target_id[selected]
        selected_query_controller_instance_id = query_controller_instance_id[selected]

        lift_start_s = time.perf_counter()
        current_lift_valid = _tracker_lift_valid_mask(
            tracks_yx=tracks_latest,
            visibility=display_visibility,
            depth=depth_for_lift,
            depth_scale_m_per_unit=float(depth_scale_m_per_unit),
            mask=lift_mask,
            depth_min_m=float(PHYSTWIN_DEPTH_MIN_M),
            depth_max_m=float(PHYSTWIN_DEPTH_MAX_M),
        )
        consistent_visible_count = queries.and_update_consistent_visible(
            current_lift_valid
        )
        if self.session.table_c2w is None:
            raise RuntimeError(
                "tracker lift requires camera-to-world table calibration"
            )
        lifted = lift_tracks_yx_to_world(
            tracks_yx=selected_tracks,
            visibility=selected_visibility,
            depth=depth_for_lift,
            intrinsics=mask_packet.intrinsics,
            c2w=self.session.table_c2w,
            depth_scale_m_per_unit=float(depth_scale_m_per_unit),
            mask=lift_mask,
            depth_min_m=float(PHYSTWIN_DEPTH_MIN_M),
            depth_max_m=float(PHYSTWIN_DEPTH_MAX_M),
        )
        lift_ms = _elapsed_ms(lift_start_s, time.perf_counter())
        source_indices = lifted.source_indices
        if len(source_indices):
            lifted_query_indices = selected[source_indices].astype(np.int64, copy=False)
            lifted_query_is_object = selected_query_is_object[source_indices]
            lifted_query_is_controller = selected_query_is_controller[source_indices]
            lifted_query_target_id = selected_query_target_id[source_indices]
            lifted_query_controller_instance_id = selected_query_controller_instance_id[
                source_indices
            ]
            lifted_marker_colors = queries.rgb_u8[lifted_query_indices]
        else:
            lifted_query_indices = np.empty((0,), dtype=np.int64)
            lifted_query_is_object = np.empty((0,), dtype=bool)
            lifted_query_is_controller = np.empty((0,), dtype=bool)
            lifted_query_target_id = np.empty((0,), dtype=np.int64)
            lifted_query_controller_instance_id = np.empty((0,), dtype=np.int64)
            lifted_marker_colors = np.empty((0, 3), dtype=np.uint8)
        hand_a_query_count = int(
            np.count_nonzero(
                lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A
            )
        )
        hand_b_query_count = int(
            np.count_nonzero(
                lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B
            )
        )
        object_query_count = int(np.count_nonzero(lifted_query_target_id == OBJECT_ID))
        done_s = time.perf_counter()
        stats = getattr(result, "stats", {}) or {}
        packet = TrackerMarkerPacket(
            seq=mask_packet.seq,
            marker_xyz_m=np.ascontiguousarray(
                lifted.points_world, dtype=np.float32
            ).reshape(-1, 3),
            marker_colors_rgb_u8=np.ascontiguousarray(
                lifted_marker_colors, dtype=np.uint8
            ).reshape(-1, 3),
            query_rgb_u8=np.ascontiguousarray(queries.rgb_u8, dtype=np.uint8).reshape(
                -1, 3
            ),
            query_points_yx=query_points,
            tracks_yx=np.ascontiguousarray(lifted.tracks_yx, dtype=np.float32).reshape(
                -1, 2
            ),
            visibility=np.ascontiguousarray(
                selected_visibility[source_indices], dtype=np.float32
            ),
            query_is_object=np.ascontiguousarray(lifted_query_is_object, dtype=bool),
            query_is_controller=np.ascontiguousarray(
                lifted_query_is_controller, dtype=bool
            ),
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            query_count=int(len(query_points)),
            consistent_visible_count=consistent_visible_count,
            model_ms=float(
                stats.get("model_run_ms", stats.get("cuda_event_ms", 0.0)) or 0.0
            ),
            lift_ms=float(lift_ms),
            e2e_ms=_elapsed_ms(started_s, done_s),
            backend=str(getattr(result, "backend", None) or adapter.name),
            query_indices=np.ascontiguousarray(lifted_query_indices, dtype=np.int64),
            query_target_id=np.ascontiguousarray(
                lifted_query_target_id, dtype=np.int64
            ),
            query_controller_instance_id=np.ascontiguousarray(
                lifted_query_controller_instance_id, dtype=np.int64
            ),
            hand_a_query_count=hand_a_query_count,
            hand_b_query_count=hand_b_query_count,
            object_query_count=object_query_count,
            all_tracks_yx=np.ascontiguousarray(tracks_latest, dtype=np.float32).reshape(
                -1, 2
            ),
            all_tracker_visibility=np.ascontiguousarray(
                visibility_latest, dtype=np.float32
            ).reshape(-1),
            all_observation_visibility=np.ascontiguousarray(
                current_lift_valid, dtype=bool
            ).reshape(-1),
            coordinate_frame=pcd_coordinate_frame(self.session.table_c2w),
        )
        return packet

    def run_latest(self) -> None:
        """Latest-frame tracker worker loop (non-lossless)."""
        try:
            adapter = self.preload.join_tracker()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={DEFAULT_TAPNET_REPO_DIR} checkpoint={DEFAULT_TAPNEXTPP_CHECKPOINT} "
                f"image_size={'256,256'} overlay_max={int(self.args.tracker_overlay_max_points)}",
                flush=True,
            )
            last_seq = -1
            while not self.stop_event.is_set():
                mask_packet = self.mask_slot.get_latest_after(last_seq)
                if mask_packet is None:
                    time.sleep(0.001)
                    continue
                last_seq = mask_packet.seq
                depth_for_lift, depth_scale = self._tracker_depth_for_lift(mask_packet)
                packet = self._build_tracker_marker_packet(
                    mask_packet,
                    adapter,
                    depth_for_lift=depth_for_lift,
                    depth_scale_m_per_unit=depth_scale,
                )
                if packet is None:
                    continue
                if (
                    self.session.headless_capture_writer is not None
                    and not self.timeline_gate.rows_gated()
                ):
                    self.session.headless_capture_writer.write_tracker(packet)
                self.stage_stats.record("tracker", packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("TAPNext++ tracker worker", exc)

    def run_lossless(self) -> None:
        """Strict same-seq tracker worker loop."""
        try:
            adapter = self.preload.join_tracker()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={DEFAULT_TAPNET_REPO_DIR} checkpoint={DEFAULT_TAPNEXTPP_CHECKPOINT} "
                f"image_size={'256,256'} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1 lossless=1",
                flush=True,
            )
            while not self.stop_event.is_set():
                processed_frame = self.lossless.processed_frame_queue.get(
                    stop_event=self.stop_event
                )
                if processed_frame is None:
                    break
                mask_packet = processed_frame.mask_packet
                packet = self._build_tracker_marker_packet(
                    mask_packet,
                    adapter,
                    depth_for_lift=processed_frame.depth_m,
                    depth_scale_m_per_unit=1.0,
                )
                if packet is None:
                    raise LosslessPipelineError(
                        f"tracker did not produce packet for seq {mask_packet.seq}"
                    )
                if not self.lossless.submit_tracker_packet(
                    packet, stop_event=self.stop_event
                ):
                    break
            self.lossless.close_tracker_side()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("lossless TAPNext++ tracker worker", exc)


__all__ = ["TrackerStage"]
