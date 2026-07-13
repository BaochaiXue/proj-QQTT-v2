"""MainDataProcessingDemo tracker mixin."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_packets import TrackerMarkerPacket
from demo_v6_2.mdp_tracker_geometry import (
    _classify_query_targets_yx,
    _latest_tracker_arrays,
    _mask_packet_hand_a_mask,
    _mask_packet_hand_b_mask,
    _select_visible_spread_indices,
    _tracker_display_visibility,
    _tracker_lift_valid_mask,
    _tracker_per_target_visibility,
    _tracker_union_mask,
)
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_pipeline_plumbing import LosslessPipelineError


class _TrackerMixin(_DemoRuntimeContract):
    """MainDataProcessingDemo tracker mixin."""

    def _build_tracker_adapter(self) -> Any:
        """Build tracker adapter."""
        config = PointTrackerAdapterConfig(
            backend=str(self.args.tracker_backend),
            device=str(self.args.tracker_device),
            tapnet_repo_dir=str(DEFAULT_TAPNET_REPO_DIR),
            tapnextpp_checkpoint=str(DEFAULT_TAPNEXTPP_CHECKPOINT),
            tapnextpp_image_size=str("256,256"),
            tapnextpp_autocast_dtype=str("fp16"),
            tapnextpp_compile=bool(False),
            tapnextpp_fast_postprocess=bool(True),
        )
        adapter = build_point_tracker_adapter_factory(config)(0)
        availability = adapter.availability()
        if not availability.available:
            raise RuntimeError(availability.reason)
        return adapter

    def _ensure_tracker_queries(
        self, mask_packet: MaskPacket, adapter: Any
    ) -> np.ndarray | None:
        """Return the ensure tracker queries."""
        if self._tracker_query_points_yx is not None:
            return self._tracker_query_points_yx
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
        self._tracker_query_points_yx = np.ascontiguousarray(
            query_points, dtype=np.float32
        )
        self._tracker_query_rgb_u8 = query_rainbow_colors_from_points_yx_rgb_u8(
            query_points
        )
        self._tracker_query_is_object = np.ascontiguousarray(
            query_is_object, dtype=bool
        )
        self._tracker_query_is_controller = np.ascontiguousarray(
            query_is_controller, dtype=bool
        )
        self._tracker_query_target_id = np.ascontiguousarray(
            query_target_id, dtype=np.int64
        )
        self._tracker_query_controller_instance_id = np.ascontiguousarray(
            query_controller_instance_id, dtype=np.int64
        )
        self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
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
        return self._tracker_query_points_yx

    def _tracker_depth_for_lift(
        self, mask_packet: MaskPacket
    ) -> tuple[np.ndarray, float]:
        """Return the tracker depth for lift."""
        if mask_packet.depth_u16 is not None:
            return mask_packet.depth_u16, float(mask_packet.depth_scale_m_per_unit)
        if mask_packet.depth_source == "ffs":
            (
                depth_m,
                _ffs_ms,
                _ffs_align_ms,
                _remote_rtt_ms,
                _server_total_ms,
                _request_kb,
                _response_kb,
            ) = self._compute_external_ffs_depth_color_m(mask_packet)
            return np.ascontiguousarray(depth_m, dtype=np.float32), 1.0
        raise RuntimeError("tracker lift requires RGB-D depth")

    def _tracker_lift_mask(self, mask_packet: MaskPacket) -> np.ndarray | None:
        """Return the tracker lift mask."""
        scope = str(DEFAULT_TRACKER_DISPLAY_SCOPE)
        if scope == TRACKER_DISPLAY_SCOPE_CONTROLLER:
            mask = np.asarray(mask_packet.controller_mask, dtype=bool)
        elif scope == TRACKER_DISPLAY_SCOPE_OBJECT:
            mask = np.asarray(mask_packet.object_mask, dtype=bool)
        else:
            mask = _tracker_union_mask(mask_packet)
        return np.ascontiguousarray(mask)

    def _build_tracker_marker_packet(
        self,
        mask_packet: MaskPacket,
        adapter: Any,
        *,
        depth_for_lift: np.ndarray,
        depth_scale_m_per_unit: float,
    ) -> TrackerMarkerPacket | None:
        """Build tracker marker packet."""
        query_points = self._ensure_tracker_queries(mask_packet, adapter)
        if query_points is None:
            return None
        assert self._tracker_query_is_object is not None
        assert self._tracker_query_is_controller is not None
        assert self._tracker_query_rgb_u8 is not None
        assert self._tracker_query_target_id is not None
        assert self._tracker_query_controller_instance_id is not None
        started_s = time.perf_counter()
        rgb = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
        result = adapter.update(rgb)
        tracks_latest, visibility_latest = _latest_tracker_arrays(result)
        query_is_object_all = np.asarray(
            self._tracker_query_is_object, dtype=bool
        ).reshape(-1)
        query_is_controller_all = np.asarray(
            self._tracker_query_is_controller, dtype=bool
        ).reshape(-1)
        query_target_id_all = np.asarray(
            self._tracker_query_target_id, dtype=np.int64
        ).reshape(-1)
        query_controller_instance_id_all = np.asarray(
            self._tracker_query_controller_instance_id,
            dtype=np.int64,
        ).reshape(-1)
        query_is_object = query_is_object_all
        query_is_controller = query_is_controller_all
        query_target_id = query_target_id_all
        query_controller_instance_id = query_controller_instance_id_all
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
        display_visibility = _tracker_display_visibility(
            target_visibility,
            query_is_object=query_is_object,
            query_is_controller=query_is_controller,
            display_scope=str(DEFAULT_TRACKER_DISPLAY_SCOPE),
        )
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
        if self._tracker_consistent_visible is None or len(
            self._tracker_consistent_visible
        ) != len(query_points):
            self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
        current_lift_valid_full = np.zeros_like(
            self._tracker_consistent_visible, dtype=bool
        )
        fitted_count = min(len(current_lift_valid), len(current_lift_valid_full))
        current_lift_valid_full[:fitted_count] = current_lift_valid[:fitted_count]
        self._tracker_consistent_visible &= current_lift_valid_full
        consistent_visible_count = int(
            np.count_nonzero(self._tracker_consistent_visible)
        )
        lifted = lift_tracks_yx_to_world(
            tracks_yx=selected_tracks,
            visibility=selected_visibility,
            depth=depth_for_lift,
            intrinsics=mask_packet.intrinsics,
            c2w=self.table_c2w
            if self.table_c2w is not None
            else np.eye(4, dtype=np.float32),
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
            lifted_marker_colors = self._tracker_query_rgb_u8[lifted_query_indices]
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
            query_rgb_u8=np.ascontiguousarray(
                self._tracker_query_rgb_u8, dtype=np.uint8
            ).reshape(-1, 3),
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
            display_scope=str(DEFAULT_TRACKER_DISPLAY_SCOPE),
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
            coordinate_frame=pcd_coordinate_frame(self.table_c2w),
        )
        return packet

    def _tracker_worker(self) -> None:
        """Return the tracker worker."""
        try:
            adapter = self._build_tracker_adapter()
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
                self.tracker_marker_slot.put(packet)
                if (
                    self.headless_capture_writer is not None
                    and not self._headless_product_rows_gated()
                ):
                    self.headless_capture_writer.write_tracker(packet)
                self.tracker_stats.record(packet.process_done_perf_s)
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("TAPNext++ tracker worker", exc)

    def _lossless_tracker_worker(self) -> None:
        """Return the lossless tracker worker."""
        try:
            adapter = self._build_tracker_adapter()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={DEFAULT_TAPNET_REPO_DIR} checkpoint={DEFAULT_TAPNEXTPP_CHECKPOINT} "
                f"image_size={'256,256'} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1 lossless=1",
                flush=True,
            )
            while not self.stop_event.is_set():
                processed_frame = self.lossless_processed_frame_queue.get(
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
                self._lossless_tracker_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity(
                    "tracker", stop_event=self.stop_event
                ):
                    break
                with self._lossless_pairer_lock:
                    pairs = self.same_seq_pairer.add_tracker_packet(packet)
                    self._publish_pairer_outputs(pairs)
            with self._lossless_pairer_lock:
                pairs = self.same_seq_pairer.close_tracker()
                self._publish_pairer_outputs(pairs)
                self._maybe_finish_lossless_processing()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error(
                    "lossless TAPNext++ tracker worker", exc
                )


__all__ = ["_TrackerMixin"]
