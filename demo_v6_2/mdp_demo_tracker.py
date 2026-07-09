"""MainDataProcessingDemo tracker mixin."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import controller_pcd_mask_erode_pixels, object_pcd_mask_erode_pixels, pcd_filter_enabled, tracker_marker_gate, tracker_marker_retirement_policy, tracker_query_source
from demo_v6_2.mdp_packets import MarkerResidualAudit, TrackerMarkerPacket, _fit_bool_array, _remaining_query_class_counts
from demo_v6_2.mdp_pcd_depth import _audit_marker_residual_subset, _classify_query_targets_yx, _latest_tracker_arrays, _mask_from_yx, _mask_packet_hand_a_mask, _mask_packet_hand_b_mask, _query_current_residual_visibility, _select_visible_spread_indices, _tracker_display_visibility, _tracker_lift_valid_mask, _tracker_per_target_visibility, _tracker_union_mask, _transform_points_c2w, apply_table_z_filter_with_yx, backproject_masked_rgbd_profiled, erode_binary_mask
from demo_v6_2.mdp_pipeline_plumbing import LosslessPipelineError


class _TrackerMixin:
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

    def _ensure_tracker_queries(self, mask_packet: MaskPacket, adapter: Any) -> np.ndarray | None:
        """Return the ensure tracker queries."""
        if self._tracker_query_points_yx is not None:
            return self._tracker_query_points_yx
        query_source = tracker_query_source(self.args)
        if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            object_query_mask, controller_query_mask = self._tracker_pcd_filter_residual_masks(mask_packet)
            union_mask = np.logical_or(object_query_mask, controller_query_mask)
        else:
            object_query_mask = np.asarray(mask_packet.object_mask, dtype=bool)
            controller_query_mask = np.asarray(mask_packet.controller_mask, dtype=bool)
            union_mask = _tracker_union_mask(mask_packet)
        object_pixels = int(np.count_nonzero(object_query_mask))
        controller_pixels = int(np.count_nonzero(controller_query_mask))
        union_pixels = int(np.count_nonzero(union_mask))
        requested = int(DEFAULT_TRACKER_QUERY_COUNT)
        if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            if union_pixels <= 0:
                raise RuntimeError(
                    "pcd_filter_residual query source produced no residual query candidates "
                    f"seq={mask_packet.seq} object={object_pixels} controller={controller_pixels}"
                )
            if requested > 0 and union_pixels < requested:
                raise RuntimeError(
                    "not enough residual query candidates for TAPNext++ initialization: "
                    f"requested={requested} residual={union_pixels} object={object_pixels} controller={controller_pixels}"
                )
        elif object_pixels <= 0 or controller_pixels <= 0 or union_pixels <= 0:
            return None
        query_points = sample_phystwin_dense(
            union_mask,
            seed=int(DEFAULT_TRACKER_SEED),
            camera_idx=0,
            torch_device="cpu",
        )
        if requested > 0 and len(query_points) > requested:
            query_points = np.ascontiguousarray(query_points[:requested], dtype=np.float32)
        if len(query_points) == 0:
            if query_source == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
                raise RuntimeError("pcd_filter_residual query source produced no sampled query points")
            return None
        hand_a_query_mask = _mask_packet_hand_a_mask(mask_packet) & controller_query_mask
        hand_b_query_mask = _mask_packet_hand_b_mask(mask_packet) & controller_query_mask
        query_is_object, query_is_controller, query_target_id, query_controller_instance_id = _classify_query_targets_yx(
            query_points,
            object_mask=object_query_mask,
            hand_a_mask=hand_a_query_mask,
            hand_b_mask=hand_b_query_mask,
            controller_mask=controller_query_mask,
        )
        adapter.initialize([], query_points)
        self._tracker_query_points_yx = np.ascontiguousarray(query_points, dtype=np.float32)
        self._tracker_query_rgb_u8 = query_rainbow_colors_from_points_yx_rgb_u8(query_points)
        self._tracker_query_is_object = np.ascontiguousarray(query_is_object, dtype=bool)
        self._tracker_query_is_controller = np.ascontiguousarray(query_is_controller, dtype=bool)
        self._tracker_query_target_id = np.ascontiguousarray(query_target_id, dtype=np.int64)
        self._tracker_query_controller_instance_id = np.ascontiguousarray(query_controller_instance_id, dtype=np.int64)
        self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
        self._tracker_query_alive_mask = np.ones((len(query_points),), dtype=bool)
        self._tracker_query_initial_seq = int(mask_packet.seq)
        print(
            "[tapnextpp-tracker] "
            f"initialized query_count={len(query_points)} requested={requested or 'phystwin_dense'} "
            f"union_pixels={union_pixels} object_pixels={object_pixels} controller_pixels={controller_pixels} "
            f"hand_a_queries={int(np.count_nonzero(query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A))} "
            f"hand_b_queries={int(np.count_nonzero(query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B))} "
            f"query_source={query_source} display_scope={DEFAULT_TRACKER_DISPLAY_SCOPE} device={self.args.tracker_device}",
            flush=True,
        )
        return self._tracker_query_points_yx

    def _tracker_depth_for_lift(self, mask_packet: MaskPacket) -> tuple[np.ndarray, float]:
        """Return the tracker depth for lift."""
        if mask_packet.depth_u16 is not None:
            return mask_packet.depth_u16, float(mask_packet.depth_scale_m_per_unit)
        if mask_packet.depth_source == "ffs":
            depth_m, _ffs_ms, _ffs_align_ms, _remote_rtt_ms, _server_total_ms, _request_kb, _response_kb = (
                self._compute_external_ffs_depth_color_m(mask_packet)
            )
            return np.ascontiguousarray(depth_m, dtype=np.float32), 1.0
        raise RuntimeError("tracker lift requires RGB-D depth")

    def _tracker_lift_mask(self, mask_packet: MaskPacket) -> np.ndarray | None:
        """Return the tracker lift mask."""
        scope = str(DEFAULT_TRACKER_DISPLAY_SCOPE)
        if scope == TRACKER_DISPLAY_SCOPE_CONTROLLER:
            mask = np.asarray(mask_packet.controller_mask, dtype=bool)
            erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        elif scope == TRACKER_DISPLAY_SCOPE_OBJECT:
            mask = np.asarray(mask_packet.object_mask, dtype=bool)
            erode_pixels = object_pcd_mask_erode_pixels(self.args)
        else:
            mask = _tracker_union_mask(mask_packet)
            erode_pixels = min(object_pcd_mask_erode_pixels(self.args), controller_pcd_mask_erode_pixels(self.args))
        if erode_pixels > 0:
            return erode_binary_mask(mask, erode_pixels=erode_pixels)
        return np.ascontiguousarray(mask)

    def _tracker_pcd_filter_residual_masks(self, mask_packet: MaskPacket) -> tuple[np.ndarray, np.ndarray]:
        """Return the tracker PCD filter residual masks."""
        if not pcd_filter_enabled(self.args):
            raise RuntimeError("pcd_filter_residual query source requires enabled sync PCD filtering")
        if str(self.args.pcd_filter_mode) != "sync":
            raise RuntimeError("pcd_filter_residual query source requires --pcd-filter-mode sync")
        if self.ray_x is None or self.ray_y is None:
            raise RuntimeError("pcd_filter_residual query source requires initialized projection grids")

        if mask_packet.depth_source == "ffs":
            depth_m, _ffs_ms, _ffs_align_ms, _remote_rtt_ms, _server_total_ms, _request_kb, _response_kb = (
                self._compute_external_ffs_depth_color_m(mask_packet)
            )
        else:
            if mask_packet.depth_u16 is None:
                raise RuntimeError("pcd_filter_residual query source requires RGB-D depth")
            depth_m = np.ascontiguousarray(
                mask_packet.depth_u16.astype(np.float32) * np.float32(mask_packet.depth_scale_m_per_unit)
            )

        stride = int(1)
        if stride > 1:
            color_bgr = mask_packet.color_bgr[::stride, ::stride]
            depth_for_pcd = depth_m[::stride, ::stride]
            controller_mask = mask_packet.controller_mask[::stride, ::stride]
            object_mask = mask_packet.object_mask[::stride, ::stride]
            ray_x_for_pcd = self.ray_x[::stride, ::stride]
            ray_y_for_pcd = self.ray_y[::stride, ::stride]
        else:
            color_bgr = mask_packet.color_bgr
            depth_for_pcd = depth_m
            controller_mask = mask_packet.controller_mask
            object_mask = mask_packet.object_mask
            ray_x_for_pcd = self.ray_x
            ray_y_for_pcd = self.ray_y

        controller_erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        object_erode_pixels = object_pcd_mask_erode_pixels(self.args)
        if controller_erode_pixels > 0:
            controller_mask = erode_binary_mask(controller_mask, erode_pixels=controller_erode_pixels)
        if object_erode_pixels > 0:
            object_mask = erode_binary_mask(object_mask, erode_pixels=object_erode_pixels)

        controller_xyz, controller_colors, controller_yx, _controller_timing = backproject_masked_rgbd_profiled(
            color_bgr=color_bgr,
            depth_m=depth_for_pcd,
            mask=controller_mask,
            ray_x=ray_x_for_pcd,
            ray_y=ray_y_for_pcd,
            depth_min_m=float(0.2),
            depth_max_m=float(1.5),
            max_points=int(60000),
            color_mode=str(self.args.pcd_color_mode),
            class_rgb=tuple(self.args.controller_color),
            rng=np.random.default_rng(int(mask_packet.seq) * 2 + 31),
            return_yx=True,
        )
        object_xyz, object_colors, object_yx, _object_timing = backproject_masked_rgbd_profiled(
            color_bgr=color_bgr,
            depth_m=depth_for_pcd,
            mask=object_mask,
            ray_x=ray_x_for_pcd,
            ray_y=ray_y_for_pcd,
            depth_min_m=float(0.2),
            depth_max_m=float(1.5),
            max_points=int(60000),
            color_mode=str(self.args.pcd_color_mode),
            class_rgb=tuple(self.args.object_color),
            rng=np.random.default_rng(int(mask_packet.seq) * 2 + 29),
            return_yx=True,
        )
        if stride > 1:
            controller_yx = np.ascontiguousarray(controller_yx * int(stride), dtype=np.int64)
            object_yx = np.ascontiguousarray(object_yx * int(stride), dtype=np.int64)

        filter_input = self._make_filter_input(
            seq=int(mask_packet.seq),
            object_xyz=object_xyz,
            object_colors=object_colors,
            object_yx=object_yx,
            controller_xyz=controller_xyz,
            controller_colors=controller_colors,
            controller_yx=controller_yx,
        )
        filter_output = self._filter_pcd_input(filter_input)
        object_xyz_world = _transform_points_c2w(filter_output.object_xyz, self.table_c2w)
        controller_xyz_world = _transform_points_c2w(filter_output.controller_xyz, self.table_c2w)
        object_yx = filter_output.object_yx
        controller_yx = filter_output.controller_yx
        if bool(self.args.enable_table_z_filter):
            classes = str(TABLE_Z_FILTER_CLASS_BOTH)
            if classes in {TABLE_Z_FILTER_CLASS_OBJECT, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    _object_xyz,
                    _object_colors,
                    object_yx,
                    _object_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    object_xyz_world,
                    filter_output.object_rgb,
                    object_yx,
                    enabled=True,
                    threshold_m=float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
                    table_z_m=TABLE_Z_M,
                )
            if classes in {TABLE_Z_FILTER_CLASS_CONTROLLER, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    _controller_xyz,
                    _controller_colors,
                    controller_yx,
                    _controller_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    controller_xyz_world,
                    filter_output.controller_rgb,
                    controller_yx,
                    enabled=True,
                    threshold_m=float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
                    table_z_m=TABLE_Z_M,
                )
        shape = tuple(mask_packet.object_mask.shape[:2])
        object_residual = _mask_from_yx(shape, object_yx)
        controller_residual = _mask_from_yx(shape, controller_yx)
        return object_residual, controller_residual

    def _ensure_tracker_query_alive_mask(self, query_count: int) -> np.ndarray:
        """Return the ensure tracker query alive mask."""
        count = max(0, int(query_count))
        if self._tracker_query_alive_mask is None or len(self._tracker_query_alive_mask) != count:
            self._tracker_query_alive_mask = np.ones((count,), dtype=bool)
            self._tracker_query_initial_seq = None
        return self._tracker_query_alive_mask

    def _current_tracker_query_alive_mask(
        self,
        *,
        current_seq: int,
        query_count: int,
        residual_visibility: np.ndarray | None,
    ) -> np.ndarray:
        """Return the current tracker query alive mask."""
        alive = self._ensure_tracker_query_alive_mask(query_count)
        if self._tracker_query_initial_seq is None:
            self._tracker_query_initial_seq = int(current_seq)
        retirement_frame = int(current_seq) > int(self._tracker_query_initial_seq)
        if (
            retirement_frame
            and residual_visibility is not None
            and tracker_marker_retirement_policy(self.args)
            == TRACKER_MARKER_RETIREMENT_POLICY_PCD_FILTER_RESIDUAL_TABLE_Z_ONCE_FALSE
        ):
            residual = np.asarray(residual_visibility, dtype=bool).reshape(-1)
            count = min(len(alive), len(residual))
            if count:
                alive[:count] &= residual[:count]
        return np.ascontiguousarray(alive.copy(), dtype=bool)

    def _build_tracker_marker_packet(self, mask_packet: MaskPacket, adapter: Any) -> TrackerMarkerPacket | None:
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
        query_is_object_all = np.asarray(self._tracker_query_is_object, dtype=bool).reshape(-1)
        query_is_controller_all = np.asarray(self._tracker_query_is_controller, dtype=bool).reshape(-1)
        query_target_id_all = np.asarray(self._tracker_query_target_id, dtype=np.int64).reshape(-1)
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
        object_residual_mask: np.ndarray | None = None
        controller_residual_mask: np.ndarray | None = None
        residual_visibility: np.ndarray | None = None
        if tracker_query_source(self.args) == TRACKER_QUERY_SOURCE_PCD_FILTER_RESIDUAL:
            object_residual_mask, controller_residual_mask = self._tracker_pcd_filter_residual_masks(mask_packet)
            residual_visibility = _query_current_residual_visibility(
                tracks_latest,
                query_is_object=query_is_object,
                query_is_controller=query_is_controller,
                object_residual_mask=object_residual_mask,
                controller_residual_mask=controller_residual_mask,
            )
            display_visibility = np.where(residual_visibility, display_visibility, 0.0).astype(np.float32, copy=False)
            lift_mask = np.logical_or(object_residual_mask, controller_residual_mask)
        query_alive_mask = self._current_tracker_query_alive_mask(
            current_seq=int(mask_packet.seq),
            query_count=len(query_points),
            residual_visibility=residual_visibility,
        )
        alive_for_display = _fit_bool_array(query_alive_mask, len(display_visibility))
        display_visibility = np.where(alive_for_display, display_visibility, 0.0).astype(np.float32, copy=False)
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
        depth_for_lift, depth_scale = self._tracker_depth_for_lift(mask_packet)
        depth_max_m = float("inf") if float(1.5) <= 0.0 else float(1.5)
        current_lift_valid = _tracker_lift_valid_mask(
            tracks_yx=tracks_latest,
            visibility=display_visibility,
            depth=depth_for_lift,
            depth_scale_m_per_unit=float(depth_scale),
            mask=lift_mask,
            depth_min_m=float(0.2),
            depth_max_m=depth_max_m,
        )
        if self._tracker_consistent_visible is None or len(self._tracker_consistent_visible) != len(query_points):
            self._tracker_consistent_visible = np.ones((len(query_points),), dtype=bool)
        current_lift_valid_full = np.zeros_like(self._tracker_consistent_visible, dtype=bool)
        fitted_count = min(len(current_lift_valid), len(current_lift_valid_full))
        current_lift_valid_full[:fitted_count] = current_lift_valid[:fitted_count]
        self._tracker_consistent_visible &= current_lift_valid_full
        consistent_visible_count = int(np.count_nonzero(self._tracker_consistent_visible))
        lifted = lift_tracks_yx_to_world(
            tracks_yx=selected_tracks,
            visibility=selected_visibility,
            depth=depth_for_lift,
            intrinsics=mask_packet.intrinsics,
            c2w=self.table_c2w if self.table_c2w is not None else np.eye(4, dtype=np.float32),
            depth_scale_m_per_unit=float(depth_scale),
            mask=lift_mask,
            depth_min_m=float(0.2),
            depth_max_m=depth_max_m,
        )
        lift_ms = _elapsed_ms(lift_start_s, time.perf_counter())
        source_indices = lifted.source_indices
        if len(source_indices):
            lifted_query_indices = selected[source_indices].astype(np.int64, copy=False)
            lifted_query_is_object = selected_query_is_object[source_indices]
            lifted_query_is_controller = selected_query_is_controller[source_indices]
            lifted_query_target_id = selected_query_target_id[source_indices]
            lifted_query_controller_instance_id = selected_query_controller_instance_id[source_indices]
            lifted_marker_colors = self._tracker_query_rgb_u8[lifted_query_indices]
        else:
            lifted_query_indices = np.empty((0,), dtype=np.int64)
            lifted_query_is_object = np.empty((0,), dtype=bool)
            lifted_query_is_controller = np.empty((0,), dtype=bool)
            lifted_query_target_id = np.empty((0,), dtype=np.int64)
            lifted_query_controller_instance_id = np.empty((0,), dtype=np.int64)
            lifted_marker_colors = np.empty((0, 3), dtype=np.uint8)
        if object_residual_mask is not None and controller_residual_mask is not None:
            marker_residual_audit = _audit_marker_residual_subset(
                lifted.tracks_yx,
                object_residual_mask=object_residual_mask,
                controller_residual_mask=controller_residual_mask,
            )
        else:
            marker_residual_audit = MarkerResidualAudit(
                pixels_yx=np.empty((0, 2), dtype=np.int64),
                valid=np.empty((0,), dtype=bool),
                violation=np.empty((0,), dtype=bool),
                checked_count=0,
                violation_count=0,
                gate=tracker_marker_gate(self.args),
            )
        hand_a_query_count = int(np.count_nonzero(lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_A))
        hand_b_query_count = int(np.count_nonzero(lifted_query_controller_instance_id == QUERY_CONTROLLER_INSTANCE_HAND_B))
        object_query_count = int(np.count_nonzero(lifted_query_target_id == OBJECT_ID))
        remaining_object_query_count, remaining_controller_query_count, remaining_hand_a_query_count, remaining_hand_b_query_count = (
            _remaining_query_class_counts(
                query_alive_mask,
                query_is_object=query_is_object_all,
                query_is_controller=query_is_controller_all,
                query_controller_instance_id=query_controller_instance_id_all,
            )
        )
        remaining_query_count = int(np.count_nonzero(query_alive_mask))
        retired_query_count = max(0, int(len(query_points)) - remaining_query_count)
        done_s = time.perf_counter()
        stats = getattr(result, "stats", {}) or {}
        packet = TrackerMarkerPacket(
            seq=mask_packet.seq,
            marker_xyz_m=np.ascontiguousarray(lifted.points_world, dtype=np.float32).reshape(-1, 3),
            marker_colors_rgb_u8=np.ascontiguousarray(lifted_marker_colors, dtype=np.uint8).reshape(-1, 3),
            query_rgb_u8=np.ascontiguousarray(self._tracker_query_rgb_u8, dtype=np.uint8).reshape(-1, 3),
            query_points_yx=query_points,
            tracks_yx=np.ascontiguousarray(lifted.tracks_yx, dtype=np.float32).reshape(-1, 2),
            visibility=np.ascontiguousarray(selected_visibility[source_indices], dtype=np.float32),
            query_is_object=np.ascontiguousarray(lifted_query_is_object, dtype=bool),
            query_is_controller=np.ascontiguousarray(lifted_query_is_controller, dtype=bool),
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            query_count=int(len(query_points)),
            consistent_visible_count=consistent_visible_count,
            model_ms=float(stats.get("model_run_ms", stats.get("cuda_event_ms", 0.0)) or 0.0),
            lift_ms=float(lift_ms),
            e2e_ms=_elapsed_ms(started_s, done_s),
            backend=str(getattr(result, "backend", None) or adapter.name),
            display_scope=str(DEFAULT_TRACKER_DISPLAY_SCOPE),
            query_indices=np.ascontiguousarray(lifted_query_indices, dtype=np.int64),
            query_target_id=np.ascontiguousarray(lifted_query_target_id, dtype=np.int64),
            query_controller_instance_id=np.ascontiguousarray(lifted_query_controller_instance_id, dtype=np.int64),
            query_all_target_id=np.ascontiguousarray(query_target_id_all, dtype=np.int64),
            query_all_controller_instance_id=np.ascontiguousarray(query_controller_instance_id_all, dtype=np.int64),
            hand_a_query_count=hand_a_query_count,
            hand_b_query_count=hand_b_query_count,
            object_query_count=object_query_count,
            marker_pixels_yx=np.ascontiguousarray(marker_residual_audit.pixels_yx, dtype=np.int64).reshape(-1, 2),
            marker_residual_valid=np.ascontiguousarray(marker_residual_audit.valid, dtype=bool),
            marker_residual_violation=np.ascontiguousarray(marker_residual_audit.violation, dtype=bool),
            marker_residual_checked_count=int(marker_residual_audit.checked_count),
            marker_residual_violation_count=int(marker_residual_audit.violation_count),
            marker_residual_gate=str(marker_residual_audit.gate),
            query_alive_mask=np.ascontiguousarray(query_alive_mask, dtype=bool),
            remaining_query_count=remaining_query_count,
            remaining_object_query_count=remaining_object_query_count,
            remaining_controller_query_count=remaining_controller_query_count,
            remaining_hand_a_query_count=remaining_hand_a_query_count,
            remaining_hand_b_query_count=remaining_hand_b_query_count,
            retired_query_count=retired_query_count,
            all_tracks_yx=np.ascontiguousarray(tracks_latest, dtype=np.float32).reshape(-1, 2),
            all_tracker_visibility=np.ascontiguousarray(visibility_latest, dtype=np.float32).reshape(-1),
            coordinate_frame=self._pcd_coordinate_frame(),
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
                f"image_size={"256,256"} overlay_max={int(self.args.tracker_overlay_max_points)}",
                flush=True,
            )
            last_seq = -1
            while not self.stop_event.is_set():
                mask_packet = self.mask_slot.get_latest_after(last_seq)
                if mask_packet is None:
                    time.sleep(0.001)
                    continue
                last_seq = mask_packet.seq
                packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if packet is None:
                    continue
                self.tracker_marker_slot.put(packet)
                if self.headless_capture_writer is not None and not self._headless_product_rows_gated():
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
                f"image_size={"256,256"} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1 lossless=1",
                flush=True,
            )
            while not self.stop_event.is_set():
                mask_packet = self.lossless_tracker_mask_queue.get(stop_event=self.stop_event)
                if mask_packet is None:
                    break
                packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if packet is None:
                    raise LosslessPipelineError(f"tracker did not produce packet for seq {mask_packet.seq}")
                self._lossless_tracker_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity("tracker", stop_event=self.stop_event):
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
                self._record_fatal_worker_error("lossless TAPNext++ tracker worker", exc)


__all__ = ["_TrackerMixin"]
