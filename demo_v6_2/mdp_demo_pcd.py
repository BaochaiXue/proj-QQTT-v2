"""MainDataProcessingDemo point-cloud mixin."""

from __future__ import annotations

from demo_v6_2.mdp_cli import (
    controller_pcd_mask_erode_pixels,
    controller_tracking_enabled,
    object_pcd_mask_erode_pixels,
    object_tracking_enabled,
)
from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_packets import MaskedPcdPacket, PcdBuildResult
from demo_v6_2.mdp_pcd_depth import (
    _mask_from_yx,
    _select_points_by_yx_mask,
    _transform_points_c2w,
    apply_table_z_filter_with_yx,
    backproject_masked_rgbd_profiled,
    build_world_z_diagnostics,
    erode_binary_mask,
)


class _PcdMixin(_DemoRuntimeContract):
    """MainDataProcessingDemo point-cloud mixin."""

    def _lossless_pcd_worker(self) -> None:
        """Return the lossless PCD worker."""
        try:
            while not self.stop_event.is_set():
                mask_packet = self.lossless_pcd_mask_queue.get(
                    stop_event=self.stop_event
                )
                if mask_packet is None:
                    break
                result = self._build_pcd_packet_from_mask(mask_packet)
                self._maybe_start_shape_prior_from_pcd_result(result)
                self._lossless_pcd_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity(
                    "pcd", stop_event=self.stop_event
                ):
                    break
                with self._lossless_pairer_lock:
                    pairs = self.same_seq_pairer.add_pcd_result(result)
                    self._publish_pairer_outputs(pairs)
            with self._lossless_pairer_lock:
                pairs = self.same_seq_pairer.close_pcd()
                self._publish_pairer_outputs(pairs)
                self._maybe_finish_lossless_processing()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("lossless PCD worker", exc)

    def _write_headless_pcd_result(
        self,
        result: PcdBuildResult,
        tracker_packet: TrackerMarkerPacket | None = None,
        *,
        gated: bool | None = None,
    ) -> None:
        """Write headless PCD result.

        ``gated`` lets callers that already consulted
        :meth:`_headless_product_rows_gated` for this frame (to skip the
        tracker sidecar) reuse that single decision — the shape-prior worker
        flips the status asynchronously, so evaluating twice per frame could
        write a row whose query_trajectory sidecar was skipped.
        """
        if self.headless_capture_writer is None or result.depth_m is None:
            return
        if result.controller_pcd_mask is None or result.object_pcd_mask is None:
            return
        if gated is None:
            gated = self._headless_product_rows_gated()
        if gated:
            self._formal_timeline_gated_frames += 1
            return
        if (
            self._formal_timeline_gated_frames
            and not self._formal_timeline_metadata_written
        ):
            # First formal frame after the shape-prior wait: record the seam so
            # downstream tools can tell warmup frame 0 from output frame 1.
            self.headless_capture_writer.update_metadata(
                {
                    "formal_timeline_gated_frame_count": int(
                        self._formal_timeline_gated_frames
                    ),
                    "formal_timeline_start_seq": int(result.packet.seq),
                }
            )
            self._formal_timeline_metadata_written = True
        if not self._warmup_anchor_row_written:
            # Mirror chunk_data_stream._row_ready_for_realtime_chunk_start: only
            # a chunk-ready row may claim the warmup frame-0 slot; invalid
            # startup rows keep writing and are trimmed by the bridge.
            self._warmup_anchor_row_written = (
                int(result.packet.controller_point_count) >= CONTROLLER_FINAL_COUNT
                and int(result.packet.object_point_count) > 0
            )
        self.headless_capture_writer.write_pcd(
            result.packet,
            depth_m=result.depth_m,
            mask_packet=result.mask_packet,
            controller_pcd_mask=result.controller_pcd_mask,
            object_pcd_mask=result.object_pcd_mask,
            pcd_stride=int(result.pcd_stride),
            pcd_mask_erode_pixels=int(result.pcd_mask_erode_pixels),
            object_pcd_mask_erode_pixels=int(result.object_pcd_mask_erode_pixels),
            controller_pcd_mask_erode_pixels=int(
                result.controller_pcd_mask_erode_pixels
            ),
            tracker_packet=tracker_packet,
            stage_fps={
                "capture_fps": float(self.capture_stats.fps),
                "seg_fps": float(self.seg_stats.fps),
                "depth_fps": float(self.depth_stats.fps),
                "pcd_fps": float(self.pcd_stats.fps),
                "tracker_fps": float(self.tracker_stats.fps),
            },
            world_z_diagnostics=result.world_z_diagnostics,
            startup_hold_s=float(getattr(self, "_startup_hold_s", 0.0)),
        )

    def _build_pcd_packet_from_mask(
        self,
        mask_packet: MaskPacket,
    ) -> PcdBuildResult:
        """Build a masked point-cloud packet from a mask/depth pair."""
        start_s = time.perf_counter()
        assert self.ray_x is not None and self.ray_y is not None
        ray_x = self.ray_x
        ray_y = self.ray_y
        if mask_packet.depth_source == "ffs":
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            remote_rtt_ms = 0.0
            remote_server_total_ms = 0.0
            remote_request_kb = 0.0
            remote_response_kb = 0.0
            depth_convert_ms = 0.0
            (
                depth_m,
                ffs_ms,
                ffs_align_ms,
                remote_rtt_ms,
                remote_server_total_ms,
                remote_request_kb,
                remote_response_kb,
            ) = self._compute_external_ffs_depth_color_m(mask_packet)
        else:
            ffs_ms = 0.0
            ffs_align_ms = 0.0
            remote_rtt_ms = 0.0
            remote_server_total_ms = 0.0
            remote_request_kb = 0.0
            remote_response_kb = 0.0
            if mask_packet.depth_u16 is None:
                raise RuntimeError("PCD packet requires RGB-D depth")
            depth_convert_start_s = time.perf_counter()
            depth_m = np.ascontiguousarray(
                mask_packet.depth_u16.astype(np.float32)
                * np.float32(mask_packet.depth_scale_m_per_unit)
            )
            depth_convert_ms = _elapsed_ms(depth_convert_start_s, time.perf_counter())

        stride = int(1)
        if stride > 1:
            color_bgr = mask_packet.color_bgr[::stride, ::stride]
            depth_for_pcd = depth_m[::stride, ::stride]
            controller_mask = mask_packet.controller_mask[::stride, ::stride]
            object_mask = mask_packet.object_mask[::stride, ::stride]
            ray_x_for_pcd = ray_x[::stride, ::stride]
            ray_y_for_pcd = ray_y[::stride, ::stride]
        else:
            color_bgr = mask_packet.color_bgr
            depth_for_pcd = depth_m
            controller_mask = mask_packet.controller_mask
            object_mask = mask_packet.object_mask
            ray_x_for_pcd = ray_x
            ray_y_for_pcd = ray_y
        pcd_mask_erode_pixels = int(DEFAULT_PCD_MASK_ERODE_PIXELS)
        controller_erode_pixels = controller_pcd_mask_erode_pixels(self.args)
        object_erode_pixels = object_pcd_mask_erode_pixels(self.args)
        if controller_erode_pixels > 0:
            controller_mask = erode_binary_mask(
                controller_mask, erode_pixels=controller_erode_pixels
            )
        if object_erode_pixels > 0:
            object_mask = erode_binary_mask(
                object_mask, erode_pixels=object_erode_pixels
            )
        empty_pcd_timing = {
            "pcd_mask_intersection_ms": 0.0,
            "pcd_select_ms": 0.0,
            "pcd_backproject_ms": 0.0,
            "pcd_color_gather_ms": 0.0,
        }
        if controller_tracking_enabled(self.args):
            controller_xyz, controller_colors, controller_yx, controller_pcd_timing = (
                backproject_masked_rgbd_profiled(
                    color_bgr=color_bgr,
                    depth_m=depth_for_pcd,
                    mask=controller_mask,
                    ray_x=ray_x_for_pcd,
                    ray_y=ray_y_for_pcd,
                    depth_min_m=float(0.2),
                    depth_max_m=float(1.5),
                    color_mode=str(self.args.pcd_color_mode),
                    class_rgb=tuple(self.args.controller_color),
                    return_yx=True,
                )
            )
            if stride > 1:
                controller_yx = np.ascontiguousarray(
                    controller_yx * int(stride), dtype=np.int64
                )
        else:
            controller_xyz = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
            controller_yx = np.empty((0, 2), dtype=np.int64)
            controller_pcd_timing = dict(empty_pcd_timing)
        if object_tracking_enabled(self.args):
            object_xyz, object_colors, object_yx, object_pcd_timing = (
                backproject_masked_rgbd_profiled(
                    color_bgr=color_bgr,
                    depth_m=depth_for_pcd,
                    mask=object_mask,
                    ray_x=ray_x_for_pcd,
                    ray_y=ray_y_for_pcd,
                    depth_min_m=float(0.2),
                    depth_max_m=float(1.5),
                    color_mode=str(self.args.pcd_color_mode),
                    class_rgb=tuple(self.args.object_color),
                    return_yx=True,
                )
            )
            if stride > 1:
                object_yx = np.ascontiguousarray(
                    object_yx * int(stride), dtype=np.int64
                )
        else:
            object_xyz = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
            object_yx = np.empty((0, 2), dtype=np.int64)
            object_pcd_timing = dict(empty_pcd_timing)
        render_controller_xyz = controller_xyz
        render_controller_colors = controller_colors
        render_controller_yx = controller_yx
        render_object_xyz = object_xyz
        render_object_colors = object_colors
        render_object_yx = object_yx
        render_controller_xyz = _transform_points_c2w(
            render_controller_xyz, self.table_c2w
        )
        render_object_xyz = _transform_points_c2w(render_object_xyz, self.table_c2w)
        hand_a_xyz = None
        hand_b_xyz = None
        if mask_packet.hand_a_mask is not None:
            hand_a_xyz = _select_points_by_yx_mask(
                render_controller_xyz,
                render_controller_yx,
                mask_packet.hand_a_mask,
            )
        if mask_packet.hand_b_mask is not None:
            hand_b_xyz = _select_points_by_yx_mask(
                render_controller_xyz,
                render_controller_yx,
                mask_packet.hand_b_mask,
            )
        world_z_diagnostics = build_world_z_diagnostics(
            object_xyz_m=render_object_xyz,
            controller_xyz_m=render_controller_xyz,
            hand_a_xyz_m=hand_a_xyz,
            hand_b_xyz_m=hand_b_xyz,
            table_z_m=TABLE_Z_M,
            thresholds_m=DEFAULT_TABLE_Z_DIAGNOSTIC_THRESHOLDS_M,
        )
        table_z_filter_stats: dict[str, Any] = {
            "enabled": bool(self.args.enable_table_z_filter),
            "threshold_m": float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
            "table_z_above_direction": TABLE_Z_ABOVE_DIRECTION,
            "classes": str(TABLE_Z_FILTER_CLASS_BOTH),
            "object": None,
            "controller": None,
        }
        if bool(self.args.enable_table_z_filter):
            classes = str(TABLE_Z_FILTER_CLASS_BOTH)
            if classes in {TABLE_Z_FILTER_CLASS_OBJECT, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    render_object_xyz,
                    render_object_colors,
                    render_object_yx,
                    object_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    render_object_xyz,
                    render_object_colors,
                    render_object_yx,
                    enabled=True,
                    threshold_m=float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
                    table_z_m=TABLE_Z_M,
                )
                table_z_filter_stats["object"] = object_table_z_stats
            if classes in {TABLE_Z_FILTER_CLASS_CONTROLLER, TABLE_Z_FILTER_CLASS_BOTH}:
                (
                    render_controller_xyz,
                    render_controller_colors,
                    render_controller_yx,
                    controller_table_z_stats,
                ) = apply_table_z_filter_with_yx(
                    render_controller_xyz,
                    render_controller_colors,
                    render_controller_yx,
                    enabled=True,
                    threshold_m=float(DEFAULT_TABLE_Z_FILTER_THRESHOLD_M),
                    table_z_m=TABLE_Z_M,
                )
                table_z_filter_stats["controller"] = controller_table_z_stats
        world_z_diagnostics["runtime_table_z_filter"] = table_z_filter_stats
        done_s = time.perf_counter()
        timing = replace(
            mask_packet.timing,
            ffs_ms=ffs_ms,
            ffs_align_ms=ffs_align_ms,
            remote_rtt_ms=remote_rtt_ms,
            remote_server_total_ms=remote_server_total_ms,
            remote_request_kb=remote_request_kb,
            remote_response_kb=remote_response_kb,
            depth_convert_ms=depth_convert_ms,
            pcd_mask_intersection_ms=float(
                controller_pcd_timing["pcd_mask_intersection_ms"]
                + object_pcd_timing["pcd_mask_intersection_ms"]
            ),
            pcd_select_ms=float(
                controller_pcd_timing["pcd_select_ms"]
                + object_pcd_timing["pcd_select_ms"]
            ),
            pcd_backproject_ms=float(
                controller_pcd_timing["pcd_backproject_ms"]
                + object_pcd_timing["pcd_backproject_ms"]
            ),
            pcd_color_gather_ms=float(
                controller_pcd_timing["pcd_color_gather_ms"]
                + object_pcd_timing["pcd_color_gather_ms"]
            ),
            pcd_ms=_elapsed_ms(start_s, done_s),
        )
        packet = MaskedPcdPacket(
            seq=mask_packet.seq,
            controller_xyz_m=render_controller_xyz,
            controller_colors_rgb_u8=render_controller_colors,
            object_xyz_m=render_object_xyz,
            object_colors_rgb_u8=render_object_colors,
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            dropped_capture_frames=mask_packet.dropped_capture_frames,
            dropped_seg_frames=self.mask_slot.dropped_count,
            timing=timing,
            coordinate_frame=pcd_coordinate_frame(self.table_c2w),
            source_timestamp_s=mask_packet.source_timestamp_s,
            source_frame_index=mask_packet.source_frame_index,
            source_step=mask_packet.source_step,
        )
        return PcdBuildResult(
            packet=packet,
            depth_m=depth_m,
            mask_packet=mask_packet,
            controller_pcd_mask=controller_mask,
            object_pcd_mask=object_mask,
            object_observation_mask=_mask_from_yx(
                tuple(mask_packet.object_mask.shape[:2]),
                render_object_yx,
            ),
            pcd_stride=stride,
            pcd_mask_erode_pixels=pcd_mask_erode_pixels,
            object_pcd_mask_erode_pixels=object_erode_pixels,
            controller_pcd_mask_erode_pixels=controller_erode_pixels,
            world_z_diagnostics=world_z_diagnostics,
        )


__all__ = ["_PcdMixin"]
