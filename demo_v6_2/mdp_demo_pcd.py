"""Canonical processed-mask geometry stage for Demo v6.2."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_packets import (
    MaskedPcdPacket,
    PcdBuildResult,
    ProcessedFramePacket,
)
from demo_v6_2.phystwin_strict_product import (
    PHYSTWIN_DEPTH_MAX_M,
    PHYSTWIN_DEPTH_MIN_M,
    apply_depth_validity_to_mask_frame,
    apply_radius_outlier_to_mask_frame,
    dense_world_pcd_grid,
)


class _PcdMixin(_DemoRuntimeContract):
    """Build the one processed frame consumed by every formal stage."""

    def _lossless_processed_frame_worker(self) -> None:
        """Build canonical frames, then fan them out to PCD pairing and tracker."""
        try:
            while not self.stop_event.is_set():
                raw_mask_packet = self.lossless.mask_queue.get(
                    stop_event=self.stop_event
                )
                if raw_mask_packet is None:
                    break
                result = self._build_processed_frame_result(raw_mask_packet)
                self._maybe_start_shape_prior_from_pcd_result(result)

                if not self.lossless.processed_frame_queue.wait_for_capacity(
                    stop_event=self.stop_event
                ):
                    break
                self.lossless.processed_frame_queue.put(result.processed_frame)

                if not self.lossless.submit_pcd_result(
                    result, stop_event=self.stop_event
                ):
                    break

            self.lossless.processed_frame_queue.close()
            self.lossless.close_pcd_side()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("processed-frame worker", exc)

    def _write_headless_pcd_result(
        self,
        result: PcdBuildResult,
        tracker_packet: TrackerMarkerPacket | None = None,
        *,
        gated: bool | None = None,
    ) -> None:
        """Write one canonical processed frame and its paired products."""
        if self.headless_capture_writer is None:
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
            self._warmup_anchor_row_written = (
                int(result.packet.controller_point_count) >= CONTROLLER_FINAL_COUNT
                and int(result.packet.object_point_count) > 0
            )
        self.headless_capture_writer.write_pcd(
            result.packet,
            processed_frame=result.processed_frame,
            tracker_packet=tracker_packet,
            stage_fps={
                "capture_fps": float(self.capture_stats.fps),
                "seg_fps": float(self.seg_stats.fps),
                "depth_fps": float(self.depth_stats.fps),
                "pcd_fps": float(self.pcd_stats.fps),
                "tracker_fps": float(self.tracker_stats.fps),
            },
            startup_hold_s=float(getattr(self, "_startup_hold_s", 0.0)),
        )

    def _metric_depth_for_mask_packet(
        self, mask_packet: MaskPacket
    ) -> tuple[np.ndarray, dict[str, float]]:
        """Resolve one color-aligned metric depth frame and timing values."""
        if mask_packet.depth_source == "ffs":
            if self.depth_engine is None:
                raise RuntimeError("FFS depth engine is not initialized")
            depth_m, ffs_ms, ffs_align_ms = self.depth_engine.compute_color_depth(
                mask_packet
            )
            return np.ascontiguousarray(depth_m, dtype=np.float32), {
                "ffs_ms": float(ffs_ms),
                "ffs_align_ms": float(ffs_align_ms),
                "remote_rtt_ms": 0.0,
                "remote_server_total_ms": 0.0,
                "remote_request_kb": 0.0,
                "remote_response_kb": 0.0,
                "depth_convert_ms": 0.0,
            }
        if mask_packet.depth_u16 is None:
            raise RuntimeError("formal processed frame requires RGB-D depth")
        started_s = time.perf_counter()
        depth_m = np.ascontiguousarray(
            mask_packet.depth_u16.astype(np.float32)
            * np.float32(mask_packet.depth_scale_m_per_unit)
        )
        return depth_m, {
            "ffs_ms": 0.0,
            "ffs_align_ms": 0.0,
            "remote_rtt_ms": 0.0,
            "remote_server_total_ms": 0.0,
            "remote_request_kb": 0.0,
            "remote_response_kb": 0.0,
            "depth_convert_ms": _elapsed_ms(started_s, time.perf_counter()),
        }

    def _build_processed_frame_result(
        self,
        mask_packet: MaskPacket,
    ) -> PcdBuildResult:
        """Build the origin-style processed mask, dense grid, and class PCDs."""
        started_s = time.perf_counter()
        if self.table_c2w is None:
            raise RuntimeError(
                "formal processed frames require camera-to-world calibration"
            )
        c2w = np.asarray(self.table_c2w, dtype=np.float32)
        if c2w.shape != (4, 4) or not np.isfinite(c2w).all():
            raise RuntimeError("camera-to-world calibration must be a finite 4x4")

        depth_m, depth_timing = self._metric_depth_for_mask_packet(mask_packet)
        rgb_u8 = np.ascontiguousarray(mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8)
        pcd_points, pcd_colors = dense_world_pcd_grid(
            depth_m=depth_m,
            color_rgb_u8=rgb_u8,
            intrinsics=mask_packet.intrinsics,
            c2w=c2w,
        )
        raw_masks = {
            "object": mask_packet.object_mask,
            "controller": mask_packet.controller_mask,
            "hand_a": mask_packet.hand_a_mask,
            "hand_b": mask_packet.hand_b_mask,
        }
        depth_valid_masks = apply_depth_validity_to_mask_frame(raw_masks, depth_m)
        processed_masks = apply_radius_outlier_to_mask_frame(
            depth_valid_masks,
            pcd_points,
        )
        object_mask = np.ascontiguousarray(processed_masks["object"], dtype=bool)
        controller_mask = np.ascontiguousarray(
            processed_masks["controller"], dtype=bool
        )
        if not np.any(object_mask):
            raise RuntimeError(
                f"processed object mask is empty at seq {mask_packet.seq}"
            )
        if not np.any(controller_mask):
            raise RuntimeError(
                f"processed controller mask is empty at seq {mask_packet.seq}"
            )

        processed_mask_packet = replace(
            mask_packet,
            controller_mask=controller_mask,
            object_mask=object_mask,
            hand_a_mask=np.ascontiguousarray(
                processed_masks.get("hand_a", np.zeros_like(controller_mask)),
                dtype=bool,
            ),
            hand_b_mask=np.ascontiguousarray(
                processed_masks.get("hand_b", np.zeros_like(controller_mask)),
                dtype=bool,
            ),
        )
        depth_valid_mask = np.ascontiguousarray(
            np.isfinite(depth_m)
            & (depth_m > np.float32(PHYSTWIN_DEPTH_MIN_M))
            & (depth_m < np.float32(PHYSTWIN_DEPTH_MAX_M)),
            dtype=bool,
        )
        processed_frame = ProcessedFramePacket(
            seq=int(mask_packet.seq),
            mask_packet=processed_mask_packet,
            depth_m=np.ascontiguousarray(depth_m, dtype=np.float32),
            depth_valid_mask=depth_valid_mask,
            pcd_points=np.ascontiguousarray(pcd_points, dtype=np.float32),
            pcd_colors=np.ascontiguousarray(pcd_colors, dtype=np.uint8),
        )

        points_grid = processed_frame.pcd_points[0]
        colors_grid = processed_frame.pcd_colors[0]
        controller_xyz = np.ascontiguousarray(
            points_grid[controller_mask], dtype=np.float32
        ).reshape(-1, 3)
        object_xyz = np.ascontiguousarray(
            points_grid[object_mask], dtype=np.float32
        ).reshape(-1, 3)
        if str(self.args.pcd_color_mode) == "class":
            controller_colors = np.tile(
                np.asarray(self.args.controller_color, dtype=np.uint8),
                (len(controller_xyz), 1),
            )
            object_colors = np.tile(
                np.asarray(self.args.object_color, dtype=np.uint8),
                (len(object_xyz), 1),
            )
        else:
            controller_colors = colors_grid[controller_mask]
            object_colors = colors_grid[object_mask]

        done_s = time.perf_counter()
        timing = replace(
            mask_packet.timing,
            **depth_timing,
            pcd_ms=_elapsed_ms(started_s, done_s),
        )
        packet = MaskedPcdPacket(
            seq=mask_packet.seq,
            controller_xyz_m=controller_xyz,
            controller_colors_rgb_u8=np.ascontiguousarray(
                controller_colors, dtype=np.uint8
            ).reshape(-1, 3),
            object_xyz_m=object_xyz,
            object_colors_rgb_u8=np.ascontiguousarray(
                object_colors, dtype=np.uint8
            ).reshape(-1, 3),
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=mask_packet.receive_perf_s,
            process_done_perf_s=done_s,
            dropped_capture_frames=mask_packet.dropped_capture_frames,
            dropped_seg_frames=self.mask_slot.dropped_count,
            timing=timing,
            coordinate_frame=TABLE_WORLD_FRAME_KIND,
            source_timestamp_s=mask_packet.source_timestamp_s,
            source_frame_index=mask_packet.source_frame_index,
            source_step=mask_packet.source_step,
        )
        return PcdBuildResult(packet=packet, processed_frame=processed_frame)


__all__ = ["_PcdMixin"]
