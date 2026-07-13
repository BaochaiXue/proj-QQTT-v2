"""Headless capture: on-disk artifact writer."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_packets import (
    ProcessedFramePacket,
    _full_tracker_arrays_for_prepared_frame,
)
from demo_v6_2.perception.mdp_tracker_geometry import (
    _mask_packet_hand_a_mask,
    _mask_packet_hand_b_mask,
)


class HeadlessCaptureWriter:
    def __init__(self, output_dir: str | Path, *, metadata: dict[str, Any]) -> None:
        """Initialize HeadlessCaptureWriter."""
        self.output_dir = _resolve_path(output_dir)
        self.prepared_only = bool(metadata.get("headless_prepared_only", False))
        self.write_input_rgb_timeline = bool(
            metadata.get("write_input_rgb_timeline", False)
        )
        self.saved_pcd_source = str(
            metadata.get("saved_pcd_source") or HEADLESS_CAPTURE_SAVED_PCD_SOURCE
        )
        self.pcd_coordinate_frame = str(
            metadata.get("pcd_coordinate_frame")
            or metadata.get("coordinate_frame")
            or COORDINATE_FRAME
        )
        self.pcd_dir = self.output_dir / "pcd"
        self.depth_dir = self.output_dir / "depth_color_m"
        self.rgb_dir = self.output_dir / "rgb"
        self.trajectory_dir = self.output_dir / "query_trajectory"
        self.mask_dir = self.output_dir / "masks"
        self.shape_prior_dir = self.output_dir / "shape_prior"
        self.prepared_phystwin_dir = self.output_dir / "prepared_phystwin"
        self.input_rgb_dir = self.output_dir / "input_rgb"
        self.frames_path = self.output_dir / "frames.jsonl"
        self.input_frames_path = self.output_dir / "input_frames.jsonl"
        self.metadata_path = self.output_dir / "metadata.json"
        self._lock = threading.Lock()
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.pcd_dir.mkdir(parents=True, exist_ok=True)
        self.depth_dir.mkdir(parents=True, exist_ok=True)
        self.rgb_dir.mkdir(parents=True, exist_ok=True)
        self.trajectory_dir.mkdir(parents=True, exist_ok=True)
        self.mask_dir.mkdir(parents=True, exist_ok=True)
        self.prepared_phystwin_dir.mkdir(parents=True, exist_ok=True)
        self.input_rgb_dir.mkdir(parents=True, exist_ok=True)
        self.frames_path.write_text("", encoding="utf-8")
        self.input_frames_path.write_text("", encoding="utf-8")
        payload = dict(metadata)
        payload["headless_capture_enabled"] = True
        payload["headless_prepared_only"] = bool(self.prepared_only)
        payload["write_input_rgb_timeline"] = bool(self.write_input_rgb_timeline)
        payload["saved_pcd_source"] = self.saved_pcd_source
        payload["saved_mask_source"] = "origin_style_processed_masks"
        payload["saved_rgb_source"] = "segmentation_color_bgr"
        payload["input_rgb_timeline"] = "input_frames.jsonl"
        payload["startup_hold_s"] = float(payload.get("startup_hold_s") or 0.0)
        payload["output_dir"] = _repo_relative_path_text(self.output_dir)
        self._metadata_payload = payload
        self._write_metadata_payload(payload)

    def _relative(self, path: Path) -> str:
        """Return the relative."""
        try:
            return str(path.relative_to(self.output_dir))
        except ValueError:
            return str(path)

    def _write_metadata_payload(self, payload: dict[str, Any]) -> None:
        """Write metadata payload."""
        tmp_path = self.metadata_path.with_name(f"{self.metadata_path.name}.tmp")
        tmp_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        tmp_path.replace(self.metadata_path)

    def update_metadata(self, values: dict[str, Any]) -> None:
        """Update metadata."""
        with self._lock:
            payload = dict(self._metadata_payload)
            payload.update(values)
            self._metadata_payload = payload
            self._write_metadata_payload(payload)

    def write_shape_prior_result(
        self, result: shape_prior_warmup.ShapePriorResult
    ) -> None:
        """Write shape prior result."""
        self.shape_prior_dir.mkdir(parents=True, exist_ok=True)
        path = self.shape_prior_dir / "points.npz"
        np.savez_compressed(
            path,
            seq=np.asarray([int(result.seq)], dtype=np.int64),
            source_seq=np.asarray(
                [-1 if result.source_seq is None else int(result.source_seq)],
                dtype=np.int64,
            ),
            source_timestamp_s=np.asarray(
                [
                    np.nan
                    if result.source_timestamp_s is None
                    else float(result.source_timestamp_s)
                ],
                dtype=np.float64,
            ),
            points_m=np.ascontiguousarray(result.points_m, dtype=np.float32).reshape(
                -1, 3
            ),
            colors_rgb_u8=np.ascontiguousarray(
                result.colors_rgb_u8, dtype=np.uint8
            ).reshape(-1, 3),
            surface_points_m=np.ascontiguousarray(
                result.surface_points_m, dtype=np.float32
            ).reshape(-1, 3),
            interior_points_m=np.ascontiguousarray(
                result.interior_points_m, dtype=np.float32
            ).reshape(-1, 3),
            metadata_json=np.asarray(
                [json.dumps(dict(result.metadata), sort_keys=True)]
            ),
        )
        values = dict(result.metadata)
        values.update(
            {
                "shape_prior_status": str(result.status),
                "shape_prior_source_seq": result.source_seq,
                "shape_prior_source_time_s": result.source_timestamp_s,
                "shape_prior_ready_seq": int(result.seq),
                "shape_prior_path": self._relative(path),
                "shape_prior_point_count": int(
                    np.asarray(result.points_m).reshape(-1, 3).shape[0]
                ),
                "shape_prior_surface_point_count": int(
                    np.asarray(result.surface_points_m).reshape(-1, 3).shape[0]
                ),
                "shape_prior_interior_point_count": int(
                    np.asarray(result.interior_points_m).reshape(-1, 3).shape[0]
                ),
            }
        )
        self.update_metadata(values)

    def write_input_frame(self, packet: FramePacket) -> None:
        """Write input frame."""
        seq_name = f"{int(packet.seq):06d}"
        rgb_path = self.input_rgb_dir / f"{seq_name}.png"
        row = {
            "seq": int(packet.seq),
            "source_timestamp_s": (
                None
                if packet.source_timestamp_s is None
                else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None
                if packet.source_frame_index is None
                else int(packet.source_frame_index)
            ),
            "source_step": None
            if packet.source_step is None
            else int(packet.source_step),
            "receive_perf_s": float(packet.receive_perf_s),
        }
        if self.write_input_rgb_timeline or not self.prepared_only:
            main_warmup.bgr_to_pil_rgb(packet.color_bgr).save(rgb_path)
            row["input_rgb_path"] = self._relative(rgb_path)
        with self._lock:
            with self.input_frames_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(row, sort_keys=True) + "\n")

    def write_pcd(
        self,
        packet: MaskedPcdPacket,
        *,
        processed_frame: ProcessedFramePacket,
        tracker_packet: TrackerMarkerPacket | None = None,
        stage_fps: dict[str, float] | None = None,
        startup_hold_s: float = 0.0,
    ) -> None:
        """Write RGB-D, PCD, masks, tracking, and prepared PhysTwin artifacts."""
        if self.prepared_only and tracker_packet is None:
            raise RuntimeError(
                "prepared-only headless capture requires a tracker packet"
            )
        fps_info = stage_fps or {}
        if int(packet.seq) != int(processed_frame.seq):
            raise ValueError(
                "headless PCD/processed-frame sequence mismatch: "
                f"pcd={packet.seq} processed={processed_frame.seq}"
            )
        depth_m = processed_frame.depth_m
        mask_packet = processed_frame.mask_packet
        seq_name = f"{int(packet.seq):06d}"
        pcd_path = self.pcd_dir / f"{seq_name}.npz"
        depth_path = self.depth_dir / f"{seq_name}.npy"
        rgb_path = self.rgb_dir / f"{seq_name}.png"
        query_path = self.trajectory_dir / f"{seq_name}.npz"
        mask_path = self.mask_dir / f"{seq_name}.npz"
        prepared_phystwin_path = self.prepared_phystwin_dir / f"{seq_name}.npz"
        if not self.prepared_only:
            main_warmup.bgr_to_pil_rgb(mask_packet.color_bgr).save(rgb_path)
            np.save(
                depth_path,
                np.ascontiguousarray(depth_m, dtype=np.float32),
            )
            np.savez_compressed(
                mask_path,
                seq=np.asarray([int(packet.seq)], dtype=np.int64),
                controller_mask=np.ascontiguousarray(
                    mask_packet.controller_mask, dtype=bool
                ),
                object_mask=np.ascontiguousarray(mask_packet.object_mask, dtype=bool),
                hand_a_mask=np.ascontiguousarray(
                    _mask_packet_hand_a_mask(mask_packet), dtype=bool
                ),
                hand_b_mask=np.ascontiguousarray(
                    _mask_packet_hand_b_mask(mask_packet), dtype=bool
                ),
                mask_source=np.asarray(["origin_style_processed_masks"]),
            )
            np.savez(
                pcd_path,
                seq=np.asarray([int(packet.seq)], dtype=np.int64),
                controller_xyz_m=np.ascontiguousarray(
                    packet.controller_xyz_m, dtype=np.float32
                ),
                controller_rgb_u8=np.ascontiguousarray(
                    packet.controller_colors_rgb_u8, dtype=np.uint8
                ),
                object_xyz_m=np.ascontiguousarray(
                    packet.object_xyz_m, dtype=np.float32
                ),
                object_rgb_u8=np.ascontiguousarray(
                    packet.object_colors_rgb_u8, dtype=np.uint8
                ),
                intrinsics=np.asarray(
                    [
                        float(packet.intrinsics.fx),
                        float(packet.intrinsics.fy),
                        float(packet.intrinsics.cx),
                        float(packet.intrinsics.cy),
                    ],
                    dtype=np.float32,
                ),
                saved_pcd_source=np.asarray([self.saved_pcd_source]),
                coordinate_frame=np.asarray(
                    [str(packet.coordinate_frame or self.pcd_coordinate_frame)]
                ),
            )
        prepared_phystwin_frame_path: str | None = None
        if tracker_packet is not None:
            full_tracks_yx, full_visibility = _full_tracker_arrays_for_prepared_frame(
                tracker_packet
            )
            mask_frame = {
                "object": np.asarray(mask_packet.object_mask, dtype=bool),
                "controller": np.asarray(mask_packet.controller_mask, dtype=bool),
                "hand_a": np.asarray(_mask_packet_hand_a_mask(mask_packet), dtype=bool),
                "hand_b": np.asarray(_mask_packet_hand_b_mask(mask_packet), dtype=bool),
            }
            prepared = prepare_phystwin_frame(
                seq=int(packet.seq),
                rgb_frame=np.ascontiguousarray(
                    mask_packet.color_bgr[:, :, ::-1], dtype=np.uint8
                ),
                depth_m=np.asarray(depth_m, dtype=np.float32),
                processed_mask_frame=mask_frame,
                pcd_points=processed_frame.pcd_points,
                pcd_colors=processed_frame.pcd_colors,
                tracks_yx=full_tracks_yx,
                visibility=full_visibility,
                query_points_yx=np.asarray(
                    tracker_packet.query_points_yx, dtype=np.float32
                ),
                source_timestamp_s=packet.source_timestamp_s,
                source_frame_index=packet.source_frame_index,
                source_step=packet.source_step,
            )
            write_prepared_phystwin_frame(prepared_phystwin_path, prepared)
            prepared_phystwin_frame_path = self._relative(prepared_phystwin_path)
        pair_process_done_s = (
            max(
                float(packet.process_done_perf_s),
                float(tracker_packet.process_done_perf_s),
            )
            if tracker_packet is not None
            else float(packet.process_done_perf_s)
        )
        row = {
            "seq": int(packet.seq),
            "source_timestamp_s": (
                None
                if packet.source_timestamp_s is None
                else float(packet.source_timestamp_s)
            ),
            "source_frame_index": (
                None
                if packet.source_frame_index is None
                else int(packet.source_frame_index)
            ),
            "source_step": None
            if packet.source_step is None
            else int(packet.source_step),
            "startup_hold_s": float(startup_hold_s),
            "pipeline_latency_ms": float(
                pair_process_done_s - float(packet.receive_perf_s)
            )
            * 1000.0,
            "capture_fps": float(fps_info.get("capture_fps", 0.0)),
            "seg_fps": float(fps_info.get("seg_fps", 0.0)),
            "depth_fps": float(fps_info.get("depth_fps", 0.0)),
            "pcd_fps": float(fps_info.get("pcd_fps", 0.0)),
            "tracker_fps": float(fps_info.get("tracker_fps", 0.0)),
            "saved_pcd_source": self.saved_pcd_source,
            "marker_count": int(tracker_packet.marker_count)
            if tracker_packet is not None
            else 0,
            "controller_point_count": int(packet.controller_point_count),
            "object_point_count": int(packet.object_point_count),
            "controller_mask_pixels": int(
                np.count_nonzero(mask_packet.controller_mask)
            ),
            "object_mask_pixels": int(np.count_nonzero(mask_packet.object_mask)),
            "hand_a_mask_pixels": int(
                np.count_nonzero(_mask_packet_hand_a_mask(mask_packet))
            ),
            "hand_b_mask_pixels": int(
                np.count_nonzero(_mask_packet_hand_b_mask(mask_packet))
            ),
            "hand_a_query_count": int(tracker_packet.hand_a_query_count)
            if tracker_packet is not None
            else 0,
            "hand_b_query_count": int(tracker_packet.hand_b_query_count)
            if tracker_packet is not None
            else 0,
            "object_query_count": int(tracker_packet.object_query_count)
            if tracker_packet is not None
            else 0,
            "query_count": int(tracker_packet.query_count)
            if tracker_packet is not None
            else 0,
            "receive_perf_s": float(packet.receive_perf_s),
            "process_done_perf_s": float(packet.process_done_perf_s),
            "pair_process_done_perf_s": float(pair_process_done_s),
            "timing": asdict(packet.timing),
        }
        if not self.prepared_only:
            row.update(
                {
                    "pcd_path": self._relative(pcd_path),
                    "depth_color_m_path": self._relative(depth_path),
                    "rgb_path": self._relative(rgb_path),
                    "query_trajectory_path": self._relative(query_path),
                    "mask_path": self._relative(mask_path),
                }
            )
        if prepared_phystwin_frame_path is not None:
            row["prepared_phystwin_frame_path"] = prepared_phystwin_frame_path
        line = json.dumps(row, sort_keys=True)
        with self._lock:
            with self.frames_path.open("a", encoding="utf-8") as handle:
                handle.write(line + "\n")

    def write_tracker(self, packet: TrackerMarkerPacket) -> None:
        """Write tracker."""
        if self.prepared_only:
            return
        seq_name = f"{int(packet.seq):06d}"
        path = self.trajectory_dir / f"{seq_name}.npz"
        np.savez(
            path,
            seq=np.asarray([int(packet.seq)], dtype=np.int64),
            query_points_yx=np.ascontiguousarray(
                packet.query_points_yx, dtype=np.float32
            ),
            query_indices=np.ascontiguousarray(packet.query_indices, dtype=np.int64),
            query_rgb_u8=np.ascontiguousarray(packet.query_rgb_u8, dtype=np.uint8),
            marker_xyz_m=np.ascontiguousarray(packet.marker_xyz_m, dtype=np.float32),
            marker_rgb_u8=np.ascontiguousarray(
                packet.marker_colors_rgb_u8, dtype=np.uint8
            ),
            tracks_yx=np.ascontiguousarray(packet.tracks_yx, dtype=np.float32),
            visibility=np.ascontiguousarray(packet.visibility, dtype=np.float32),
            query_is_object=np.ascontiguousarray(packet.query_is_object, dtype=bool),
            query_is_controller=np.ascontiguousarray(
                packet.query_is_controller, dtype=bool
            ),
            query_target_id=np.ascontiguousarray(
                packet.query_target_id, dtype=np.int64
            ),
            query_controller_instance_id=np.ascontiguousarray(
                packet.query_controller_instance_id, dtype=np.int64
            ),
            all_tracks_yx=np.ascontiguousarray(
                packet.all_tracks_yx, dtype=np.float32
            ).reshape(-1, 2),
            all_tracker_visibility=np.ascontiguousarray(
                packet.all_tracker_visibility, dtype=np.float32
            ).reshape(-1),
            all_observation_visibility=np.ascontiguousarray(
                packet.all_observation_visibility, dtype=bool
            ).reshape(-1),
            query_count=np.asarray([int(packet.query_count)], dtype=np.int64),
            consistent_visible_count=np.asarray(
                [int(packet.consistent_visible_count)], dtype=np.int64
            ),
            hand_a_query_count=np.asarray(
                [int(packet.hand_a_query_count)], dtype=np.int64
            ),
            hand_b_query_count=np.asarray(
                [int(packet.hand_b_query_count)], dtype=np.int64
            ),
            object_query_count=np.asarray(
                [int(packet.object_query_count)], dtype=np.int64
            ),
            model_ms=np.asarray([float(packet.model_ms)], dtype=np.float32),
            lift_ms=np.asarray([float(packet.lift_ms)], dtype=np.float32),
            e2e_ms=np.asarray([float(packet.e2e_ms)], dtype=np.float32),
            coordinate_frame=np.asarray(
                [str(packet.coordinate_frame or self.pcd_coordinate_frame)]
            ),
        )


__all__ = [
    "HeadlessCaptureWriter",
]
