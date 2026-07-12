"""MainDataProcessingDemo point-cloud/filter mixin."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import controller_pcd_mask_erode_pixels, controller_tracking_enabled, object_pcd_mask_erode_pixels, object_tracking_enabled, pcd_filter_enabled
from demo_v6_2.mdp_packets import MaskedPcdPacket, PcdBuildResult, PcdFilterTelemetry
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_pcd_depth import _mask_from_yx, _select_points_by_yx_mask, _transform_points_c2w, apply_table_z_filter_with_yx, backproject_masked_rgbd_profiled, build_world_z_diagnostics, erode_binary_mask


class _PcdMixin(_DemoRuntimeContract):
    """MainDataProcessingDemo point-cloud/filter mixin."""

    def _lossless_pcd_worker(self) -> None:
        """Return the lossless PCD worker."""
        rng = np.random.default_rng()
        try:
            while not self.stop_event.is_set():
                mask_packet = self.lossless_pcd_mask_queue.get(stop_event=self.stop_event)
                if mask_packet is None:
                    break
                result = self._build_pcd_packet_from_mask(
                    mask_packet,
                    rng=rng,
                    require_filter_seq=True,
                )
                self._maybe_start_shape_prior_from_pcd_result(result)
                self._lossless_pcd_results += 1
                if not self.same_seq_pairer.wait_for_side_capacity("pcd", stop_event=self.stop_event):
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
        """Create filter input."""
        object_cap = 0 if int(self.args.object_filter_cap) == 0 else int(self.object_filter_budget.cap)
        controller_cap = 0 if int(self.args.controller_filter_cap) == 0 else int(self.controller_filter_budget.cap)
        return FilterInput(
            seq=int(seq),
            object_xyz=np.asarray(object_xyz, dtype=np.float32),
            object_rgb=np.asarray(object_colors, dtype=np.uint8),
            controller_xyz=np.asarray(controller_xyz, dtype=np.float32),
            controller_rgb=np.asarray(controller_colors, dtype=np.uint8),
            object_cap=object_cap,
            controller_cap=controller_cap,
            object_voxel_size_m=float(self.args.object_filter_voxel_m),
            controller_voxel_size_m=float(self.args.controller_filter_voxel_m),
            object_yx=np.asarray(
                object_yx if object_yx is not None else np.empty((0, 2), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1, 2),
            controller_yx=np.asarray(
                controller_yx if controller_yx is not None else np.empty((0, 2), dtype=np.int64),
                dtype=np.int64,
            ).reshape(-1, 2),
        )

    def _apply_single_pcd_filter(
        self,
        *,
        points: np.ndarray,
        colors: np.ndarray,
        yx: np.ndarray | None = None,
        mode: str,
        cap: int,
        voxel_size_m: float,
        keep_components: int,
        min_retain_ratio: float,
        min_raw_retain_ratio: float,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
        """Apply single PCD filter."""
        raw_points = np.asarray(points, dtype=np.float32).reshape(-1, 3)
        raw_colors = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
        raw_yx = (
            np.asarray(yx, dtype=np.int64).reshape(-1, 2)
            if yx is not None
            else np.empty((0, 2), dtype=np.int64)
        )
        if len(raw_yx) not in {0, len(raw_points)}:
            raise ValueError("yx must have the same first dimension as points when provided")

        def select_yx(source_yx: np.ndarray, indices: np.ndarray) -> np.ndarray:
            """Select YX."""
            if len(source_yx) == 0:
                return np.empty((0, 2), dtype=np.int64)
            return np.ascontiguousarray(source_yx[np.asarray(indices, dtype=np.int64)], dtype=np.int64).reshape(-1, 2)

        cap_start_s = time.perf_counter()
        cap_indices = voxel_cap_indices(
            raw_points,
            max_points=int(cap),
            voxel_size_m=float(voxel_size_m),
            rng=rng,
        )
        capped_points = np.ascontiguousarray(raw_points[cap_indices], dtype=np.float32).reshape(-1, 3)
        capped_colors = np.ascontiguousarray(raw_colors[cap_indices], dtype=np.uint8).reshape(-1, 3)
        capped_yx = select_yx(raw_yx, cap_indices)
        cap_ms = _elapsed_ms(cap_start_s, time.perf_counter())
        raw_point_count = int(len(raw_points))
        capped_point_count = int(len(capped_points))

        fallback_to_capped = False
        fallback_reason = ""
        fallback_source = "none"
        cap_raw_retain_ratio = float(capped_point_count / max(1, raw_point_count))
        if (
            mode != PCD_FILTER_NONE
            and float(min_raw_retain_ratio) > 0.0
            and raw_point_count > 0
            and capped_point_count < raw_point_count
            and cap_raw_retain_ratio < float(min_raw_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
            return filtered_points, filtered_colors, filtered_yx, {
                "mode": str(mode),
                "raw_points": raw_point_count,
                "cap_points": capped_point_count,
                "output_points": int(len(filtered_points)),
                "filter_output_points": capped_point_count,
                "filter_retain_ratio": 1.0 if capped_point_count > 0 else 0.0,
                "raw_retain_ratio": cap_raw_retain_ratio,
                "min_retain_ratio": float(min_retain_ratio),
                "min_raw_retain_ratio": float(min_raw_retain_ratio),
                "fallback_to_capped": True,
                "fallback_reason": "skip_filter_low_cap_raw_retain_ratio",
                "fallback_source": "raw",
                "cap": int(cap),
                "voxel_size_m": float(voxel_size_m),
                "keep_components": int(keep_components),
                "cap_ms": float(cap_ms),
                "filter_ms": 0.0,
            }

        filter_start_s = time.perf_counter()
        if mode == PCD_FILTER_NONE:
            filtered_points = np.asarray(capped_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = capped_colors
            filtered_yx = capped_yx
        elif mode == PCD_FILTER_VOXEL_DENSITY:
            density_indices = voxel_density_indices(
                capped_points,
                voxel_size_m=float(voxel_size_m),
                min_points_per_voxel=int(self.args.voxel_density_min_points),
            )
            filtered_points = np.asarray(capped_points[density_indices], dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.asarray(capped_colors[density_indices], dtype=np.uint8).reshape(-1, 3)
            filtered_yx = select_yx(capped_yx, density_indices)
        elif mode == PCD_FILTER_PT_FILTER:
            from demo_v6_2.utils.pcd_postprocess import (
                apply_phystwin_like_radius_postprocess_with_trace,
            )

            filtered_points, filtered_colors, _unused_stats, trace = apply_phystwin_like_radius_postprocess_with_trace(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
            )
            kept_indices = np.flatnonzero(np.asarray(trace["kept_mask"], dtype=bool).reshape(-1))
            filtered_yx = select_yx(capped_yx, kept_indices)
        elif mode == PCD_FILTER_ENHANCED_PT:
            from demo_v6_2.utils.pcd_postprocess import (
                apply_enhanced_phystwin_like_postprocess_with_trace,
            )

            filtered_points, filtered_colors, _unused_stats, trace = apply_enhanced_phystwin_like_postprocess_with_trace(
                points=capped_points,
                colors=capped_colors,
                enabled=True,
                radius_m=float(self.args.filter_radius_m),
                nb_points=int(self.args.filter_nb_points),
                component_voxel_size_m=float(self.args.enhanced_component_voxel_size_m),
                keep_near_main_gap_m=float(self.args.enhanced_keep_near_main_gap_m),
                keep_top_n_components=int(keep_components),
            )
            kept_indices = np.flatnonzero(np.asarray(trace["kept_mask"], dtype=bool).reshape(-1))
            filtered_yx = select_yx(capped_yx, kept_indices)
        else:
            raise ValueError(f"unsupported PCD filter mode: {mode}")

        filter_ms = _elapsed_ms(filter_start_s, time.perf_counter())
        filtered_points = np.ascontiguousarray(filtered_points, dtype=np.float32).reshape(-1, 3)
        filtered_colors = np.ascontiguousarray(filtered_colors, dtype=np.uint8).reshape(-1, 3)
        filtered_yx = np.ascontiguousarray(filtered_yx, dtype=np.int64).reshape(-1, 2)
        filter_output_points = int(len(filtered_points))
        retain_ratio = float(filter_output_points / max(1, capped_point_count))
        raw_retain_ratio = float(filter_output_points / max(1, raw_point_count))
        if filter_output_points == 0 and int(len(capped_points)) > 0:
            if float(min_raw_retain_ratio) > 0.0:
                filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
                filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
                fallback_reason = "empty_filter_output_raw"
                fallback_source = "raw"
            else:
                filtered_points = np.ascontiguousarray(capped_points, dtype=np.float32).reshape(-1, 3)
                filtered_colors = np.ascontiguousarray(capped_colors, dtype=np.uint8).reshape(-1, 3)
                filtered_yx = np.ascontiguousarray(capped_yx, dtype=np.int64).reshape(-1, 2)
                fallback_reason = "empty_filter_output"
                fallback_source = "capped"
            fallback_to_capped = True
        elif (
            float(min_raw_retain_ratio) > 0.0
            and raw_point_count > 0
            and raw_retain_ratio < float(min_raw_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(raw_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(raw_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(raw_yx, dtype=np.int64).reshape(-1, 2)
            fallback_to_capped = True
            fallback_reason = "low_filter_raw_retain_ratio"
            fallback_source = "raw"
        elif (
            float(min_retain_ratio) > 0.0
            and capped_point_count > 0
            and retain_ratio < float(min_retain_ratio)
        ):
            filtered_points = np.ascontiguousarray(capped_points, dtype=np.float32).reshape(-1, 3)
            filtered_colors = np.ascontiguousarray(capped_colors, dtype=np.uint8).reshape(-1, 3)
            filtered_yx = np.ascontiguousarray(capped_yx, dtype=np.int64).reshape(-1, 2)
            fallback_to_capped = True
            fallback_reason = "low_filter_retain_ratio"
            fallback_source = "capped"
        return filtered_points, filtered_colors, filtered_yx, {
            "mode": str(mode),
            "raw_points": raw_point_count,
            "cap_points": capped_point_count,
            "output_points": int(len(filtered_points)),
            "filter_output_points": filter_output_points,
            "filter_retain_ratio": retain_ratio,
            "raw_retain_ratio": raw_retain_ratio,
            "min_retain_ratio": float(min_retain_ratio),
            "min_raw_retain_ratio": float(min_raw_retain_ratio),
            "fallback_to_capped": bool(fallback_to_capped),
            "fallback_reason": fallback_reason,
            "fallback_source": fallback_source,
            "cap": int(cap),
            "voxel_size_m": float(voxel_size_m),
            "keep_components": int(keep_components),
            "cap_ms": float(cap_ms),
            "filter_ms": float(filter_ms),
        }

    def _filter_pcd_input(self, item: FilterInput) -> FilterOutput:
        """Return the filter PCD input."""
        started_s = time.perf_counter()
        object_points, object_colors, object_yx, object_stats = self._apply_single_pcd_filter(
            points=item.object_xyz,
            colors=item.object_rgb,
            yx=item.object_yx,
            mode=str(self.args.object_filter),
            cap=int(item.object_cap),
            voxel_size_m=float(item.object_voxel_size_m),
            keep_components=int(self.args.object_filter_keep_components),
            min_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RETAIN_RATIO),
            min_raw_retain_ratio=float(DEFAULT_OBJECT_FILTER_MIN_RAW_RETAIN_RATIO),
            rng=np.random.default_rng(int(item.seq) * 2 + 17),
        )
        controller_points, controller_colors, controller_yx, controller_stats = self._apply_single_pcd_filter(
            points=item.controller_xyz,
            colors=item.controller_rgb,
            yx=item.controller_yx,
            mode=str(self.args.controller_filter),
            cap=int(item.controller_cap),
            voxel_size_m=float(item.controller_voxel_size_m),
            keep_components=int(self.args.controller_filter_keep_components),
            min_retain_ratio=float(DEFAULT_CONTROLLER_FILTER_MIN_RETAIN_RATIO),
            min_raw_retain_ratio=float(DEFAULT_CONTROLLER_FILTER_MIN_RAW_RETAIN_RATIO),
            rng=np.random.default_rng(int(item.seq) * 2 + 19),
        )
        done_s = time.perf_counter()
        filter_ms = _elapsed_ms(started_s, done_s)
        if float(12.0) > 0:
            self.object_filter_budget.update(float(object_stats["filter_ms"] + object_stats["cap_ms"]))
            self.controller_filter_budget.update(float(controller_stats["filter_ms"] + controller_stats["cap_ms"]))
        return FilterOutput(
            seq=int(item.seq),
            object_xyz=object_points,
            object_rgb=object_colors,
            controller_xyz=controller_points,
            controller_rgb=controller_colors,
            filter_ms=float(filter_ms),
            created_perf_s=float(item.created_perf_s),
            output_perf_s=done_s,
            object_yx=object_yx,
            controller_yx=controller_yx,
            stats={
                "object": object_stats,
                "controller": controller_stats,
                "object_filter": str(self.args.object_filter),
                "controller_filter": str(self.args.controller_filter),
            },
        )

    def _filter_worker_stats(self) -> dict[str, Any]:
        """Return the filter worker stats."""
        worker = self.filter_worker
        if worker is None:
            return {
                "busy": False,
                "submit_fps": self.filter_submit_stats.fps,
                "output_fps": self.filter_output_stats.fps,
                "pending_replace_count": 0,
            }
        stats = worker.stats
        return {
            "busy": bool(stats.get("busy", False)),
            "submit_fps": float(stats.get("submit_fps", self.filter_submit_stats.fps)),
            "output_fps": float(stats.get("output_fps", self.filter_output_stats.fps)),
            "pending_replace_count": int(stats.get("pending_replace_count", 0)) + int(self._filter_submit_skip_count),
        }

    def _filter_output_is_fresh(self, *, packet_seq: int, output: FilterOutput) -> bool:
        """Return the filter output is fresh."""
        age_frames = max(0, int(packet_seq) - int(output.seq))
        return age_frames <= int(self.args.filter_max_age_frames)

    def _filter_telemetry_from_output(
        self,
        *,
        packet_seq: int,
        output: FilterOutput | None,
        using_filtered: bool,
        object_raw_points: int,
        object_cap_points: int,
        controller_raw_points: int,
        controller_cap_points: int,
    ) -> PcdFilterTelemetry:
        """Return the filter telemetry from output."""
        worker_stats = self._filter_worker_stats()
        if output is None:
            return PcdFilterTelemetry(
                enabled=pcd_filter_enabled(self.args),
                mode=str(self.args.pcd_filter_mode if pcd_filter_enabled(self.args) else PCD_FILTER_NONE),
                object_raw_points=int(object_raw_points),
                object_cap_points=int(object_cap_points),
                object_output_points=int(object_cap_points),
                object_prefallback_points=int(object_cap_points),
                object_raw_retain_ratio=1.0 if int(object_raw_points) > 0 else 0.0,
                controller_raw_points=int(controller_raw_points),
                controller_cap_points=int(controller_cap_points),
                controller_output_points=int(controller_cap_points),
                controller_prefallback_points=int(controller_cap_points),
                controller_raw_retain_ratio=1.0 if int(controller_raw_points) > 0 else 0.0,
                object_filter_cap=int(self.object_filter_budget.cap),
                controller_filter_cap=int(self.controller_filter_budget.cap),
                filter_submit_fps=float(worker_stats["submit_fps"]),
                filter_output_fps=float(worker_stats["output_fps"]),
                filter_queue_drop=int(worker_stats["pending_replace_count"]),
                filter_busy=bool(worker_stats["busy"]),
            )

        object_stats = dict(output.stats.get("object", {}))
        controller_stats = dict(output.stats.get("controller", {}))
        age_ms = max(0.0, _elapsed_ms(output.output_perf_s, time.perf_counter()))
        return PcdFilterTelemetry(
            enabled=pcd_filter_enabled(self.args),
            mode=str(self.args.pcd_filter_mode),
            render_using_filtered=bool(using_filtered),
            filter_seq=int(output.seq),
            filter_age_frames=max(0, int(packet_seq) - int(output.seq)),
            filter_age_ms=float(age_ms),
            filter_ms=float(output.filter_ms),
            object_filter_ms=float(object_stats.get("filter_ms", 0.0)),
            controller_filter_ms=float(controller_stats.get("filter_ms", 0.0)),
            object_raw_points=int(object_stats.get("raw_points", object_raw_points)),
            object_cap_points=int(object_stats.get("cap_points", object_cap_points)),
            object_output_points=int(object_stats.get("output_points", object_cap_points)),
            object_prefallback_points=int(object_stats.get("filter_output_points", object_cap_points)),
            object_raw_retain_ratio=float(object_stats.get("raw_retain_ratio", 0.0)),
            object_fallback_reason=str(object_stats.get("fallback_reason", "")),
            controller_raw_points=int(controller_stats.get("raw_points", controller_raw_points)),
            controller_cap_points=int(controller_stats.get("cap_points", controller_cap_points)),
            controller_output_points=int(controller_stats.get("output_points", controller_cap_points)),
            controller_prefallback_points=int(controller_stats.get("filter_output_points", controller_cap_points)),
            controller_raw_retain_ratio=float(controller_stats.get("raw_retain_ratio", 0.0)),
            controller_fallback_reason=str(controller_stats.get("fallback_reason", "")),
            object_filter_cap=int(object_stats.get("cap", self.object_filter_budget.cap)),
            controller_filter_cap=int(controller_stats.get("cap", self.controller_filter_budget.cap)),
            filter_submit_fps=float(worker_stats["submit_fps"]),
            filter_output_fps=float(worker_stats["output_fps"]),
            filter_queue_drop=int(worker_stats["pending_replace_count"]),
            filter_busy=bool(worker_stats["busy"]),
        )

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
        if self._formal_timeline_gated_frames and not self._formal_timeline_metadata_written:
            # First formal frame after the shape-prior wait: record the seam so
            # downstream tools can tell warmup frame 0 from output frame 1.
            self.headless_capture_writer.update_metadata(
                {
                    "formal_timeline_gated_frame_count": int(self._formal_timeline_gated_frames),
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
            controller_pcd_mask_erode_pixels=int(result.controller_pcd_mask_erode_pixels),
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
        *,
        rng: np.random.Generator,
        require_filter_seq: bool = False,
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
                mask_packet.depth_u16.astype(np.float32) * np.float32(mask_packet.depth_scale_m_per_unit)
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
            controller_mask = erode_binary_mask(controller_mask, erode_pixels=controller_erode_pixels)
        if object_erode_pixels > 0:
            object_mask = erode_binary_mask(object_mask, erode_pixels=object_erode_pixels)
        empty_pcd_timing = {
            "pcd_mask_intersection_ms": 0.0,
            "pcd_select_ms": 0.0,
            "pcd_point_cap_ms": 0.0,
            "pcd_backproject_ms": 0.0,
            "pcd_color_gather_ms": 0.0,
            "pcd_raw_points": 0.0,
            "pcd_cap_points": 0.0,
        }
        if controller_tracking_enabled(self.args):
            controller_xyz, controller_colors, controller_yx, controller_pcd_timing = backproject_masked_rgbd_profiled(
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
                rng=rng,
                return_yx=True,
            )
            if stride > 1:
                controller_yx = np.ascontiguousarray(controller_yx * int(stride), dtype=np.int64)
        else:
            controller_xyz = np.empty((0, 3), dtype=np.float32)
            controller_colors = np.empty((0, 3), dtype=np.uint8)
            controller_yx = np.empty((0, 2), dtype=np.int64)
            controller_pcd_timing = dict(empty_pcd_timing)
        if object_tracking_enabled(self.args):
            object_xyz, object_colors, object_yx, object_pcd_timing = backproject_masked_rgbd_profiled(
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
                rng=rng,
                return_yx=True,
            )
            if stride > 1:
                object_yx = np.ascontiguousarray(object_yx * int(stride), dtype=np.int64)
        else:
            object_xyz = np.empty((0, 3), dtype=np.float32)
            object_colors = np.empty((0, 3), dtype=np.uint8)
            object_yx = np.empty((0, 2), dtype=np.int64)
            object_pcd_timing = dict(empty_pcd_timing)
        controller_raw_points = int(controller_pcd_timing.get("pcd_raw_points", len(controller_xyz)))
        controller_cap_points = int(controller_pcd_timing.get("pcd_cap_points", len(controller_xyz)))
        object_raw_points = int(object_pcd_timing.get("pcd_raw_points", len(object_xyz)))
        object_cap_points = int(object_pcd_timing.get("pcd_cap_points", len(object_xyz)))
        render_controller_xyz = controller_xyz
        render_controller_colors = controller_colors
        render_controller_yx = controller_yx
        render_object_xyz = object_xyz
        render_object_colors = object_colors
        render_object_yx = object_yx
        filter_output: FilterOutput | None = None
        using_filtered = False

        if pcd_filter_enabled(self.args):
            if str(self.args.pcd_filter_mode) == "sync":
                filter_input = self._make_filter_input(
                    seq=mask_packet.seq,
                    object_xyz=object_xyz,
                    object_colors=object_colors,
                    object_yx=object_yx,
                    controller_xyz=controller_xyz,
                    controller_colors=controller_colors,
                    controller_yx=controller_yx,
                )
                self.filter_submit_stats.record()
                filter_output = self._filter_pcd_input(filter_input)
                self.filter_output_stats.record(filter_output.output_perf_s)
                render_controller_xyz = filter_output.controller_xyz
                render_controller_colors = filter_output.controller_rgb
                render_controller_yx = filter_output.controller_yx
                render_object_xyz = filter_output.object_xyz
                render_object_colors = filter_output.object_rgb
                # Keep XYZ/colors/YX aligned; downstream observation masks are
                # rebuilt from render_object_yx.
                render_object_yx = filter_output.object_yx
                using_filtered = True
            elif str(self.args.pcd_filter_mode) == "async":
                worker = self.filter_worker
                if worker is not None:
                    latest = worker.latest_output()
                    if latest is not None:
                        filter_output = latest
                        if int(latest.seq) != self._last_filter_output_seq_recorded:
                            self.filter_output_stats.record(latest.output_perf_s)
                            self._last_filter_output_seq_recorded = int(latest.seq)
                        filter_matches = int(latest.seq) == int(mask_packet.seq)
                        if filter_matches or (
                            not bool(require_filter_seq)
                            and self._filter_output_is_fresh(packet_seq=mask_packet.seq, output=latest)
                        ):
                            render_controller_xyz = latest.controller_xyz
                            render_controller_colors = latest.controller_rgb
                            render_controller_yx = latest.controller_yx
                            render_object_xyz = latest.object_xyz
                            render_object_colors = latest.object_rgb
                            # Keep XYZ/colors/YX aligned; downstream observation
                            # masks are rebuilt from render_object_yx.
                            render_object_yx = latest.object_yx
                            using_filtered = True
                    if mask_packet.seq % int(self.args.filter_every_n) == 0:
                        if not worker.is_busy():
                            worker.submit_latest(
                                self._make_filter_input(
                                    seq=mask_packet.seq,
                                    object_xyz=object_xyz,
                                    object_colors=object_colors,
                                    object_yx=object_yx,
                                    controller_xyz=controller_xyz,
                                    controller_colors=controller_colors,
                                    controller_yx=controller_yx,
                                )
                            )
                            self.filter_submit_stats.record()
                        else:
                            self._filter_submit_skip_count += 1
            elif str(self.args.pcd_filter_mode) != "none":
                raise ValueError(f"unsupported --pcd-filter-mode {self.args.pcd_filter_mode!r}")

        filter_telemetry = self._filter_telemetry_from_output(
            packet_seq=mask_packet.seq,
            output=filter_output,
            using_filtered=using_filtered,
            object_raw_points=object_raw_points,
            object_cap_points=object_cap_points,
            controller_raw_points=controller_raw_points,
            controller_cap_points=controller_cap_points,
        )
        render_controller_xyz = _transform_points_c2w(render_controller_xyz, self.table_c2w)
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
            pcd_select_ms=float(controller_pcd_timing["pcd_select_ms"] + object_pcd_timing["pcd_select_ms"]),
            pcd_point_cap_ms=float(
                controller_pcd_timing["pcd_point_cap_ms"] + object_pcd_timing["pcd_point_cap_ms"]
            ),
            pcd_backproject_ms=float(
                controller_pcd_timing["pcd_backproject_ms"] + object_pcd_timing["pcd_backproject_ms"]
            ),
            pcd_color_gather_ms=float(
                controller_pcd_timing["pcd_color_gather_ms"] + object_pcd_timing["pcd_color_gather_ms"]
            ),
            pcd_filter_ms=float(filter_telemetry.filter_ms),
            object_filter_ms=float(filter_telemetry.object_filter_ms),
            controller_filter_ms=float(filter_telemetry.controller_filter_ms),
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
            filter_telemetry=filter_telemetry,
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

    def _pcd_worker(self) -> None:
        """Return the PCD worker."""
        last_seq = -1
        rng = np.random.default_rng()
        while not self.stop_event.is_set():
            mask_packet = self.mask_slot.get_latest_after(last_seq)
            if mask_packet is None:
                time.sleep(0.001)
                continue
            last_seq = mask_packet.seq
            try:
                result = self._build_pcd_packet_from_mask(mask_packet, rng=rng)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] PCD frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            self._maybe_start_shape_prior_from_pcd_result(result)
            result = replace(result, packet=self._packet_with_shape_prior_state(result.packet))
            self.pcd_slot.put(result.packet)
            self._maybe_write_shape_prior_headless_result()
            self._write_headless_pcd_result(result)
            self.pcd_stats.record(result.packet.process_done_perf_s)


__all__ = ["_PcdMixin"]
