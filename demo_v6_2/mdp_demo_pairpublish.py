"""MainDataProcessingDemo pairing/publish/depth mixin."""
from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import lossless_enabled
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_packets import DepthProfilePacket
from demo_v6_2.mdp_pipeline_plumbing import LosslessPipelineError


class _PairPublishMixin(_DemoRuntimeContract):
    """MainDataProcessingDemo pairing/publish/depth mixin."""

    def _publish_strict_pair(self, pair: PairedBuildResult) -> None:
        """Publish one validated same-sequence PCD and tracker result."""
        pcd_result = pair.pcd_result
        tracker_packet = pair.tracker_packet
        self._maybe_start_shape_prior_from_pcd_result(pcd_result, from_strict_pair=True)
        pcd_result = replace(
            pcd_result,
            packet=self._packet_with_shape_prior_state(pcd_result.packet),
        )
        self.pcd_stats.record(pcd_result.packet.process_done_perf_s)
        self.tracker_stats.record(tracker_packet.process_done_perf_s)
        self._lossless_pairs_emitted += 1
        if pair.seq == 0:
            self._lossless_first_pair_published.set()
        if self.headless_capture_writer is not None:
            self._maybe_write_shape_prior_headless_result()
            # One gate decision per frame: the row and its query_trajectory
            # sidecar must agree even if the prior flips ready mid-frame.
            rows_gated = self._headless_product_rows_gated()
            if not rows_gated:
                self.headless_capture_writer.write_tracker(tracker_packet)
            self._write_headless_pcd_result(
                pcd_result, tracker_packet=tracker_packet, gated=rows_gated
            )

    def _publish_pairer_outputs(self, pairs: list[PairedBuildResult]) -> None:
        """Publish pairer outputs."""
        for pair in pairs:
            self.lossless_pair_output_queue.put(pair)

    def _publish_ordered_lossless_pair(self, pair: PairedBuildResult) -> None:
        """Publish ordered lossless pair."""
        seq = int(pair.seq)
        with self._lossless_publish_condition:
            while seq != self._lossless_next_publish_seq:
                if seq < self._lossless_next_publish_seq:
                    raise LosslessPipelineError(
                        f"lossless publish received stale seq {seq}, expected {self._lossless_next_publish_seq}"
                    )
                if self.stop_event.is_set():
                    return
                self._lossless_publish_condition.wait(timeout=0.05)
        self._publish_strict_pair(pair)
        with self._lossless_publish_condition:
            expected = self._lossless_next_publish_seq
            if seq != expected:
                raise LosslessPipelineError(f"lossless publish expected seq {expected}, got {seq}")
            self._lossless_next_publish_seq += 1
            self._lossless_publish_condition.notify_all()

    def _maybe_finish_lossless_processing(self) -> None:
        """Maybe start or update finish lossless processing."""
        if not lossless_enabled(self.args):
            return
        if self.same_seq_pairer.done and not self._lossless_processing_done.is_set():
            self.lossless_pair_output_queue.close()

    def _finish_lossless_output(self) -> None:
        """Finish lossless output."""
        if not lossless_enabled(self.args):
            return
        if not self._lossless_processing_done.is_set():
            self._lossless_processing_done.set()

    def _lossless_pair_output_worker(self) -> None:
        """Return the lossless pair output worker."""
        try:
            while not self.stop_event.is_set():
                pair = self.lossless_pair_output_queue.get(stop_event=self.stop_event)
                if pair is None:
                    break
                self._publish_ordered_lossless_pair(pair)
            self._finish_lossless_output()
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("lossless pair output worker", exc)

    def _strict_paired_worker(self) -> None:
        """Return the strict paired worker."""
        try:
            adapter = self._build_tracker_adapter()
            print(
                "[tapnextpp-tracker] "
                f"backend={adapter.name} device={self.args.tracker_device} "
                f"repo={DEFAULT_TAPNET_REPO_DIR} checkpoint={DEFAULT_TAPNEXTPP_CHECKPOINT} "
                f"image_size={"256,256"} overlay_max={int(self.args.tracker_overlay_max_points)} "
                "strict_sync=1",
                flush=True,
            )
            last_seq = -1
            rng = np.random.default_rng()
            while not self.stop_event.is_set():
                mask_packet = self.mask_slot.get_latest_after(last_seq)
                if mask_packet is None:
                    time.sleep(0.001)
                    continue
                last_seq = mask_packet.seq
                try:
                    pcd_result = self._build_pcd_packet_from_mask(
                        mask_packet,
                        rng=rng,
                        require_filter_seq=True,
                    )
                except Exception as exc:
                    if not self.stop_event.is_set():
                        print(f"[WARN] strict PCD frame {mask_packet.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                    continue
                self._maybe_start_shape_prior_from_pcd_result(pcd_result)
                tracker_packet = self._build_tracker_marker_packet(mask_packet, adapter)
                if tracker_packet is None:
                    continue
                self._publish_strict_pair(
                    PairedBuildResult(
                        seq=int(mask_packet.seq),
                        pcd_result=pcd_result,
                        tracker_packet=tracker_packet,
                    )
                )
        except Exception as exc:
            if not self.stop_event.is_set():
                self._record_fatal_worker_error("strict same-seq tracker/PCD", exc)

    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish mask packet."""
        self.mask_slot.put(packet)
        if lossless_enabled(self.args):
            if not self.lossless_pcd_mask_queue.wait_for_capacity(stop_event=self.stop_event):
                return
            if not self.lossless_tracker_mask_queue.wait_for_capacity(stop_event=self.stop_event):
                return
            self.lossless_pcd_mask_queue.put(packet)
            self.lossless_tracker_mask_queue.put(packet)
            self._lossless_segmented_frames += 1

    def _depth_profile_worker(self) -> None:
        """Return the depth profile worker."""
        last_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_seq)
            if frame is None:
                time.sleep(0.001)
                continue
            last_seq = frame.seq
            if frame.depth_source != "ffs":
                continue
            try:
                (
                    _depth_m,
                    ffs_ms,
                    ffs_align_ms,
                    remote_rtt_ms,
                    remote_server_total_ms,
                    remote_request_kb,
                    remote_response_kb,
                ) = self._compute_external_ffs_depth_color_m(frame)
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(f"[WARN] FFS depth profile frame {frame.seq} failed: {type(exc).__name__}: {exc}", flush=True)
                continue
            done_s = time.perf_counter()
            packet = DepthProfilePacket(
                seq=frame.seq,
                receive_perf_s=frame.receive_perf_s,
                process_done_perf_s=done_s,
                dropped_capture_frames=self.capture_slot.dropped_count,
                timing=replace(
                    frame.timing,
                    ffs_ms=ffs_ms,
                    ffs_align_ms=ffs_align_ms,
                    remote_rtt_ms=remote_rtt_ms,
                    remote_server_total_ms=remote_server_total_ms,
                    remote_request_kb=remote_request_kb,
                    remote_response_kb=remote_response_kb,
                ),
            )
            self.depth_profile_slot.put(packet)
            self.depth_stats.record(done_s)

    def _compute_external_ffs_depth_color_m(
        self,
        packet: MaskPacket | FramePacket,
    ) -> tuple[np.ndarray, float, float, float, float, float, float]:
        """Compute external FFS depth color m."""
        depth_color_m, ffs_ms, ffs_align_ms = self._compute_ffs_depth_color_m(packet)
        return depth_color_m, ffs_ms, ffs_align_ms, 0.0, 0.0, 0.0, 0.0

    def _get_cached_local_ffs_depth(self, seq: int) -> tuple[np.ndarray, float, float] | None:
        """Return the get cached local FFS depth."""
        cached = self._local_ffs_depth_cache.get(int(seq))
        if cached is None:
            return None
        self._local_ffs_depth_cache.move_to_end(int(seq))
        return cached

    def _put_cached_local_ffs_depth(self, seq: int, value: tuple[np.ndarray, float, float]) -> None:
        """Return the put cached local FFS depth."""
        self._local_ffs_depth_cache[int(seq)] = value
        self._local_ffs_depth_cache.move_to_end(int(seq))
        while len(self._local_ffs_depth_cache) > DEFAULT_LOCAL_FFS_DEPTH_CACHE_FRAMES:
            self._local_ffs_depth_cache.popitem(last=False)

    def _compute_ffs_depth_color_m(self, packet: MaskPacket | FramePacket) -> tuple[np.ndarray, float, float]:
        """Compute FFS depth color m."""
        runner = self.ffs_runner
        if runner is None:
            raise RuntimeError("FFS runner is not initialized")
        if (
            packet.ir_left_u8 is None
            or packet.ir_right_u8 is None
            or packet.k_ir_left is None
            or packet.t_ir_left_to_color is None
            or packet.k_color is None
            or packet.ir_baseline_m <= 0
        ):
            raise RuntimeError("FFS packet is missing IR stereo calibration/data")

        with self._local_ffs_lock:
            cached = self._get_cached_local_ffs_depth(int(packet.seq))
            if cached is not None:
                return cached

            ffs_start_s = time.perf_counter()
            output = runner.run_pair(
                packet.ir_left_u8,
                packet.ir_right_u8,
                K_ir_left=packet.k_ir_left,
                baseline_m=float(packet.ir_baseline_m),
            )
            ffs_done_s = time.perf_counter()
            depth_ir_left_m = np.asarray(output["depth_ir_left_m"], dtype=np.float32)
            k_ir_left_used = np.asarray(output.get("K_ir_left_used", packet.k_ir_left), dtype=np.float32)
            align_start_s = time.perf_counter()
            aligner = self._get_ir_to_color_aligner(
                depth_shape=depth_ir_left_m.shape,
                color_shape=packet.color_bgr.shape[:2],
                k_ir_left=k_ir_left_used,
                t_ir_left_to_color=packet.t_ir_left_to_color,
                k_color=packet.k_color,
            )
            depth_color_m = np.ascontiguousarray(aligner.align(depth_ir_left_m), dtype=np.float32)
            align_done_s = time.perf_counter()
            result = (
                depth_color_m,
                _elapsed_ms(ffs_start_s, ffs_done_s),
                _elapsed_ms(align_start_s, align_done_s),
            )
            self._put_cached_local_ffs_depth(int(packet.seq), result)
            return result


__all__ = ["_PairPublishMixin"]
