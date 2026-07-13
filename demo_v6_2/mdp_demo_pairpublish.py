"""MainDataProcessingDemo pairing/publish/depth mixin."""

from __future__ import annotations

from demo_v6_2.mdp_constants import *  # noqa: F401,F403
from demo_v6_2.mdp_cli import lossless_enabled
from demo_v6_2.mdp_demo_contract import _DemoRuntimeContract
from demo_v6_2.mdp_packets import DepthProfilePacket


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
        if pair.seq == 0:
            self.lossless.first_pair_published.set()
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

    def _publish_ordered_lossless_pair(self, pair: PairedBuildResult) -> None:
        """Publish ordered lossless pair."""
        seq = int(pair.seq)
        if not self.lossless.wait_publish_turn(seq, stop_event=self.stop_event):
            return
        self._publish_strict_pair(pair)
        self.lossless.finish_publish_turn(seq)

    def _lossless_pair_output_worker(self) -> None:
        """Return the lossless pair output worker."""
        try:
            while not self.stop_event.is_set():
                pair = self.lossless.pair_output_queue.get(stop_event=self.stop_event)
                if pair is None:
                    break
                self._publish_ordered_lossless_pair(pair)
            self.lossless.finish_output()
        except Exception as exc:
            if not self.stop_event.is_set():
                self.fatal.record("lossless pair output worker", exc)

    def _publish_mask_packet(self, packet: MaskPacket) -> None:
        """Publish raw masks to diagnostics and the canonical formal stage."""
        self.mask_slot.put(packet)
        if lossless_enabled(self.args):
            self.lossless.publish_mask(packet, stop_event=self.stop_event)

    def _depth_profile_worker(self) -> None:
        """Return the depth profile worker."""
        last_seq = -1
        while not self.stop_event.is_set():
            frame = self.capture_slot.get_latest_after(last_seq)
            if frame is None:
                time.sleep(0.001)
                continue
            last_seq = frame.seq
            if frame.depth_source != "ffs" or self.depth_engine is None:
                continue
            try:
                _depth_m, ffs_ms, ffs_align_ms = self.depth_engine.compute_color_depth(
                    frame
                )
            except Exception as exc:
                if not self.stop_event.is_set():
                    print(
                        f"[WARN] FFS depth profile frame {frame.seq} failed: "
                        f"{type(exc).__name__}: {exc}",
                        flush=True,
                    )
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
                ),
            )
            self.depth_profile_slot.put(packet)
            self.depth_stats.record(done_s)


__all__ = ["_PairPublishMixin"]
