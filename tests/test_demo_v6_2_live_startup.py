"""Tests for Demo v6.2 strict live startup ordering."""

from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from demo_v6_2 import mdp_demo_capture, mdp_demo_lifecycle
from demo_v6_2.mdp_packets import FramePacket, PipelineTiming


class _ColorFrame:
    def get_data(self) -> np.ndarray:
        return np.zeros((2, 2, 3), dtype=np.uint8)


class _FrameSet:
    def get_color_frame(self) -> _ColorFrame:
        return _ColorFrame()


class LiveStartupTests(unittest.TestCase):
    def test_live_lossless_startup_waits_for_first_pair(self) -> None:
        first_pair = mock.Mock()
        first_pair.wait.return_value = True
        demo = SimpleNamespace(
            args=SimpleNamespace(track_mode="controller-object"),
            stop_event=threading.Event(),
            _lossless_enabled=mock.Mock(return_value=True),
            _lossless_first_pair_published=first_pair,
        )

        completed = (
            mdp_demo_lifecycle._LifecycleMixin._wait_for_lossless_startup_pair(
                demo
            )
        )

        self.assertTrue(completed)
        first_pair.wait.assert_called_once_with(timeout=0.01)

    def test_live_sampler_starts_after_first_pair(self) -> None:
        stop_event = threading.Event()
        order: list[str] = []
        sampler = mock.Mock()
        sampler.pop_due.return_value = None

        def wait_for_first_pair(*, on_wait_tick=None) -> bool:
            order.append("first_pair")
            return True

        def start_sampler(*, first_publish_s: float) -> None:
            self.assertGreater(first_publish_s, 0.0)
            order.append("sampler_start")
            stop_event.set()

        sampler.start.side_effect = start_sampler
        runtime = SimpleNamespace(
            pipeline=SimpleNamespace(wait_for_frames=mock.Mock(return_value=_FrameSet())),
            align=None,
            intrinsics=mock.sentinel.intrinsics,
            depth_scale_m_per_unit=0.001,
            k_ir_left=None,
            t_ir_left_to_color=None,
            k_color=np.eye(3, dtype=np.float32),
            ir_baseline_m=0.0,
        )
        demo = SimpleNamespace(
            runtime=runtime,
            args=SimpleNamespace(
                input_source="live",
                depth_source="none",
                track_mode="controller-object",
            ),
            stop_event=stop_event,
            _lossless_enabled=mock.Mock(return_value=True),
            _lossless_input_fps=mock.Mock(return_value=5.0),
            _publish_capture_packet=mock.Mock(),
            _first_frame_segmented=SimpleNamespace(
                wait=mock.Mock(return_value=True)
            ),
            _wait_for_lossless_startup_pair=mock.Mock(
                side_effect=wait_for_first_pair
            ),
            _lossless_capture_done=mock.Mock(),
            lossless_frame_queue=mock.Mock(),
            _record_fatal_worker_error=mock.Mock(),
        )

        with mock.patch.object(
            mdp_demo_capture,
            "LiveLatestFrameSampler",
            return_value=sampler,
        ):
            mdp_demo_capture._CaptureMixin._capture_worker(demo)

        self.assertEqual(order, ["first_pair", "sampler_start"])
        demo._publish_capture_packet.assert_called_once()
        # The startup-pair wait now carries a display-only warm-up preview
        # pump so the RGB preview stays live while the capture loop is blocked.
        demo._wait_for_lossless_startup_pair.assert_called_once_with(
            on_wait_tick=mock.ANY
        )
        demo._record_fatal_worker_error.assert_not_called()

    def test_warmup_preview_pump_is_display_only(self) -> None:
        # While blocked on the frame-0 segmentation barrier the capture loop
        # keeps the RGB preview live: each tick grabs a color frame and pushes
        # a display-only packet to input_preview_slot with a monotonic seq,
        # never routing it into the pipeline (capture_slot / lossless queue).
        stop_event = threading.Event()
        preview_puts: list = []
        input_preview_slot = mock.Mock()
        input_preview_slot.put.side_effect = lambda packet: preview_puts.append(packet)
        capture_slot = mock.Mock()
        sampler = mock.Mock()
        sampler.pop_due.return_value = None
        sampler.start.side_effect = lambda *, first_publish_s: stop_event.set()

        runtime = SimpleNamespace(
            pipeline=SimpleNamespace(
                wait_for_frames=mock.Mock(return_value=_FrameSet())
            ),
            align=None,
            intrinsics=mock.sentinel.intrinsics,
            depth_scale_m_per_unit=0.001,
            k_ir_left=None,
            t_ir_left_to_color=None,
            k_color=np.eye(3, dtype=np.float32),
            ir_baseline_m=0.0,
        )
        demo = SimpleNamespace(
            runtime=runtime,
            args=SimpleNamespace(
                input_source="live",
                depth_source="none",
                track_mode="controller-object",
            ),
            stop_event=stop_event,
            input_preview_slot=input_preview_slot,
            capture_slot=capture_slot,
            _input_preview_publish_seq=0,
            _lossless_enabled=mock.Mock(return_value=True),
            # Huge fps -> tiny pacing period so every tick publishes a preview.
            _lossless_input_fps=mock.Mock(return_value=1_000_000.0),
            _publish_capture_packet=mock.Mock(),
            _first_frame_segmented=SimpleNamespace(
                wait=mock.Mock(side_effect=[False, False, True])
            ),
            _wait_for_lossless_startup_pair=mock.Mock(return_value=True),
            _lossless_capture_done=mock.Mock(),
            lossless_frame_queue=mock.Mock(),
            _record_fatal_worker_error=mock.Mock(),
        )
        # Bind the real reseq helper so the pump exercises the shared seq path.
        demo._put_preview_slot_frame = (
            mdp_demo_capture._CaptureMixin._put_preview_slot_frame.__get__(demo)
        )

        with mock.patch.object(
            mdp_demo_capture,
            "LiveLatestFrameSampler",
            return_value=sampler,
        ):
            mdp_demo_capture._CaptureMixin._capture_worker(demo)

        # Two barrier ticks -> two display-only preview frames, monotonic seq>0.
        self.assertEqual([1, 2], [packet.seq for packet in preview_puts])
        self.assertTrue(
            all(
                packet.depth_u16 is None and packet.ir_left_u8 is None
                for packet in preview_puts
            )
        )
        # Only frame 0 goes through the real publish path; the pump never
        # touches capture_slot or the lossless queue.
        demo._publish_capture_packet.assert_called_once()
        demo.capture_slot.put.assert_not_called()
        demo._record_fatal_worker_error.assert_not_called()

    def test_preview_slot_seq_monotonic_across_producers(self) -> None:
        # Regression: the pump must not lock resumed live output out of the
        # preview. A real LatestSlot consumer (WarmupRgbPreview reads via
        # get_latest_after) must accept frame 0, every pump frame, AND the
        # resumed output frames whose output_seq restarts at 1 — all routed
        # through the shared _put_preview_slot_frame monotonic seq.
        from demo_v6_2.utils.concurrency import LatestSlot

        demo = SimpleNamespace(
            input_preview_slot=LatestSlot(),
            _input_preview_publish_seq=0,
        )
        put_preview = mdp_demo_capture._CaptureMixin._put_preview_slot_frame.__get__(
            demo
        )

        def frame(seq: int) -> FramePacket:
            return FramePacket(
                seq=seq,
                color_bgr=np.zeros((2, 2, 3), dtype=np.uint8),
                depth_source="none",
                intrinsics=mock.sentinel.intrinsics,
                depth_scale_m_per_unit=0.001,
                receive_perf_s=0.0,
                timing=PipelineTiming(),
            )

        # frame 0 (output_seq 0), 3 pump frames, then resumed output_seq 1,2,3.
        # A WarmupRgbPreview-style consumer tracks last_seq from -1 and must
        # never be starved (get_latest_after returns the fresh frame each time).
        last_seq = -1
        for original_seq in (0, 0, 0, 0, 1, 2, 3):
            put_preview(frame(original_seq))
            latest = demo.input_preview_slot.get_latest_after(last_seq)
            self.assertIsNotNone(
                latest, "consumer must accept every published preview frame"
            )
            last_seq = int(latest.seq)
        self.assertEqual(7, demo._input_preview_publish_seq)
        self.assertEqual(7, last_seq)


if __name__ == "__main__":
    unittest.main()
