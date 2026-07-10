"""Tests for Demo v6.2 strict live startup ordering."""

from __future__ import annotations

from types import SimpleNamespace
import threading
import unittest
from unittest import mock

import numpy as np

from demo_v6_2 import mdp_demo_capture, mdp_demo_lifecycle


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

        def wait_for_first_pair() -> bool:
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
        demo._wait_for_lossless_startup_pair.assert_called_once_with()
        demo._record_fatal_worker_error.assert_not_called()


if __name__ == "__main__":
    unittest.main()
