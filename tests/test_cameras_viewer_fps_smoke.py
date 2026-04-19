from __future__ import annotations

from collections import deque
import unittest

from cameras_viewer import (
    _compute_measured_fps,
    _format_panel_label_lines,
    _update_recent_frame_times,
)


class CamerasViewerFpsSmokeTest(unittest.TestCase):
    def test_measured_fps_is_zero_for_empty_history(self) -> None:
        self.assertEqual(_compute_measured_fps(deque()), 0.0)

    def test_measured_fps_is_zero_for_single_sample(self) -> None:
        self.assertEqual(_compute_measured_fps(deque([1.0])), 0.0)

    def test_measured_fps_matches_stable_30hz_timestamps(self) -> None:
        frame_times = deque([0.0, 1.0 / 30.0, 2.0 / 30.0, 3.0 / 30.0])
        self.assertAlmostEqual(_compute_measured_fps(frame_times), 30.0, places=4)

    def test_update_recent_frame_times_prunes_stale_samples(self) -> None:
        frame_times = deque([0.0, 0.2, 0.6])
        _update_recent_frame_times(frame_times, now_s=1.3, frame_received=False, window_s=1.0)
        self.assertEqual(list(frame_times), [0.6])

    def test_label_lines_are_deterministic(self) -> None:
        line1, line2 = _format_panel_label_lines(
            serial="239222300433",
            usb_desc="3.2",
            stream_w=848,
            stream_h=480,
            configured_fps=30.0,
            measured_fps=28.75,
            measured_sample_count=5,
        )
        self.assertEqual(line1, "239222300433 usb=3.2 848x480@30.0fps")
        self.assertEqual(line2, "configured: 30.0 | measured: 28.8")

    def test_label_lines_show_warming_before_enough_samples(self) -> None:
        _, line2 = _format_panel_label_lines(
            serial="239222300433",
            usb_desc="3.2",
            stream_w=848,
            stream_h=480,
            configured_fps=15.0,
            measured_fps=0.0,
            measured_sample_count=1,
        )
        self.assertEqual(line2, "configured: 15.0 | measured: warming")


if __name__ == "__main__":
    unittest.main()
