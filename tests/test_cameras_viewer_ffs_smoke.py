from __future__ import annotations

from collections import deque
import queue
import unittest

import numpy as np

from cameras_viewer_FFS import (
    _compute_measured_fps,
    _format_panel_label_lines,
    _put_latest,
    _reproject_ffs_depth_to_color,
    _update_recent_frame_times,
)


class CamerasViewerFfsSmokeTest(unittest.TestCase):
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
            capture_fps=15.2,
            capture_sample_count=5,
            ffs_fps=3.1,
            ffs_sample_count=5,
        )
        self.assertEqual(line1, "239222300433 usb=3.2 848x480@30.0fps")
        self.assertEqual(line2, "capture: 15.2 | ffs: 3.1")

    def test_label_lines_show_warming_before_enough_samples(self) -> None:
        _, line2 = _format_panel_label_lines(
            serial="239222300433",
            usb_desc="3.2",
            stream_w=848,
            stream_h=480,
            configured_fps=30.0,
            capture_fps=0.0,
            capture_sample_count=1,
            ffs_fps=0.0,
            ffs_sample_count=1,
        )
        self.assertEqual(line2, "capture: warming | ffs: warming")

    def test_put_latest_replaces_pending_item(self) -> None:
        q: queue.Queue[object] = queue.Queue(maxsize=1)
        _put_latest(q, {"frame": 1})
        _put_latest(q, {"frame": 2})
        self.assertEqual(q.qsize(), 1)
        self.assertEqual(q.get_nowait(), {"frame": 2})

    def test_reproject_ffs_depth_to_color_keeps_identity_mapping(self) -> None:
        depth_ir_left_m = np.array([[0.0, 0.0, 0.0], [0.0, 1.25, 0.0], [0.0, 0.0, 0.0]], dtype=np.float32)
        K = np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        depth_color = _reproject_ffs_depth_to_color(
            depth_ir_left_m,
            K_ir_left=K,
            T_ir_left_to_color=T,
            K_color=K,
            output_shape=(3, 3),
        )
        self.assertEqual(depth_color.shape, (3, 3))
        self.assertAlmostEqual(float(depth_color[1, 1]), 1.25, places=6)
        self.assertEqual(float(depth_color[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
