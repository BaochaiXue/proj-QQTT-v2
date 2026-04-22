from __future__ import annotations

from collections import deque
import unittest
from unittest import mock

import cameras_viewer
from cameras_viewer import (
    _build_camera_state,
    _compute_display_target_size,
    _compute_measured_fps,
    _format_depth_debug_lines,
    _format_panel_label_lines,
    _get_screen_size,
    _render_panel,
    _update_recent_frame_times,
)


class CamerasViewerFpsSmokeTest(unittest.TestCase):
    def setUp(self) -> None:
        cameras_viewer._SCREEN_SIZE_CACHE = None

    def tearDown(self) -> None:
        cameras_viewer._SCREEN_SIZE_CACHE = None

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

    def test_depth_debug_lines_are_deterministic(self) -> None:
        line1, line2 = _format_depth_debug_lines(
            measured_fps=21.54,
            measured_sample_count=5,
        )
        self.assertEqual(line1, "Depth render disabled")
        self.assertEqual(line2, "depth fps: 21.5")

    def test_depth_debug_lines_show_warming_before_enough_samples(self) -> None:
        _, line2 = _format_depth_debug_lines(
            measured_fps=0.0,
            measured_sample_count=1,
        )
        self.assertEqual(line2, "depth fps: warming")

    def test_screen_size_probe_is_cached(self) -> None:
        with mock.patch("cameras_viewer._query_screen_size", return_value=(1600, 900)) as probe:
            self.assertEqual(_get_screen_size(), (1600, 900))
            self.assertEqual(_get_screen_size(), (1600, 900))
        probe.assert_called_once_with()

    def test_compute_display_target_size_keeps_grid_when_it_fits(self) -> None:
        self.assertEqual(
            _compute_display_target_size(
                grid_height=600,
                grid_width=800,
                screen_size=(1920, 1080),
            ),
            (800, 600),
        )

    def test_compute_display_target_size_scales_to_screen_bounds(self) -> None:
        self.assertEqual(
            _compute_display_target_size(
                grid_height=1200,
                grid_width=1600,
                screen_size=(1000, 800),
            ),
            (827, 620),
        )

    def test_build_camera_state_initializes_threaded_fields(self) -> None:
        state = _build_camera_state(
            serial="239222300433",
            usb_desc="3.2",
            pipeline=object(),
            align=object(),
            stream_w=848,
            stream_h=480,
            fps_used=30,
        )
        self.assertEqual(state["capture_seq"], 0)
        self.assertEqual(state["last_rendered_capture_seq"], 0)
        self.assertEqual(state["measured_fps"], 0.0)
        self.assertIsNone(state["last_color"])
        self.assertIsNotNone(state["lock"])

    def test_render_panel_uses_latest_buffer_and_marks_rendered_seq(self) -> None:
        state = _build_camera_state(
            serial="239222300433",
            usb_desc="3.2",
            pipeline=object(),
            align=object(),
            stream_w=848,
            stream_h=480,
            fps_used=30,
        )
        state["capture_seq"] = 2
        state["last_rendered_capture_seq"] = 1
        state["last_color"] = cameras_viewer._runtime_imports()[1].zeros((480, 848, 3), dtype="uint8")
        state["last_depth"] = cameras_viewer._runtime_imports()[1].zeros((480, 848), dtype="uint16")
        state["last_depth_scale_m_per_unit"] = 0.001

        with mock.patch("cameras_viewer.time.perf_counter", return_value=1.0):
            with mock.patch("cameras_viewer._make_panel", return_value="panel") as make_panel:
                result = _render_panel(
                    state,
                    width=848,
                    height=480,
                    depth_vis_min_m=0.1,
                    depth_vis_max_m=3.0,
                    depth_render_mode="colormap",
                )

        self.assertEqual(result, "panel")
        self.assertEqual(state["last_rendered_capture_seq"], 2)
        self.assertEqual(len(state["frame_times"]), 1)
        self.assertEqual(state["measured_fps"], 0.0)
        self.assertEqual(
            make_panel.call_args.kwargs["label_lines"],
            ("239222300433 usb=3.2 848x480@30.0fps", "configured: 30.0 | measured: warming"),
        )

    def test_render_panel_placeholder_mode_skips_depth_render(self) -> None:
        state = _build_camera_state(
            serial="239222300433",
            usb_desc="3.2",
            pipeline=object(),
            align=object(),
            stream_w=848,
            stream_h=480,
            fps_used=30,
        )
        state["capture_seq"] = 2
        state["last_rendered_capture_seq"] = 1
        state["last_color"] = cameras_viewer._runtime_imports()[1].zeros((480, 848, 3), dtype="uint8")
        state["last_depth"] = cameras_viewer._runtime_imports()[1].zeros((480, 848), dtype="uint16")
        state["last_depth_scale_m_per_unit"] = 0.001

        with mock.patch("cameras_viewer.time.perf_counter", return_value=1.0):
            with mock.patch("cameras_viewer._make_panel") as make_panel:
                with mock.patch("cameras_viewer._make_depth_debug_bottom", return_value="bottom") as debug_bottom:
                    with mock.patch("cameras_viewer._compose_panel", return_value="panel") as compose_panel:
                        result = _render_panel(
                            state,
                            width=848,
                            height=480,
                            depth_vis_min_m=0.1,
                            depth_vis_max_m=3.0,
                            depth_render_mode="fps_placeholder",
                        )

        self.assertEqual(result, "panel")
        make_panel.assert_not_called()
        debug_bottom.assert_called_once_with(
            width=848,
            height=480,
            measured_fps=0.0,
            measured_sample_count=1,
        )
        compose_panel.assert_called_once()


if __name__ == "__main__":
    unittest.main()
