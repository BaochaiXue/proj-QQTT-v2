"""Live data-process viewer + CvGuiLoop: pure observer, latest-wins, one GUI thread."""

from __future__ import annotations

import threading
import time
import unittest
from types import SimpleNamespace

import numpy as np

from demo_v6_2.mdp.gui_loop import CvGuiLoop
from demo_v6_2.mdp.live_viewer import LiveDataProcessViewer, render_pair_frame
from demo_v6_2.mdp.warmup_preview import WarmupRgbPreview
from demo_v6_2.utils.concurrency import LatestSlot


class _FakeCv2:
    """Records GUI calls; draws nothing."""

    WINDOW_AUTOSIZE = 1
    FONT_HERSHEY_SIMPLEX = 0
    LINE_AA = 16
    INTER_AREA = 3

    def __init__(self) -> None:
        self.shown: list[tuple[str, tuple[int, ...]]] = []
        self.windows: list[str] = []
        self.destroyed: list[str] = []
        self.waitkey_calls = 0

    def namedWindow(self, name, flags=None):  # noqa: N802
        self.windows.append(str(name))

    def imshow(self, name, frame):
        self.shown.append((str(name), tuple(frame.shape)))

    def waitKey(self, ms):  # noqa: N802
        self.waitkey_calls += 1
        time.sleep(min(int(ms), 5) / 1000.0)
        return -1

    def destroyWindow(self, name):  # noqa: N802
        self.destroyed.append(str(name))

    def putText(self, image, text, org, font, scale, color, thickness, line):  # noqa: N802
        return None

    def resize(self, image, size, interpolation=None):
        width, height = size
        return np.zeros((height, width, image.shape[2]), dtype=image.dtype)


class _BrokenCv2(_FakeCv2):
    def imshow(self, name, frame):
        raise RuntimeError("no display")


class _FakeGui:
    """Records submit/close_window calls from viewer/preview clients."""

    def __init__(self) -> None:
        self.submitted: list[tuple[str, tuple[int, ...]]] = []
        self.closed: list[str] = []

    def submit(self, window_name, frame):
        self.submitted.append((str(window_name), tuple(frame.shape)))

    def close_window(self, window_name):
        self.closed.append(str(window_name))


def _make_pair(seq: int, *, height: int = 12, width: int = 16) -> SimpleNamespace:
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[4:8, 4:9] = True
    hand_a = np.zeros((height, width), dtype=bool)
    hand_a[1:3, 1:3] = True
    hand_b = np.zeros((height, width), dtype=bool)
    hand_b[9:11, 12:14] = True
    controller = np.logical_or(hand_a, hand_b)
    mask_packet = SimpleNamespace(
        seq=seq,
        color_bgr=np.full((height, width, 3), 120, dtype=np.uint8),
        object_mask=object_mask,
        controller_mask=controller,
        hand_a_mask=hand_a,
        hand_b_mask=hand_b,
    )
    intrinsics = SimpleNamespace(fx=10.0, fy=10.0, cx=width / 2.0, cy=height / 2.0)
    pcd_packet = SimpleNamespace(
        seq=seq,
        intrinsics=intrinsics,
        coordinate_frame="camera_color_frame",
        object_xyz_m=np.asarray([[0.0, 0.0, 0.5], [0.05, 0.02, 0.6]], np.float32),
        object_colors_rgb_u8=np.asarray([[250, 10, 10], [10, 250, 10]], np.uint8),
        controller_xyz_m=np.asarray([[-0.02, 0.01, 0.4]], np.float32),
        controller_colors_rgb_u8=np.asarray([[10, 10, 250]], np.uint8),
        object_point_count=2,
        controller_point_count=1,
        shape_prior_points_m=np.asarray([[0.01, 0.01, 0.55]], np.float32),
        shape_prior_status="running",
        receive_perf_s=100.0,
        process_done_perf_s=100.150,
        timing=SimpleNamespace(mask_ms=20.0, pcd_ms=30.0),
    )
    tracker_packet = SimpleNamespace(
        seq=seq,
        all_tracks_yx=np.asarray([[2.0, 3.0], [6.0, 7.0], [50.0, 90.0]], np.float32),
        all_observation_visibility=np.asarray([True, True, False]),
        query_rgb_u8=np.asarray(
            [[255, 0, 0], [0, 255, 0], [0, 0, 255]], np.uint8
        ),
        process_done_perf_s=100.200,
        model_ms=8.0,
        lift_ms=2.0,
    )
    return SimpleNamespace(
        seq=seq,
        pcd_result=SimpleNamespace(
            pcd_packet=pcd_packet,
            processed_frame=SimpleNamespace(seq=seq, mask_packet=mask_packet),
        ),
        tracker_packet=tracker_packet,
    )


def _wait_until(predicate, timeout_s=3.0) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class RenderPairFrameTests(unittest.TestCase):
    def test_composite_shape_and_input_purity(self) -> None:
        pair = _make_pair(3)
        mask_packet = pair.pcd_result.processed_frame.mask_packet
        color_before = mask_packet.color_bgr.copy()
        object_mask_before = mask_packet.object_mask.copy()
        tracks_before = pair.tracker_packet.all_tracks_yx.copy()
        colors_before = pair.tracker_packet.query_rgb_u8.copy()
        object_xyz_before = pair.pcd_result.pcd_packet.object_xyz_m.copy()

        frame = render_pair_frame(
            pair,
            cv2=_FakeCv2(),
            table_c2w=None,
            fps_snapshot={"capture_fps": 5.0, "pcd_fps": 4.9},
            display_dropped=7,
        )
        self.assertEqual(frame.ndim, 3)
        # Pure observer: nothing reachable from the pair was modified.
        np.testing.assert_array_equal(mask_packet.color_bgr, color_before)
        np.testing.assert_array_equal(mask_packet.object_mask, object_mask_before)
        np.testing.assert_array_equal(pair.tracker_packet.all_tracks_yx, tracks_before)
        np.testing.assert_array_equal(pair.tracker_packet.query_rgb_u8, colors_before)
        np.testing.assert_array_equal(
            pair.pcd_result.pcd_packet.object_xyz_m, object_xyz_before
        )

    def test_world_frame_projects_through_table_c2w(self) -> None:
        pair = _make_pair(1)
        pair.pcd_result.pcd_packet.coordinate_frame = "table_world_z0"
        frame = render_pair_frame(
            pair,
            cv2=_FakeCv2(),
            table_c2w=np.eye(4, dtype=np.float32),
            fps_snapshot={},
            display_dropped=0,
        )
        self.assertEqual(frame.ndim, 3)


class LiveDataProcessViewerTests(unittest.TestCase):
    def _viewer(self, slot, gui, *, enabled=True):
        return LiveDataProcessViewer(
            pair_slot=slot,
            stage_stats=SimpleNamespace(fps_snapshot=lambda: {}),
            gui=gui,
            stop_event=threading.Event(),
            enabled=enabled,
            table_c2w=lambda: None,
            cv2_module=_FakeCv2(),
        )

    def test_submits_latest_and_drops_stale_display_frames(self) -> None:
        slot = LatestSlot()
        for seq in range(6):
            slot.put(_make_pair(seq))
        gui = _FakeGui()
        viewer = self._viewer(slot, gui)
        viewer.start()
        try:
            self.assertTrue(_wait_until(lambda: len(gui.submitted) >= 1))
        finally:
            viewer.close()
        self.assertEqual(gui.submitted[0][0], LiveDataProcessViewer.WINDOW_NAME)
        # close() tombstones the window at the GUI loop.
        self.assertEqual(gui.closed, [LiveDataProcessViewer.WINDOW_NAME])

    def test_disabled_viewer_never_starts_a_thread(self) -> None:
        viewer = self._viewer(LatestSlot(), _FakeGui(), enabled=False)
        viewer.start()
        self.assertIsNone(viewer._thread)

    def test_render_failure_disables_without_raising(self) -> None:
        slot = LatestSlot()
        bad = _make_pair(0)
        bad.tracker_packet.all_tracks_yx = "not-an-array"
        slot.put(bad)
        viewer = self._viewer(slot, _FakeGui())
        viewer.start()
        thread = viewer._thread
        self.assertIsNotNone(thread)
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())


class CvGuiLoopTests(unittest.TestCase):
    def test_lazy_start_and_latest_wins(self) -> None:
        fake = _FakeCv2()
        loop = CvGuiLoop(stop_event=threading.Event(), cv2_module=fake)
        self.assertIsNone(loop._thread)
        frame_a = np.zeros((4, 4, 3), np.uint8)
        frame_b = np.ones((4, 4, 3), np.uint8)
        loop.submit("w", frame_a)
        loop.submit("w", frame_b)  # replaces an unshown frame silently
        self.assertIsNotNone(loop._thread)
        try:
            self.assertTrue(_wait_until(lambda: len(fake.shown) >= 1))
        finally:
            loop.shutdown()
        self.assertEqual(fake.windows, ["w"])
        self.assertEqual(fake.destroyed, ["w"])

    def test_close_window_tombstones_future_submits(self) -> None:
        fake = _FakeCv2()
        loop = CvGuiLoop(stop_event=threading.Event(), cv2_module=fake)
        loop.submit("w", np.zeros((4, 4, 3), np.uint8))
        try:
            self.assertTrue(_wait_until(lambda: len(fake.shown) >= 1))
            loop.close_window("w")
            self.assertTrue(_wait_until(lambda: "w" in fake.destroyed))
            shown_before = len(fake.shown)
            loop.submit("w", np.zeros((4, 4, 3), np.uint8))
            time.sleep(0.2)
            self.assertEqual(len(fake.shown), shown_before)
        finally:
            loop.shutdown()

    def test_backend_failure_disables_all_windows_silently(self) -> None:
        loop = CvGuiLoop(stop_event=threading.Event(), cv2_module=_BrokenCv2())
        loop.submit("w", np.zeros((4, 4, 3), np.uint8))
        thread = loop._thread
        self.assertIsNotNone(thread)
        thread.join(timeout=3.0)
        self.assertFalse(thread.is_alive())
        self.assertTrue(loop._failed)
        loop.submit("w2", np.zeros((4, 4, 3), np.uint8))  # ignored, no restart

    def test_windowless_phase_does_not_busy_spin_on_waitkey(self) -> None:
        fake = _FakeCv2()
        loop = CvGuiLoop(stop_event=threading.Event(), cv2_module=fake)
        loop.submit("w", np.zeros((4, 4, 3), np.uint8))
        try:
            self.assertTrue(_wait_until(lambda: len(fake.shown) >= 1))
            loop.close_window("w")
            self.assertTrue(_wait_until(lambda: "w" in fake.destroyed))
            calls_before = fake.waitkey_calls
            time.sleep(0.3)
            # Windowless pacing must come from the close-event wait, not
            # from spinning waitKey (which returns instantly sans windows).
            self.assertEqual(fake.waitkey_calls, calls_before)
        finally:
            loop.shutdown()


class WarmupPreviewClientTests(unittest.TestCase):
    def test_preview_composes_and_submits_through_gui(self) -> None:
        slot = LatestSlot()
        slot.put(
            SimpleNamespace(
                seq=0, color_bgr=np.zeros((10, 10, 3), np.uint8)
            )
        )
        gui = _FakeGui()
        preview = WarmupRgbPreview(
            input_preview_slot=slot,
            gui=gui,
            stop_event=threading.Event(),
            enabled=True,
            cv2_module=_FakeCv2(),
        )
        preview.start()
        try:
            self.assertTrue(_wait_until(lambda: len(gui.submitted) >= 1))
        finally:
            preview.close()
        self.assertEqual(gui.submitted[0][0], WarmupRgbPreview.WINDOW_NAME)
        self.assertEqual(gui.closed, [WarmupRgbPreview.WINDOW_NAME])


class FormalStageTapTests(unittest.TestCase):
    def test_publish_strict_pair_feeds_live_viz_slot(self) -> None:
        from demo_v6_2.mdp.formal_products import FormalProductStage

        slot = LatestSlot()
        stage = FormalProductStage.__new__(FormalProductStage)
        stage.live_viz_slot = slot
        stage.shape_prior = SimpleNamespace(
            maybe_start_from_pcd_result=lambda result: None,
            packet_with_state=lambda packet: SimpleNamespace(
                seq=packet.seq,
                process_done_perf_s=packet.process_done_perf_s,
                shape_prior_status="ready",
            ),
        )
        stage.stage_stats = SimpleNamespace(record=lambda *a, **k: None)
        stage.lossless = SimpleNamespace(first_pair_published=threading.Event())
        stage.session = SimpleNamespace(headless_capture_writer=None)

        pair = SimpleNamespace(
            seq=4,
            pcd_result=SimpleNamespace(
                pcd_packet=SimpleNamespace(seq=4, process_done_perf_s=1.0),
                processed_frame=SimpleNamespace(
                    mask_packet=SimpleNamespace(seq=4)
                ),
            ),
            tracker_packet=SimpleNamespace(seq=4, process_done_perf_s=1.1),
        )
        import demo_v6_2.mdp.formal_products as formal_module

        original_replace = formal_module.replace
        formal_module.replace = lambda obj, **kw: SimpleNamespace(
            **{**obj.__dict__, **kw}
        )
        try:
            stage._publish_strict_pair(pair)
        finally:
            formal_module.replace = original_replace

        published = slot.get_latest_after(-1)
        self.assertIsNotNone(published)
        self.assertEqual(int(published.seq), 4)
        self.assertEqual(
            published.pcd_result.pcd_packet.shape_prior_status, "ready"
        )


class ConfigChainTests(unittest.TestCase):
    def test_flag_defaults_and_forwarding(self) -> None:
        from pathlib import Path

        from demo_v6_2 import main_cli
        from demo_v6_2.main_subprocess import build_main_data_processing_command
        from demo_v6_2.mdp import cli as mdp_cli
        from demo_v6_2.orchestration.main_config import (
            DEFAULT_LIVE_DATAPROCESS_VIEWER,
        )

        self.assertTrue(DEFAULT_LIVE_DATAPROCESS_VIEWER)
        camera_args = mdp_cli.build_parser().parse_args([])
        self.assertTrue(camera_args.live_dataprocess_viewer)

        orch_args = main_cli.build_parser().parse_args(
            ["--no-live-dataprocess-viewer"]
        )
        command = build_main_data_processing_command(
            orch_args,
            capture_dir=Path("/tmp/capture"),
            profile_json=Path("/tmp/profile.json"),
        )
        self.assertIn("--no-live-dataprocess-viewer", command)


if __name__ == "__main__":
    unittest.main()
