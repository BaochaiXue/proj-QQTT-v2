from __future__ import annotations

import argparse
from collections import deque
import queue
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from cameras_viewer_FFS import (
    DEFAULT_FFS_TRT_MODEL_DIR,
    _compute_measured_fps,
    _drain_shared_worker_next_request,
    _format_ffs_backend_startup_note,
    _format_runtime_stats_lines,
    _format_panel_label_lines,
    _fit_grid_for_window,
    _put_latest,
    _reproject_ffs_depth_to_color,
    _resolve_ffs_worker_kwargs,
    _summarize_runtime_stats,
    _update_recent_frame_times,
    parse_args,
)
from data_process.depth_backends.fast_foundation_stereo import (
    apply_tensorrt_image_transform,
    resolve_tensorrt_image_transform,
    run_forward_on_non_default_cuda_stream,
    undo_tensorrt_disparity_transform,
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

    def test_shared_worker_request_drain_round_robins_across_camera_queues(self) -> None:
        q0: queue.Queue[object] = queue.Queue(maxsize=1)
        q1: queue.Queue[object] = queue.Queue(maxsize=1)
        q2: queue.Queue[object] = queue.Queue(maxsize=1)
        q0.put_nowait({"capture_seq": 1})
        q1.put_nowait({"capture_seq": 2})
        q2.put_nowait({"capture_seq": 3})
        request_queues = {0: q0, 1: q1, 2: q2}

        camera_idx, payload, cursor, closed = _drain_shared_worker_next_request(
            camera_order=[0, 1, 2],
            request_queues=request_queues,
            closed_camera_indices=set(),
            start_cursor=0,
        )
        self.assertEqual(camera_idx, 0)
        self.assertEqual(payload, {"capture_seq": 1})
        self.assertEqual(cursor, 1)
        self.assertEqual(closed, set())

        camera_idx, payload, cursor, closed = _drain_shared_worker_next_request(
            camera_order=[0, 1, 2],
            request_queues=request_queues,
            closed_camera_indices=closed,
            start_cursor=cursor,
        )
        self.assertEqual(camera_idx, 1)
        self.assertEqual(payload, {"capture_seq": 2})
        self.assertEqual(cursor, 2)
        self.assertEqual(closed, set())

    def test_shared_worker_request_drain_marks_closed_queues_and_continues(self) -> None:
        q0: queue.Queue[object] = queue.Queue(maxsize=1)
        q1: queue.Queue[object] = queue.Queue(maxsize=1)
        q0.put_nowait(None)
        q1.put_nowait({"capture_seq": 9})
        camera_idx, payload, cursor, closed = _drain_shared_worker_next_request(
            camera_order=[0, 1],
            request_queues={0: q0, 1: q1},
            closed_camera_indices=set(),
            start_cursor=0,
        )
        self.assertEqual(camera_idx, 1)
        self.assertEqual(payload, {"capture_seq": 9})
        self.assertEqual(cursor, 0)
        self.assertEqual(closed, {0})

    def test_fit_grid_for_window_uses_stable_screen_bounded_helper(self) -> None:
        grid = np.zeros((960, 1696, 3), dtype=np.uint8)
        sentinel = np.zeros((720, 1272, 3), dtype=np.uint8)
        with mock.patch("cameras_viewer_FFS._fit_grid_for_display", return_value=sentinel) as fit_mock:
            result = _fit_grid_for_window(grid, window_name="RealSense FFS Viewer")
        fit_mock.assert_called_once()
        self.assertIs(result, sentinel)

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

    def test_runtime_stats_summary_aggregates_per_camera_rates(self) -> None:
        runtime_stats = _summarize_runtime_stats(
            [
                {
                    "camera_idx": 0,
                    "serial": "cam0",
                    "capture_fps": 33.1,
                    "capture_sample_count": 10,
                    "ffs_fps": 2.6,
                    "ffs_sample_count": 10,
                    "latest_inference_ms": 381.2,
                    "seq_gap": 14,
                    "worker_error": None,
                },
                {
                    "camera_idx": 1,
                    "serial": "cam1",
                    "capture_fps": 33.1,
                    "capture_sample_count": 10,
                    "ffs_fps": 2.7,
                    "ffs_sample_count": 10,
                    "latest_inference_ms": 376.8,
                    "seq_gap": 13,
                    "worker_error": None,
                },
            ]
        )
        self.assertEqual(runtime_stats["camera_count"], 2)
        self.assertAlmostEqual(runtime_stats["aggregate_capture_fps"], 66.2, places=6)
        self.assertAlmostEqual(runtime_stats["aggregate_ffs_fps"], 5.3, places=6)

    def test_runtime_stats_lines_are_deterministic(self) -> None:
        runtime_stats = {
            "camera_count": 2,
            "aggregate_capture_fps": 66.2,
            "aggregate_ffs_fps": 5.3,
            "per_camera": [
                {
                    "camera_idx": 0,
                    "serial": "239222300412",
                    "capture_fps": 33.1,
                    "capture_sample_count": 10,
                    "ffs_fps": 2.6,
                    "ffs_sample_count": 10,
                    "latest_inference_ms": 381.2,
                    "seq_gap": 14,
                    "worker_error": None,
                },
                {
                    "camera_idx": 1,
                    "serial": "239222303506",
                    "capture_fps": 33.1,
                    "capture_sample_count": 10,
                    "ffs_fps": 2.7,
                    "ffs_sample_count": 10,
                    "latest_inference_ms": 376.8,
                    "seq_gap": 13,
                    "worker_error": "RuntimeError: sample",
                },
            ],
        }
        lines = _format_runtime_stats_lines(elapsed_s=5.0, runtime_stats=runtime_stats)
        self.assertEqual(lines[0], "[stats t=5.0s cams=2] capture_sum=66.2 ffs_sum=5.3")
        self.assertEqual(
            lines[1],
            "[stats cam0 239222300412] capture=33.1 ffs=2.6 infer_ms=381.2 seq_gap=14",
        )
        self.assertEqual(
            lines[2],
            "[stats cam1 239222303506] capture=33.1 ffs=2.7 infer_ms=376.8 seq_gap=13 error=RuntimeError: sample",
        )

    def test_resolve_pytorch_worker_kwargs_keeps_existing_runtime_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "ffs_repo"
            repo.mkdir()
            model_path = root / "model.pth"
            model_path.write_bytes(b"stub")
            args = argparse.Namespace(
                ffs_backend="pytorch",
                ffs_repo=repo,
                ffs_model_path=model_path,
                ffs_scale=0.75,
                ffs_valid_iters=4,
                ffs_max_disp=192,
                ffs_trt_model_dir=None,
                ffs_trt_root=None,
            )
            worker_kwargs = _resolve_ffs_worker_kwargs(args)

        self.assertEqual(worker_kwargs["runner_backend"], "pytorch")
        self.assertEqual(worker_kwargs["model_path"], str(model_path.resolve()))
        self.assertAlmostEqual(worker_kwargs["ffs_scale"], 0.75)
        self.assertEqual(worker_kwargs["ffs_valid_iters"], 4)
        self.assertEqual(worker_kwargs["ffs_max_disp"], 192)
        self.assertIsNone(worker_kwargs["trt_model_dir"])

    def test_parse_args_defaults_to_tensorrt_and_repo_local_engine_dir(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "ffs_repo"
            repo.mkdir()
            with mock.patch("sys.argv", ["cameras_viewer_FFS.py", "--ffs_repo", str(repo)]):
                args = parse_args()

        self.assertEqual(args.ffs_backend, "tensorrt")
        self.assertEqual(args.ffs_trt_model_dir, DEFAULT_FFS_TRT_MODEL_DIR)
        self.assertEqual(args.ffs_worker_mode, "per_camera")

    def test_parse_args_accepts_shared_worker_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "ffs_repo"
            repo.mkdir()
            with mock.patch(
                "sys.argv",
                ["cameras_viewer_FFS.py", "--ffs_repo", str(repo), "--ffs_worker_mode", "shared"],
            ):
                args = parse_args()

        self.assertEqual(args.ffs_worker_mode, "shared")

    def test_resolve_tensorrt_worker_kwargs_loads_engine_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            repo = root / "ffs_repo"
            repo.mkdir()
            trt_model_dir = root / "trt_model"
            trt_model_dir.mkdir()
            (trt_model_dir / "onnx.yaml").write_text(
                "image_size: [480, 640]\nvalid_iters: 4\nmax_disp: 192\n",
                encoding="utf-8",
            )
            (trt_model_dir / "feature_runner.engine").write_bytes(b"stub")
            (trt_model_dir / "post_runner.engine").write_bytes(b"stub")
            args = argparse.Namespace(
                ffs_backend="tensorrt",
                ffs_repo=repo,
                ffs_model_path=None,
                ffs_scale=1.0,
                ffs_valid_iters=8,
                ffs_max_disp=256,
                ffs_trt_model_dir=trt_model_dir,
                ffs_trt_root=None,
            )
            worker_kwargs = _resolve_ffs_worker_kwargs(args)

        self.assertEqual(worker_kwargs["runner_backend"], "tensorrt")
        self.assertEqual(worker_kwargs["trt_model_dir"], str(trt_model_dir.resolve()))
        self.assertEqual(worker_kwargs["trt_engine_height"], 480)
        self.assertEqual(worker_kwargs["trt_engine_width"], 640)
        self.assertIsNone(worker_kwargs["model_path"])

    def test_tensorrt_startup_note_reports_resize_when_engine_differs(self) -> None:
        note = _format_ffs_backend_startup_note(
            runner_backend="tensorrt",
            stream_w=848,
            stream_h=480,
            worker_kwargs={
                "trt_engine_height": 448,
                "trt_engine_width": 640,
            },
        )
        self.assertEqual(
            note,
            "TensorRT engine 640x448; capture 848x480 will be resized before inference.",
        )

    def test_tensorrt_startup_note_reports_symmetric_padding_for_848_capture(self) -> None:
        note = _format_ffs_backend_startup_note(
            runner_backend="tensorrt",
            stream_w=848,
            stream_h=480,
            worker_kwargs={
                "trt_engine_height": 480,
                "trt_engine_width": 864,
            },
        )
        self.assertEqual(
            note,
            "TensorRT engine 864x480; capture 848x480 will be symmetrically padded to 864x480 before inference.",
        )

    def test_tensorrt_transform_resolves_pad_for_848_capture_and_864_engine(self) -> None:
        transform = resolve_tensorrt_image_transform(
            input_height=480,
            input_width=848,
            engine_height=480,
            engine_width=864,
        )
        self.assertEqual(transform["mode"], "pad")
        self.assertEqual(transform["pad_left"], 8)
        self.assertEqual(transform["pad_right"], 8)
        self.assertEqual(transform["scale_x"], 1.0)
        self.assertEqual(transform["scale_y"], 1.0)

    def test_tensorrt_pad_transform_replication_and_unpad_are_deterministic(self) -> None:
        transform = resolve_tensorrt_image_transform(
            input_height=480,
            input_width=848,
            engine_height=480,
            engine_width=864,
        )
        image = np.zeros((480, 848, 3), dtype=np.uint8)
        image[:, 0, :] = 17
        image[:, -1, :] = 29
        padded = apply_tensorrt_image_transform(image, transform=transform)
        self.assertEqual(padded.shape, (480, 864, 3))
        self.assertTrue(np.all(padded[:, :8, :] == 17))
        self.assertTrue(np.all(padded[:, -8:, :] == 29))

        disparity = np.arange(480 * 864, dtype=np.float32).reshape(480, 864)
        cropped = undo_tensorrt_disparity_transform(disparity, transform=transform)
        self.assertEqual(cropped.shape, (480, 848))
        np.testing.assert_array_equal(cropped, disparity[:, 8:-8])

    def test_non_default_cuda_stream_helper_waits_and_restores_order(self) -> None:
        events: list[object] = []

        class FakeCurrentStream:
            def wait_stream(self, stream: object) -> None:
                events.append(("current_wait_stream", stream))

        class FakeInferenceStream:
            def wait_stream(self, stream: object) -> None:
                events.append(("inference_wait_stream", stream))

        class FakeStreamContext:
            def __init__(self, stream: object) -> None:
                self._stream = stream

            def __enter__(self) -> None:
                events.append(("enter_stream", self._stream))

            def __exit__(self, exc_type, exc, tb) -> None:
                events.append(("exit_stream", self._stream))

        fake_current_stream = FakeCurrentStream()
        fake_inference_stream = FakeInferenceStream()

        class FakeCuda:
            def current_stream(self) -> object:
                events.append("current_stream")
                return fake_current_stream

            def stream(self, stream: object) -> FakeStreamContext:
                events.append(("stream_context", stream))
                return FakeStreamContext(stream)

        class FakeTorch:
            cuda = FakeCuda()

        def fake_forward(**kwargs: object) -> dict[str, object]:
            events.append(("forward", kwargs))
            return {"ok": True}

        result = run_forward_on_non_default_cuda_stream(
            torch_module=FakeTorch(),
            stream=fake_inference_stream,
            forward_fn=fake_forward,
            image1="left",
            image2="right",
        )

        self.assertEqual(result, {"ok": True})
        self.assertEqual(
            events,
            [
                "current_stream",
                ("inference_wait_stream", fake_current_stream),
                ("stream_context", fake_inference_stream),
                ("enter_stream", fake_inference_stream),
                ("forward", {"image1": "left", "image2": "right"}),
                ("exit_stream", fake_inference_stream),
                ("current_wait_stream", fake_inference_stream),
            ],
        )


if __name__ == "__main__":
    unittest.main()
