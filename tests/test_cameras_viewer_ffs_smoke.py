from __future__ import annotations

import argparse
from collections import deque
import queue
from pathlib import Path
import tempfile
import unittest

import numpy as np

from cameras_viewer_FFS import (
    _compute_measured_fps,
    _format_ffs_backend_startup_note,
    _format_runtime_stats_lines,
    _format_panel_label_lines,
    _put_latest,
    _reproject_ffs_depth_to_color,
    _resolve_ffs_worker_kwargs,
    _summarize_runtime_stats,
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
                "trt_engine_height": 480,
                "trt_engine_width": 640,
            },
        )
        self.assertEqual(
            note,
            "TensorRT engine 640x480; capture 848x480 will be resized before inference.",
        )


if __name__ == "__main__":
    unittest.main()
