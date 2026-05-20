from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import numpy as np

from qqtt.tracking.backends.litetracker_onnx_adapter import (
    REQUIRED_LITETRACKER_ONNX_FILES,
    OnnxLiteTrackerAdapter,
)


class _FakeSession:
    def __init__(self, providers):
        self._providers = list(providers)

    def get_providers(self):
        return list(self._providers)


class _FakeOnnxLiteTracker:
    def __init__(self, onnx_dir: str, providers=None):
        self.onnx_dir = str(onnx_dir)
        self.providers = list(providers or ["CUDAExecutionProvider", "CPUExecutionProvider"])
        self.fnet_session = _FakeSession(self.providers)
        self.calls = 0
        self.reset_calls = 0
        self.last_frame_shape: tuple[int, ...] | None = None
        self.last_queries = None

    def reset(self) -> None:
        self.reset_calls += 1

    def __call__(self, frame, queries):
        import torch

        self.calls += 1
        self.last_frame_shape = tuple(frame.shape)
        self.last_queries = queries.detach().cpu().numpy()
        batch_size, query_count, _ = queries.shape
        coords_xy = queries[..., 1:3]
        visibility = torch.ones((batch_size, 1, query_count), dtype=torch.float32)
        confidence = torch.full((batch_size, 1, query_count), 0.75, dtype=torch.float32)
        return coords_xy[:, None], visibility, confidence


def _write_dummy_onnx_dir(root: Path) -> None:
    root.mkdir(parents=True, exist_ok=True)
    for name in REQUIRED_LITETRACKER_ONNX_FILES:
        (root / name).write_bytes(b"dummy")


class LiteTrackerOnnxAdapterContractTest(unittest.TestCase):
    def test_availability_reports_missing_onnx_dir(self) -> None:
        adapter = OnnxLiteTrackerAdapter(onnx_dir=None)

        availability = adapter.availability()

        self.assertFalse(availability.available)
        self.assertIn("--litetracker-onnx-dir", availability.reason)

    def test_update_batch_is_not_supported_yet(self) -> None:
        adapter = OnnxLiteTrackerAdapter(tracker_cls=_FakeOnnxLiteTracker)

        with self.assertRaisesRegex(NotImplementedError, "serial adapter does not support batch-views"):
            adapter.initialize_batch({0: np.zeros((1, 2), dtype=np.float32)})
        with self.assertRaisesRegex(NotImplementedError, "serial adapter does not support batch-views"):
            adapter.update_batch({0: np.zeros((4, 4, 3), dtype=np.uint8)})

    def test_serial_update_preserves_yx_xy_yx_and_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx_dir = Path(tmp)
            _write_dummy_onnx_dir(onnx_dir)
            adapter = OnnxLiteTrackerAdapter(
                camera_idx=2,
                onnx_dir=str(onnx_dir),
                tracker_cls=_FakeOnnxLiteTracker,
                optimization_level=5,
            )
            query_points_yx = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
            adapter.initialize([], query_points_yx)

            result = adapter.update(np.zeros((8, 9, 3), dtype=np.uint8))

            fake = adapter._tracker
            self.assertIsInstance(fake, _FakeOnnxLiteTracker)
            self.assertEqual(fake.reset_calls, 1)
            self.assertEqual(fake.calls, 2)
            self.assertEqual(fake.last_frame_shape, (1, 3, 8, 9))
            np.testing.assert_allclose(
                fake.last_queries[0],
                np.array([[0.0, 20.0, 10.0], [0.0, 40.0, 30.0]], dtype=np.float32),
            )
            self.assertEqual(result.camera_idx, 2)
            self.assertEqual(result.tracks_yx.shape, (1, 2, 2))
            self.assertEqual(result.visibility.shape, (1, 2))
            np.testing.assert_allclose(result.tracks_yx[0], query_points_yx)
            self.assertEqual(result.stats["tracker_backend"], "litetracker")
            self.assertEqual(result.stats["litetracker_runtime"], "onnx-cuda")
            self.assertEqual(result.stats["onnx_provider"], "CUDAExecutionProvider")
            self.assertEqual(result.stats["onnx_providers"], ["CUDAExecutionProvider", "CPUExecutionProvider"])
            self.assertIn("model_run_ms", result.stats)
            self.assertIn("litetracker_model_ms", result.stats)
            self.assertIn("litetracker_e2e_ms", result.stats)
            self.assertEqual(result.stats["litetracker_onnx_optimization_level"], 5)
            self.assertEqual(result.stats["litetracker_onnx_opset"], 17)
            self.assertEqual(result.stats["litetracker_onnx_opset_actual"], 18)

    def test_initialize_rejects_bad_query_shape(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            onnx_dir = Path(tmp)
            _write_dummy_onnx_dir(onnx_dir)
            adapter = OnnxLiteTrackerAdapter(onnx_dir=str(onnx_dir), tracker_cls=_FakeOnnxLiteTracker)

            with self.assertRaisesRegex(ValueError, "shape"):
                adapter.initialize([], np.zeros((3,), dtype=np.float32))


if __name__ == "__main__":
    unittest.main()
