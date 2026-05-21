from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
import subprocess
import unittest

import numpy as np

from qqtt.tracking.backends.cotracker3_adapter import CoTracker3Adapter
from qqtt.tracking.backends.cotracker3_online import CoTracker3OnlineBackend
from qqtt.tracking.backends.litetracker_adapter import LiteTrackerAdapter
from qqtt.tracking.backends.locotrack_adapter import LocoTrackAdapter
from qqtt.tracking.backends.tapnextpp_adapter import TAPNextPPAdapter
from qqtt.tracking.backends.point_tracker_adapter import (
    PointTrackerAdapterConfig,
    build_point_tracker_adapter_factory,
    effective_legacy_update_mode,
    normalize_litetracker_runtime,
    normalize_tracker_backend,
    normalize_tracker_batch_query_count_policy,
    normalize_tracker_execution_mode,
    tracker_backend_spec,
)
from qqtt.tracking.base import TrackingResult


class _FakeCoTrackerBackend:
    def __init__(self) -> None:
        self.query_points = np.empty((0, 2), dtype=np.float32)
        self.query_points_by_camera: dict[int, np.ndarray] = {}

    def availability(self):
        from qqtt.tracking.base import BackendAvailability

        return BackendAvailability("fake", True, "ok")

    def is_available(self) -> bool:
        return True

    def availability_reason(self) -> str:
        return "ok"

    def warmup(self):
        return {"total_ms": 1.0}

    def initialize(self, frames, query_points_yx, masks=None) -> None:
        _ = frames, masks
        self.query_points = np.asarray(query_points_yx, dtype=np.float32)

    def update(self, frame) -> TrackingResult:
        _ = frame
        return TrackingResult(
            tracks_yx=self.query_points[None, :, :],
            visibility=np.ones((1, len(self.query_points)), dtype=np.float32),
            query_points_yx=self.query_points,
            stats={"stream_status": "published"},
        )

    def initialize_batch(self, query_points_yx_by_camera) -> None:
        self.query_points_by_camera = {
            int(camera_idx): np.asarray(points, dtype=np.float32)
            for camera_idx, points in query_points_yx_by_camera.items()
        }

    def update_batch(self, frames_by_camera):
        return {
            int(camera_idx): TrackingResult(
                tracks_yx=points[None, :, :],
                visibility=np.ones((1, len(points)), dtype=np.float32),
                camera_idx=int(camera_idx),
                query_points_yx=points,
                stats={"stream_status": "published", "update_mode": "batch"},
            )
            for camera_idx, points in self.query_points_by_camera.items()
        }


class _FakeLiteTrackerModel:
    def __init__(self) -> None:
        self.calls = 0
        self.reset_calls = 0
        self.last_frame_shape: tuple[int, ...] | None = None
        self.last_queries = None

    def init_video_online_processing(self) -> None:
        self.reset_calls += 1

    def __call__(self, frame, queries):
        import torch

        self.calls += 1
        self.last_frame_shape = tuple(frame.shape)
        self.last_queries = queries.detach().cpu()
        batch_size, query_count, _ = queries.shape
        batch_offsets = torch.arange(batch_size, device=queries.device, dtype=queries.dtype).reshape(batch_size, 1, 1)
        coords_xy = queries[..., 1:3] + batch_offsets
        visibility = torch.ones((batch_size, 1, query_count), device=queries.device, dtype=queries.dtype)
        confidence = torch.full((batch_size, 1, query_count), 0.75, device=queries.device, dtype=queries.dtype)
        return coords_xy[:, None], visibility, confidence


class _FakeLiteTrackerWrapper:
    def __init__(self) -> None:
        import torch

        self.device = "cpu"
        self.dtype = torch.float32
        self.model = _FakeLiteTrackerModel()
        self.queries = None
        self.is_first_frame = True


class _FakeLiteTrackerBatchPatchModel:
    def __init__(self) -> None:
        import torch

        self.corr_levels = 1
        self.corr_radius = 0
        self.stride = 1
        self.model_resolution = (8, 8)
        self.latent_dim = 2
        self.inv_sigmoid_true_val = 4.6
        self.iters = 1
        self.online_ind = 0
        self.track_feat_cache = [torch.empty(0)]
        self.ema_flow_buffer = torch.empty(0)
        self.coords_buffer = torch.empty(0)
        self.vis_buffer = torch.empty(0)
        self.conf_buffer = torch.empty(0)
        self.forward_window_coords_shapes: list[tuple[int, ...]] = []

    def fnet(self, frame):
        import torch

        batch_size = int(frame.shape[0])
        return torch.ones(
            (batch_size, self.latent_dim, self.model_resolution[0], self.model_resolution[1]),
            device=frame.device,
            dtype=frame.dtype,
        )

    def get_track_feat(self, fmaps, queried_coords, support_radius: int = 0):
        import torch

        _ = fmaps, support_radius
        batch_size, query_count, _coord_dim = queried_coords.shape
        support = torch.ones(
            (batch_size, 1, query_count, self.latent_dim),
            device=queried_coords.device,
            dtype=queried_coords.dtype,
        )
        return support[:, None, 0], support

    def forward_window(
        self,
        *,
        fmaps_pyramid,
        coords,
        track_feat_support_pyramid,
        queried_frames,
        vis,
        conf,
        is_track_previsouly_initialized,
        iters,
    ):
        _ = fmaps_pyramid, track_feat_support_pyramid, queried_frames, is_track_previsouly_initialized, iters
        self.forward_window_coords_shapes.append(tuple(coords.shape))
        if self.online_ind == 0:
            self.coords_buffer = coords
            self.vis_buffer = vis
            self.conf_buffer = conf
        else:
            self.coords_buffer = np_concat_torch_time(self.coords_buffer, coords)
            self.vis_buffer = np_concat_torch_time(self.vis_buffer, vis)
            self.conf_buffer = np_concat_torch_time(self.conf_buffer, conf)
        return coords, vis[..., 0], conf[..., 0]


def np_concat_torch_time(left, right):
    import torch

    return torch.cat([left, right], dim=1)


class _FakeLocoTrackModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_video_shape: tuple[int, ...] | None = None
        self.last_queries = None
        self.last_kwargs: dict[str, object] = {}

    def inference(self, video, query_points, **kwargs):
        import torch

        self.calls += 1
        self.last_video_shape = tuple(video.shape)
        self.last_queries = query_points.detach().cpu().numpy()
        self.last_kwargs = dict(kwargs)
        batch_size, frames, _height, _width, _channels = video.shape
        query_count = int(query_points.shape[1])
        queries_np = self.last_queries.astype(np.float32)
        tracks_xy = np.zeros((batch_size, query_count, frames, 2), dtype=np.float32)
        occlusion = np.zeros((batch_size, query_count, frames), dtype=bool)
        for batch_idx in range(batch_size):
            for query_idx in range(query_count):
                y = float(queries_np[batch_idx, query_idx, 1])
                x = float(queries_np[batch_idx, query_idx, 2])
                for frame_idx in range(frames):
                    tracks_xy[batch_idx, query_idx, frame_idx, 0] = x + batch_idx
                    tracks_xy[batch_idx, query_idx, frame_idx, 1] = y + frame_idx
        if query_count > 1:
            occlusion[:, 1, -1] = True
        return {
            "tracks": torch.from_numpy(tracks_xy),
            "occlusion": torch.from_numpy(occlusion),
        }


class _FakeTAPNextPPModel:
    def __init__(self) -> None:
        self.calls = 0
        self.last_video_shape: tuple[int, ...] | None = None
        self.last_query_shape: tuple[int, ...] | None = None
        self.last_queries = None
        self.last_used_state = False
        self.call_history: list[dict[str, object]] = []

    def __call__(self, *, video, query_points=None, state=None):
        import torch

        self.calls += 1
        self.last_video_shape = tuple(video.shape)
        self.last_query_shape = None if query_points is None else tuple(query_points.shape)
        self.last_queries = None if query_points is None else query_points.detach().cpu().numpy()
        self.last_used_state = state is not None
        source_queries = query_points if query_points is not None else state["query_points"]
        batch_size = int(video.shape[0])
        query_count = int(source_queries.shape[1])
        batch_offsets = torch.arange(batch_size, device=video.device, dtype=source_queries.dtype).reshape(batch_size, 1)
        tracks_xy = torch.empty((batch_size, 1, query_count, 2), device=video.device, dtype=source_queries.dtype)
        tracks_xy[:, 0, :, 0] = source_queries[:, :, 2] + batch_offsets
        tracks_xy[:, 0, :, 1] = source_queries[:, :, 1] + 2.0 * batch_offsets
        visible_logits = torch.ones((batch_size, 1, query_count, 1), device=video.device, dtype=source_queries.dtype)
        if query_count > 1:
            visible_logits[:, 0, 1, 0] = -1.0
        next_state = {"query_points": source_queries, "step": self.calls}
        self.call_history.append(
            {
                "video_shape": self.last_video_shape,
                "query_shape": self.last_query_shape,
                "used_state": self.last_used_state,
            }
        )
        return tracks_xy, torch.zeros((batch_size, 1, query_count, 512), device=video.device), visible_logits, next_state


class PointTrackerAdaptersTest(unittest.TestCase):
    def test_backend_normalization_and_specs(self) -> None:
        self.assertEqual(normalize_tracker_backend("co-tracker3"), "cotracker3_online")
        self.assertEqual(normalize_tracker_backend("track_on2"), "trackon2")
        self.assertEqual(normalize_tracker_backend("lite-tracker"), "litetracker")
        self.assertEqual(normalize_tracker_backend("loco-track-s"), "locotrack")
        self.assertEqual(normalize_tracker_backend("tapnext++"), "tapnextpp")
        self.assertEqual(normalize_tracker_backend("tap-next-plus-plus"), "tapnextpp")
        self.assertTrue(tracker_backend_spec("cotracker3_online").supports_batch_views)
        self.assertTrue(tracker_backend_spec("trackon2").supports_batch_views)
        self.assertTrue(tracker_backend_spec("litetracker").supports_batch_views)
        self.assertEqual(tracker_backend_spec("litetracker").batch_support_status, "experimental_batch_views")
        self.assertEqual(tracker_backend_spec("locotrack").family, "locotrack")
        self.assertTrue(tracker_backend_spec("locotrack").supports_batch_views)
        self.assertFalse(tracker_backend_spec("locotrack").supports_online)
        self.assertEqual(tracker_backend_spec("locotrack").batch_support_status, "windowed_batch_views")
        self.assertEqual(tracker_backend_spec("tapnextpp").family, "tapnext")
        self.assertTrue(tracker_backend_spec("tapnextpp").supports_batch_views)
        self.assertTrue(tracker_backend_spec("tapnextpp").supports_online)
        self.assertEqual(tracker_backend_spec("tapnextpp").batch_support_status, "true_online_batch_views")

    def test_execution_mode_and_policy_normalization(self) -> None:
        self.assertEqual(normalize_tracker_execution_mode("batch"), "batch-views")
        self.assertEqual(effective_legacy_update_mode("batch-views"), "batch")
        self.assertEqual(normalize_tracker_batch_query_count_policy("min_common"), "min-common")
        self.assertEqual(normalize_litetracker_runtime("onnx_cuda"), "onnx-cuda")
        self.assertEqual(normalize_litetracker_runtime("torch"), "pytorch")

    def test_factory_returns_external_adapter_shells(self) -> None:
        trackon = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="trackon2"))(-1)
        lite = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="litetracker"))(-1)
        loco = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="locotrack"))(-1)
        tapnext = build_point_tracker_adapter_factory(PointTrackerAdapterConfig(backend="tapnextpp"))(-1)

        self.assertEqual(trackon.name, "trackon2")
        self.assertFalse(trackon.availability().available)
        self.assertEqual(lite.name, "litetracker")
        self.assertFalse(lite.availability().available)
        self.assertIn("--litetracker-weights", lite.availability().reason)
        self.assertEqual(loco.name, "locotrack")
        self.assertEqual(type(loco).__name__, "LocoTrackAdapter")
        self.assertFalse(loco.availability().available)
        self.assertIn("--locotrack-repo-dir", loco.availability().reason)
        self.assertIn("install_locotrack_s_demo_3_1_max.sh", loco.availability().reason)
        self.assertEqual(tapnext.name, "tapnextpp")
        self.assertEqual(type(tapnext).__name__, "TAPNextPPAdapter")
        self.assertFalse(tapnext.availability().available)
        self.assertIn("--tapnet-repo-dir", tapnext.availability().reason)
        self.assertIn("install_tapnextpp_demo_3_1_max.sh", tapnext.availability().reason)

        lite_onnx = build_point_tracker_adapter_factory(
            PointTrackerAdapterConfig(backend="litetracker", litetracker_runtime="onnx-cuda")
        )(-1)
        self.assertEqual(type(lite_onnx).__name__, "OnnxLiteTrackerAdapter")
        self.assertFalse(lite_onnx.availability().available)
        self.assertIn("--litetracker-onnx-dir", lite_onnx.availability().reason)

    def test_point_tracker_adapter_config_roundtrips_locotrack_fields(self) -> None:
        config = PointTrackerAdapterConfig(
            backend="locotrack",
            locotrack_repo_dir="/tmp/locotrack/locotrack_pytorch",
            locotrack_checkpoint="/tmp/locotrack_small.ckpt",
            locotrack_model_size="small",
            locotrack_window_frames=12,
            locotrack_resolution=(320, 256),
            locotrack_query_chunk_size=128,
            locotrack_autocast_dtype="fp16",
        )
        restored = PointTrackerAdapterConfig(**asdict(config))

        self.assertEqual(restored, config)

    def test_point_tracker_adapter_config_roundtrips_tapnextpp_fields(self) -> None:
        config = PointTrackerAdapterConfig(
            backend="tapnextpp",
            tapnet_repo_dir="/tmp/tapnet",
            tapnextpp_checkpoint="/tmp/tapnextpp_ckpt.pt",
            tapnextpp_image_size=(256, 256),
            tapnextpp_autocast_dtype="fp16",
            tapnextpp_use_certainty=True,
            tapnextpp_certainty_radius=6,
            tapnextpp_certainty_threshold=0.4,
            tapnextpp_compile=True,
            tapnextpp_reset_on_reinitialize=False,
        )
        restored = PointTrackerAdapterConfig(**asdict(config))

        self.assertEqual(restored, config)

    def test_locotrack_adapter_availability_fails_clearly_when_missing(self) -> None:
        adapter = LocoTrackAdapter(device="cpu")
        availability = adapter.availability()

        self.assertFalse(availability.available)
        self.assertIn("--locotrack-repo-dir", availability.reason)
        self.assertIn("--locotrack-checkpoint", availability.reason)
        self.assertIn("install_locotrack_s_demo_3_1_max.sh", availability.reason)

    def test_tapnextpp_adapter_availability_fails_clearly_when_missing(self) -> None:
        adapter = TAPNextPPAdapter(device="cpu")
        availability = adapter.availability()

        self.assertFalse(availability.available)
        self.assertIn("--tapnet-repo-dir", availability.reason)
        self.assertIn("--tapnextpp-checkpoint", availability.reason)
        self.assertIn("install_tapnextpp_demo_3_1_max.sh", availability.reason)

    def test_tapnextpp_fake_model_serial_online_shapes_and_visibility(self) -> None:
        fake = _FakeTAPNextPPModel()
        adapter = TAPNextPPAdapter(device="cpu", image_size=(256, 256), autocast_dtype="fp32")
        adapter._model = fake
        query = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        adapter.initialize([], query)
        adapter.update(np.zeros((256, 256, 3), dtype=np.uint8))
        result = adapter.update(np.ones((256, 256, 3), dtype=np.uint8))

        self.assertEqual(fake.calls, 2)
        self.assertEqual(fake.call_history[0]["video_shape"], (1, 1, 256, 256, 3))
        self.assertEqual(fake.call_history[0]["query_shape"], (1, 2, 3))
        self.assertFalse(bool(fake.call_history[0]["used_state"]))
        self.assertEqual(fake.call_history[1]["video_shape"], (1, 1, 256, 256, 3))
        self.assertIsNone(fake.call_history[1]["query_shape"])
        self.assertTrue(bool(fake.call_history[1]["used_state"]))
        np.testing.assert_allclose(fake.last_queries if fake.last_queries is not None else query, query)
        first_query = adapter._query_points_tyx.detach().cpu().numpy()[0, 0]
        np.testing.assert_allclose(first_query, np.array([0.0, 10.0, 20.0], dtype=np.float32))
        self.assertEqual(result.tracks_yx.shape, (1, 2, 2))
        np.testing.assert_allclose(result.tracks_yx[0, 0], np.array([10.0, 20.0], dtype=np.float32))
        self.assertEqual(result.visibility.shape, (1, 2))
        self.assertEqual(float(result.visibility[0, 0]), 1.0)
        self.assertEqual(float(result.visibility[0, 1]), 0.0)
        self.assertEqual(result.stats["mode"], "tapnextpp_online_serial")
        self.assertEqual(result.stats["update_mode"], "serial")

    def test_tapnextpp_fake_model_batch_views_single_call_and_camera_split(self) -> None:
        fake = _FakeTAPNextPPModel()
        adapter = TAPNextPPAdapter(device="cpu", image_size=(256, 256), autocast_dtype="fp32")
        adapter._model = fake
        query_points = {
            0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            1: np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32),
            2: np.array([[12.0, 22.0], [32.0, 42.0]], dtype=np.float32),
        }
        adapter.initialize_batch(query_points)
        frames = {idx: np.zeros((256, 256, 3), dtype=np.uint8) for idx in query_points}

        results = adapter.update_batch(frames)

        self.assertEqual(fake.calls, 1)
        self.assertEqual(fake.last_video_shape, (3, 1, 256, 256, 3))
        self.assertEqual(fake.last_query_shape, (3, 2, 3))
        np.testing.assert_allclose(fake.last_queries[2, 0], np.array([0.0, 12.0, 22.0], dtype=np.float32))
        self.assertEqual(tuple(results), (0, 1, 2))
        np.testing.assert_allclose(results[0].tracks_yx[0, 0], np.array([10.0, 20.0], dtype=np.float32))
        np.testing.assert_allclose(results[1].tracks_yx[0, 0], np.array([13.0, 22.0], dtype=np.float32))
        np.testing.assert_allclose(results[2].tracks_yx[0, 0], np.array([16.0, 24.0], dtype=np.float32))
        self.assertEqual(float(results[2].visibility[0, 1]), 0.0)
        self.assertEqual(results[0].stats["mode"], "tapnextpp_online_batch_views")
        self.assertEqual(results[0].stats["update_mode"], "batch")
        self.assertEqual(results[0].stats["batch_size"], 3)
        self.assertEqual(results[0].stats["batch_camera_ids"], [0, 1, 2])
        self.assertEqual(results[0].stats["tapnextpp_model_calls"], 1)

    def test_tapnextpp_batch_views_rejects_unequal_query_count(self) -> None:
        adapter = TAPNextPPAdapter(device="cpu")
        adapter._model = _FakeTAPNextPPModel()

        with self.assertRaisesRegex(ValueError, "requires equal query counts"):
            adapter.initialize_batch(
                {
                    0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
                    1: np.array([[11.0, 21.0]], dtype=np.float32),
                }
            )

    def test_locotrack_fake_model_serial_window_shapes_and_visibility(self) -> None:
        fake = _FakeLocoTrackModel()
        adapter = LocoTrackAdapter(
            device="cpu",
            window_frames=3,
            resolution=(256, 256),
            query_chunk_size=128,
        )
        adapter._model = fake
        query = np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32)
        adapter.initialize([], query)
        adapter.update(np.zeros((4, 5, 3), dtype=np.uint8))
        result = adapter.update(np.ones((4, 5, 3), dtype=np.uint8))

        self.assertEqual(fake.calls, 2)
        self.assertEqual(fake.last_video_shape, (1, 2, 4, 5, 3))
        self.assertEqual(tuple(fake.last_queries.shape), (1, 2, 3))
        np.testing.assert_allclose(fake.last_queries[0, 0], np.array([0.0, 10.0, 20.0], dtype=np.float32))
        self.assertEqual(fake.last_kwargs["query_format"], "tyx")
        self.assertEqual(fake.last_kwargs["query_chunk_size"], 128)
        self.assertEqual(fake.last_kwargs["resolution"], (256, 256))
        self.assertEqual(result.tracks_yx.shape, (2, 2, 2))
        np.testing.assert_allclose(result.tracks_yx[-1, 0], np.array([11.0, 20.0], dtype=np.float32))
        self.assertEqual(result.visibility.shape, (2, 2))
        self.assertEqual(float(result.visibility[-1, 0]), 1.0)
        self.assertEqual(float(result.visibility[-1, 1]), 0.0)
        self.assertEqual(result.stats["mode"], "locotrack_windowed_serial")
        self.assertEqual(result.stats["update_mode"], "serial")

    def test_locotrack_fake_model_batch_views_single_call_and_camera_split(self) -> None:
        fake = _FakeLocoTrackModel()
        adapter = LocoTrackAdapter(device="cpu", window_frames=4, resolution=(256, 256))
        adapter._model = fake
        query_points = {
            0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            1: np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32),
            2: np.array([[12.0, 22.0], [32.0, 42.0]], dtype=np.float32),
        }
        adapter.initialize_batch(query_points)
        frames = {idx: np.zeros((6, 7, 3), dtype=np.uint8) for idx in query_points}

        results = adapter.update_batch(frames)

        self.assertEqual(fake.calls, 1)
        self.assertEqual(fake.last_video_shape, (3, 1, 6, 7, 3))
        self.assertEqual(tuple(fake.last_queries.shape), (3, 2, 3))
        np.testing.assert_allclose(fake.last_queries[2, 0], np.array([0.0, 12.0, 22.0], dtype=np.float32))
        self.assertEqual(tuple(results), (0, 1, 2))
        np.testing.assert_allclose(results[0].tracks_yx[-1, 0], np.array([10.0, 20.0], dtype=np.float32))
        np.testing.assert_allclose(results[1].tracks_yx[-1, 0], np.array([11.0, 22.0], dtype=np.float32))
        np.testing.assert_allclose(results[2].tracks_yx[-1, 0], np.array([12.0, 24.0], dtype=np.float32))
        self.assertEqual(float(results[2].visibility[-1, 1]), 0.0)
        self.assertEqual(results[0].stats["mode"], "locotrack_windowed_batch_views")
        self.assertEqual(results[0].stats["update_mode"], "batch")
        self.assertEqual(results[0].stats["batch_size"], 3)
        self.assertEqual(results[0].stats["batch_camera_ids"], [0, 1, 2])

    def test_locotrack_batch_views_rejects_unequal_query_count(self) -> None:
        adapter = LocoTrackAdapter(device="cpu")
        adapter._model = _FakeLocoTrackModel()

        with self.assertRaisesRegex(ValueError, "requires equal query counts"):
            adapter.initialize_batch(
                {
                    0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
                    1: np.array([[11.0, 21.0]], dtype=np.float32),
                }
            )

    def test_locotrack_install_script_help_does_not_install_torch_by_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts/env/install_locotrack_s_demo_3_1_max.sh"
        completed = subprocess.run(
            [str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = script.read_text(encoding="utf-8")

        self.assertIn("live-inference dependencies only", completed.stdout)
        self.assertIn("does not reinstall torch", completed.stdout)
        self.assertNotIn("torch==", text)
        self.assertIn("Do not reinstall torch/torchvision/torchaudio", text)

    def test_tapnextpp_install_script_help_does_not_install_torch_by_default(self) -> None:
        root = Path(__file__).resolve().parents[1]
        script = root / "scripts/env/install_tapnextpp_demo_3_1_max.sh"
        completed = subprocess.run(
            [str(script), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        text = script.read_text(encoding="utf-8")

        self.assertIn("Install TAPNext++", completed.stdout)
        self.assertIn("do not reinstall torch", completed.stdout.lower())
        self.assertNotIn("torch==", text)
        self.assertIn("Do not reinstall torch/torchvision/torchaudio", text)

    def test_cotracker_adapter_serial_and_batch_shapes(self) -> None:
        adapter = CoTracker3Adapter(backend=_FakeCoTrackerBackend())
        query = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
        adapter.initialize([], query)
        serial = adapter.update(np.zeros((4, 4, 3), dtype=np.uint8))

        self.assertEqual(serial.backend, "cotracker3_online")
        self.assertEqual(serial.tracks_yx.shape, (1, 2, 2))

        batch = CoTracker3Adapter(backend=_FakeCoTrackerBackend())
        batch.initialize_batch({0: query, 1: query[:1]})
        results = batch.update_batch({0: np.zeros((4, 4, 3), dtype=np.uint8), 1: np.zeros((4, 4, 3), dtype=np.uint8)})

        self.assertEqual(set(results), {0, 1})
        self.assertEqual(results[0].backend, "cotracker3_online")
        self.assertEqual(results[1].tracks_yx.shape, (1, 1, 2))

    def test_cotracker_online_batch_tensors_are_contiguous(self) -> None:
        frames = [
            np.zeros((3, 4, 5, 3), dtype=np.uint8),
            np.ones((3, 4, 5, 3), dtype=np.uint8),
        ]
        video = CoTracker3OnlineBackend._batch_frames_to_torch_video(frames, device="cpu")
        queries = CoTracker3OnlineBackend._batch_queries_yx_to_torch(
            {
                0: np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
                1: np.array([[5.0, 6.0]], dtype=np.float32),
                2: np.array([[7.0, 8.0]], dtype=np.float32),
            },
            camera_ids=(0, 1, 2),
            device="cpu",
        )

        self.assertEqual(tuple(video.shape), (3, 2, 3, 4, 5))
        self.assertTrue(video.is_contiguous())
        self.assertEqual(tuple(queries.shape), (3, 2, 3))
        self.assertTrue(queries.is_contiguous())

    def test_cotracker_online_batch_results_preserve_camera_order_and_xy_yx_round_trip(self) -> None:
        backend = CoTracker3OnlineBackend(device="cpu", model=object())
        query_points_by_camera = {
            0: np.array([[100.0, 200.0]], dtype=np.float32),
            1: np.array([[110.0, 210.0]], dtype=np.float32),
            2: np.array([[120.0, 220.0]], dtype=np.float32),
        }
        backend._batch_camera_ids = (0, 1, 2)
        backend._batch_query_points_yx_by_camera = query_points_by_camera
        backend._batch_query_counts_by_camera = {0: 1, 1: 1, 2: 1}
        backend._batch_total_frames = 1

        tracks_xy = np.array(
            [
                [[[200.0, 100.0]]],
                [[[210.0, 110.0]]],
                [[[220.0, 120.0]]],
            ],
            dtype=np.float32,
        )
        visibility = np.ones((3, 1, 1), dtype=np.float32)

        results = backend._tracks_to_batch_results(
            tracks_xy=tracks_xy,
            visibility=visibility,
            run_ms=1.0,
            step=1,
            window_len=2,
        )

        self.assertEqual(tuple(results), (0, 1, 2))
        for camera_idx, expected_yx in query_points_by_camera.items():
            result = results[camera_idx]
            self.assertEqual(result.camera_idx, camera_idx)
            np.testing.assert_allclose(result.tracks_yx[0, 0], expected_yx[0])
            np.testing.assert_allclose(result.query_points_yx[0], expected_yx[0])
            self.assertEqual(result.stats["batch_camera_ids"], [0, 1, 2])

    def test_cotracker_online_forward_window_patch_makes_expanded_coords_contiguous(self) -> None:
        import torch

        class _Core:
            def __init__(self) -> None:
                self.contiguous_seen: dict[str, bool] = {}

            def forward_window(self, *, coords, track_feat_support_pyramid, vis=None, conf=None, attention_mask=None):
                self.contiguous_seen = {
                    "coords": bool(coords.is_contiguous()),
                    "track_feat_support": bool(track_feat_support_pyramid[0].is_contiguous()),
                    "vis": bool(vis.is_contiguous()),
                    "conf": bool(conf.is_contiguous()),
                    "attention_mask": bool(attention_mask.is_contiguous()),
                }
                return "ok"

        class _Predictor:
            def __init__(self) -> None:
                self.model = _Core()

        predictor = _Predictor()
        CoTracker3OnlineBackend._patch_online_model_for_batch_views(predictor)
        coords = torch.zeros((2, 1, 3, 2)).expand(2, 4, 3, 2)
        track_feat = torch.zeros((2, 4, 3, 5)).transpose(1, 2)
        vis = torch.zeros((2, 1, 3, 1)).expand(2, 4, 3, 1)
        conf = torch.zeros((2, 1, 3, 1)).expand(2, 4, 3, 1)
        attention_mask = torch.zeros((2, 1, 3)).expand(2, 4, 3)

        self.assertFalse(coords.is_contiguous())
        self.assertEqual(
            predictor.model.forward_window(
                coords=coords,
                track_feat_support_pyramid=[track_feat],
                vis=vis,
                conf=conf,
                attention_mask=attention_mask,
            ),
            "ok",
        )
        self.assertEqual(
            predictor.model.contiguous_seen,
            {
                "coords": True,
                "track_feat_support": True,
                "vis": True,
                "conf": True,
                "attention_mask": True,
            },
        )

    def test_cotracker_online_cuda_cache_is_reserved_for_oom_recovery(self) -> None:
        self.assertTrue(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("CUDA out of memory.")))
        self.assertTrue(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("CUBLAS out of memory.")))
        self.assertFalse(CoTracker3OnlineBackend._is_cuda_oom_error(RuntimeError("shape mismatch")))

    def test_litetracker_batch_views_preserve_camera_order_and_xy_yx_round_trip(self) -> None:
        fake = _FakeLiteTrackerWrapper()
        adapter = LiteTrackerAdapter()
        adapter._tracker = fake
        query_points = {
            0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
            1: np.array([[11.0, 21.0], [31.0, 41.0]], dtype=np.float32),
            2: np.array([[12.0, 22.0], [32.0, 42.0]], dtype=np.float32),
        }
        adapter.initialize_batch(query_points)
        frames = {idx: np.zeros((4, 5, 3), dtype=np.uint8) for idx in query_points}

        results = adapter.update_batch(frames)

        self.assertEqual(tuple(results), (0, 1, 2))
        self.assertEqual(fake.model.reset_calls, 1)
        self.assertEqual(fake.model.calls, 2)
        self.assertEqual(fake.model.last_frame_shape, (3, 3, 4, 5))
        self.assertEqual(tuple(fake.model.last_queries.shape), (3, 2, 3))
        np.testing.assert_allclose(fake.model.last_queries[1, 0].numpy(), np.array([0.0, 21.0, 11.0], dtype=np.float32))
        np.testing.assert_allclose(results[0].tracks_yx[0], query_points[0])
        np.testing.assert_allclose(results[1].tracks_yx[0], query_points[1] + 1.0)
        np.testing.assert_allclose(results[2].tracks_yx[0], query_points[2] + 2.0)
        self.assertEqual(results[2].stats["update_mode"], "batch")
        self.assertEqual(results[2].stats["lite_batch_size"], 3)
        self.assertEqual(results[2].stats["lite_effective_query_count"], 2)

    def test_litetracker_batch_views_rejects_unequal_query_count(self) -> None:
        adapter = LiteTrackerAdapter()
        adapter._tracker = _FakeLiteTrackerWrapper()
        with self.assertRaisesRegex(ValueError, "requires equal query counts"):
            adapter.initialize_batch(
                {
                    0: np.array([[10.0, 20.0], [30.0, 40.0]], dtype=np.float32),
                    1: np.array([[11.0, 21.0]], dtype=np.float32),
                }
            )

    def test_litetracker_batch_forward_patch_keeps_time_axis_for_b_greater_than_one(self) -> None:
        import torch

        model = _FakeLiteTrackerBatchPatchModel()
        LiteTrackerAdapter._patch_model_for_batch_views(model)
        frame = torch.zeros((3, 3, 8, 8), dtype=torch.float32)
        queries = torch.zeros((3, 2, 3), dtype=torch.float32)
        queries[..., 1] = 2.0
        queries[..., 2] = 3.0

        model.forward(frame, queries)
        model.forward(frame, queries)

        self.assertTrue(getattr(model, "_qqtt_batch_view_forward_patch", False))
        self.assertEqual(model.forward_window_coords_shapes, [(3, 1, 2, 2), (3, 1, 2, 2)])
        self.assertEqual(tuple(model.coords_buffer.shape), (3, 2, 2, 2))


if __name__ == "__main__":
    unittest.main()
