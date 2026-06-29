from __future__ import annotations

import contextlib
import io
import subprocess
import sys
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]


class DemoV51ShapePriorSimplificationTests(unittest.TestCase):
    def test_warmup_split_modules_are_local_files(self) -> None:
        expected_files = (
            ROOT / "demo_v5_1" / "shape_prior.py",
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "shape_prior_worker.py",
            ROOT / "demo_v5_1" / "runtime_warmup.py",
        )
        for path in expected_files:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

        removed_files = (
            ROOT / "demo_v5_1" / "shape_prior_runtime.py",
            ROOT / "demo_v5_1" / "shape_prior_rpc.py",
            ROOT / "demo_v5_1" / "single_view_shape_prior_sampling.py",
            ROOT / "demo_v5_1" / "single_view_shape_align.py",
        )
        for path in removed_files:
            with self.subTest(path=path.name):
                self.assertFalse(path.exists())

        for path in (
            ROOT / "demo_v5_1" / "shape_prior.py",
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "shape_prior_worker.py",
            ROOT / "demo_v5_1" / "realtime_dense_track.py",
            ROOT / "demo_v5_1" / "runtime_warmup.py",
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("qqtt.demo.shape_prior", source)
                self.assertNotIn("from qqtt.demo import shape_prior", source)
                self.assertNotIn("services.shape_prior_remote", source)

    def test_shape_prior_warmup_module_owns_lifecycle_helpers(self) -> None:
        shape_prior_source = (ROOT / "demo_v5_1" / "shape_prior.py").read_text(
            encoding="utf-8"
        )
        warmup_source = (
            ROOT / "demo_v5_1" / "shape_prior_warmup.py"
        ).read_text(encoding="utf-8")
        worker_source = (
            ROOT / "demo_v5_1" / "shape_prior_worker.py"
        ).read_text(encoding="utf-8")

        for token in (
            "class ShapePriorRemoteClient",
            "class ShapePriorWarmupManager",
            "def default_profile",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, shape_prior_source)
                self.assertIn(token, warmup_source)

        self.assertIn("def prepare_shape_prior_worker_startup", warmup_source)
        self.assertNotIn("def _prepare_worker_startup", worker_source)

    def test_runtime_warmup_module_owns_first_frame_and_startup_helpers(self) -> None:
        runtime_source = (ROOT / "demo_v5_1" / "runtime_warmup.py").read_text(
            encoding="utf-8"
        )
        realtime_source = (
            ROOT / "demo_v5_1" / "realtime_dense_track.py"
        ).read_text(encoding="utf-8")

        for token in (
            "class InitialMaskBundle",
            "def run_sam31_first_frame_mask_bundle",
            "def resolve_initial_mask_bundle",
            "def prepare_runtime_services_and_source",
            "def prepare_runtime_projection_and_capture",
            "def prepare_segmentation_warmup",
        ):
            with self.subTest(token=token):
                self.assertIn(token, runtime_source)

        for token in (
            "class InitialMaskBundle",
            "def run_sam31_first_frame_mask_bundle",
            "def resolve_initial_mask_bundle",
            "def load_binary_mask",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, realtime_source)

    def test_prepare_shape_prior_worker_startup_preloads_models(self) -> None:
        from demo_v5_1.shape_prior_warmup import (
            prepare_shape_prior_worker_startup,
        )

        class FakeWorker:
            def __init__(self) -> None:
                self.preload_calls = 0
                self._startup_metadata: dict[str, object] = {}

            def preload_models(self) -> dict[str, object]:
                self.preload_calls += 1
                self._startup_metadata["preload_called"] = True
                return self.startup_metadata()

            def startup_metadata(self) -> dict[str, object]:
                return dict(self._startup_metadata)

        worker = FakeWorker()
        metadata = prepare_shape_prior_worker_startup(worker, preload_models=True)

        self.assertEqual(1, worker.preload_calls)
        self.assertTrue(metadata["preload_called"])
        self.assertTrue(metadata["worker_preloaded_models"])
        self.assertIn("worker_ready_ms", metadata)

    def test_shape_prior_warmup_manager_ready_and_failed_profiles(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        request = shape_prior_warmup.ShapePriorFrame0Request(
            seq=7,
            source_timestamp_s=12.5,
            input_source="test",
            depth_backend="test",
            depth_source_internal="test",
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            object_observation_mask=None,
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        class ReadyClient:
            def request_shape_prior(
                self,
                frame0: shape_prior_warmup.ShapePriorFrame0Request,
            ) -> shape_prior_warmup.ShapePriorResult:
                return shape_prior_warmup.ShapePriorResult(
                    seq=int(frame0.seq),
                    source_seq=int(frame0.seq),
                    source_timestamp_s=frame0.source_timestamp_s,
                    status=shape_prior_warmup.SHAPE_PRIOR_STATUS_READY,
                    points_m=np.ones((1, 3), dtype=np.float32),
                    colors_rgb_u8=np.zeros((1, 3), dtype=np.uint8),
                    metadata={"worker": "ready"},
                )

        ready_manager = shape_prior_warmup.ShapePriorWarmupManager(
            enabled=True,
            client=ReadyClient(),
        )
        self.assertTrue(ready_manager.maybe_submit(request))
        ready_manager.wait(1.0)
        self.assertIsNotNone(ready_manager.ready_result())
        ready_profile = ready_manager.profile()
        self.assertEqual(
            shape_prior_warmup.SHAPE_PRIOR_STATUS_READY,
            ready_profile["shape_prior_status"],
        )
        self.assertEqual("ready", ready_profile["worker"])

        class FailedClient:
            def request_shape_prior(
                self,
                frame0: shape_prior_warmup.ShapePriorFrame0Request,
            ) -> shape_prior_warmup.ShapePriorResult:
                raise RuntimeError("boom")

        failed_manager = shape_prior_warmup.ShapePriorWarmupManager(
            enabled=True,
            client=FailedClient(),
        )
        self.assertTrue(failed_manager.maybe_submit(request))
        failed_manager.wait(1.0)
        failed_profile = failed_manager.profile()
        self.assertIsNone(failed_manager.ready_result())
        self.assertEqual(
            shape_prior_warmup.SHAPE_PRIOR_STATUS_FAILED,
            failed_profile["shape_prior_status"],
        )
        self.assertEqual("boom", failed_profile["shape_prior_error"])

    def test_removed_demo_v51_shape_prior_cli_flags_are_rejected(self) -> None:
        from demo_v5_1 import main as runner

        parser = runner.build_parser()
        removed_flags = (
            ("--shape-prior-execution", "local-subprocess"),
            ("--shape-prior-start-policy", "after-teardown"),
            ("--shape-prior-worker-futurephystwin-root", "vendor/demo_runtime"),
            ("--shape-prior-worker-warmup-models",),
            ("--shape-prior-worker-debug",),
        )
        for args in removed_flags:
            with self.subTest(args=args):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    parser.parse_args(list(args))
                self.assertEqual(2, error.exception.code)

        parsed = parser.parse_args([])
        self.assertFalse(hasattr(parsed, "shape_prior_execution"))
        self.assertFalse(hasattr(parsed, "shape_prior_start_policy"))
        self.assertFalse(hasattr(parsed, "shape_prior_worker_futurephystwin_root"))
        self.assertFalse(hasattr(parsed, "shape_prior_worker_warmup_models"))
        self.assertFalse(hasattr(parsed, "shape_prior_worker_debug"))

    def test_worker_help_excludes_removed_debug_and_compat_flags(self) -> None:
        completed = subprocess.run(
            [sys.executable, str(ROOT / "demo_v5_1" / "shape_prior_worker.py"), "--help"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        for option in (
            "--futurephystwin-root",
            "--echo-observation",
            "--warmup-models",
            "--debug",
        ):
            with self.subTest(option=option):
                self.assertNotIn(option, completed.stdout)

    def test_shape_prior_uses_single_npz_frame_not_8_frame_protocol(self) -> None:
        from demo_v5_1.shape_prior import ShapePriorFrame0Request
        from demo_v5_1.shape_prior import pack_shape_prior_request
        from demo_v5_1.shape_prior import unpack_shape_prior_request

        request = ShapePriorFrame0Request(
            seq=0,
            source_timestamp_s=None,
            input_source="test",
            depth_backend="test",
            depth_source_internal="test",
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            object_observation_mask=None,
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )
        parts = pack_shape_prior_request(request)
        self.assertEqual(1, len(parts))
        self.assertEqual(0, unpack_shape_prior_request(parts).seq)

        source = (ROOT / "demo_v5_1" / "shape_prior.py").read_text(
            encoding="utf-8"
        )
        self.assertNotIn("PROTOCOL_NAME", source)
        self.assertNotIn("expected 8 frames", source)

    def test_frame0_request_runtime_drops_rgbd_normalization_layer(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior.py").read_text(
            encoding="utf-8"
        )

        for token in (
            "ShapePriorSnapshot",
            "normalize_snapshot",
            "_as_rgb_u8",
            "_as_mask",
            "rgb payload must",
            "mask/depth shapes must match RGB",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)

    def test_shape_prior_sampling_accepts_trimesh_main_path(self) -> None:
        import trimesh

        from demo_v5_1.shape_prior import sample_shape_prior_points

        mesh = trimesh.creation.box(extents=(0.10, 0.10, 0.10))
        reference_points = np.asarray(mesh.sample(600), dtype=np.float32)

        samples = sample_shape_prior_points(
            mesh,
            reference_points,
            target_surface_points=32,
            target_interior_points=16,
            max_dist_m=0.20,
        )

        self.assertGreater(samples.surface_points_m.shape[0], 0)
        self.assertEqual((3,), samples.surface_points_m.shape[1:])
        self.assertEqual((3,), samples.interior_points_m.shape[1:])
        self.assertEqual(
            "sam3d-single-view",
            samples.metadata["shape_prior_sampling_backend"],
        )

    def test_shape_prior_file_drops_legacy_and_fallback_exports(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior.py").read_text(encoding="utf-8")

        for token in (
            "ShapePriorSnapshot",
            "shape_prior_runtime",
            "shape_prior_rpc",
            "single_view_shape_align",
            "single_view_shape_prior_sampling",
            "sample_legacy_single_view_shape_prior_points",
            "SimpleShapeMesh",
            "filter_points_by_nn_distance",
            "_sort_by_reference_distance",
            "_dedupe_points",
        ):
            with self.subTest(token=token):
                self.assertNotIn(token, source)


if __name__ == "__main__":
    unittest.main()
