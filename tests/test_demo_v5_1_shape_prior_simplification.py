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
    def test_shape_prior_path_is_two_local_files(self) -> None:
        expected_files = (
            ROOT / "demo_v5_1" / "shape_prior.py",
            ROOT / "demo_v5_1" / "shape_prior_worker.py",
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

        total_lines = sum(
            len(path.read_text(encoding="utf-8").splitlines())
            for path in expected_files
        )
        self.assertLessEqual(total_lines, 1000)

        for path in (
            ROOT / "demo_v5_1" / "shape_prior.py",
            ROOT / "demo_v5_1" / "shape_prior_worker.py",
            ROOT / "demo_v5_1" / "realtime_dense_track.py",
        ):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("qqtt.demo.shape_prior", source)
                self.assertNotIn("from qqtt.demo import shape_prior", source)
                self.assertNotIn("services.shape_prior_remote", source)

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
