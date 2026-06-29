from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _frame0_request():
    from demo_v5_1 import shape_prior_warmup

    rgb = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 255]],
        ],
        dtype=np.uint8,
    )
    return shape_prior_warmup.ShapePriorFrame0Request(
        seq=7,
        source_timestamp_s=12.5,
        input_source="test",
        depth_backend="test",
        depth_source_internal="test",
        rgb_u8=rgb,
        object_mask=np.array([[True, False], [True, False]], dtype=bool),
        object_observation_mask=None,
        controller_mask=np.array([[False, True], [False, False]], dtype=bool),
        depth_color_m=np.ones((2, 2), dtype=np.float32),
        k_color=np.eye(3, dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
    )


class DemoV51ShapePriorSimplificationTests(unittest.TestCase):
    def test_shape_prior_pipeline_modules_are_local_files(self) -> None:
        expected_files = (
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "shape_prior_generate.py",
            ROOT / "demo_v5_1" / "shape_prior_align.py",
            ROOT / "demo_v5_1" / "shape_prior_match_pairs.py",
            ROOT / "demo_v5_1" / "shape_prior_sample.py",
            ROOT / "demo_v5_1" / "runtime_warmup.py",
        )
        for path in expected_files:
            with self.subTest(path=path.name):
                self.assertTrue(path.is_file())

        removed_files = (
            ROOT / "demo_v5_1" / "shape_prior.py",
            ROOT / "demo_v5_1" / "shape_prior_worker.py",
            ROOT / "demo_v5_1" / "shape_prior_runtime.py",
            ROOT / "demo_v5_1" / "shape_prior_rpc.py",
            ROOT / "demo_v5_1" / "single_view_shape_prior_sampling.py",
            ROOT / "demo_v5_1" / "single_view_shape_align.py",
        )
        for path in removed_files:
            with self.subTest(path=path.name):
                self.assertFalse(path.exists())

        for path in expected_files + (ROOT / "demo_v5_1" / "realtime_dense_track.py",):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("qqtt.demo.shape_prior", source)
                self.assertNotIn("services.shape_prior_remote", source)

    def test_shape_prior_align_uses_v51_match_pairs(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior_align.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            "from demo_v5_1.shape_prior_match_pairs import image_pair_matching",
            source,
        )
        self.assertNotIn("from match_pairs import image_pair_matching", source)
        self.assertNotIn("range(3)", source)

    def test_shape_prior_warmup_writes_single_camera_case(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = shape_prior_warmup.write_shape_prior_case(
                _frame0_request(),
                case_root=Path(tmpdir),
                case_name="case",
                controller_name="hand",
            )
            case = Path(paths["case"])

            self.assertTrue((case / "color" / "0" / "0.png").is_file())
            self.assertTrue((case / "shape" / "sam3d_input_rgba.png").is_file())
            self.assertTrue((case / "mask" / "0" / "0" / "0.png").is_file())
            self.assertTrue((case / "mask" / "processed_masks.pkl").is_file())
            pcd = np.load(case / "pcd" / "0.npz")
            self.assertEqual((1, 2, 2, 3), pcd["points"].shape)

            rgba = np.asarray(Image.open(case / "shape" / "sam3d_input_rgba.png"))
            self.assertEqual(255, int(rgba[0, 0, 3]))
            self.assertEqual(0, int(rgba[0, 1, 3]))

            with (case / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            self.assertEqual(1, len(processed_masks[0]))
            mask_info = json.loads(
                (case / "mask" / "mask_info_0.json").read_text(encoding="utf-8")
            )
            self.assertEqual({"0": "stuffed animal", "1": "hand"}, mask_info)

    def test_shape_prior_warmup_manager_ready_and_failed_profiles(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        class ReadyClient:
            def request_shape_prior(
                self,
                frame0: shape_prior_warmup.ShapePriorFrame0Request,
            ) -> shape_prior_warmup.ShapePriorResult:
                return shape_prior_warmup.ShapePriorResult(
                    seq=int(frame0.seq),
                    source_seq=int(frame0.seq),
                    source_timestamp_s=frame0.source_timestamp_s,
                    status=shape_prior_warmup.STATUS_READY,
                    points_m=np.ones((1, 3), dtype=np.float32),
                    colors_rgb_u8=np.zeros((1, 3), dtype=np.uint8),
                    metadata={"worker": "ready"},
                )

        ready_manager = shape_prior_warmup.ShapePriorWarmupManager(
            enabled=True,
            client=ReadyClient(),
        )
        self.assertTrue(ready_manager.maybe_submit(_frame0_request()))
        ready_manager.wait(1.0)
        self.assertIsNotNone(ready_manager.ready_result())
        ready_profile = ready_manager.profile()
        self.assertEqual(
            shape_prior_warmup.STATUS_READY,
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
        self.assertTrue(failed_manager.maybe_submit(_frame0_request()))
        failed_manager.wait(1.0)
        failed_profile = failed_manager.profile()
        self.assertIsNone(failed_manager.ready_result())
        self.assertEqual(
            shape_prior_warmup.STATUS_FAILED,
            failed_profile["shape_prior_status"],
        )
        self.assertEqual("boom", failed_profile["shape_prior_error"])

    def test_shape_prior_sampling_uses_origin_counts(self) -> None:
        from demo_v5_1 import data_process_chunk_writer
        from demo_v5_1 import shape_prior_sample
        from demo_v5_1 import shape_prior_warmup

        self.assertEqual(1024, shape_prior_sample.DEFAULT_SURFACE_POINTS)
        self.assertEqual(10000, shape_prior_sample.INTERIOR_CANDIDATE_POINTS)
        self.assertEqual(1024, shape_prior_warmup.DEFAULT_SURFACE_POINT_COUNT)
        self.assertFalse(hasattr(shape_prior_sample, "DEFAULT_INTERIOR_POINTS"))
        self.assertFalse(hasattr(shape_prior_warmup, "DEFAULT_INTERIOR_POINT_COUNT"))

        parser = shape_prior_sample.build_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            parser.parse_args(
                [
                    "--base_path",
                    "x",
                    "--case_name",
                    "y",
                    "--num_interior_points",
                    "1000",
                ]
            )
        self.assertEqual(2, error.exception.code)

        metrics = data_process_chunk_writer.DATA_PROCESS_SAM3D_METRICS
        self.assertEqual(1024, metrics["shape_prior_target_surface_points"])
        self.assertEqual(10000, metrics["shape_prior_interior_candidate_points"])
        self.assertNotIn("shape_prior_target_interior_points", metrics)

        quality = data_process_chunk_writer._quality_manifest_fields(
            {
                "object_points": np.zeros((1, 1, 3), dtype=np.float64),
                "controller_points": np.zeros((1, 1, 3), dtype=np.float64),
                "surface_points": np.zeros((1, 3), dtype=np.float64),
                "interior_points": np.zeros((1, 3), dtype=np.float64),
            },
            {"controller_mask": np.ones((1,), dtype=bool)},
        )
        self.assertTrue(quality["shape_prior_complete"])
        self.assertNotIn("shape_prior_target_counts_met", quality)

    def test_shape_prior_has_no_backend_default(self) -> None:
        from demo_v5_1 import data_process_chunk_writer
        from demo_v5_1 import shape_prior_warmup

        self.assertFalse(hasattr(shape_prior_warmup, "SHAPE_BACKEND_SAM3D_OBJECTS"))
        self.assertNotIn("shape_backend", shape_prior_warmup.default_profile(enabled=True))
        self.assertNotIn("shape_backend", shape_prior_warmup.default_profile(enabled=False))
        self.assertNotIn(
            "shape_prior_sampling_backend",
            data_process_chunk_writer.DATA_PROCESS_SAM3D_METRICS,
        )

        checked_files = (
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "realtime_dense_track.py",
            ROOT / "demo_v5_1" / "data_process_chunk_writer.py",
        )
        for path in checked_files:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("shape_backend", source)
                self.assertNotIn("SHAPE_BACKEND", source)

    def test_shape_prior_status_names_are_short(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        expected_statuses = {
            "DISABLED": "disabled",
            "PENDING": "pending",
            "RUNNING": "running",
            "READY": "ready",
            "FAILED": "failed",
        }
        for name, value in expected_statuses.items():
            with self.subTest(name=name):
                self.assertEqual(value, getattr(shape_prior_warmup, f"STATUS_{name}"))
                self.assertFalse(
                    hasattr(shape_prior_warmup, f"SHAPE_PRIOR_STATUS_{name}")
                )

    def test_shape_prior_controller_name_has_no_warmup_default(self) -> None:
        from demo_v5_1 import realtime_dense_track
        from demo_v5_1 import shape_prior_warmup

        self.assertFalse(
            hasattr(shape_prior_warmup, "DEFAULT_SHAPE_PRIOR_CONTROLLER_NAME")
        )
        self.assertEqual("shape_prior_frame0", shape_prior_warmup.CASE_NAME)
        self.assertFalse(hasattr(shape_prior_warmup, "DEFAULT_SHAPE_PRIOR_CASE_NAME"))

        with self.assertRaisesRegex(ValueError, "controller_name"):
            shape_prior_warmup.ShapePriorLocalClient(
                case_root=ROOT,
                controller_name="",
            )

        parser = realtime_dense_track.build_parser()
        args = parser.parse_args(["--shape-prior-warmup"])
        self.assertIsNone(args.shape_prior_controller_name)
        with self.assertRaisesRegex(ValueError, "--shape-prior-controller-name"):
            realtime_dense_track.validate_args(args)

    def test_removed_demo_v51_shape_prior_cli_flags_are_rejected(self) -> None:
        from demo_v5_1 import main as runner

        parser = runner.build_parser()
        removed_flags = (
            ("--realtime-gpu-mode", "single"),
            ("--warmup-gpu-mode", "dual"),
            ("--" + "-".join(("camera", "cuda", "visible", "devices")), "0"),
            (
                "--" + "-".join(("shape", "prior", "cuda", "visible", "devices")),
                "1",
            ),
            ("--shape-prior-worker-mode", "managed"),
            (
                "--"
                + "-".join(
                    ("shape", "prior", "worker", "future" + "phystwin", "root")
                ),
                "vendor/demo_runtime",
            ),
            ("--shape-prior-worker-warmup-models",),
            ("--shape-prior-worker-debug",),
            ("--shape-prior-endpoint", "tcp://127.0.0.1:7100"),
            ("--shape-prior-device", "cuda:0"),
            ("--" + "-".join(("optimization", "mode")), "continuous"),
            (
                "--" + "-".join(("realtime", "phystwin", "root")),
                "_".join(("realtime", "phystwin")),
            ),
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
        self.assertFalse(hasattr(parsed, "realtime_gpu_mode"))
        self.assertFalse(hasattr(parsed, "warmup_gpu_mode"))
        self.assertFalse(hasattr(parsed, "_".join(("camera", "cuda", "visible", "devices"))))
        self.assertFalse(
            hasattr(parsed, "_".join(("shape", "prior", "cuda", "visible", "devices")))
        )
        self.assertFalse(hasattr(parsed, "shape_prior_worker_mode"))
        self.assertFalse(hasattr(parsed, "shape_prior_endpoint"))
        self.assertFalse(hasattr(parsed, "shape_prior_device"))
        self.assertFalse(hasattr(parsed, "_".join(("optimization", "mode"))))
        self.assertFalse(hasattr(parsed, "_".join(("realtime", "phystwin", "root"))))


if __name__ == "__main__":
    unittest.main()
