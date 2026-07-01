from __future__ import annotations

import argparse
import contextlib
from dataclasses import replace
import io
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]


def _frame0_request():
    from demo_v5_1 import shape_prior_warmup

    height = 12
    width = 12
    rgb = np.zeros((height, width, 3), dtype=np.uint8)
    rgb[:, :, 0] = 255
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[:6, :7] = True
    controller_mask = np.zeros((height, width), dtype=bool)
    controller_mask[6:12, :7] = True
    depth_m = np.ones((height, width), dtype=np.float32)
    return shape_prior_warmup.ShapePriorFrame0Request(
        seq=7,
        source_timestamp_s=12.5,
        input_source="test",
        depth_backend="test",
        depth_source_internal="test",
        rgb_u8=rgb,
        object_mask=object_mask,
        object_observation_mask=None,
        controller_mask=controller_mask,
        depth_color_m=depth_m,
        k_color=np.array(
            [[10000.0, 0.0, 0.0], [0.0, 10000.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        ),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
    )


def _frame0_request_with_isolated_object_outlier():
    frame0 = _frame0_request()
    object_mask = np.asarray(frame0.object_mask, dtype=bool).copy()
    depth_m = np.asarray(frame0.depth_color_m, dtype=np.float32).copy()
    object_mask[-1, -1] = True
    depth_m[-1, -1] = np.float32(2.0)
    return replace(
        frame0,
        object_mask=object_mask,
        depth_color_m=depth_m,
    )


class DemoV51ShapePriorSimplificationTests(unittest.TestCase):
    def test_shape_prior_pipeline_modules_are_local_files(self) -> None:
        expected_files = (
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "shape_prior_generate.py",
            ROOT / "demo_v5_1" / "shape_prior_align.py",
            ROOT / "demo_v5_1" / "shape_prior_match_pairs.py",
            ROOT / "demo_v5_1" / "shape_prior_sample.py",
            ROOT / "demo_v5_1" / "sam31_image_segmentation.py",
            ROOT / "demo_v5_1" / "main_warmup.py",
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

        for path in expected_files + (ROOT / "demo_v5_1" / "main_data_processing.py",):
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("qqtt.demo.shape_prior", source)
                self.assertNotIn("services.shape_prior_remote", source)
                self.assertNotIn("scripts.harness.support.sam31_mask_helper", source)

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

    def test_shape_prior_align_defaults_to_origin_full_flow(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior_align.py").read_text(
            encoding="utf-8"
        )

        self.assertIn('"--single_view_alignment"', source)
        self.assertIn('choices=("full", "conservative")', source)
        self.assertIn('default="full"', source)
        self.assertIn("original PhysTwin alignment flow", source)
        self.assertIn("align_full_vendor_compatible", source)
        self.assertNotIn("align_multiview_vendor_compatible", source)

    def test_shape_prior_align_conservative_mode_is_opt_in(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior_align.py").read_text(
            encoding="utf-8"
        )

        self.assertIn(
            'if camera_count == 1 and args.single_view_alignment == "conservative":',
            source,
        )
        self.assertIn("align_single_view_conservative", source)
        self.assertIn("pcd camera count does not match processed mask count", source)
        self.assertNotIn("unsupported two-camera shape-prior alignment mode", source)

    def test_shape_prior_warmup_writes_single_camera_case(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = shape_prior_warmup.write_shape_prior_case(
                _frame0_request(),
                case_root=Path(tmpdir),
                case_name="case",
                object_name="stuffed animal",
                controller_name="hand",
            )
            case = Path(paths["case"])

            self.assertTrue((case / "color" / "0" / "0.png").is_file())
            self.assertTrue((case / "shape").is_dir())
            self.assertFalse((case / "shape" / "sam3d_input_rgba.png").exists())
            self.assertTrue((case / "mask" / "0" / "0" / "0.png").is_file())
            self.assertTrue((case / "mask" / "processed_masks.pkl").is_file())
            pcd = np.load(case / "pcd" / "0.npz")
            self.assertEqual((1, 12, 12, 3), pcd["points"].shape)
            self.assertEqual((1, 12, 12), pcd["masks"].shape)

            with (case / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            self.assertEqual(1, len(processed_masks[0]))
            mask_info = json.loads(
                (case / "mask" / "mask_info_0.json").read_text(encoding="utf-8")
            )
            self.assertEqual({"0": "stuffed animal", "1": "hand"}, mask_info)
            metadata = json.loads(
                (case / "metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual("negative", metadata["table_z_above_direction"])

    def test_shape_prior_warmup_radius_cleans_processed_masks_not_dense_pcd(
        self,
    ) -> None:
        from demo_v5_1 import shape_prior_warmup

        with tempfile.TemporaryDirectory() as tmpdir:
            paths = shape_prior_warmup.write_shape_prior_case(
                _frame0_request_with_isolated_object_outlier(),
                case_root=Path(tmpdir),
                case_name="case",
                object_name="stuffed animal",
                controller_name="hand",
            )
            case = Path(paths["case"])

            pcd = np.load(case / "pcd" / "0.npz")
            self.assertEqual((1, 12, 12, 3), pcd["points"].shape)
            with (case / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            processed_object = processed_masks[0][0]["object"]
            self.assertTrue(np.all(processed_object[:6, :7]))
            self.assertFalse(processed_object[-1, -1])

    def test_shape_prior_warmup_documents_origin_radius_mask_parity(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior_warmup.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("data_process_origin/data_process_mask.py", source)
        self.assertIn("_apply_radius_outlier_to_mask_frame", source)
        self.assertIn("radius_m=0.01", source)
        self.assertIn("nb_points=40", source)
        self.assertIn("processed_masks.pkl", source)

    def test_main_data_processing_keeps_filtered_object_yx_aligned(self) -> None:
        source = (ROOT / "demo_v5_1" / "main_data_processing.py").read_text(
            encoding="utf-8"
        )

        self.assertIn("render_object_yx = filter_output.object_yx", source)
        self.assertIn("render_object_yx = latest.object_yx", source)

    def test_sam31_image_segmentation_matches_origin_rgba_semantics(self) -> None:
        from demo_v5_1 import sam31_image_segmentation

        rgb = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        mask = np.array([[True, False], [False, True]], dtype=bool)

        def fake_run_image_segmentation(**_kwargs):
            return {
                "masks_by_label": {"stuffed animal": [mask]},
                "parsed_prompts": ["stuffed animal"],
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            input_path = tmp / "high_resolution.png"
            output_path = tmp / "masked_image.png"
            Image.fromarray(rgb).save(input_path)

            with mock.patch.object(
                sam31_image_segmentation,
                "run_image_segmentation",
                side_effect=fake_run_image_segmentation,
            ):
                sam31_image_segmentation.segment_image_to_origin_rgba(
                    img_path=input_path,
                    text_prompt="stuffed animal",
                    output_path=output_path,
                )

            rgba = np.asarray(Image.open(output_path).convert("RGBA"))
            np.testing.assert_array_equal(rgb[0, 0], rgba[0, 0, :3])
            np.testing.assert_array_equal([0, 0, 0], rgba[0, 1, :3])
            self.assertEqual(255, int(rgba[0, 0, 3]))
            self.assertEqual(0, int(rgba[0, 1, 3]))
            self.assertEqual(255, int(rgba[1, 1, 3]))

    def test_shape_prior_client_uses_origin_input_command_chain(self) -> None:
        from demo_v5_1 import sam31_image_segmentation
        from demo_v5_1 import shape_prior_warmup

        commands: list[list[str]] = []
        segment_calls: list[dict[str, object]] = []

        def arg_value(command: list[str], flag: str) -> str:
            return command[command.index(flag) + 1]

        def fake_run_stage(command: list[str], *, env: dict[str, str]) -> float:
            del env
            commands.append(command)
            module = command[command.index("-m") + 1]
            if module == "demo_v5_1.utils.image_upscale":
                Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                    arg_value(command, "--output_path")
                )
            elif module == "demo_v5_1.shape_prior_sample":
                case = Path(arg_value(command, "--base_path")) / arg_value(
                    command,
                    "--case_name",
                )
                with (case / "final_data.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "surface_points": np.zeros((1, 3), dtype=np.float32),
                            "interior_points": np.ones((1, 3), dtype=np.float32),
                        },
                        handle,
                        protocol=pickle.HIGHEST_PROTOCOL,
                    )
            return float(len(commands))

        def fake_segment_image_to_origin_rgba(**kwargs):
            segment_calls.append(dict(kwargs))
            Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(
                kwargs["output_path"]
            )
            return Path(kwargs["output_path"])

        with tempfile.TemporaryDirectory() as tmpdir:
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=Path(tmpdir),
                object_name="stuffed animal",
                controller_name="hand",
                points_npz=Path(tmpdir) / "points.npz",
                sam31_device="cuda",
            )
            with (
                mock.patch.object(
                    shape_prior_warmup,
                    "_run_stage",
                    side_effect=fake_run_stage,
                ),
                mock.patch.object(
                    sam31_image_segmentation,
                    "segment_image_to_origin_rgba",
                    side_effect=fake_segment_image_to_origin_rgba,
                ),
            ):
                result = client.request_shape_prior(_frame0_request())

            self.assertTrue(result.ready)
            modules = [command[command.index("-m") + 1] for command in commands]
            self.assertEqual(
                [
                    "demo_v5_1.utils.image_upscale",
                    "demo_v5_1.shape_prior_generate",
                    "demo_v5_1.shape_prior_align",
                    "demo_v5_1.shape_prior_sample",
                ],
                modules,
            )
            self.assertEqual(1, len(segment_calls))
            self.assertEqual(
                Path(tmpdir) / "shape_prior_frame0" / "shape" / "high_resolution.png",
                Path(str(segment_calls[0]["img_path"])),
            )
            self.assertEqual("stuffed animal", segment_calls[0]["text_prompt"])
            self.assertEqual("cuda", segment_calls[0]["device"])
            self.assertTrue(segment_calls[0]["reuse_model"])

            generate_command = commands[1]
            self.assertEqual(
                str(Path(tmpdir) / "shape_prior_frame0" / "shape" / "masked_image.png"),
                arg_value(generate_command, "--img_path"),
            )
            self.assertFalse(
                (Path(tmpdir) / "shape_prior_frame0" / "shape" / "sam3d_input_rgba.png")
                .exists()
            )
            self.assertNotIn(
                "sam3d_input_rgba",
                "\n".join(" ".join(command) for command in commands),
            )

    def test_shape_prior_points_npz_contains_only_prior_points(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        surface = np.asarray([[1.0, 2.0, 3.0]], dtype=np.float32)
        interior = np.asarray([[4.0, 5.0, 6.0], [7.0, 8.0, 9.0]], dtype=np.float32)
        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / "shape_prior" / "points.npz"
            shape_prior_warmup.write_shape_prior_points_npz(
                output_path,
                surface_points=surface,
                interior_points=interior,
            )

            data = np.load(output_path)
            np.testing.assert_array_equal(surface, data["surface_points"])
            np.testing.assert_array_equal(interior, data["interior_points"])
            np.testing.assert_array_equal(
                np.concatenate([surface, interior], axis=0),
                data["points"],
            )

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
        from demo_v5_1 import chunk_data_payload
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

        metrics = chunk_data_payload.DATA_PROCESS_SAM3D_METRICS
        self.assertEqual(1024, metrics["shape_prior_target_surface_points"])
        self.assertEqual(10000, metrics["shape_prior_interior_candidate_points"])
        self.assertNotIn("shape_prior_target_interior_points", metrics)

        quality = chunk_data_payload._quality_manifest_fields(
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
        from demo_v5_1 import chunk_data_payload
        from demo_v5_1 import shape_prior_warmup

        self.assertFalse(hasattr(shape_prior_warmup, "SHAPE_BACKEND_SAM3D_OBJECTS"))
        self.assertNotIn(
            "shape_backend", shape_prior_warmup.default_profile(enabled=True)
        )
        self.assertNotIn(
            "shape_backend", shape_prior_warmup.default_profile(enabled=False)
        )
        self.assertNotIn(
            "shape_prior_sampling_backend",
            chunk_data_payload.DATA_PROCESS_SAM3D_METRICS,
        )

        checked_files = (
            ROOT / "demo_v5_1" / "shape_prior_warmup.py",
            ROOT / "demo_v5_1" / "main_data_processing.py",
            ROOT / "demo_v5_1" / "chunk_data_payload.py",
        )
        for path in checked_files:
            source = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("shape_backend", source)
                self.assertNotIn("SHAPE_BACKEND", source)

    def test_table_z_above_direction_is_fixed_to_origin_convention(self) -> None:
        from demo_v5_1 import main_data_processing

        main_source = (ROOT / "demo_v5_1" / "main_data_processing.py").read_text(
            encoding="utf-8"
        )
        warmup_source = (ROOT / "demo_v5_1" / "shape_prior_warmup.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("--table-z-above-direction", main_source)
        self.assertIn('TABLE_Z_ABOVE_DIRECTION = "negative"', main_source)
        self.assertIn('TABLE_Z_ABOVE_DIRECTION = "negative"', warmup_source)

        parser = main_data_processing.build_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            parser.parse_args(["--table-z-above-direction", "positive"])
        self.assertEqual(2, error.exception.code)
        self.assertFalse(hasattr(parser.parse_args([]), "table_z_above_direction"))

    def test_sam31_runtime_release_uses_named_module_api(self) -> None:
        main_warmup_source = (ROOT / "demo_v5_1" / "main_warmup.py").read_text(
            encoding="utf-8"
        )
        sam31_source = (
            ROOT / "demo_v5_1" / "sam31_image_segmentation.py"
        ).read_text(encoding="utf-8")

        self.assertNotIn("sys.modules.get", main_warmup_source)
        self.assertNotIn("helper =", main_warmup_source)
        self.assertNotIn("getattr(helper", main_warmup_source)
        self.assertIn(
            "release_sam31_image_segmentation_runtime",
            main_warmup_source,
        )
        self.assertIn(
            "def release_sam31_image_segmentation_runtime",
            sam31_source,
        )

    def test_shape_prior_keeps_sam31_first_frame_runtime_cached(self) -> None:
        from demo_v5_1 import main_warmup
        from demo_v5_1 import sam31_image_segmentation

        mask = np.array([[True, False], [False, True]], dtype=bool)
        args = argparse.Namespace(
            track_mode="controller-object",
            object_prompt="stuffed animal",
            controller_prompt="hand",
            controller_instance_mode="single",
            shape_prior_warmup=True,
            device="cuda",
        )

        def fake_run_image_segmentation(**kwargs):
            return {
                "masks_by_label": {
                    "stuffed animal": [mask],
                    "hand": [~mask],
                },
                "timing_ms": {},
            }

        with (
            mock.patch.object(
                sam31_image_segmentation,
                "run_image_segmentation",
                side_effect=fake_run_image_segmentation,
            ) as run_mock,
            mock.patch.object(
                main_warmup,
                "trim_sam31_cuda_allocator",
                return_value=1.0,
            ) as trim_mock,
            mock.patch.object(
                main_warmup,
                "release_sam31_runtime_resources",
                return_value=1.0,
            ) as release_mock,
        ):
            bundle = main_warmup.run_sam31_first_frame_mask_bundle(
                np.zeros((2, 2, 3), dtype=np.uint8),
                args,
            )

        self.assertTrue(np.array_equal(mask, bundle.object_mask))
        self.assertTrue(run_mock.call_args.kwargs["reuse_model"])
        trim_mock.assert_called_once_with("cuda")
        release_mock.assert_not_called()

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
        from demo_v5_1 import main_data_processing
        from demo_v5_1 import shape_prior_warmup

        self.assertFalse(
            hasattr(shape_prior_warmup, "DEFAULT_SHAPE_PRIOR_CONTROLLER_NAME")
        )
        self.assertEqual("shape_prior_frame0", shape_prior_warmup.CASE_NAME)
        self.assertFalse(hasattr(shape_prior_warmup, "DEFAULT_SHAPE_PRIOR_CASE_NAME"))

        with self.assertRaisesRegex(ValueError, "controller_name"):
            shape_prior_warmup.ShapePriorLocalClient(
                case_root=ROOT,
                object_name="stuffed animal",
                controller_name="",
            )

        parser = main_data_processing.build_parser()
        args = parser.parse_args(["--shape-prior-warmup"])
        self.assertIsNone(args.shape_prior_controller_name)
        with self.assertRaisesRegex(ValueError, "--shape-prior-controller-name"):
            main_data_processing.validate_args(args)

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
                + "-".join(("shape", "prior", "worker", "future" + "phystwin", "root")),
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
        self.assertFalse(
            hasattr(parsed, "_".join(("camera", "cuda", "visible", "devices")))
        )
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
