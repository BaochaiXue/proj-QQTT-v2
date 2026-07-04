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


def _command_arg(command: list[str], flag: str) -> str:
    return command[command.index(flag) + 1]


def _command_module(command: list[str]) -> str:
    return _command_arg(command, "-m")


def _write_blank_image(path: Path, shape: tuple[int, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.zeros(shape, dtype=np.uint8)).save(path)


def _write_dummy_shape_prior_final_data(case: Path) -> None:
    case.mkdir(parents=True, exist_ok=True)
    with (case / "final_data.pkl").open("wb") as handle:
        pickle.dump(
            {
                "surface_points": np.zeros((1, 3), dtype=np.float32),
                "interior_points": np.ones((1, 3), dtype=np.float32),
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )


def _fake_segment_image_to_origin_rgba(calls: list[dict[str, object]] | None = None):
    def fake_segment_image_to_origin_rgba(**kwargs):
        if calls is not None:
            calls.append(dict(kwargs))
        _write_blank_image(Path(kwargs["output_path"]), (2, 2, 4))
        return Path(kwargs["output_path"])

    return fake_segment_image_to_origin_rgba


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

        self.assertNotIn('"--single_view_alignment"', source)
        self.assertNotIn("single_view_alignment", source)
        self.assertIn("original PhysTwin alignment flow", source)
        self.assertIn("align_full_vendor_compatible", source)
        self.assertNotIn("align_multiview_vendor_compatible", source)

    def test_shape_prior_align_has_no_single_view_bypass_mode(self) -> None:
        source = (ROOT / "demo_v5_1" / "shape_prior_align.py").read_text(
            encoding="utf-8"
        )

        self.assertNotIn("align_single_view_conservative", source)
        self.assertNotIn("single_view_conservative", source)
        self.assertNotIn("alignment_mode", source)
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

        def fake_run_stage(command: list[str], *, env: dict[str, str]) -> float:
            del env
            commands.append(command)
            module = _command_module(command)
            if module == "demo_v5_1.utils.image_upscale":
                _write_blank_image(
                    Path(_command_arg(command, "--output_path")),
                    (2, 2, 3),
                )
            elif module == "demo_v5_1.shape_prior_sample":
                case = Path(_command_arg(command, "--base_path")) / _command_arg(
                    command,
                    "--case_name",
                )
                _write_dummy_shape_prior_final_data(case)
            return float(len(commands))

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
                    side_effect=_fake_segment_image_to_origin_rgba(segment_calls),
                ),
            ):
                result = client.request_shape_prior(_frame0_request())

            self.assertTrue(result.ready)
            modules = [_command_module(command) for command in commands]
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
                _command_arg(generate_command, "--img_path"),
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

    def test_controller_instance_mode_cli_is_removed(self) -> None:
        from demo_v5_1 import main_data_processing

        parser = main_data_processing.build_parser()
        with (
            contextlib.redirect_stderr(io.StringIO()),
            self.assertRaises(SystemExit) as error,
        ):
            parser.parse_args(["--controller-instance-mode", "single"])
        self.assertEqual(2, error.exception.code)
        self.assertFalse(hasattr(parser.parse_args([]), "controller_instance_mode"))

    def test_edgetam_mask_logit_threshold_cli_is_runtime_parameter(self) -> None:
        from demo_v5_1 import main_data_processing

        parser = main_data_processing.build_parser()
        default_args = parser.parse_args([])
        loose_args = parser.parse_args(["--edgetam-mask-logit-threshold", "-0.5"])

        self.assertEqual(0.0, default_args.edgetam_mask_logit_threshold)
        self.assertEqual(-0.5, loose_args.edgetam_mask_logit_threshold)

        infinite_args = parser.parse_args(["--edgetam-mask-logit-threshold", "inf"])
        with self.assertRaisesRegex(ValueError, "edgetam-mask-logit-threshold"):
            main_data_processing.validate_args(infinite_args)

    def test_edgetam_mask_logit_threshold_controls_binarization(self) -> None:
        from demo_v5_1 import main_data_processing

        output = argparse.Namespace(object_ids=[main_data_processing.OBJECT_ID])
        post_masks = [
            np.asarray(
                [
                    [-0.25, 0.0],
                    [0.10, -0.75],
                ],
                dtype=np.float32,
            )
        ]

        default_masks = main_data_processing.extract_object_masks_from_hf_output(
            output,
            post_masks,
        )
        loose_masks = main_data_processing.extract_object_masks_from_hf_output(
            output,
            post_masks,
            mask_logit_threshold=-0.5,
        )

        self.assertTrue(
            np.array_equal(
                default_masks[main_data_processing.OBJECT_ID],
                np.asarray([[False, False], [True, False]], dtype=bool),
            )
        )
        self.assertTrue(
            np.array_equal(
                loose_masks[main_data_processing.OBJECT_ID],
                np.asarray([[True, True], [True, False]], dtype=bool),
            )
        )

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

        object_mask = np.zeros((4, 4), dtype=bool)
        object_mask[1:3, 1:3] = True
        hand_a_mask = np.zeros((4, 4), dtype=bool)
        hand_a_mask[0:2, 0] = True
        hand_b_mask = np.zeros((4, 4), dtype=bool)
        hand_b_mask[2:4, 3] = True
        args = argparse.Namespace(
            track_mode="controller-object",
            object_prompt="stuffed animal",
            controller_prompt="hand",
            shape_prior_warmup=True,
            device="cuda",
        )

        def fake_run_image_segmentation(**kwargs):
            return {
                "masks_by_label": {
                    "stuffed animal": [object_mask],
                    "hand": [hand_b_mask, hand_a_mask],
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

        self.assertTrue(np.array_equal(object_mask, bundle.object_mask))
        self.assertTrue(np.array_equal(hand_a_mask, bundle.hand_a_mask))
        self.assertTrue(np.array_equal(hand_b_mask, bundle.hand_b_mask))
        self.assertTrue(
            np.array_equal(
                np.logical_or(hand_a_mask, hand_b_mask),
                bundle.controller_mask,
            )
        )
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


class _FakePrewarmWorker:
    """Stands in for a --wait-signal stage subprocess in orchestrator tests."""

    def __init__(self, args: list[str], *, on_go=None) -> None:
        self.args = list(args)
        self.stdin = mock.Mock()
        self.returncode: int | None = None
        self.signals: list[str] = []
        self._on_go = on_go
        self.stdin.write.side_effect = self.signals.append

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int:
        del timeout
        if self._on_go is not None and "GO\n" in self.signals:
            self._on_go()
        self.returncode = 0
        return 0

    def terminate(self) -> None:
        self.returncode = -15

    def kill(self) -> None:
        self.returncode = -9


class DemoV51ShapePriorPrewarmWorkerTests(unittest.TestCase):
    def _client(self, tmpdir: str):
        from demo_v5_1 import shape_prior_warmup

        return shape_prior_warmup.ShapePriorLocalClient(
            case_root=Path(tmpdir),
            object_name="stuffed animal",
            controller_name="hand",
            points_npz=Path(tmpdir) / "points.npz",
            sam31_device="cuda",
        )

    def test_prewarm_spawns_wait_signal_workers_and_close_reaps(self) -> None:
        from demo_v5_1 import shape_prior_warmup

        spawned: list[tuple[list[str], dict[str, str]]] = []

        def fake_popen(command, *, cwd, env, stdin, text):
            del cwd, stdin, text
            spawned.append((list(command), dict(env)))
            return _FakePrewarmWorker(command)

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir)
            with mock.patch.object(
                shape_prior_warmup.subprocess, "Popen", side_effect=fake_popen
            ):
                client.prewarm()
                client.prewarm()  # idempotent: must not respawn

            self.assertEqual(3, len(spawned))
            commands = client._stage_commands()
            for stage, (command, env) in zip(
                shape_prior_warmup.PREWARM_STAGES, spawned
            ):
                self.assertEqual([*commands[stage], "--wait-signal"], command)
                self.assertEqual(
                    client.cuda_visible_devices, env["CUDA_VISIBLE_DEVICES"]
                )

            workers = dict(client._prewarm_workers)
            client.close()
            self.assertEqual({}, client._prewarm_workers)
            for worker in workers.values():
                self.assertIn("EXIT\n", worker.signals)

    def test_request_shape_prior_runs_prewarmed_workers(self) -> None:
        from demo_v5_1 import sam31_image_segmentation
        from demo_v5_1 import shape_prior_warmup

        cold_commands: list[list[str]] = []

        def fake_run_stage(command: list[str], *, env: dict[str, str]) -> float:
            del env
            cold_commands.append(command)
            self.assertEqual("demo_v5_1.shape_prior_sample", _command_module(command))
            case = Path(_command_arg(command, "--base_path")) / _command_arg(
                command,
                "--case_name",
            )
            _write_dummy_shape_prior_final_data(case)
            return 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir)
            commands = client._stage_commands()
            shape_dir = Path(tmpdir) / "shape_prior_frame0" / "shape"

            def write_high_resolution() -> None:
                _write_blank_image(shape_dir / "high_resolution.png", (2, 2, 3))

            workers = {
                shape_prior_warmup.PREWARM_STAGE_UPSCALE: _FakePrewarmWorker(
                    commands[shape_prior_warmup.PREWARM_STAGE_UPSCALE],
                    on_go=write_high_resolution,
                ),
                shape_prior_warmup.PREWARM_STAGE_GENERATE: _FakePrewarmWorker(
                    commands[shape_prior_warmup.PREWARM_STAGE_GENERATE]
                ),
                shape_prior_warmup.PREWARM_STAGE_ALIGN: _FakePrewarmWorker(
                    commands[shape_prior_warmup.PREWARM_STAGE_ALIGN]
                ),
            }
            client._prewarm_workers.update(workers)

            with (
                mock.patch.object(
                    shape_prior_warmup,
                    "_run_stage",
                    side_effect=fake_run_stage,
                ),
                mock.patch.object(
                    sam31_image_segmentation,
                    "segment_image_to_origin_rgba",
                    side_effect=_fake_segment_image_to_origin_rgba(),
                ),
            ):
                result = client.request_shape_prior(_frame0_request())

            self.assertTrue(result.ready)
            # Only the sample stage remains a cold subprocess.
            self.assertEqual(1, len(cold_commands))
            self.assertEqual(
                list(shape_prior_warmup.PREWARM_STAGES),
                result.metadata["shape_prior_prewarmed_stages"],
            )
            self.assertEqual({}, client._prewarm_workers)
            for worker in workers.values():
                self.assertIn("GO\n", worker.signals)
                worker.stdin.close.assert_called_once()

    def test_request_fails_fast_when_prewarmed_worker_died(self) -> None:
        from demo_v5_1 import sam31_image_segmentation
        from demo_v5_1 import shape_prior_warmup

        with tempfile.TemporaryDirectory() as tmpdir:
            client = self._client(tmpdir)
            dead = _FakePrewarmWorker(["upscale-cmd"])
            dead.returncode = 3
            client._prewarm_workers[shape_prior_warmup.PREWARM_STAGE_UPSCALE] = dead
            with (
                mock.patch.object(
                    shape_prior_warmup,
                    "_run_stage",
                    side_effect=AssertionError("cold path must not run"),
                ),
                mock.patch.object(
                    sam31_image_segmentation,
                    "segment_image_to_origin_rgba",
                    side_effect=AssertionError("segment must not run"),
                ),
                self.assertRaisesRegex(RuntimeError, "exited before GO"),
            ):
                client.request_shape_prior(_frame0_request())

    def test_stage_prewarm_wait_for_go_protocol(self) -> None:
        from demo_v5_1.utils import stage_prewarm

        for stdin_text, expected_run in (("GO\n", True), ("EXIT\n", False), ("", False)):
            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(stage_prewarm.sys, "stdin", io.StringIO(stdin_text)),
            ):
                self.assertIs(
                    expected_run,
                    stage_prewarm.wait_for_go("test-stage"),
                )

        with (
            contextlib.redirect_stdout(io.StringIO()),
            mock.patch.object(stage_prewarm.sys, "stdin", io.StringIO("BOGUS\n")),
            self.assertRaisesRegex(ValueError, "unexpected signal"),
        ):
            stage_prewarm.wait_for_go("test-stage")


if __name__ == "__main__":
    unittest.main()
