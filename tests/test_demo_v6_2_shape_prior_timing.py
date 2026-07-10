"""Focused tests for Demo v6.2 shape-prior timing instrumentation."""

from __future__ import annotations

import tempfile
import pickle
import time
from types import SimpleNamespace
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

from demo_v6_2 import mdp_demo_segwarmup
from demo_v6_2 import sam31_image_segmentation
from demo_v6_2 import shape_prior_generate
from demo_v6_2 import shape_prior_timing
from demo_v6_2 import shape_prior_warmup


def _frame0_request(*, with_runtime_timing: bool = True):
    now_s = time.perf_counter()
    timing = {
        "warmup_runtime_start_perf_s": now_s - 0.040,
        "frame_receive_perf_s": now_s - 0.030,
        "frame_mask_ready_perf_s": now_s - 0.020,
        "frame_pcd_ready_perf_s": now_s - 0.010,
    }
    if not with_runtime_timing:
        timing = {key: None for key in timing}
    return shape_prior_warmup.ShapePriorFrame0Request(
        seq=0,
        source_timestamp_s=12.5,
        input_source="fake-live",
        depth_backend="native-realsense",
        depth_source_internal="realsense",
        rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
        object_mask=np.ones((2, 2), dtype=bool),
        object_observation_mask=np.ones((2, 2), dtype=bool),
        controller_mask=np.ones((2, 2), dtype=bool),
        depth_color_m=np.ones((2, 2), dtype=np.float32),
        k_color=np.eye(3, dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
        frame0_pipeline_timing_ms={"mask_ms": 3.0, "pcd_ms": 2.0},
        frame0_perception_profile={"edgetam_runtime_init": {"total_ms": 8.0}},
        **timing,
    )


class ShapePriorTimingHelperTests(unittest.TestCase):
    def test_valid_analysis_ranks_stages_and_reports_unattributed_time(self) -> None:
        entries = [
            {
                "stage": "write_case",
                "start_offset_ms": 0.0,
                "end_offset_ms": 5.0,
                "duration_ms": 5.0,
            },
            {
                "stage": "generate",
                "start_offset_ms": 6.0,
                "end_offset_ms": 15.0,
                "duration_ms": 9.0,
            },
        ]

        analysis = shape_prior_timing.build_critical_path_analysis(
            entries,
            total_ms=20.0,
        )

        self.assertEqual(
            shape_prior_timing.SHAPE_PRIOR_TIMING_SCHEMA_VERSION,
            analysis["schema_version"],
        )
        self.assertEqual(14.0, analysis["accounted_ms"])
        self.assertEqual(6.0, analysis["unattributed_ms"])
        self.assertEqual("generate", analysis["bottleneck"]["stage"])
        self.assertEqual(
            ["generate", "write_case"],
            [item["stage"] for item in analysis["ranking"]],
        )
        self.assertAlmostEqual(45.0, analysis["ranking"][0]["share_percent"])

    def test_analysis_rejects_negative_timing(self) -> None:
        with self.assertRaisesRegex(ValueError, "finite non-negative"):
            shape_prior_timing.build_critical_path_analysis(
                [
                    {
                        "stage": "generate",
                        "start_offset_ms": 0.0,
                        "end_offset_ms": 1.0,
                        "duration_ms": -1.0,
                    }
                ],
                total_ms=1.0,
            )

    def test_completed_stage_profile_rejects_wrong_stage(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            profile_path = Path(tmpdir) / "generate.json"
            shape_prior_timing.write_stage_profile(
                profile_path,
                stage="generate",
                status=shape_prior_timing.STAGE_PROFILE_STATUS_COMPLETED,
                execution_mode="cold",
                timing_ms={"model_load_ms": 2.0, "total_ms": 3.0},
            )

            with self.assertRaisesRegex(ValueError, "stage mismatch"):
                shape_prior_timing.load_completed_stage_profile(
                    profile_path,
                    expected_stage="align",
                )

    def test_completed_stage_profile_requires_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            missing = Path(tmpdir) / "missing.json"
            with self.assertRaisesRegex(FileNotFoundError, "did not write"):
                shape_prior_timing.load_completed_stage_profile(
                    missing,
                    expected_stage="sample",
                )


class Sam31ExportTimingTests(unittest.TestCase):
    def test_segment_image_returns_inference_and_export_timing(self) -> None:
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        raw_bgr = np.asarray(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        masks = [
            np.asarray([[True, False], [False, False]]),
            np.asarray([[False, True], [False, False]]),
        ]
        inference_timing = {
            "cache_hit": True,
            "model_load_ms": 0.0,
            "set_image_ms": 12.0,
            "total_ms": 25.0,
        }
        segmentation_result = {
            "masks_by_label": {"stuffed animal": masks},
            "timing_ms": inference_timing,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / "input.png"
            output_path = Path(tmpdir) / "masked.png"
            Image.fromarray(rgb).save(input_path)
            with (
                mock.patch.object(
                    sam31_image_segmentation,
                    "run_image_segmentation",
                    return_value=segmentation_result,
                ) as run_segmentation,
                mock.patch.object(
                    sam31_image_segmentation.cv2,
                    "imread",
                    return_value=raw_bgr,
                ),
                mock.patch.object(
                    sam31_image_segmentation.cv2,
                    "imwrite",
                    return_value=True,
                ) as write_image,
                mock.patch.object(
                    sam31_image_segmentation.time,
                    "perf_counter",
                    side_effect=[1.0, 1.1, 1.3, 1.4, 1.7, 1.8, 2.0, 2.2],
                ),
            ):
                result_path, timing = (
                    sam31_image_segmentation.segment_image_to_origin_rgba(
                        img_path=input_path,
                        text_prompt="stuffed animal",
                        output_path=output_path,
                        device="cuda",
                        reuse_model=True,
                    )
                )

        self.assertEqual(output_path, result_path)
        self.assertEqual(inference_timing, timing["inference"])
        self.assertAlmostEqual(200.0, timing["input_read_ms"])
        self.assertAlmostEqual(300.0, timing["mask_union_ms"])
        self.assertAlmostEqual(200.0, timing["output_write_ms"])
        self.assertAlmostEqual(1200.0, timing["total_ms"])
        run_segmentation.assert_called_once()
        self.assertTrue(run_segmentation.call_args.kwargs["reuse_model"])

        written_rgba = write_image.call_args.args[1]
        np.testing.assert_array_equal(written_rgba[0, :, :3], raw_bgr[0])
        np.testing.assert_array_equal(written_rgba[0, :, 3], [255, 255])
        np.testing.assert_array_equal(written_rgba[1, :, :], 0)


class Sam3dGenerateContractTests(unittest.TestCase):
    def test_pipeline_disables_layout_postprocess_and_preserves_mesh_options(
        self,
    ) -> None:
        rgba = np.zeros((2, 2, 4), dtype=np.uint8)
        rgba[0, 0, 3] = 255
        pipeline = mock.Mock()
        pipeline.run.return_value = {"glb": [mock.Mock()]}
        inference = SimpleNamespace(_pipeline=pipeline)

        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            input_path = root / "masked.png"
            Image.fromarray(rgba).save(input_path)
            args = SimpleNamespace(
                img_path=str(input_path),
                output_dir=str(root / "shape"),
                seed=42,
                skip_visualization=True,
            )

            shape_prior_generate.run_sam3d_shape_prior(
                args,
                infer=inference,
                timing_ms={},
            )

        pipeline.run.assert_called_once_with(
            mock.ANY,
            mock.ANY,
            seed=42,
            with_mesh_postprocess=True,
            with_texture_baking=True,
            with_layout_postprocess=False,
            use_vertex_color=False,
        )


class ShapePriorClientTimingTests(unittest.TestCase):
    def test_client_aggregates_complete_critical_path_and_pre_submit(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=root,
                object_name="stuffed animal",
                controller_name="hand",
                points_npz=root / "points.npz",
            )
            case = root / shape_prior_warmup.CASE_NAME
            shape_dir = case / "shape"

            def fake_write_case(*_args, **_kwargs):
                shape_dir.mkdir(parents=True)
                return {"case": case, "shape": shape_dir}

            def fake_pipeline_stage(stage, *_args, **_kwargs):
                if stage == shape_prior_warmup.PREWARM_STAGE_UPSCALE:
                    Image.fromarray(np.zeros((2, 2, 3), dtype=np.uint8)).save(
                        shape_dir / "high_resolution.png"
                    )
                shape_prior_timing.write_stage_profile(
                    client._stage_profile_path(stage),
                    stage=stage,
                    status=shape_prior_timing.STAGE_PROFILE_STATUS_COMPLETED,
                    execution_mode="cold",
                    timing_ms={"active_work_ms": 1.0, "total_ms": 1.0},
                )
                return 1.0, {
                    "execution_mode": "cold",
                    "critical_path_ms": 1.0,
                    "go_wall_time_s": None,
                }

            def fake_segment(**kwargs):
                output = Path(kwargs["output_path"])
                Image.fromarray(np.zeros((2, 2, 4), dtype=np.uint8)).save(output)
                return output, {"inference": {"cache_hit": True}, "total_ms": 1.0}

            def fake_sample(_command, *, env):
                del env
                with (case / "final_data.pkl").open("wb") as handle:
                    pickle.dump(
                        {
                            "surface_points": np.ones((2, 3), dtype=np.float32),
                            "interior_points": np.zeros((1, 3), dtype=np.float32),
                        },
                        handle,
                    )
                shape_prior_timing.write_stage_profile(
                    client._stage_profile_path("sample"),
                    stage="sample",
                    status=shape_prior_timing.STAGE_PROFILE_STATUS_COMPLETED,
                    execution_mode="cold",
                    timing_ms={"output_write_ms": 1.0, "total_ms": 1.0},
                )
                return 1.0

            with (
                mock.patch.object(
                    shape_prior_warmup,
                    "write_shape_prior_case",
                    side_effect=fake_write_case,
                ),
                mock.patch.object(
                    client,
                    "_run_stage_maybe_prewarmed",
                    side_effect=fake_pipeline_stage,
                ),
                mock.patch.object(
                    sam31_image_segmentation,
                    "segment_image_to_origin_rgba",
                    side_effect=fake_segment,
                ),
                mock.patch.object(
                    shape_prior_warmup,
                    "_run_stage",
                    side_effect=fake_sample,
                ),
            ):
                result = client.request_shape_prior(_frame0_request())

        self.assertTrue(result.ready)
        analysis = result.metadata["shape_prior_timing"]
        self.assertEqual(
            [
                "case_write",
                "upscale",
                "segment_image",
                "generate",
                "align",
                "sample",
                "result_finalize",
            ],
            [entry["stage"] for entry in analysis["critical_path"]],
        )
        self.assertTrue(analysis["pre_submit"]["available"])
        self.assertEqual(
            {"mask_ms": 3.0, "pcd_ms": 2.0},
            analysis["pre_submit"]["frame0_pipeline_timing_ms"],
        )
        self.assertGreaterEqual(analysis["unattributed_ms"], 0.0)
        self.assertEqual(3, result.metadata["shape_prior_point_count"])

    def test_prewarm_readiness_reports_lead_and_startup_tail(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            client = shape_prior_warmup.ShapePriorLocalClient(
                case_root=tmpdir,
                object_name="stuffed animal",
                controller_name="hand",
            )
            profile_path = client._stage_profile_path("upscale")
            shape_prior_timing.write_stage_profile(
                profile_path,
                stage="upscale",
                status=shape_prior_timing.STAGE_PROFILE_STATUS_COMPLETED,
                execution_mode="prewarmed",
                timing_ms={"total_ms": 1.0},
                ready_wall_time_s=10.0,
            )
            ready_before = client._completed_stage_details(
                "upscale",
                orchestration={
                    "execution_mode": "prewarmed",
                    "critical_path_ms": 1.0,
                    "go_wall_time_s": 10.25,
                },
            )
            startup_tail = client._completed_stage_details(
                "upscale",
                orchestration={
                    "execution_mode": "prewarmed",
                    "critical_path_ms": 1.0,
                    "go_wall_time_s": 9.75,
                },
            )

        self.assertTrue(ready_before["orchestration"]["ready_before_go"])
        self.assertAlmostEqual(
            250.0,
            ready_before["orchestration"]["ready_lead_ms"],
        )
        self.assertFalse(startup_tail["orchestration"]["ready_before_go"])
        self.assertAlmostEqual(
            250.0,
            startup_tail["orchestration"]["startup_tail_on_critical_path_ms"],
        )


class ShapePriorManagerTimingTests(unittest.TestCase):
    def test_finished_banner_prints_once_when_formal_gate_opens(self) -> None:
        result = SimpleNamespace(ready=True)
        demo = SimpleNamespace(
            shape_prior_manager=mock.Mock(),
            headless_capture_writer=mock.Mock(),
            _shape_prior_written=False,
            _shape_prior_profile_payload=mock.Mock(return_value={}),
            _write_shape_prior_profile_json=mock.Mock(),
            _status=mock.Mock(),
        )
        demo.shape_prior_manager.ready_result.return_value = result

        with mock.patch("builtins.print") as print_output:
            mdp_demo_segwarmup._SegWarmupMixin._maybe_write_shape_prior_headless_result(
                demo
            )
            mdp_demo_segwarmup._SegWarmupMixin._maybe_write_shape_prior_headless_result(
                demo
            )

        print_output.assert_called_once_with(
            mdp_demo_segwarmup.WARMUP_FINISHED_BANNER,
            flush=True,
        )
        demo.headless_capture_writer.write_shape_prior_result.assert_called_once_with(
            result
        )
        demo.shape_prior_manager.mark_gate_open.assert_called_once_with()

    def test_default_profile_has_no_unmeasured_legacy_zero_timings(self) -> None:
        profile = shape_prior_warmup.default_profile(enabled=True)

        self.assertNotIn("shape_prior_submit_ms", profile)
        self.assertNotIn("first_mask_depth_pair_ms", profile)
        self.assertNotIn("first_strict_pair_ms", profile)

    def test_manager_records_request_ready_and_gate_durations(self) -> None:
        class ReadyClient:
            def request_shape_prior(self, frame0):
                return shape_prior_warmup.ShapePriorResult(
                    seq=frame0.seq,
                    source_seq=frame0.seq,
                    source_timestamp_s=frame0.source_timestamp_s,
                    status=shape_prior_warmup.STATUS_READY,
                    points_m=np.ones((1, 3), dtype=np.float32),
                    colors_rgb_u8=np.zeros((1, 3), dtype=np.uint8),
                    metadata={"shape_prior_timing": {"schema_version": 1}},
                )

        manager = shape_prior_warmup.ShapePriorWarmupManager(
            enabled=True,
            client=ReadyClient(),
        )
        self.assertTrue(manager.maybe_submit(_frame0_request()))
        self.assertIsNotNone(manager.wait(1.0))
        manager.mark_gate_open()
        profile = manager.profile()

        self.assertEqual(shape_prior_warmup.STATUS_READY, profile["shape_prior_status"])
        self.assertGreaterEqual(profile["shape_prior_request_total_ms"], 0.0)
        self.assertGreaterEqual(
            profile["warmup_runtime_start_to_shape_prior_ready_ms"],
            profile["shape_prior_request_total_ms"],
        )
        self.assertGreaterEqual(
            profile["warmup_shape_prior_ready_to_gate_open_ms"],
            0.0,
        )
        self.assertGreaterEqual(
            profile["warmup_total_ms"],
            profile["warmup_runtime_start_to_shape_prior_ready_ms"],
        )


if __name__ == "__main__":
    unittest.main()
