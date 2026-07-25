from __future__ import annotations

from contextlib import redirect_stderr
import io
import inspect
import json
from pathlib import Path
import pickle
from types import SimpleNamespace
import tempfile
import unittest

import numpy as np

from demo_v6_2 import main_cli
from demo_v6_2.mdp import cli as mdp_cli
from demo_v6_2.mdp import packets as mdp_packets
from demo_v6_2.mdp.formal_products import FormalProductStage
from demo_v6_2.mdp.runtime import MainDataProcessingDemo
from demo_v6_2.shape_prior import case as shape_prior_case
from demo_v6_2.main_subprocess import build_main_data_processing_command
from demo_v6_2.orchestration.run_config import OrchestratorRunConfig, dry_run_contract
from demo_v6_2.mdp.packets import PairedBuildResult
from demo_v6_2.mdp.packets import (
    MaskPacket,
    PipelineTiming,
    _full_tracker_arrays_for_prepared_frame,
)
from demo_v6_2.utils.camera import CameraIntrinsics
from demo_v6_2.phystwin_strict_product import (
    PHYSTWIN_DEPTH_MAX_M,
    PHYSTWIN_DEPTH_MIN_M,
    apply_depth_validity_to_mask_frame,
    apply_radius_outlier_to_mask_frame,
    prepare_phystwin_frame,
)


class RootFacadeTests(unittest.TestCase):
    EXPECTED_ROOT_PYTHON_FILES = frozenset(
        {
            "main.py",
            "main_cli.py",
            "main_data_processing.py",
            "main_options.py",
            "main_subprocess.py",
            "phystwin_shen_launch.py",
            "phystwin_strict_product.py",
            "pipeline_status.py",
            "tracking.py",
        }
    )

    def test_root_python_files_are_pipeline_facade_only(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        demo_root = repo_root / "demo_v6_2"
        root_python_files = {
            path.name for path in demo_root.glob("*.py") if path.name != "__init__.py"
        }
        self.assertEqual(root_python_files, self.EXPECTED_ROOT_PYTHON_FILES)

        pipeline_text = (demo_root / "PIPELINE.md").read_text(encoding="utf-8")
        q2_start = pipeline_text.index("## 摄像头与逐帧 I/O（Q2–Q7）")
        pipeline_answers = pipeline_text[q2_start:]
        missing_citations = sorted(
            name
            for name in self.EXPECTED_ROOT_PYTHON_FILES
            if name not in pipeline_answers
        )
        self.assertEqual(missing_citations, [])


class RuntimeInputModeTests(unittest.TestCase):
    @staticmethod
    def _mask_packet(
        *, object_mask: np.ndarray, controller_mask: np.ndarray
    ) -> MaskPacket:
        height, width = object_mask.shape
        return MaskPacket(
            seq=0,
            color_bgr=np.zeros((height, width, 3), dtype=np.uint8),
            depth_source="realsense",
            intrinsics=CameraIntrinsics(
                fx=1000.0,
                fy=1000.0,
                cx=float(width - 1) / 2.0,
                cy=float(height - 1) / 2.0,
            ),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=0.0,
            process_done_perf_s=0.0,
            dropped_capture_frames=0,
            timing=PipelineTiming(),
            controller_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            object_mask=np.ascontiguousarray(object_mask, dtype=bool),
            hand_a_mask=np.ascontiguousarray(controller_mask, dtype=bool),
            hand_b_mask=np.zeros_like(controller_mask, dtype=bool),
            depth_u16=np.full((height, width), 1000, dtype=np.uint16),
        )

    @staticmethod
    def _formal_stage(*, with_calibration: bool = True) -> FormalProductStage:
        return FormalProductStage(
            args=SimpleNamespace(
                pcd_color_mode="rgb",
                controller_color=(255, 0, 0),
                object_color=(0, 0, 255),
            ),
            session=SimpleNamespace(
                table_c2w=np.eye(4, dtype=np.float32) if with_calibration else None,
                headless_capture_writer=None,
                depth_engine=None,
            ),
            lossless=SimpleNamespace(),
            stage_stats=SimpleNamespace(),
            timeline_gate=SimpleNamespace(),
            shape_prior=SimpleNamespace(),
            capture=SimpleNamespace(startup_hold_s=0.0),
            stop_event=SimpleNamespace(),
            fatal=SimpleNamespace(),
        )

    def test_orchestrator_accepts_only_fake_live_and_live(self) -> None:
        parser = main_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)
            self.assertFalse(hasattr(args, "source_headless_capture"))
        default_args = parser.parse_args([])
        self.assertNotIn(
            "source_headless_capture",
            dry_run_contract(default_args, OrchestratorRunConfig.from_args(default_args)),
        )

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--source-headless-capture", "/tmp/capture"])
        self.assertEqual(error.exception.code, 2)

    def test_camera_runtime_accepts_only_fake_live_and_live(self) -> None:
        parser = mdp_cli.build_parser()
        supported_modes = {"fake-live", "live"}
        input_source_action = next(
            action for action in parser._actions if action.dest == "input_source"
        )
        self.assertEqual(set(input_source_action.choices or ()), supported_modes)
        self.assertIn(parser.parse_args([]).input_source, supported_modes)

        for input_source in supported_modes:
            args = parser.parse_args(["--input-source", input_source])
            self.assertEqual(args.input_source, input_source)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--input-source", "recording"])
        self.assertEqual(error.exception.code, 2)

        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit) as error:
                parser.parse_args(["--recording-case", "/tmp/case"])
        self.assertEqual(error.exception.code, 2)

        self.assertTrue(mdp_cli._is_fake_live_input_source("fake-live"))
        self.assertFalse(mdp_cli._is_fake_live_input_source("live"))
        self.assertFalse(mdp_cli._is_fake_live_input_source("recording"))

    def test_masked_pcd_requires_tracker(self) -> None:
        parser = mdp_cli.build_parser()
        defaults = parser.parse_args([])
        self.assertEqual(defaults.pcd_mode, "masked")
        self.assertEqual(defaults.tracker_backend, "tapnextpp")

        calibration_args = ["--table-calibrate", "table_calibrate.pkl"]
        no_tracker = parser.parse_args([*calibration_args, "--tracker-backend", "none"])
        with self.assertRaisesRegex(
            ValueError,
            "--pcd-mode masked requires --tracker-backend tapnextpp",
        ):
            mdp_cli.validate_and_normalize_args(no_tracker)

        no_track_mode = parser.parse_args([*calibration_args, "--track-mode", "none"])
        with self.assertRaisesRegex(
            ValueError,
            "--pcd-mode masked requires an enabled --track-mode",
        ):
            mdp_cli.validate_and_normalize_args(no_track_mode)

    def test_formal_runtime_requires_camera_to_world_calibration(self) -> None:
        args = mdp_cli.build_parser().parse_args([])
        with self.assertRaisesRegex(
            ValueError,
            "requires --table-calibrate",
        ):
            mdp_cli.validate_and_normalize_args(args)

    def test_legacy_latest_frame_pcd_worker_is_absent(self) -> None:
        self.assertFalse(hasattr(FormalProductStage, "_pcd_worker"))

    def test_runtime_pcd_filter_surface_is_absent(self) -> None:
        parser = mdp_cli.build_parser()
        options = set(parser._option_string_actions)
        removed_options = {
            "--enable-pcd-filter",
            "--pcd-filter-mode",
            "--pcd-filter-preset",
            "--object-filter",
            "--controller-filter",
            "--object-filter-cap",
            "--controller-filter-cap",
            "--object-filter-keep-components",
            "--controller-filter-keep-components",
            "--object-filter-voxel-m",
            "--controller-filter-voxel-m",
            "--filter-every-n",
            "--filter-max-age-frames",
            "--voxel-density-min-points",
            "--filter-radius-m",
            "--filter-nb-points",
            "--enhanced-component-voxel-size-m",
            "--enhanced-keep-near-main-gap-m",
            "--tracker-retire-filtered-markers",
            "--enable-table-z-filter",
            "--disable-table-z-filter",
            "--object-pcd-mask-erode-pixels",
            "--controller-pcd-mask-erode-pixels",
        }
        self.assertFalse(removed_options & options)
        self.assertFalse(hasattr(FormalProductStage, "_make_filter_input"))
        self.assertFalse(hasattr(FormalProductStage, "_filter_pcd_input"))
        self.assertFalse(hasattr(FormalProductStage, "_apply_single_pcd_filter"))
        self.assertFalse(hasattr(mdp_packets, "PcdFilterTelemetry"))

        orchestrator_args = main_cli.build_parser().parse_args([])
        command = build_main_data_processing_command(
            orchestrator_args,
            capture_dir=Path("/tmp/demo-v6-2-capture"),
            profile_json=Path("/tmp/demo-v6-2-profile.json"),
        )
        self.assertFalse(removed_options & set(command))

    def test_metadata_only_camera_cli_surface_is_absent(self) -> None:
        parser = mdp_cli.build_parser()
        options = set(parser._option_string_actions)
        removed_options = {
            "--demo-visual-mode",
            "--runtime-product-name",
            "--metadata-demo-version",
            "--metadata-reference-pipeline",
        }
        self.assertFalse(removed_options & options)

        command = build_main_data_processing_command(
            main_cli.build_parser().parse_args([]),
            capture_dir=Path("/tmp/demo-v6-2-capture"),
            profile_json=Path("/tmp/demo-v6-2-profile.json"),
        )
        self.assertFalse(removed_options & set(command))

    def test_capture_metadata_contains_only_semantic_contract_fields(self) -> None:
        runtime = object.__new__(MainDataProcessingDemo)
        runtime.args = SimpleNamespace(
            input_source="fake-live",
            depth_source="realsense",
            headless_prepared_only=True,
            write_input_rgb_timeline=True,
        )
        runtime.mode = SimpleNamespace(
            depth_backend_label="realsense",
            lossless_enabled=True,
            lossless_input_fps=5.0,
        )
        runtime.session = SimpleNamespace(
            camera_runtime=SimpleNamespace(
                serial="test-camera",
                intrinsics=CameraIntrinsics(
                    fx=100.0,
                    fy=101.0,
                    cx=20.0,
                    cy=21.0,
                ),
                k_color=np.eye(3, dtype=np.float32),
            ),
            recording_source=SimpleNamespace(
                effective_fps=5.0,
                frame_count=12,
            ),
            width=40,
            height=30,
            table_c2w=np.eye(4, dtype=np.float32),
        )
        runtime.shape_prior_manager = SimpleNamespace(
            profile_payload=lambda: {
                "shape_prior_status": "pending",
                "shape_prior_error": None,
                "unused_profile_value": 123,
            }
        )

        metadata = runtime._build_headless_capture_metadata()

        self.assertEqual(
            set(metadata),
            {
                "input_source",
                "replay_fps",
                "recording_frame_count",
                "depth_source",
                "depth_source_internal",
                "depth_backend",
                "headless_prepared_only",
                "write_input_rgb_timeline",
                "shape_prior_status",
                "shape_prior_error",
                "lossless_input_fps",
                "saved_pcd_source",
                "serial",
                "width",
                "height",
                "pcd_coordinate_frame",
                "camera_to_world_c2w",
                "intrinsics",
                "k_color",
            },
        )

    def test_origin_depth_gate_uses_strict_02_to_15_meter_bounds(self) -> None:
        depth_m = np.asarray(
            [[PHYSTWIN_DEPTH_MIN_M, 1.0, PHYSTWIN_DEPTH_MAX_M]],
            dtype=np.float32,
        )
        mask = np.ones_like(depth_m, dtype=bool)
        processed = apply_depth_validity_to_mask_frame(
            {"object": mask, "controller": mask}, depth_m
        )
        expected = np.asarray([[False, True, False]], dtype=bool)
        np.testing.assert_array_equal(processed["object"], expected)
        np.testing.assert_array_equal(processed["controller"], expected)

    def test_canonical_stage_keeps_the_only_pt_mask_filter(self) -> None:
        prepare_parameters = inspect.signature(prepare_phystwin_frame).parameters
        self.assertIn("processed_mask_frame", prepare_parameters)
        self.assertIn("pcd_points", prepare_parameters)
        self.assertNotIn("mask_frame", prepare_parameters)
        self.assertNotIn("intrinsics", prepare_parameters)
        self.assertNotIn("c2w", prepare_parameters)

        points_grid = np.zeros((1, 41, 3), dtype=np.float32)
        points_grid[0, :40, 2] = 1.0
        points_grid[0, 40] = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        object_mask = np.ones((1, 41), dtype=bool)
        controller_mask = np.zeros((1, 41), dtype=bool)

        cleaned = apply_radius_outlier_to_mask_frame(
            {"object": object_mask, "controller": controller_mask},
            points_grid,
        )

        self.assertTrue(np.all(cleaned["object"][0, :40]))
        self.assertFalse(cleaned["object"][0, 40])

    def test_processed_hands_are_subsets_of_cleaned_controller(self) -> None:
        points_grid = np.zeros((1, 41, 3), dtype=np.float32)
        points_grid[0, :40, 2] = 1.0
        points_grid[0, 40] = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)
        controller = np.ones((1, 41), dtype=bool)
        cleaned = apply_radius_outlier_to_mask_frame(
            {
                "object": np.ones((1, 41), dtype=bool),
                "controller": controller,
                "hand_a": controller,
                "hand_b": np.zeros_like(controller),
            },
            points_grid,
        )
        self.assertFalse(cleaned["controller"][0, 40])
        self.assertFalse(cleaned["hand_a"][0, 40])
        self.assertTrue(np.all(~cleaned["hand_a"] | cleaned["controller"]))

    def test_object_controller_overlap_keeps_origin_semantics(self) -> None:
        points_grid = np.zeros((1, 40, 3), dtype=np.float32)
        points_grid[0, :, 2] = 1.0
        shared_mask = np.ones((1, 40), dtype=bool)
        cleaned = apply_radius_outlier_to_mask_frame(
            {"object": shared_mask, "controller": shared_mask},
            points_grid,
        )
        np.testing.assert_array_equal(cleaned["object"], shared_mask)
        np.testing.assert_array_equal(cleaned["controller"], shared_mask)

    def test_runtime_pcd_uses_the_canonical_processed_masks(self) -> None:
        object_mask = np.zeros((10, 10), dtype=bool)
        object_mask[:, :5] = True
        controller_mask = ~object_mask
        result = self._formal_stage()._build_processed_frame_result(
            self._mask_packet(
                object_mask=object_mask,
                controller_mask=controller_mask,
            )
        )

        np.testing.assert_array_equal(
            result.processed_frame.mask_packet.object_mask,
            object_mask,
        )
        np.testing.assert_array_equal(
            result.processed_frame.mask_packet.controller_mask,
            controller_mask,
        )
        self.assertEqual(result.pcd_packet.object_point_count, 50)
        self.assertEqual(result.pcd_packet.controller_point_count, 50)
        self.assertEqual(result.processed_frame.pcd_points.shape, (1, 10, 10, 3))

    def test_empty_processed_class_fails_without_raw_mask_fallback(self) -> None:
        object_mask = np.zeros((10, 10), dtype=bool)
        object_mask[0, 0] = True
        controller_mask = np.ones((10, 10), dtype=bool)
        with self.assertRaisesRegex(RuntimeError, "processed object mask is empty"):
            self._formal_stage()._build_processed_frame_result(
                self._mask_packet(
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                )
            )

    def test_processed_frame_fails_without_camera_to_world(self) -> None:
        object_mask = np.zeros((10, 10), dtype=bool)
        object_mask[:, :5] = True
        controller_mask = ~object_mask
        with self.assertRaisesRegex(RuntimeError, "camera-to-world calibration"):
            self._formal_stage(with_calibration=False)._build_processed_frame_result(
                self._mask_packet(
                    object_mask=object_mask,
                    controller_mask=controller_mask,
                )
            )

    def test_prepared_visibility_uses_design_spec_observation_gate(self) -> None:
        tracks, visibility = _full_tracker_arrays_for_prepared_frame(
            SimpleNamespace(
                query_points_yx=np.zeros((2, 2), dtype=np.float32),
                all_tracks_yx=np.ones((2, 2), dtype=np.float32),
                all_tracker_visibility=np.ones((2,), dtype=bool),
                all_observation_visibility=np.asarray([True, False]),
                tracks_yx=np.empty((0, 2), dtype=np.float32),
                visibility=np.empty((0,), dtype=bool),
                query_indices=np.empty((0,), dtype=np.int64),
            )
        )
        self.assertEqual(tracks.shape, (2, 2))
        np.testing.assert_array_equal(visibility, [True, False])

    def test_prepared_tracker_arrays_do_not_fallback_to_sparse_state(self) -> None:
        with self.assertRaisesRegex(
            ValueError,
            "requires full processed tracker arrays",
        ):
            _full_tracker_arrays_for_prepared_frame(
                SimpleNamespace(
                    query_points_yx=np.zeros((2, 2), dtype=np.float32),
                    all_tracks_yx=np.empty((0, 2), dtype=np.float32),
                    all_observation_visibility=np.empty((0,), dtype=bool),
                    tracks_yx=np.zeros((2, 2), dtype=np.float32),
                    visibility=np.ones((2,), dtype=bool),
                    query_indices=np.arange(2, dtype=np.int64),
                )
            )

    def test_shape_prior_writer_does_not_repeat_pt_mask_filtering(self) -> None:
        object_mask = np.asarray([[True, False]], dtype=bool)
        controller_mask = np.asarray([[False, True]], dtype=bool)
        points_world_m = np.asarray(
            [[[0.0, 0.0, 1.0], [0.001, 0.0, 1.0]]],
            dtype=np.float32,
        )
        request = shape_prior_case.ShapePriorFrame0Request(
            seq=0,
            source_timestamp_s=0.0,
            input_source="fake-live",
            depth_backend="realsense",
            depth_source_internal="realsense",
            rgb_u8=np.zeros((1, 2, 3), dtype=np.uint8),
            object_mask=object_mask,
            controller_mask=controller_mask,
            depth_color_m=np.ones((1, 2), dtype=np.float32),
            depth_valid_mask=np.ones((1, 2), dtype=bool),
            points_world_m=points_world_m,
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = shape_prior_case.write_shape_prior_case(
                request,
                case_root=Path(temporary_directory),
                case_name="case",
                object_name="stuffed animal",
                controller_name="hand",
            )
            with (paths["case"] / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            case_metadata = json.loads(
                (paths["case"] / "metadata.json").read_text(encoding="utf-8")
            )

        np.testing.assert_array_equal(processed_masks[0][0]["object"], object_mask)
        np.testing.assert_array_equal(
            processed_masks[0][0]["controller"], controller_mask
        )
        self.assertEqual(
            set(case_metadata),
            {
                "frame_num",
                "intrinsics",
                "input_source",
                "depth_backend",
                "depth_source_internal",
            },
        )


class VolumeSampleTests(unittest.TestCase):
    def test_unified_sampling_matches_origin_reference(self) -> None:
        """sample_origin_unified_structure vs a literal origin reimplementation.

        The reference reproduces data_process_origin/data_process_sample.py
        L47-L119 verbatim (np.unique value order, z>0 clamp, raw-candidate
        min_bound, ONE shared grid, object -> surface -> interior priority).
        """
        from demo_v6_2 import tracking

        rng = np.random.default_rng(11)
        obj = rng.uniform(-0.05, 0.05, size=(3, 60, 3)).astype(np.float64)
        obj[0, 5] = obj[0, 4]  # exact duplicate for the unique step
        obj[0, 7, 2] = 0.004  # positive z exercises the above-table clamp
        surface = rng.uniform(-0.06, 0.06, size=(40, 3))
        interior = rng.uniform(-0.06, 0.06, size=(80, 3))
        voxel = 0.005

        # --- literal origin reference ---
        ref_obj = obj.copy()
        unique_idx = np.unique(ref_obj[0], axis=0, return_index=True)[1]
        ref_obj = ref_obj[:, unique_idx, :]
        ref_obj[ref_obj[..., 2] > 0, 2] = 0
        all_points = np.concatenate([surface, interior, ref_obj[0]], axis=0)
        min_bound = np.min(all_points, axis=0)
        grid_flag: dict[tuple, int] = {}
        ref_object_local = []
        for i in range(ref_obj.shape[1]):
            key = tuple(np.floor((ref_obj[0, i] - min_bound) / voxel).astype(int))
            if key not in grid_flag:
                grid_flag[key] = 1
                ref_object_local.append(i)
        ref_surface = []
        for i in range(surface.shape[0]):
            key = tuple(np.floor((surface[i] - min_bound) / voxel).astype(int))
            if key not in grid_flag:
                grid_flag[key] = 1
                ref_surface.append(surface[i])
        ref_interior = []
        for i in range(interior.shape[0]):
            key = tuple(np.floor((interior[i] - min_bound) / voxel).astype(int))
            if key not in grid_flag:
                grid_flag[key] = 1
                ref_interior.append(interior[i])

        unified = tracking.sample_origin_unified_structure(
            obj, surface, interior, volume_sample_size=voxel
        )
        np.testing.assert_array_equal(
            unified.object_source_indices, unique_idx[ref_object_local]
        )
        np.testing.assert_array_equal(unified.surface_points, np.asarray(ref_surface))
        np.testing.assert_array_equal(unified.interior_points, np.asarray(ref_interior))
        np.testing.assert_array_equal(unified.min_bound, min_bound)

    def test_unified_sampling_priority_and_raw_min_bound(self) -> None:
        from demo_v6_2 import tracking

        voxel = 0.005
        # object + surface share a voxel -> object wins; surface + interior
        # share another -> surface wins.
        obj = np.asarray([[[0.001, 0.001, -0.001]]], dtype=np.float64)
        surface = np.asarray(
            [[0.002, 0.002, -0.002], [0.0125, 0.001, -0.001]], dtype=np.float64
        )
        interior = np.asarray([[0.0135, 0.002, -0.002]], dtype=np.float64)
        unified = tracking.sample_origin_unified_structure(
            obj, surface, interior, volume_sample_size=voxel
        )
        np.testing.assert_array_equal(unified.object_source_indices, [0])
        np.testing.assert_array_equal(unified.surface_points, [[0.0125, 0.001, -0.001]])
        self.assertEqual(unified.interior_points.shape[0], 0)

        # A raw extreme candidate anchors min_bound even when occupancy later
        # rejects it (the exact defect of the old two-pass split).
        extreme = np.asarray([[-0.1, -0.1, -0.1]], dtype=np.float64)
        anchored = tracking.sample_origin_unified_structure(
            np.asarray([[[-0.0999, -0.0999, -0.0999]]], dtype=np.float64),
            extreme,
            None,
            volume_sample_size=voxel,
        )
        np.testing.assert_array_equal(anchored.min_bound, extreme[0])
        # Same voxel as the object -> the extreme surface candidate is
        # rejected by occupancy, yet it anchored the grid.
        self.assertEqual(anchored.surface_points.shape[0], 0)
        np.testing.assert_array_equal(anchored.object_source_indices, [0])

    def test_volume_sample_size_is_a_single_authoritative_config(self) -> None:
        from demo_v6_2 import main_cli
        from demo_v6_2.main_subprocess import build_main_data_processing_command
        from demo_v6_2.orchestration.main_config import DEFAULT_VOLUME_SAMPLE_SIZE_M
        from demo_v6_2.shape_prior import sample as shape_prior_sample
        from demo_v6_2.streaming.data_payload import DATA_PROCESS_SAM3D_METRICS

        # config/default.yaml is the only default location; 0.005 preserves
        # data_process_origin volume-sampling parity.
        self.assertEqual(DEFAULT_VOLUME_SAMPLE_SIZE_M, 0.005)

        # The unified sampling happens ONLY in the orchestrator's tracking
        # runtime at chunk 0: the camera CLI and the sample stage (raw
        # candidates only) no longer carry a voxel-size knob at all.
        camera_args = mdp_cli.build_parser().parse_args([])
        self.assertFalse(hasattr(camera_args, "volume_sample_size_m"))
        sample_args = shape_prior_sample.build_parser().parse_args(
            ["--base_path", "/tmp/x", "--case_name", "case"]
        )
        self.assertFalse(hasattr(sample_args, "volume_sample_size"))
        orch_args = main_cli.build_parser().parse_args(
            ["--volume-sample-size-m", "0.0075"]
        )
        command = build_main_data_processing_command(
            orch_args,
            capture_dir=Path("/tmp/capture"),
            profile_json=Path("/tmp/profile.json"),
        )
        self.assertNotIn("--volume-sample-size-m", command)

        # The runtime consumes it, and the manifest constants no longer carry
        # a hardcoded copy (the true value is injected per run).
        tracking = __import__("demo_v6_2.tracking", fromlist=["TrackingRuntime"])
        self.assertEqual(
            tracking.TrackingRuntime(volume_sample_size=0.0075).volume_sample_size,
            0.0075,
        )
        self.assertNotIn("object_volume_sample_size_m", DATA_PROCESS_SAM3D_METRICS)
        self.assertNotIn(
            "shape_prior_volume_sample_size_m", DATA_PROCESS_SAM3D_METRICS
        )


class PairedBuildResultTests(unittest.TestCase):
    def test_rejects_mixed_sequence_results(self) -> None:
        for component in ("pcd", "mask", "tracker"):
            sequences = {"pcd": 7, "mask": 7, "tracker": 7}
            sequences[component] = 8
            with self.subTest(component=component):
                with self.assertRaisesRegex(
                    ValueError,
                    "strict same-seq build result mismatch",
                ):
                    PairedBuildResult(
                        seq=7,
                        pcd_result=SimpleNamespace(
                            pcd_packet=SimpleNamespace(seq=sequences["pcd"]),
                            processed_frame=SimpleNamespace(
                                mask_packet=SimpleNamespace(seq=sequences["mask"])
                            ),
                        ),
                        tracker_packet=SimpleNamespace(seq=sequences["tracker"]),
                    )

    def test_legacy_render_packet_is_absent(self) -> None:
        self.assertFalse(hasattr(mdp_packets, "PairedRenderPacket"))


class WebVisualizerWiringTests(unittest.TestCase):
    def test_web_frontend_builds_web_viewer_command(self) -> None:
        from demo_v6_2 import main_cli
        from demo_v6_2.main_subprocess import build_visualizer_command

        args = main_cli.build_parser().parse_args([])
        command = build_visualizer_command(args)
        self.assertIn(
            str(Path("demo_v6_2") / "visualization" / "visualize_track_web.py"),
            command,
        )
        self.assertEqual(command[command.index("--port") + 1], "8767")
        self.assertEqual(command[command.index("--host") + 1], "127.0.0.1")
        self.assertNotIn("--layout", command)

    def test_window_frontend_keeps_legacy_command(self) -> None:
        from demo_v6_2 import main_cli
        from demo_v6_2.main_subprocess import build_visualizer_command

        args = main_cli.build_parser().parse_args(
            ["--visualizer-frontend", "window"]
        )
        command = build_visualizer_command(args)
        self.assertIn(
            str(Path("demo_v6_2") / "visualization" / "visualize_track.py"),
            command,
        )
        self.assertIn("--layout", command)

    def test_web_frontend_starts_immediately_for_any_layout(self) -> None:
        from demo_v6_2 import main_cli
        from demo_v6_2.orchestration.run_config import OrchestratorRunConfig

        for layout in ("side-by-side", "output-only"):
            args = main_cli.build_parser().parse_args(
                [
                    "--downstream-mode",
                    "demo_visualizer",
                    "--visualizer-layout",
                    layout,
                ]
            )
            config = OrchestratorRunConfig.from_args(args)
            self.assertEqual(config.visualizer_frontend, "web")
            self.assertEqual(
                config.visualizer_start_policy, "immediate_after_camera_start"
            )

        window_args = main_cli.build_parser().parse_args(
            [
                "--downstream-mode",
                "demo_visualizer",
                "--visualizer-frontend",
                "window",
                "--visualizer-layout",
                "output-only",
            ]
        )
        window_config = OrchestratorRunConfig.from_args(window_args)
        self.assertEqual(
            window_config.visualizer_start_policy,
            "after_first_committed_online_chunk",
        )

    def test_web_port_is_validated_and_off_phystwin_defaults(self) -> None:
        from demo_v6_2 import main_cli
        from demo_v6_2.orchestration.run_config import OrchestratorRunConfig

        args = main_cli.build_parser().parse_args(
            [
                "--downstream-mode",
                "demo_visualizer",
                "--visualizer-web-port",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "visualizer-web-port"):
            OrchestratorRunConfig.from_args(args)

        default_port = main_cli.build_parser().parse_args([]).visualizer_web_port
        self.assertNotIn(int(default_port), (8765, 8766))


if __name__ == "__main__":
    unittest.main()
