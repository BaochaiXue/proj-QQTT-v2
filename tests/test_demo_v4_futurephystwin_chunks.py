from __future__ import annotations

import json
from pathlib import Path
import pickle
import tempfile
import threading
import time
import unittest
from contextlib import redirect_stdout
import io
from unittest import mock

import numpy as np
from PIL import Image

import demo_v4.futurephystwin_chunk_writer as chunk_writer
from demo_v4.futurephystwin_chunk_writer import (
    FuturePhysTwinChunk,
    validate_futurephystwin_case,
    write_futurephystwin_chunk_case,
)
from demo_v4.headless_chunk_bridge import (
    _read_json_file_stable,
    stream_chunks_from_headless_capture,
    write_chunks_from_headless_capture,
)
from demo_v4.realtime_futurephystwin_chunks import (
    _contract,
    build_demo32_realtime_command,
    build_parser,
    main as demo_v4_main,
    resolve_chunk_frame_count,
    resolve_demo32_source_replay_fps,
    resolve_demo32_cuda_visible_devices,
    select_validation_chunk_cases,
)
from qqtt.demo.single_view_shape_prior_sampling import (
    SimpleShapeMesh,
    sample_data_process_sam3d_single_view_shape_prior_points,
)
from qqtt.demo import phystwin_strict_product as strict


def _rgb_frames(frame_count: int, height: int = 4, width: int = 5) -> list[np.ndarray]:
    frames: list[np.ndarray] = []
    for idx in range(frame_count):
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[..., 0] = np.uint8(20 + idx)
        frame[..., 1] = np.uint8(40 + idx)
        frame[..., 2] = np.uint8(60 + idx)
        frames.append(frame)
    return frames


def _processed_masks(frame_count: int, height: int = 4, width: int = 5) -> list[list[dict[str, np.ndarray]]]:
    frames: list[list[dict[str, np.ndarray]]] = []
    for _idx in range(frame_count):
        object_mask = np.zeros((height, width), dtype=bool)
        controller_mask = np.zeros((height, width), dtype=bool)
        object_mask[1:3, 1:3] = True
        controller_mask[0:2, 3:5] = True
        frames.append([{"object": object_mask, "controller": controller_mask}])
    return frames


def _track_process_data(frame_count: int) -> dict[str, np.ndarray]:
    object_points = np.zeros((frame_count, 4, 3), dtype=np.float64)
    object_points[:, :, 0] = np.array([0.000, 0.002, 0.010, 0.020])
    object_points[:, :, 1] = np.array([0.000, 0.001, 0.000, 0.010])
    object_points[:, :, 2] = np.array([-0.020, -0.020, -0.030, -0.040])
    for frame_idx in range(frame_count):
        object_points[frame_idx, :, 0] += float(frame_idx) * 0.001

    controller_points = np.zeros((frame_count, 30, 3), dtype=np.float64)
    controller_points[:, :, 0] = np.linspace(0.05, 0.10, 30)
    controller_points[:, :, 1] = np.linspace(0.02, 0.04, 30)
    controller_points[:, :, 2] = -0.05
    for frame_idx in range(frame_count):
        controller_points[frame_idx, :, 1] += float(frame_idx) * 0.001

    return {
        "object_points": object_points,
        "object_colors": np.ones((frame_count, 4, 3), dtype=np.float64) * 0.5,
        "object_visibilities": np.ones((frame_count, 4), dtype=bool),
        "object_motions_valid": np.ones((frame_count, 4), dtype=bool),
        "controller_points": controller_points,
        "controller_mask": np.ones((30,), dtype=bool),
    }


def _futurephystwin_chunk(frame_count: int = 3) -> FuturePhysTwinChunk:
    return FuturePhysTwinChunk(
        rgb_frames=_rgb_frames(frame_count),
        processed_masks=_processed_masks(frame_count),
        track_process_data=_track_process_data(frame_count),
        intrinsics=np.array([[600.0, 0.0, 2.0], [0.0, 601.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
        tracks_yx=np.zeros((frame_count, 4, 2), dtype=np.float32),
        tracker_visibility=np.ones((frame_count, 4), dtype=bool),
        queries_txy=np.zeros((4, 3), dtype=np.float32),
        surface_points=np.array([[0.0, 0.0, -0.02], [0.01, 0.0, -0.03]], dtype=np.float64),
        interior_points=np.array([[0.005, 0.0, -0.025]], dtype=np.float64),
        fps=5,
        serial_number="demo-v4-single-camera",
        depth_backend="native-realsense",
        depth_source_internal="realsense",
    )


class FuturePhysTwinChunkWriterTest(unittest.TestCase):
    def test_demo_v4_parser_defaults_to_fake_live_25_frame_chunks_and_shape_prior(self) -> None:
        args = build_parser().parse_args(["--dry-run"])

        self.assertEqual(args.input_source, "fake-live")
        self.assertEqual(args.replay_fps, 5.0)
        self.assertIsNone(args.demo32_source_replay_fps)
        self.assertEqual(resolve_demo32_source_replay_fps(args), 5.0)
        self.assertEqual(args.chunk_seconds, 5.0)
        self.assertIsNone(args.chunk_frame_count)
        self.assertEqual(resolve_chunk_frame_count(args), 25)
        self.assertTrue(args.shape_prior_warmup)
        self.assertEqual(args.depth_backend, "native-realsense")
        self.assertIsNone(args.max_chunks)
        self.assertEqual(args.gpu_mode, "single")
        self.assertIsNone(args.realtime_gpu_mode)
        self.assertEqual(args.warmup_gpu_mode, "dual")
        self.assertIsNone(args.demo32_cuda_visible_devices)
        self.assertEqual(resolve_demo32_cuda_visible_devices(args), "0")
        self.assertEqual(args.demo32_device, "cuda")
        self.assertEqual(args.demo32_tracker_device, "cuda")
        self.assertEqual(args.demo32_dtype, "bfloat16")
        self.assertEqual(args.shape_prior_start_policy, "async-after-first-mask-depth-pair")
        self.assertEqual(_contract(args)["realtime_gpu_mode"], "single")
        self.assertEqual(_contract(args)["warmup_gpu_mode"], "dual")
        self.assertEqual(_contract(args)["shape_prior_device"], "cuda:1")
        self.assertTrue(args.mask_radius_outlier_filter)
        self.assertEqual(args.mask_radius_outlier_radius_m, 0.01)
        self.assertEqual(args.mask_radius_outlier_nb_points, 40)
        self.assertEqual(str(args.futurephystwin_base_path), "/home/xinjie/FuturePhysTwin/data/demo_v4_chunks")

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=resolve_chunk_frame_count(args),
        )
        self.assertEqual(command[command.index("--shape-prior-device") + 1], "cuda:1")
        self.assertEqual(command[command.index("--duration-s") + 1], "0.000")

    def test_demo_v4_chunk_seconds_controls_frame_count_and_capture_duration(self) -> None:
        args = build_parser().parse_args(
            [
                "--chunk-seconds",
                "8",
                "--replay-fps",
                "5",
                "--max-chunks",
                "3",
                "--capture-extra-seconds",
                "1",
            ]
        )

        self.assertEqual(resolve_chunk_frame_count(args), 40)
        self.assertEqual(_contract(args)["chunk_seconds"], 8.0)
        self.assertEqual(_contract(args)["chunk_frame_count"], 40)

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=resolve_chunk_frame_count(args),
        )
        self.assertEqual(command[command.index("--duration-s") + 1], "25.000")

    def test_demo_v4_can_decouple_source_pacing_from_output_fps(self) -> None:
        args = build_parser().parse_args(
            [
                "--replay-fps",
                "5",
                "--demo32-source-replay-fps",
                "5.2",
                "--max-chunks",
                "3",
                "--capture-extra-seconds",
                "1",
            ]
        )

        self.assertEqual(resolve_chunk_frame_count(args), 25)
        self.assertEqual(resolve_demo32_source_replay_fps(args), 5.2)
        self.assertEqual(_contract(args)["replay_fps"], 5.0)
        self.assertEqual(_contract(args)["demo32_source_replay_fps"], 5.2)
        self.assertEqual(_contract(args)["demo32_source_replay_fps_override"], 5.2)
        self.assertEqual(_contract(args)["demo32_lossless_input_fps"], 5.2)

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=resolve_chunk_frame_count(args),
        )
        self.assertEqual(command[command.index("--replay-fps") + 1], "5.2")
        self.assertEqual(command[command.index("--lossless-input-fps") + 1], "5.2")
        self.assertEqual(command[command.index("--duration-s") + 1], "15.423")

    def test_demo_v4_chunk_frame_count_override_keeps_valid_time_contract(self) -> None:
        args = build_parser().parse_args(
            [
                "--chunk-seconds",
                "8",
                "--chunk-frame-count",
                "12",
                "--replay-fps",
                "5",
                "--max-chunks",
                "2",
                "--capture-extra-seconds",
                "0",
            ]
        )

        self.assertEqual(resolve_chunk_frame_count(args), 12)
        self.assertEqual(_contract(args)["chunk_seconds"], 8.0)
        self.assertEqual(_contract(args)["chunk_frame_count"], 12)

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=resolve_chunk_frame_count(args),
        )
        self.assertEqual(command[command.index("--duration-s") + 1], "4.800")

    def test_demo_v4_rejects_nonpositive_chunk_seconds_even_with_frame_override(self) -> None:
        with self.assertRaisesRegex(ValueError, "chunk seconds must be positive"):
            demo_v4_main(
                [
                    "--dry-run",
                    "--chunk-seconds",
                    "0",
                    "--chunk-frame-count",
                    "10",
                ]
            )

    def test_demo_v4_rejects_nonpositive_replay_fps_for_time_chunks(self) -> None:
        with self.assertRaisesRegex(ValueError, "replay fps must be positive"):
            demo_v4_main(["--dry-run", "--replay-fps", "0"])

    def test_demo_v4_rejects_nonpositive_demo32_source_replay_fps(self) -> None:
        with self.assertRaisesRegex(ValueError, "Demo 3.2 source replay fps must be positive"):
            demo_v4_main(["--dry-run", "--demo32-source-replay-fps", "0"])

    def test_demo_v4_gpu_mode_resolves_single_dual_and_explicit_override(self) -> None:
        single_args = build_parser().parse_args(["--dry-run"])
        dual_args = build_parser().parse_args(["--gpu-mode", "dual", "--dry-run"])
        override_args = build_parser().parse_args(
            ["--gpu-mode", "dual", "--demo32-cuda-visible-devices", "0", "--dry-run"]
        )

        self.assertEqual(resolve_demo32_cuda_visible_devices(single_args), "0")
        self.assertEqual(resolve_demo32_cuda_visible_devices(dual_args), "1")
        self.assertEqual(resolve_demo32_cuda_visible_devices(override_args), "0")
        self.assertEqual(_contract(dual_args)["gpu_mode"], "dual")
        self.assertEqual(_contract(dual_args)["demo32_cuda_visible_devices"], "1")
        self.assertEqual(_contract(override_args)["demo32_cuda_visible_devices_override"], "0")

    def test_demo_v4_routes_dual_warmup_with_single_realtime(self) -> None:
        args = build_parser().parse_args(
            [
                "--realtime-gpu-mode",
                "single",
                "--warmup-gpu-mode",
                "dual",
                "--dry-run",
            ]
        )

        self.assertEqual(resolve_demo32_cuda_visible_devices(args), "0")
        self.assertEqual(_contract(args)["realtime_gpu_mode"], "single")
        self.assertEqual(_contract(args)["warmup_gpu_mode"], "dual")
        self.assertEqual(_contract(args)["shape_prior_device"], "cuda:1")

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=25,
        )
        self.assertEqual(command[command.index("--shape-prior-device") + 1], "cuda:1")

    def test_demo_v4_builds_full_fake_realtime_demo32_command(self) -> None:
        args = build_parser().parse_args(
            [
                "--futurephystwin-base-path",
                "result/demo_v4/cases",
                "--case-prefix",
                "rt",
                "--shape-prior-endpoint",
                "tcp://worker:7100",
            ]
        )

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/capture"),
            profile_json=Path("result/demo_v4/shape_profile.json"),
            chunk_frame_count=25,
        )

        self.assertIn("demo_v3_2/realtime_single_camera_ffs_masked_pcd.py", command[1])
        self.assertEqual(command[command.index("--input-source") + 1], "fake-live")
        self.assertEqual(command[command.index("--replay-fps") + 1], "5.0")
        self.assertEqual(command[command.index("--depth-backend") + 1], "native-realsense")
        self.assertEqual(command[command.index("--device") + 1], "cuda")
        self.assertEqual(command[command.index("--dtype") + 1], "bfloat16")
        self.assertEqual(command[command.index("--tracker-device") + 1], "cuda")
        self.assertEqual(command[command.index("--render-mode") + 1], "none")
        self.assertEqual(command[command.index("--headless-capture-dir") + 1], "result/demo_v4/capture")
        self.assertEqual(command[command.index("--tracking-product-backend") + 1], "phystwin-strict-tracking")
        self.assertEqual(command[command.index("--track-mode") + 1], "controller-object")
        self.assertEqual(command[command.index("--tracker-backend") + 1], "tapnextpp")
        self.assertEqual(command[command.index("--shape-prior-endpoint") + 1], "tcp://worker:7100")
        self.assertIn("--shape-prior-warmup", command)
        self.assertIn("--shape-prior-skip-route-visualizations", command)

    def test_demo_v4_live_command_enables_strict_headless_tracking_explicitly(self) -> None:
        args = build_parser().parse_args(
            [
                "--input-source",
                "live",
                "--futurephystwin-base-path",
                "result/demo_v4/cases",
                "--case-prefix",
                "rt_live",
            ]
        )

        command = build_demo32_realtime_command(
            args,
            capture_dir=Path("result/demo_v4/live_capture"),
            profile_json=Path("result/demo_v4/live_shape_profile.json"),
            chunk_frame_count=25,
        )

        self.assertEqual(command[command.index("--input-source") + 1], "live")
        self.assertEqual(command[command.index("--render-mode") + 1], "none")
        self.assertEqual(command[command.index("--headless-capture-dir") + 1], "result/demo_v4/live_capture")
        self.assertEqual(command[command.index("--tracking-product-backend") + 1], "phystwin-strict-tracking")
        self.assertEqual(command[command.index("--track-mode") + 1], "controller-object")
        self.assertEqual(command[command.index("--tracker-backend") + 1], "tapnextpp")

    def test_demo_v4_cli_converts_existing_headless_capture_to_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=4)
            base_path = root / "cases"

            with redirect_stdout(io.StringIO()):
                exit_code = demo_v4_main(
                    [
                        "--source-headless-capture",
                        str(capture),
                        "--futurephystwin-base-path",
                        str(base_path),
                        "--case-prefix",
                        "demo_v4_cli",
                        "--chunk-frame-count",
                        "2",
                        "--max-chunks",
                        "1",
                        "--surface-points-npy",
                        str(self._write_points(root / "surface.npy", [[0.0, 0.0, -0.02]])),
                        "--interior-points-npy",
                        str(self._write_points(root / "interior.npy", [[0.01, 0.0, -0.03]])),
                        "--no-mask-radius-outlier-filter",
                    ]
                )

            self.assertEqual(exit_code, 0)
            manifest_path = base_path / "demo_v4_cli_chunks_manifest.json"
            self.assertTrue(manifest_path.is_file())
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(manifest["chunk_count"], 1)
            self.assertEqual(manifest["chunks"][0]["case_name"], "demo_v4_cli_chunk_0001")
            self.assertTrue(validate_futurephystwin_case(base_path / "demo_v4_cli_chunk_0001")["valid"])

    def test_chunk_writer_publishes_ready_marker_after_validation(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)

            write_futurephystwin_chunk_case(base_path, "ready_chunk", _futurephystwin_chunk())

            self.assertTrue((base_path / "ready_chunk" / "READY").is_file())
            self.assertFalse((base_path / ".publishing").exists())

    def test_validate_futurephystwin_case_can_require_ready_marker(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)
            write_futurephystwin_chunk_case(base_path, "ready_required", _futurephystwin_chunk())
            case_dir = base_path / "ready_required"

            self.assertTrue(validate_futurephystwin_case(case_dir, require_ready=True)["valid"])
            (case_dir / "READY").unlink()

            with self.assertRaisesRegex(ValueError, "READY"):
                validate_futurephystwin_case(case_dir, require_ready=True)

    def test_chunk_writer_does_not_expose_final_case_while_materializing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp)
            final_case = base_path / "atomic_chunk"
            original_write_rgb_frames = chunk_writer._write_rgb_frames
            final_case_visible_during_write: list[bool] = []

            def observing_write_rgb_frames(case_dir: Path, rgb_frames: list[np.ndarray]) -> None:
                final_case_visible_during_write.append(final_case.exists())
                original_write_rgb_frames(case_dir, rgb_frames)

            with mock.patch.object(chunk_writer, "_write_rgb_frames", side_effect=observing_write_rgb_frames):
                write_futurephystwin_chunk_case(base_path, "atomic_chunk", _futurephystwin_chunk())

            self.assertEqual(final_case_visible_during_write, [False])
            self.assertTrue((final_case / "READY").is_file())

    def test_writer_emits_futurephystwin_case_root_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "futurephystwin_cases"
            frame_count = 5
            surface_points = np.array(
                [[0.0, 0.0, -0.01], [0.01, 0.0, -0.02], [0.0, 0.01, -0.03]],
                dtype=np.float64,
            )
            interior_points = np.array([[0.005, 0.005, -0.025]], dtype=np.float64)
            chunk = FuturePhysTwinChunk(
                rgb_frames=_rgb_frames(frame_count),
                processed_masks=_processed_masks(frame_count),
                track_process_data=_track_process_data(frame_count),
                intrinsics=np.array([[600.0, 0.0, 2.0], [0.0, 601.0, 1.5], [0.0, 0.0, 1.0]], dtype=np.float32),
                camera_to_world_c2w=np.eye(4, dtype=np.float32),
                tracks_yx=np.zeros((frame_count, 4, 2), dtype=np.float32),
                tracker_visibility=np.ones((frame_count, 4), dtype=bool),
                queries_txy=np.zeros((4, 3), dtype=np.float32),
                surface_points=surface_points,
                interior_points=interior_points,
                fps=5,
                serial_number="demo-v4-single-camera",
                depth_backend="native-realsense",
                depth_source_internal="realsense",
            )

            manifest = write_futurephystwin_chunk_case(base_path, "demo_v4_chunk_0001", chunk)

            case_dir = base_path / "demo_v4_chunk_0001"
            self.assertEqual(manifest["case_name"], "demo_v4_chunk_0001")
            self.assertEqual(manifest["frame_count"], frame_count)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["motion_neighbor_dist_m"], 0.01)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["motion_min_neighbors"], 5)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["motion_similarity_m"], 0.005)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["controller_fps_count"], 30)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["object_volume_sample_size_m"], 0.005)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["mask_radius_outlier_radius_m"], 0.01)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["mask_radius_outlier_nb_points"], 40)
            self.assertEqual(
                manifest["data_process_sam3d_metrics"]["shape_prior_sampling_backend"],
                "sam3d-single-view",
            )
            self.assertEqual(manifest["data_process_sam3d_metrics"]["shape_prior_configured_max_dist_m"], 0.05)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["shape_prior_effective_max_dist_m"], 0.05)
            self.assertEqual(
                manifest["data_process_sam3d_metrics"]["shape_prior_distance_policy"],
                "canonical_single_view_configured",
            )
            self.assertTrue(manifest["data_process_sam3d_metrics"]["offline_single_view_parity"])
            self.assertEqual(
                manifest["data_process_sam3d_metrics"]["shape_prior_sampling_source"],
                "data_process_sam3d/data_process_sample.py",
            )
            self.assertEqual(manifest["data_process_sam3d_metrics"]["shape_prior_target_surface_points"], 700)
            self.assertEqual(manifest["data_process_sam3d_metrics"]["shape_prior_target_interior_points"], 1000)
            self.assertFalse(manifest["data_process_sam3d_metrics"]["shape_prior_uses_mvsam3d"])
            self.assertEqual(manifest["object_point_count"], 3)
            self.assertEqual(manifest["controller_point_count"], 30)
            self.assertEqual(manifest["controller_candidate_count"], 30)
            self.assertEqual(manifest["controller_valid_candidate_count"], 30)
            self.assertTrue(manifest["shape_prior_fields_present"])
            self.assertFalse(manifest["shape_prior_target_counts_met"])
            self.assertEqual(manifest["first_frame_zero_object_points"], 0)
            self.assertEqual(manifest["first_frame_zero_controller_points"], 0)
            for relative in (
                "final_data.pkl",
                "track_process_data.pkl",
                "calibrate.pkl",
                "metadata.json",
                "split.json",
                "color/0/0.png",
                "mask/processed_masks.pkl",
                "tracking/0.npz",
                "cotracker/0.npz",
                "manifest.json",
            ):
                self.assertTrue((case_dir / relative).is_file(), relative)

            with (case_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertEqual(set(final_data), {
                "controller_mask",
                "controller_points",
                "object_colors",
                "object_motions_valid",
                "object_points",
                "object_visibilities",
                "surface_points",
                "interior_points",
            })
            self.assertEqual(final_data["object_points"].shape[0], frame_count)
            self.assertEqual(final_data["controller_points"].shape, (frame_count, 30, 3))
            np.testing.assert_allclose(final_data["surface_points"], surface_points)
            np.testing.assert_allclose(final_data["interior_points"], interior_points)

            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            self.assertIn("controller_mask", track_process)
            self.assertEqual(track_process["controller_mask"].shape, (30,))

            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["fps"], 5)
            self.assertEqual(metadata["frame_num"], frame_count)
            self.assertEqual(metadata["serial_numbers"], ["demo-v4-single-camera"])
            self.assertEqual(np.asarray(metadata["intrinsics"]).shape, (1, 3, 3))

            split = json.loads((case_dir / "split.json").read_text(encoding="utf-8"))
            self.assertEqual(split["frame_len"], frame_count)
            self.assertEqual(split["train"], [0, max(1, int(frame_count * 0.7))])
            self.assertEqual(split["test"], [max(1, int(frame_count * 0.7)), frame_count])

            summary = validate_futurephystwin_case(case_dir)
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["frame_count"], frame_count)
            self.assertEqual(summary["surface_point_count"], len(surface_points))
            self.assertEqual(summary["interior_point_count"], len(interior_points))

    def test_validator_rejects_missing_shape_prior_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            case_dir = Path(tmp) / "bad_case"
            case_dir.mkdir()
            frame_count = 1
            final_data = {
                "controller_mask": np.ones((30,), dtype=bool),
                "controller_points": np.zeros((frame_count, 30, 3), dtype=np.float32),
                "object_colors": np.zeros((frame_count, 1, 3), dtype=np.float32),
                "object_motions_valid": np.ones((frame_count, 1), dtype=bool),
                "object_points": np.zeros((frame_count, 1, 3), dtype=np.float32),
                "object_visibilities": np.ones((frame_count, 1), dtype=bool),
                "interior_points": np.zeros((0, 3), dtype=np.float32),
            }
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(final_data, handle)

            with self.assertRaisesRegex(ValueError, "surface_points"):
                validate_futurephystwin_case(case_dir)

    def test_headless_capture_bridge_writes_multiple_futurephystwin_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=4)

            base_path = root / "futurephystwin_cases"
            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=base_path,
                case_prefix="demo_v4_capture",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
            )

            self.assertEqual([item["case_name"] for item in manifests], [
                "demo_v4_capture_chunk_0001",
                "demo_v4_capture_chunk_0002",
            ])
            for manifest in manifests:
                case_dir = base_path / manifest["case_name"]
                summary = validate_futurephystwin_case(case_dir)
                self.assertTrue(summary["valid"])
                self.assertEqual(summary["frame_count"], 2)
                self.assertEqual(summary["controller_point_count"], 30)
                self.assertEqual(summary["surface_point_count"], 1)
                self.assertEqual(summary["interior_point_count"], 1)

    def test_prepared_frame_helper_exports_chunk_compatible_arrays(self) -> None:
        height, width = 8, 40
        rgb = np.full((height, width, 3), 120, dtype=np.uint8)
        depth = np.ones((height, width), dtype=np.float32)
        depth[1, 0] = 0.0
        object_mask = np.zeros((height, width), dtype=bool)
        controller_mask = np.zeros((height, width), dtype=bool)
        object_mask[1, :6] = True
        controller_mask[3, :32] = True
        query_points = np.array(
            [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
            dtype=np.float32,
        )

        frame = strict.prepare_phystwin_frame(
            seq=9,
            rgb_frame=rgb,
            depth_m=depth,
            mask_frame={"object": object_mask, "controller": controller_mask},
            tracks_yx=query_points,
            visibility=np.ones((len(query_points),), dtype=bool),
            query_points_yx=query_points,
            intrinsics={"fx": 1000.0, "fy": 1000.0, "cx": 0.0, "cy": 0.0},
            c2w=np.eye(4, dtype=np.float32),
            mask_radius_outlier_filter=False,
            source_timestamp_s=12.5,
            source_frame_index=7,
            source_step=42,
        )

        self.assertEqual(frame.seq, 9)
        self.assertEqual(frame.rgb_frame.shape, (height, width, 3))
        self.assertEqual(frame.pcd_points.shape, (1, height, width, 3))
        self.assertEqual(frame.pcd_colors.shape, (1, height, width, 3))
        self.assertEqual(frame.tracks_yx.shape, (38, 2))
        self.assertEqual(frame.visibility.shape, (38,))
        self.assertFalse(bool(frame.processed_mask_frame["object"][1, 0]))
        self.assertTrue(bool(frame.processed_mask_frame["controller"][3, 0]))
        self.assertEqual(frame.source_timestamp_s, 12.5)
        self.assertEqual(frame.source_frame_index, 7)
        self.assertEqual(frame.source_step, 42)

    def test_headless_capture_bridge_uses_prepared_frames_without_window_reprocessing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_prepared_only_headless_capture(root / "capture", frame_count=2)
            base_path = root / "futurephystwin_cases"

            with mock.patch("demo_v4.headless_chunk_bridge._load_rgb", side_effect=AssertionError("legacy RGB IO")):
                with mock.patch("demo_v4.headless_chunk_bridge._load_mask_frame", side_effect=AssertionError("legacy mask IO")):
                    with mock.patch(
                        "demo_v4.headless_chunk_bridge.strict.dense_world_pcd_grid",
                        side_effect=AssertionError("legacy dense PCD"),
                    ):
                        manifests = write_chunks_from_headless_capture(
                            capture,
                            base_path=base_path,
                            case_prefix="demo_v4_prepared",
                            chunk_frame_count=2,
                            surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                            interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                        )

            self.assertEqual(len(manifests), 1)
            self.assertEqual(manifests[0]["chunk_materialization_source"], "prepared_phystwin_frame")
            self.assertEqual(manifests[0]["prepared_frame_count"], 2)
            self.assertEqual(manifests[0]["legacy_reprocess_frame_count"], 0)
            summary = validate_futurephystwin_case(base_path / "demo_v4_prepared_chunk_0001")
            self.assertTrue(summary["valid"])
            self.assertEqual(summary["frame_count"], 2)
            self.assertEqual(summary["controller_point_count"], 30)

    def test_headless_capture_bridge_can_skip_dense_final_pcd_for_final_data_cadence(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_prepared_only_headless_capture(root / "capture", frame_count=2)
            base_path = root / "futurephystwin_cases"

            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=base_path,
                case_prefix="demo_v4_final_data_only",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                write_final_pcd=False,
            )

            case_dir = base_path / manifests[0]["case_name"]
            summary = validate_futurephystwin_case(case_dir)
            self.assertTrue(summary["valid"])
            self.assertFalse((case_dir / "pcd").exists())

    def test_stable_json_reader_retries_empty_metadata_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "metadata.json"
            path.write_text("", encoding="utf-8")

            def finish_write() -> None:
                path.write_text(json.dumps({"shape_prior_status": "ready"}), encoding="utf-8")

            timer = threading.Timer(0.05, finish_write)
            timer.start()
            try:
                metadata = _read_json_file_stable(
                    path,
                    deadline_s=time.monotonic() + 1.0,
                    poll_interval_s=0.01,
                )
            finally:
                timer.cancel()

            self.assertEqual(metadata["shape_prior_status"], "ready")

    def test_headless_capture_bridge_filters_depth_invalid_mask_pixels(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=2)
            for seq in range(2):
                depth_path = capture / "depth_color_m" / f"{seq:06d}.npy"
                depth = np.load(depth_path)
                depth[3, 0] = 0.0
                np.save(depth_path, depth)

            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=root / "cases",
                case_prefix="demo_v4_depth_valid",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
            )

            case_dir = root / "cases" / manifests[0]["case_name"]
            with (case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            self.assertFalse(bool(processed_masks[0][0]["controller"][3, 0]))
            with (case_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertTrue(np.all(np.linalg.norm(final_data["controller_points"][0], axis=1) > 1e-9))

    def test_headless_capture_bridge_applies_radius_outlier_mask_refinement(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=2)
            for seq in range(2):
                depth_path = capture / "depth_color_m" / f"{seq:06d}.npy"
                depth = np.load(depth_path)
                depth[7, 39] = 10.0
                np.save(depth_path, depth)

                mask_path = capture / "masks" / f"{seq:06d}.npz"
                payload = np.load(mask_path, allow_pickle=False)
                controller = np.asarray(payload["controller_mask"], dtype=bool)
                hand_a = np.asarray(payload["hand_a_mask"], dtype=bool)
                controller[7, 39] = True
                hand_a[7, 39] = True
                np.savez(
                    mask_path,
                    object_mask=np.asarray(payload["object_mask"], dtype=bool),
                    controller_mask=controller,
                    hand_a_mask=hand_a,
                    hand_b_mask=np.asarray(payload["hand_b_mask"], dtype=bool),
                )

            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=root / "cases",
                case_prefix="demo_v4_radius",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=True,
                mask_radius_outlier_radius_m=0.01,
                mask_radius_outlier_nb_points=5,
            )

            case_dir = root / "cases" / manifests[0]["case_name"]
            with (case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            self.assertFalse(bool(processed_masks[0][0]["controller"][7, 39]))
            self.assertTrue(bool(processed_masks[0][0]["controller"][3, 0]))

    def test_writer_rejects_zero_depth_controller_placeholders(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "cases"
            frame_count = 2
            track = _track_process_data(frame_count)
            track["controller_points"][:, 0, :] = 0.0
            chunk = FuturePhysTwinChunk(
                rgb_frames=_rgb_frames(frame_count),
                processed_masks=_processed_masks(frame_count),
                track_process_data=track,
                intrinsics=np.eye(3, dtype=np.float32),
                camera_to_world_c2w=np.eye(4, dtype=np.float32),
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                fps=5,
            )

            with self.assertRaisesRegex(ValueError, "zero-depth"):
                write_futurephystwin_chunk_case(base_path, "zero_controller", chunk)

    def test_headless_capture_bridge_flushes_chunks_as_streaming_buffers_fill(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=4)
            base_path = root / "cases"
            seen: list[str] = []

            def on_chunk(manifest: dict[str, object]) -> None:
                seen.append(str(manifest["case_name"]))
                if len(seen) == 1:
                    self.assertTrue((base_path / "demo_v4_stream_chunk_0001" / "final_data.pkl").is_file())
                    self.assertFalse((base_path / "demo_v4_stream_chunk_0002" / "final_data.pkl").exists())

            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=base_path,
                case_prefix="demo_v4_stream",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
                on_chunk_written=on_chunk,
            )

            self.assertEqual(seen, ["demo_v4_stream_chunk_0001", "demo_v4_stream_chunk_0002"])
            self.assertEqual([item["chunk_ready_source_seq"] for item in manifests], [1, 3])

    def test_streaming_bridge_tails_realtime_frames_until_capture_process_finishes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=0)
            rows = self._headless_rows(capture, frame_count=4)
            base_path = root / "cases"
            emitted: list[str] = []
            state = {"poll": 0, "done": False}

            def pump() -> None:
                next_index = state["poll"]
                if next_index < len(rows):
                    with (capture / "frames.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rows[next_index]) + "\n")
                    state["poll"] += 1
                else:
                    state["done"] = True

            manifests = stream_chunks_from_headless_capture(
                capture,
                base_path=base_path,
                case_prefix="demo_v4_tail",
                chunk_frame_count=2,
                fps=5,
                max_chunks=2,
                capture_finished=lambda: bool(state["done"]),
                before_poll=pump,
                poll_interval_s=0.0,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
                on_chunk_written=lambda manifest: emitted.append(str(manifest["case_name"])),
            )

            self.assertEqual(emitted, ["demo_v4_tail_chunk_0001", "demo_v4_tail_chunk_0002"])
            self.assertEqual([item["chunk_ready_source_seq"] for item in manifests], [1, 3])

    def test_streaming_manifest_records_ready_publish_cadence_and_backlog(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=0)
            rows = self._headless_rows(capture, frame_count=6)
            base_path = root / "cases"
            state = {"primed": False, "done": False}

            def pump() -> None:
                if not state["primed"]:
                    with (capture / "frames.jsonl").open("a", encoding="utf-8") as handle:
                        for row in rows:
                            handle.write(json.dumps(row) + "\n")
                    state["primed"] = True
                else:
                    state["done"] = True

            manifests = stream_chunks_from_headless_capture(
                capture,
                base_path=base_path,
                case_prefix="demo_v4_cadence",
                chunk_frame_count=2,
                fps=5,
                max_chunks=1,
                capture_finished=lambda: bool(state["done"]),
                before_poll=pump,
                poll_interval_s=0.0,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
            )

            self.assertEqual(len(manifests), 1)
            manifest = manifests[0]
            self.assertEqual(manifest["source_window_start_s"], 0.0)
            self.assertEqual(manifest["source_window_end_s"], 0.4)
            for key in (
                "window_closed_wall_s",
                "track_finalize_done_wall_s",
                "final_data_written_wall_s",
                "validation_done_wall_s",
                "atomic_rename_done_wall_s",
                "publish_latency_ms",
            ):
                self.assertIn(key, manifest)
            self.assertGreaterEqual(manifest["track_finalize_done_wall_s"], manifest["window_closed_wall_s"])
            self.assertGreaterEqual(manifest["final_data_written_wall_s"], manifest["track_finalize_done_wall_s"])
            self.assertGreaterEqual(manifest["validation_done_wall_s"], manifest["final_data_written_wall_s"])
            self.assertGreaterEqual(manifest["atomic_rename_done_wall_s"], manifest["validation_done_wall_s"])
            self.assertGreaterEqual(manifest["materialize_end_wall_s"], manifest["materialize_start_wall_s"])
            self.assertEqual(manifest["publish_wall_s"], manifest["atomic_rename_done_wall_s"])
            self.assertGreaterEqual(manifest["materialize_latency_ms"], 0)
            self.assertAlmostEqual(
                manifest["publish_latency_ms"],
                (manifest["atomic_rename_done_wall_s"] - manifest["window_closed_wall_s"]) * 1000.0,
                places=3,
            )
            self.assertAlmostEqual(
                manifest["publish_lag_ms"],
                (manifest["atomic_rename_done_wall_s"] - manifest["source_window_end_s"]) * 1000.0,
                places=3,
            )
            self.assertEqual(manifest["backlog_chunks"], 2)

            manifest_path = base_path / "demo_v4_cadence_chunk_0001" / "manifest.json"
            persisted = json.loads(manifest_path.read_text(encoding="utf-8"))
            self.assertEqual(persisted["backlog_chunks"], 2)
            self.assertEqual(persisted["publish_wall_s"], manifest["publish_wall_s"])
            self.assertEqual(persisted["atomic_rename_done_wall_s"], manifest["atomic_rename_done_wall_s"])
            self.assertEqual(persisted["publish_latency_ms"], manifest["publish_latency_ms"])

    def test_streaming_bridge_waits_for_shape_prior_when_required(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=0)
            rows = self._headless_rows(capture, frame_count=2)
            state = {"poll": 0, "shape_written": False}

            def pump() -> None:
                if state["poll"] < len(rows):
                    with (capture / "frames.jsonl").open("a", encoding="utf-8") as handle:
                        handle.write(json.dumps(rows[state["poll"]]) + "\n")
                elif not state["shape_written"]:
                    shape_dir = capture / "shape_prior"
                    shape_dir.mkdir(exist_ok=True)
                    np.savez(
                        shape_dir / "points.npz",
                        points_m=np.array([[0.0, 0.0, -0.02]], dtype=np.float32),
                        colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                        surface_points_m=np.array([[0.0, 0.0, -0.02]], dtype=np.float32),
                        interior_points_m=np.array([[0.01, 0.0, -0.03]], dtype=np.float32),
                    )
                    metadata = json.loads((capture / "metadata.json").read_text(encoding="utf-8"))
                    metadata["shape_prior_status"] = "ready"
                    metadata["shape_prior_path"] = "shape_prior/points.npz"
                    (capture / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
                    state["shape_written"] = True
                state["poll"] += 1

            manifests = stream_chunks_from_headless_capture(
                capture,
                base_path=root / "cases",
                case_prefix="demo_v4_shape_wait",
                chunk_frame_count=2,
                fps=5,
                max_chunks=1,
                capture_finished=lambda: bool(state["shape_written"]),
                before_poll=pump,
                poll_interval_s=0.0,
                require_shape_prior=True,
                shape_prior_wait_timeout_s=1.0,
                mask_radius_outlier_filter=False,
            )

            self.assertEqual(len(manifests), 1)
            summary = validate_futurephystwin_case(root / "cases" / "demo_v4_shape_wait_chunk_0001")
            self.assertEqual(summary["surface_point_count"], 1)
            self.assertEqual(summary["interior_point_count"], 1)

    def test_demo_v4_cli_launches_demo32_and_streams_chunks_without_source_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = root / "capture"
            base_path = root / "cases"

            class FakeProcess:
                returncode = 0

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def fake_stream(capture_dir_arg, **kwargs):
                self.assertEqual(Path(capture_dir_arg), capture_dir)
                np.testing.assert_allclose(kwargs["surface_points"], np.array([[0.0, 0.0, -0.02]], dtype=np.float64))
                np.testing.assert_allclose(kwargs["interior_points"], np.array([[0.01, 0.0, -0.03]], dtype=np.float64))
                manifest = {
                    "case_name": "demo_v4_rt_chunk_0001",
                    "frame_count": 25,
                    "futurephystwin_case_root": str(base_path / "demo_v4_rt_chunk_0001"),
                    "publish_wall_s": 12.5,
                    "source_window_end_s": 5.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                return [manifest]

            with mock.patch("demo_v4.realtime_futurephystwin_chunks.subprocess.Popen", return_value=FakeProcess()) as popen:
                with mock.patch("demo_v4.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v4_main(
                            [
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v4_rt",
                                "--demo32-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--demo32-source-replay-fps",
                                "5.2",
                                "--demo32-lossless-max-backlog-seconds",
                                "24",
                                "--surface-points-npy",
                                str(self._write_points(root / "surface.npy", [[0.0, 0.0, -0.02]])),
                                "--interior-points-npy",
                                str(self._write_points(root / "interior.npy", [[0.01, 0.0, -0.03]])),
                            ]
                        )

            self.assertEqual(exit_code, 0)
            command = popen.call_args.args[0]
            env = popen.call_args.kwargs["env"]
            self.assertEqual(command[command.index("--input-source") + 1], "fake-live")
            self.assertEqual(command[command.index("--replay-fps") + 1], "5.2")
            self.assertEqual(command[command.index("--track-mode") + 1], "controller-object")
            self.assertEqual(command[command.index("--tracker-backend") + 1], "tapnextpp")
            self.assertEqual(command[command.index("--headless-capture-dir") + 1], str(capture_dir))
            self.assertEqual(command[command.index("--lossless-max-backlog-seconds") + 1], "24.0")
            self.assertIn("--headless-prepared-only", command)
            self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "0")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["mode"], "full-fake-realtime-camera")
            self.assertEqual(summary["gpu_mode"], "single")
            self.assertEqual(summary["demo32_cuda_visible_devices"], "0")
            self.assertEqual(summary["demo32_source_replay_fps"], 5.2)
            self.assertEqual(summary["demo32_source_replay_fps_override"], 5.2)
            self.assertEqual(summary["demo32_lossless_input_fps"], 5.2)
            self.assertEqual(summary["demo32_lossless_max_backlog_seconds"], 24.0)
            self.assertTrue(summary["demo32_headless_prepared_only"])
            self.assertEqual(summary["chunk_count"], 1)
            self.assertEqual(summary["first_ready_chunk_wall_s"], 12.5)
            self.assertEqual(summary["first_shape_prior_ready_chunk_wall_s"], 12.5)
            self.assertEqual(summary["max_backlog_chunks"], 0)

    def test_demo_v4_cli_can_use_dual_warmup_single_realtime_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = root / "capture"
            base_path = root / "cases"

            class FakeProcess:
                returncode = 0

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def fake_stream(_capture_dir_arg, **_kwargs):
                return [
                    {
                        "case_name": "demo_v4_combo_chunk_0001",
                        "frame_count": 25,
                        "futurephystwin_case_root": str(base_path / "demo_v4_combo_chunk_0001"),
                        "publish_wall_s": 18.0,
                        "source_window_end_s": 5.0,
                        "backlog_chunks": 0,
                        "shape_prior_complete": True,
                    }
                ]

            with mock.patch("demo_v4.realtime_futurephystwin_chunks.subprocess.Popen", return_value=FakeProcess()) as popen:
                with mock.patch("demo_v4.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v4_main(
                            [
                                "--realtime-gpu-mode",
                                "single",
                                "--warmup-gpu-mode",
                                "dual",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v4_combo",
                                "--demo32-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            command = popen.call_args.args[0]
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertEqual(command[command.index("--shape-prior-device") + 1], "cuda:1")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["realtime_gpu_mode"], "single")
            self.assertEqual(summary["warmup_gpu_mode"], "dual")
            self.assertEqual(summary["demo32_cuda_visible_devices"], "0")
            self.assertEqual(summary["shape_prior_device"], "cuda:1")
            self.assertEqual(summary["first_ready_chunk_wall_s"], 18.0)

    def test_demo_v4_cli_dual_gpu_mode_routes_demo32_to_gpu1(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = root / "capture"
            base_path = root / "cases"

            class FakeProcess:
                returncode = 0

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def fake_stream(_capture_dir_arg, **_kwargs):
                return [
                    {
                        "case_name": "demo_v4_dual_chunk_0001",
                        "frame_count": 25,
                        "futurephystwin_case_root": str(base_path / "demo_v4_dual_chunk_0001"),
                    }
                ]

            with mock.patch("demo_v4.realtime_futurephystwin_chunks.subprocess.Popen", return_value=FakeProcess()) as popen:
                with mock.patch("demo_v4.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v4_main(
                            [
                                "--gpu-mode",
                                "dual",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v4_dual",
                                "--demo32-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--surface-points-npy",
                                str(self._write_points(root / "surface.npy", [[0.0, 0.0, -0.02]])),
                                "--interior-points-npy",
                                str(self._write_points(root / "interior.npy", [[0.01, 0.0, -0.03]])),
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["gpu_mode"], "dual")
            self.assertEqual(summary["demo32_cuda_visible_devices"], "1")

    def test_demo_v4_cli_live_launches_demo32_headless_strict_capture(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture_dir = root / "live_capture"
            base_path = root / "cases"

            class FakeProcess:
                returncode = 0

                def poll(self):
                    return 0

                def wait(self):
                    return 0

            def fake_stream(capture_dir_arg, **kwargs):
                self.assertEqual(Path(capture_dir_arg), capture_dir)
                return [
                    {
                        "case_name": "demo_v4_live_chunk_0001",
                        "frame_count": 25,
                        "futurephystwin_case_root": str(base_path / "demo_v4_live_chunk_0001"),
                    }
                ]

            with mock.patch("demo_v4.realtime_futurephystwin_chunks.subprocess.Popen", return_value=FakeProcess()) as popen:
                with mock.patch("demo_v4.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v4_main(
                            [
                                "--input-source",
                                "live",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v4_live",
                                "--demo32-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            command = popen.call_args.args[0]
            self.assertEqual(command[command.index("--input-source") + 1], "live")
            self.assertEqual(command[command.index("--track-mode") + 1], "controller-object")
            self.assertEqual(command[command.index("--tracker-backend") + 1], "tapnextpp")
            self.assertEqual(command[command.index("--headless-capture-dir") + 1], str(capture_dir))
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["mode"], "full-live-camera")
            self.assertEqual(summary["chunk_count"], 1)

    def test_headless_capture_bridge_expands_active_query_subsets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            capture = self._write_minimal_headless_capture(root / "capture", frame_count=2, sparse_second_frame=True)
            manifests = write_chunks_from_headless_capture(
                capture,
                base_path=root / "cases",
                case_prefix="demo_v4_sparse",
                chunk_frame_count=2,
                surface_points=np.array([[0.0, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.array([[0.01, 0.0, -0.03]], dtype=np.float64),
                mask_radius_outlier_filter=False,
            )

            self.assertEqual(len(manifests), 1)
            tracking = np.load(root / "cases" / "demo_v4_sparse_chunk_0001" / "tracking" / "0.npz")
            self.assertEqual(tracking["tracks"].shape, (2, 38, 2))
            self.assertFalse(bool(tracking["visibility"][1, 0]))
            summary = validate_futurephystwin_case(root / "cases" / "demo_v4_sparse_chunk_0001")
            self.assertTrue(summary["valid"])

    def test_writer_uses_shape_prior_in_volume_sampling_min_bound(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base_path = Path(tmp) / "cases"
            frame_count = 2
            track = _track_process_data(frame_count)
            track["object_points"] = np.array(
                [
                    [[0.0000, 0.0, -0.02], [0.0049, 0.0, -0.02]],
                    [[0.0010, 0.0, -0.02], [0.0059, 0.0, -0.02]],
                ],
                dtype=np.float64,
            )
            track["object_colors"] = np.ones((frame_count, 2, 3), dtype=np.float64)
            track["object_visibilities"] = np.ones((frame_count, 2), dtype=bool)
            track["object_motions_valid"] = np.ones((frame_count, 2), dtype=bool)
            chunk = FuturePhysTwinChunk(
                rgb_frames=_rgb_frames(frame_count),
                processed_masks=_processed_masks(frame_count),
                track_process_data=track,
                intrinsics=np.eye(3, dtype=np.float32),
                camera_to_world_c2w=np.eye(4, dtype=np.float32),
                surface_points=np.array([[-0.0010, 0.0, -0.02]], dtype=np.float64),
                interior_points=np.empty((0, 3), dtype=np.float64),
                fps=5,
            )

            write_futurephystwin_chunk_case(base_path, "shape_min_bound", chunk)

            with (base_path / "shape_min_bound" / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertEqual(final_data["object_points"].shape[1], 2)

    def test_single_view_sam3d_sampling_matches_data_process_sam3d_targets(self) -> None:
        vertices = np.array(
            [
                [-0.05, -0.05, -0.10],
                [0.05, -0.05, -0.10],
                [0.05, 0.05, -0.10],
                [-0.05, 0.05, -0.10],
                [-0.05, -0.05, 0.0],
                [0.05, -0.05, 0.0],
                [0.05, 0.05, 0.0],
                [-0.05, 0.05, 0.0],
            ],
            dtype=np.float32,
        )
        faces = np.array(
            [
                [0, 1, 2],
                [0, 2, 3],
                [4, 6, 5],
                [4, 7, 6],
                [0, 4, 5],
                [0, 5, 1],
                [1, 5, 6],
                [1, 6, 2],
                [2, 6, 7],
                [2, 7, 3],
                [3, 7, 4],
                [3, 4, 0],
            ],
            dtype=np.int64,
        )
        mesh = SimpleShapeMesh(vertices=vertices, faces=faces)
        axis = np.linspace(-0.045, 0.045, 10, dtype=np.float32)
        reference_points = np.stack(
            np.meshgrid(axis, axis, np.linspace(-0.095, -0.005, 10, dtype=np.float32), indexing="ij"),
            axis=-1,
        ).reshape(-1, 3)

        samples = sample_data_process_sam3d_single_view_shape_prior_points(
            mesh,
            reference_points,
            target_surface_points=700,
            target_interior_points=1000,
            shape_prior_max_dist_m=0.08,
        )

        self.assertEqual(samples.surface_points_m.shape, (700, 3))
        self.assertEqual(samples.interior_points_m.shape, (1000, 3))
        self.assertEqual(samples.metadata["single_view_shape_prior_sampling_backend"], "sam3d-single-view")
        self.assertFalse(samples.metadata["uses_mvsam3d"])
        self.assertEqual(samples.metadata["shape_prior_target_surface_points"], 700)
        self.assertEqual(samples.metadata["shape_prior_target_interior_points"], 1000)
        self.assertEqual(samples.metadata["shape_prior_configured_max_dist_m"], 0.08)
        self.assertEqual(samples.metadata["shape_prior_effective_max_dist_m"], 0.08)
        self.assertEqual(samples.metadata["shape_prior_distance_policy"], "canonical_single_view_configured")
        self.assertTrue(samples.metadata["offline_single_view_parity"])

    def test_select_validation_chunks_uses_second_last_and_fifth_last(self) -> None:
        manifests = [{"case_name": f"chunk_{idx:04d}"} for idx in range(1, 8)]

        selected = select_validation_chunk_cases(manifests)

        self.assertEqual(selected, ["chunk_0006", "chunk_0003"])

    def _write_points(self, path: Path, points: list[list[float]]) -> Path:
        np.save(path, np.asarray(points, dtype=np.float64))
        return path

    def _write_minimal_headless_capture(
        self,
        capture: Path,
        *,
        frame_count: int,
        sparse_second_frame: bool = False,
    ) -> Path:
        for name in ("masks", "depth_color_m", "rgb", "query_trajectory"):
            (capture / name).mkdir(parents=True, exist_ok=True)
        metadata = {
            "depth_backend": "native-realsense",
            "depth_source_internal": "realsense",
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 0.0, "cy": 0.0},
            "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
        }
        (capture / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

        height, width = 8, 40
        object_mask = np.zeros((height, width), dtype=bool)
        controller_mask = np.zeros((height, width), dtype=bool)
        object_mask[1, :6] = True
        controller_mask[3, :32] = True
        query_points = np.array(
            [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
            dtype=np.float32,
        )
        rows: list[dict[str, object]] = []
        for seq in range(frame_count):
            np.save(capture / "depth_color_m" / f"{seq:06d}.npy", np.ones((height, width), dtype=np.float32))
            Image.fromarray(np.full((height, width, 3), 100 + seq, dtype=np.uint8), mode="RGB").save(
                capture / "rgb" / f"{seq:06d}.png"
            )
            np.savez(
                capture / "masks" / f"{seq:06d}.npz",
                object_mask=object_mask,
                controller_mask=controller_mask,
                hand_a_mask=controller_mask,
                hand_b_mask=np.zeros_like(controller_mask),
            )
            if sparse_second_frame and seq == 1:
                active_indices = np.arange(1, len(query_points), dtype=np.int64)
                np.savez(
                    capture / "query_trajectory" / f"{seq:06d}.npz",
                    seq=np.asarray([seq], dtype=np.int64),
                    query_points_yx=query_points,
                    query_indices=active_indices,
                    tracks_yx=query_points[active_indices],
                    visibility=np.ones((len(active_indices),), dtype=bool),
                )
            else:
                np.savez(
                    capture / "query_trajectory" / f"{seq:06d}.npz",
                    seq=np.asarray([seq], dtype=np.int64),
                    query_points_yx=query_points,
                    all_tracks_yx=query_points,
                    all_tracker_visibility=np.ones((len(query_points),), dtype=bool),
                )
            rows.append(
                {
                    "seq": seq,
                    "depth_color_m_path": f"depth_color_m/{seq:06d}.npy",
                    "rgb_path": f"rgb/{seq:06d}.png",
                    "mask_path": f"masks/{seq:06d}.npz",
                    "query_trajectory_path": f"query_trajectory/{seq:06d}.npz",
                }
            )
        with (capture / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return capture

    def _write_prepared_only_headless_capture(self, capture: Path, *, frame_count: int) -> Path:
        (capture / "prepared_phystwin").mkdir(parents=True, exist_ok=True)
        metadata = {
            "depth_backend": "native-realsense",
            "depth_source_internal": "realsense",
            "intrinsics": {"fx": 1000.0, "fy": 1000.0, "cx": 0.0, "cy": 0.0},
            "camera_to_world_c2w": np.eye(4, dtype=np.float32).tolist(),
        }
        (capture / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
        height, width = 8, 40
        object_mask = np.zeros((height, width), dtype=bool)
        controller_mask = np.zeros((height, width), dtype=bool)
        object_mask[1, :6] = True
        controller_mask[3, :32] = True
        query_points = np.array(
            [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
            dtype=np.float32,
        )
        rows: list[dict[str, object]] = []
        for seq in range(frame_count):
            frame = strict.prepare_phystwin_frame(
                seq=seq,
                rgb_frame=np.full((height, width, 3), 100 + seq, dtype=np.uint8),
                depth_m=np.ones((height, width), dtype=np.float32),
                mask_frame={"object": object_mask, "controller": controller_mask},
                tracks_yx=query_points,
                visibility=np.ones((len(query_points),), dtype=bool),
                query_points_yx=query_points,
                intrinsics=metadata["intrinsics"],
                c2w=np.eye(4, dtype=np.float32),
                mask_radius_outlier_filter=False,
            )
            prepared_path = capture / "prepared_phystwin" / f"{seq:06d}.npz"
            strict.write_prepared_phystwin_frame(prepared_path, frame)
            rows.append(
                {
                    "seq": seq,
                    "prepared_phystwin_frame_path": f"prepared_phystwin/{seq:06d}.npz",
                }
            )
        with (capture / "frames.jsonl").open("w", encoding="utf-8") as handle:
            for row in rows:
                handle.write(json.dumps(row) + "\n")
        return capture

    def _headless_rows(self, capture: Path, *, frame_count: int) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        height, width = 8, 40
        object_mask = np.zeros((height, width), dtype=bool)
        controller_mask = np.zeros((height, width), dtype=bool)
        object_mask[1, :6] = True
        controller_mask[3, :32] = True
        query_points = np.array(
            [[1.0, float(x)] for x in range(6)] + [[3.0, float(x)] for x in range(32)],
            dtype=np.float32,
        )
        for seq in range(frame_count):
            np.save(capture / "depth_color_m" / f"{seq:06d}.npy", np.ones((height, width), dtype=np.float32))
            Image.fromarray(np.full((height, width, 3), 100 + seq, dtype=np.uint8), mode="RGB").save(
                capture / "rgb" / f"{seq:06d}.png"
            )
            np.savez(
                capture / "masks" / f"{seq:06d}.npz",
                object_mask=object_mask,
                controller_mask=controller_mask,
                hand_a_mask=controller_mask,
                hand_b_mask=np.zeros_like(controller_mask),
            )
            np.savez(
                capture / "query_trajectory" / f"{seq:06d}.npz",
                seq=np.asarray([seq], dtype=np.int64),
                query_points_yx=query_points,
                all_tracks_yx=query_points,
                all_tracker_visibility=np.ones((len(query_points),), dtype=bool),
            )
            rows.append(
                {
                    "seq": seq,
                    "depth_color_m_path": f"depth_color_m/{seq:06d}.npy",
                    "rgb_path": f"rgb/{seq:06d}.png",
                    "mask_path": f"masks/{seq:06d}.npz",
                    "query_trajectory_path": f"query_trajectory/{seq:06d}.npz",
                }
            )
        return rows


if __name__ == "__main__":
    unittest.main()
