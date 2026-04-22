from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import cv2
import numpy as np

from scripts.harness.run_ffs_static_replay_matrix import (
    CAMERA_IDS,
    ENGINE_NAMES,
    MODEL_NAMES,
    ExperimentConfig,
    RoundCameraBenchmarkJob,
    SCALE_VALUES,
    VALID_ITERS_VALUES,
    _benchmark_round_camera_job,
    build_experiment_id,
    build_experiment_matrix,
    compose_plain_matrix_board,
    flatten_fps_values,
    load_replay_frames_for_camera,
    overall_mean_fps,
    resolve_trt_size_for_scale,
)


class StaticReplayMatrixSmokeTests(unittest.TestCase):
    def test_matrix_has_expected_size_and_unique_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            ffs_repo = root / "Fast-FoundationStereo"
            weights_root = ffs_repo / "weights"
            for model_name in MODEL_NAMES:
                model_dir = weights_root / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                (model_dir / "model_best_bp2_serialize.pth").write_text("stub", encoding="utf-8")

            configs = build_experiment_matrix(
                ffs_repo=ffs_repo,
                artifacts_root=root / "artifacts",
                max_disp=192,
            )
            self.assertEqual(len(configs), len(ENGINE_NAMES) * len(MODEL_NAMES) * len(SCALE_VALUES) * len(VALID_ITERS_VALUES))
            self.assertEqual(len({config.experiment_id for config in configs}), len(configs))
            self.assertIn(
                build_experiment_id(
                    engine="single_engine_fp32",
                    model_name="23-36-37",
                    scale=1.0,
                    valid_iters=4,
                ),
                {config.experiment_id for config in configs},
            )
            self.assertTrue(
                all(str(config.artifact_dir).startswith(str((root / "artifacts").resolve())) for config in configs)
            )

    def test_trt_scale_policy(self) -> None:
        self.assertEqual(resolve_trt_size_for_scale(1.0), (480, 864))
        self.assertEqual(resolve_trt_size_for_scale(0.75), (384, 640))
        self.assertEqual(resolve_trt_size_for_scale(0.5), (256, 448))
        with self.assertRaises(ValueError):
            resolve_trt_size_for_scale(0.33)

    def test_overall_mean_uses_all_nine_values(self) -> None:
        fps_by_round = {
            "Round 1": {0: 10.0, 1: 20.0, 2: 30.0},
            "Round 2": {0: 40.0, 1: 50.0, 2: 60.0},
            "Round 3": {0: 70.0, 1: 80.0, 2: 90.0},
        }
        self.assertEqual(len(flatten_fps_values(fps_by_round)), 9)
        self.assertAlmostEqual(overall_mean_fps(fps_by_round), 50.0)

    def test_plain_board_layout_uses_requested_grid(self) -> None:
        tile = np.full((24, 32, 3), 127, dtype=np.uint8)
        board = compose_plain_matrix_board(
            image_rows=[
                [tile.copy(), tile.copy(), tile.copy()],
                [tile.copy(), tile.copy(), tile.copy()],
                [tile.copy(), tile.copy(), tile.copy()],
            ],
            row_headers=["Round 1", "Round 2", "Round 3"],
            column_headers=[f"Cam {int(camera_idx) + 1}" for camera_idx in CAMERA_IDS],
            title="demo",
        )
        self.assertEqual(board.ndim, 3)
        self.assertEqual(board.shape[2], 3)
        self.assertGreater(board.shape[0], 3 * tile.shape[0])
        self.assertGreater(board.shape[1], 3 * tile.shape[1])

    def test_current_ppt_contract_is_one_slide_per_experiment(self) -> None:
        experiment_count = len(MODEL_NAMES) * len(SCALE_VALUES) * len(VALID_ITERS_VALUES) * len(ENGINE_NAMES)
        self.assertEqual(experiment_count, 54)
        self.assertEqual(experiment_count * 1, 54)

    def test_load_replay_frames_for_camera_sorts_numeric_frame_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            left_dir = case_dir / "ir_left" / "0"
            right_dir = case_dir / "ir_right" / "0"
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in (10, 2, 1):
                image = np.full((4, 5), frame_idx, dtype=np.uint8)
                cv2.imwrite(str(left_dir / f"{frame_idx}.png"), image)
                cv2.imwrite(str(right_dir / f"{frame_idx}.png"), image)

            replay_frames = load_replay_frames_for_camera(case_dir=case_dir, camera_idx=0)

        self.assertEqual([frame.frame_idx for frame in replay_frames], [1, 2, 10])

    def test_benchmark_round_camera_job_reads_frames_and_applies_warmup_contract(self) -> None:
        class _FakeRunner:
            def __init__(self) -> None:
                self.calls: list[tuple[np.ndarray, np.ndarray, np.ndarray, float]] = []

            def run_pair(self, left_image, right_image, *, K_ir_left, baseline_m, audit_mode=False):
                self.calls.append(
                    (
                        np.asarray(left_image).copy(),
                        np.asarray(right_image).copy(),
                        np.asarray(K_ir_left).copy(),
                        float(baseline_m),
                    )
                )
                call_index = len(self.calls)
                disparity_seed = np.full(np.asarray(left_image).shape, float(call_index), dtype=np.float32)
                return {
                    "depth_ir_left_m": disparity_seed,
                    "K_ir_left_used": np.asarray(K_ir_left, dtype=np.float32),
                }

        with tempfile.TemporaryDirectory() as tmpdir:
            case_dir = Path(tmpdir)
            left_dir = case_dir / "ir_left" / "0"
            right_dir = case_dir / "ir_right" / "0"
            left_dir.mkdir(parents=True, exist_ok=True)
            right_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(12):
                image = np.full((3, 4), frame_idx, dtype=np.uint8)
                cv2.imwrite(str(left_dir / f"{frame_idx}.png"), image)
                cv2.imwrite(str(right_dir / f"{frame_idx}.png"), image)

            config = ExperimentConfig(
                experiment_id="demo",
                engine="single_engine_fp32",
                model_name="23-36-37",
                model_path="/tmp/model.pth",
                scale=1.0,
                valid_iters=4,
                max_disp=192,
                engine_height=480,
                engine_width=864,
                artifact_dir="/tmp/artifacts",
            )
            job = RoundCameraBenchmarkJob(
                config=config,
                ffs_repo="/tmp/ffs_repo",
                round_label="Round 1",
                case_dir=str(case_dir),
                camera_idx=0,
                frame_idx=10,
                k_ir_left=np.eye(3, dtype=np.float32),
                k_color=np.eye(3, dtype=np.float32),
                t_ir_left_to_color=np.eye(4, dtype=np.float32),
                baseline_m=0.095,
                color_output_shape=(5, 6),
            )
            fake_runner = _FakeRunner()

            with mock.patch(
                "scripts.harness.run_ffs_static_replay_matrix.build_runner_for_experiment",
                return_value=fake_runner,
            ), mock.patch(
                "scripts.harness.run_ffs_static_replay_matrix._torch_synchronize",
                return_value=None,
            ), mock.patch(
                "scripts.harness.run_ffs_static_replay_matrix._cleanup_runner",
                return_value=None,
            ), mock.patch(
                "scripts.harness.run_ffs_static_replay_matrix.align_depth_to_color",
                return_value=np.full((5, 6), 7.0, dtype=np.float32),
            ), mock.patch(
                "scripts.harness.run_ffs_static_replay_matrix.time.perf_counter",
                side_effect=[100.0, 103.0],
            ):
                result = _benchmark_round_camera_job(job)

        self.assertEqual(result.round_label, "Round 1")
        self.assertEqual(result.camera_idx, 0)
        self.assertAlmostEqual(result.fps, 4.0, places=6)
        self.assertEqual(result.depth_color_m.shape, (5, 6))
        self.assertTrue(np.allclose(result.depth_color_m, 7.0))
        self.assertEqual(len(fake_runner.calls), 22)
        self.assertTrue(np.array_equal(fake_runner.calls[0][0], np.full((3, 4), 0, dtype=np.uint8)))
        self.assertTrue(np.array_equal(fake_runner.calls[-1][0], np.full((3, 4), 11, dtype=np.uint8)))


if __name__ == "__main__":
    unittest.main()
