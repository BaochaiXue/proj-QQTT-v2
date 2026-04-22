from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from scripts.harness.run_ffs_static_replay_matrix import (
    CAMERA_IDS,
    ENGINE_NAMES,
    MODEL_NAMES,
    SCALE_VALUES,
    VALID_ITERS_VALUES,
    build_experiment_id,
    build_experiment_matrix,
    compose_plain_matrix_board,
    flatten_fps_values,
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

    def test_trt_scale_policy(self) -> None:
        self.assertEqual(resolve_trt_size_for_scale(1.0), (480, 864))
        self.assertEqual(resolve_trt_size_for_scale(0.5), (256, 448))
        with self.assertRaises(ValueError):
            resolve_trt_size_for_scale(0.75)

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


if __name__ == "__main__":
    unittest.main()
