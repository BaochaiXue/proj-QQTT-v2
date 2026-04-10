from __future__ import annotations

import unittest

from data_process.visualization.professor_triptych import select_professor_angle, select_truth_camera_pair


class ProfessorTriptychAngleSelectionSmokeTest(unittest.TestCase):
    def test_auto_angle_prefers_supported_high_support_low_mismatch(self) -> None:
        selected = select_professor_angle(
            [
                {
                    "step_idx": 0,
                    "angle_deg": 30.0,
                    "is_supported": False,
                    "mean_multi_camera_support": 0.95,
                    "mean_mismatch_residual_m": 0.01,
                    "mean_mismatch_p90_m": 0.02,
                },
                {
                    "step_idx": 1,
                    "angle_deg": 18.0,
                    "is_supported": True,
                    "mean_multi_camera_support": 0.84,
                    "mean_mismatch_residual_m": 0.015,
                    "mean_mismatch_p90_m": 0.03,
                },
                {
                    "step_idx": 2,
                    "angle_deg": 4.0,
                    "is_supported": True,
                    "mean_multi_camera_support": 0.84,
                    "mean_mismatch_residual_m": 0.018,
                    "mean_mismatch_p90_m": 0.04,
                },
            ],
            angle_mode="auto",
            angle_deg=None,
        )
        self.assertEqual(int(selected["step_idx"]), 1)

    def test_explicit_angle_selects_nearest_step(self) -> None:
        selected = select_professor_angle(
            [
                {"step_idx": 0, "angle_deg": -20.0, "is_supported": True, "mean_multi_camera_support": 0.7, "mean_mismatch_residual_m": 0.02, "mean_mismatch_p90_m": 0.03},
                {"step_idx": 1, "angle_deg": 5.0, "is_supported": True, "mean_multi_camera_support": 0.8, "mean_mismatch_residual_m": 0.01, "mean_mismatch_p90_m": 0.02},
                {"step_idx": 2, "angle_deg": 22.0, "is_supported": True, "mean_multi_camera_support": 0.9, "mean_mismatch_residual_m": 0.02, "mean_mismatch_p90_m": 0.04},
            ],
            angle_mode="explicit",
            angle_deg=7.0,
        )
        self.assertEqual(int(selected["step_idx"]), 1)

    def test_truth_pair_prefers_high_overlap_then_gap(self) -> None:
        selected = select_truth_camera_pair(
            [
                {
                    "pair": (0, 1),
                    "mean_valid_ratio": 0.55,
                    "residual_gap": 0.4,
                    "native": {"residual_mean": 5.0},
                    "ffs": {"residual_mean": 4.2},
                },
                {
                    "pair": (0, 2),
                    "mean_valid_ratio": 0.72,
                    "residual_gap": 0.1,
                    "native": {"residual_mean": 3.1},
                    "ffs": {"residual_mean": 2.9},
                },
            ]
        )
        self.assertEqual(tuple(selected["pair"]), (0, 2))


if __name__ == "__main__":
    unittest.main()
