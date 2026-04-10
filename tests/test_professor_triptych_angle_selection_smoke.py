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
                    "object_projected_area_ratio": 0.40,
                    "object_bbox_fill_ratio": 0.46,
                    "object_multi_camera_support_ratio": 0.91,
                    "object_mismatch_residual_m": 0.01,
                    "context_dominance_penalty": 0.10,
                    "silhouette_penalty": 0.08,
                    "final_score": 2.4,
                },
                {
                    "step_idx": 1,
                    "angle_deg": 18.0,
                    "is_supported": True,
                    "object_projected_area_ratio": 0.24,
                    "object_bbox_fill_ratio": 0.31,
                    "object_multi_camera_support_ratio": 0.72,
                    "object_mismatch_residual_m": 0.018,
                    "context_dominance_penalty": 0.18,
                    "silhouette_penalty": 0.12,
                    "final_score": 1.55,
                },
                {
                    "step_idx": 2,
                    "angle_deg": 4.0,
                    "is_supported": True,
                    "object_projected_area_ratio": 0.08,
                    "object_bbox_fill_ratio": 0.11,
                    "object_multi_camera_support_ratio": 0.81,
                    "object_mismatch_residual_m": 0.010,
                    "context_dominance_penalty": 0.62,
                    "silhouette_penalty": 0.78,
                    "final_score": 0.32,
                },
            ],
            angle_mode="auto",
            angle_deg=None,
        )
        self.assertEqual(int(selected["step_idx"]), 1)

    def test_explicit_angle_selects_nearest_step(self) -> None:
        selected = select_professor_angle(
            [
                {"step_idx": 0, "angle_deg": -20.0, "is_supported": True, "object_projected_area_ratio": 0.1, "object_bbox_fill_ratio": 0.12, "object_multi_camera_support_ratio": 0.7, "object_mismatch_residual_m": 0.02, "context_dominance_penalty": 0.4, "silhouette_penalty": 0.5, "final_score": 0.2},
                {"step_idx": 1, "angle_deg": 5.0, "is_supported": True, "object_projected_area_ratio": 0.2, "object_bbox_fill_ratio": 0.24, "object_multi_camera_support_ratio": 0.8, "object_mismatch_residual_m": 0.01, "context_dominance_penalty": 0.2, "silhouette_penalty": 0.2, "final_score": 1.1},
                {"step_idx": 2, "angle_deg": 22.0, "is_supported": True, "object_projected_area_ratio": 0.18, "object_bbox_fill_ratio": 0.2, "object_multi_camera_support_ratio": 0.9, "object_mismatch_residual_m": 0.02, "context_dominance_penalty": 0.3, "silhouette_penalty": 0.3, "final_score": 0.9},
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
                    "object_warp_valid_ratio_native": 0.35,
                    "object_warp_valid_ratio_ffs": 0.38,
                    "object_residual_mean_native": 16.0,
                    "object_residual_mean_ffs": 15.0,
                    "object_edge_weighted_residual_mean_native": 18.0,
                    "object_edge_weighted_residual_mean_ffs": 17.2,
                    "object_overlap_area": 420.0,
                    "pair_object_visibility_score": 0.19,
                    "native": {"residual_mean": 5.0},
                    "ffs": {"residual_mean": 4.2},
                },
                {
                    "pair": (0, 2),
                    "mean_valid_ratio": 0.72,
                    "residual_gap": 0.1,
                    "object_warp_valid_ratio_native": 0.62,
                    "object_warp_valid_ratio_ffs": 0.66,
                    "object_residual_mean_native": 9.0,
                    "object_residual_mean_ffs": 8.2,
                    "object_edge_weighted_residual_mean_native": 10.0,
                    "object_edge_weighted_residual_mean_ffs": 9.1,
                    "object_overlap_area": 980.0,
                    "pair_object_visibility_score": 0.48,
                    "native": {"residual_mean": 3.1},
                    "ffs": {"residual_mean": 2.9},
                },
            ]
        )
        self.assertEqual(tuple(selected["pair"]), (0, 2))


if __name__ == "__main__":
    unittest.main()
