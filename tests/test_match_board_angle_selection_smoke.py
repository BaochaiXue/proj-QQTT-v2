from __future__ import annotations

import unittest

from data_process.visualization.match_board import select_match_angle


class MatchBoardAngleSelectionSmokeTest(unittest.TestCase):
    def test_auto_selection_prefers_match_quality_over_large_but_bad_context_view(self) -> None:
        selected = select_match_angle(
            [
                {
                    "step_idx": 0,
                    "angle_deg": 42.0,
                    "is_supported": True,
                    "object_projected_area_ratio": 0.30,
                    "object_bbox_fill_ratio": 0.70,
                    "object_multi_camera_support_ratio": 0.18,
                    "object_mismatch_residual_m": 0.030,
                    "context_dominance_penalty": 0.55,
                    "silhouette_penalty": 0.15,
                    "final_score": -0.35,
                },
                {
                    "step_idx": 1,
                    "angle_deg": 12.0,
                    "is_supported": True,
                    "object_projected_area_ratio": 0.22,
                    "object_bbox_fill_ratio": 0.64,
                    "object_multi_camera_support_ratio": 0.42,
                    "object_mismatch_residual_m": 0.012,
                    "context_dominance_penalty": 0.18,
                    "silhouette_penalty": 0.22,
                    "final_score": 1.31,
                },
            ],
            angle_mode="auto",
            angle_deg=None,
        )
        self.assertEqual(int(selected["step_idx"]), 1)

    def test_explicit_selection_uses_nearest_angle(self) -> None:
        selected = select_match_angle(
            [
                {"step_idx": 0, "angle_deg": -20.0, "is_supported": True, "object_projected_area_ratio": 0.1, "object_bbox_fill_ratio": 0.2, "object_multi_camera_support_ratio": 0.2, "object_mismatch_residual_m": 0.02, "context_dominance_penalty": 0.2, "silhouette_penalty": 0.2, "final_score": 0.2},
                {"step_idx": 1, "angle_deg": 8.0, "is_supported": True, "object_projected_area_ratio": 0.2, "object_bbox_fill_ratio": 0.3, "object_multi_camera_support_ratio": 0.3, "object_mismatch_residual_m": 0.01, "context_dominance_penalty": 0.1, "silhouette_penalty": 0.1, "final_score": 0.9},
            ],
            angle_mode="explicit",
            angle_deg=10.0,
        )
        self.assertEqual(int(selected["step_idx"]), 1)


if __name__ == "__main__":
    unittest.main()
