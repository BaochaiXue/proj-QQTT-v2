from __future__ import annotations

import unittest

from data_process.visualization.selection_contracts import (
    build_angle_selection_summary,
    build_truth_pair_selection_summary,
    select_angle_candidate,
    select_truth_pair_candidate,
)


class SelectionContractsSmokeTest(unittest.TestCase):
    def test_select_angle_candidate_prefers_ranked_supported_step(self) -> None:
        steps = [
            {"step_idx": 0, "angle_deg": -20.0, "is_supported": False, "final_score": 9.0},
            {"step_idx": 1, "angle_deg": 10.0, "is_supported": True, "final_score": 7.0},
            {"step_idx": 2, "angle_deg": 25.0, "is_supported": True, "final_score": 8.0},
        ]
        selected = select_angle_candidate(
            steps,
            angle_mode="auto",
            angle_deg=None,
            ranking_key=lambda item: (-float(item["final_score"]), int(item["step_idx"])),
        )
        self.assertEqual(selected["step_idx"], 2)

    def test_truth_pair_selection_summary_round_trips_expected_fields(self) -> None:
        pair = {
            "pair": (0, 2),
            "mean_valid_ratio": 0.7,
            "residual_gap": 0.1,
            "object_warp_valid_ratio_native": 0.6,
            "object_warp_valid_ratio_ffs": 0.8,
            "object_residual_mean_native": 1.3,
            "object_residual_mean_ffs": 1.0,
            "object_edge_weighted_residual_mean_native": 1.8,
            "object_edge_weighted_residual_mean_ffs": 1.2,
            "object_overlap_area": 120.0,
            "pair_object_visibility_score": 0.9,
            "native": {"valid_warped_pixel_ratio": 0.7},
            "ffs": {"valid_warped_pixel_ratio": 0.8},
        }
        selected = select_truth_pair_candidate([pair])
        payload = build_truth_pair_selection_summary(selected)
        self.assertEqual(payload["src_camera_idx"], 0)
        self.assertEqual(payload["dst_camera_idx"], 2)
        self.assertIn("pair_object_visibility_score", payload)

    def test_angle_selection_summary_uses_shared_fields(self) -> None:
        step = {
            "step_idx": 3,
            "angle_deg": 15.0,
            "is_supported": True,
            "object_projected_area_ratio": 0.4,
            "object_bbox_fill_ratio": 0.3,
            "object_multi_camera_support_ratio": 0.9,
            "object_mismatch_residual_m": 0.01,
            "context_dominance_penalty": 0.2,
            "silhouette_penalty": 0.1,
            "final_score": 1.5,
        }
        payload = build_angle_selection_summary(mode="auto", selected_step=step, candidate_count=8)
        self.assertEqual(payload["selected_step_idx"], 3)
        self.assertEqual(payload["candidate_count"], 8)


if __name__ == "__main__":
    unittest.main()
