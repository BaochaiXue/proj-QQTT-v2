from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from data_process.visualization.types import (
    AngleSelectionSummary,
    CompareCaseSelection,
    DebugArtifactSet,
    DisplayFrameContract,
    ProductArtifactSet,
    RenderOutputSpec,
    TruthPairSelectionSummary,
    ViewConfig,
)


class VisualTypesContractSmokeTest(unittest.TestCase):
    def test_render_output_spec_to_dict(self) -> None:
        spec = RenderOutputSpec(
            name="geom",
            render_mode="neutral_gray_shaded",
            video_name="orbit_compare_geom.mp4",
            gif_name="orbit_compare_geom.gif",
            sheet_name="turntable_keyframes_geom.png",
            frames_dir_name="frames_geom",
        )
        payload = spec.to_dict()
        self.assertEqual(payload["name"], "geom")
        self.assertEqual(payload["render_mode"], "neutral_gray_shaded")
        self.assertTrue(payload["write_video"])

    def test_compare_case_selection_to_dict(self) -> None:
        selection = CompareCaseSelection(
            aligned_root=Path("data"),
            native_case_dir=Path("data/native"),
            ffs_case_dir=Path("data/ffs"),
            same_case_mode=False,
            native_frame_idx=0,
            ffs_frame_idx=0,
            camera_ids=[0, 1, 2],
            serial_numbers=["a", "b", "c"],
            native_c2w=[np.eye(4, dtype=np.float32) for _ in range(3)],
        )
        payload = selection.to_dict()
        self.assertEqual(payload["camera_ids"], [0, 1, 2])
        self.assertEqual(payload["serial_numbers"], ["a", "b", "c"])

    def test_view_config_to_dict(self) -> None:
        view = ViewConfig(
            view_name="oblique",
            label="Oblique",
            center=np.zeros((3,), dtype=np.float32),
            camera_position=np.array([1.0, 1.0, 1.0], dtype=np.float32),
            up=np.array([0.0, 0.0, 1.0], dtype=np.float32),
            radius=1.5,
            camera_idx=0,
        )
        payload = view.to_dict()
        self.assertEqual(payload["view_name"], "oblique")
        self.assertEqual(payload["camera_idx"], 0)

    def test_display_frame_contract_to_dict(self) -> None:
        contract = DisplayFrameContract(
            display_frame="semantic_world",
            calibration_world_frame_kind="charuco_board_world_c2w",
            uses_semantic_world=True,
            semantic_world_frame_kind="semantic_world",
            overview_display_frame_kind="semantic_world_topdown_display",
            notes=["demo"],
        )
        payload = contract.to_dict()
        self.assertEqual(payload["display_frame"], "semantic_world")
        self.assertTrue(payload["uses_semantic_world"])

    def test_selection_and_artifact_contracts_to_dict(self) -> None:
        angle = AngleSelectionSummary(
            mode="auto",
            selected_step_idx=1,
            selected_angle_deg=12.0,
            selected_is_supported=True,
            object_projected_area_ratio=0.4,
            object_bbox_fill_ratio=0.3,
            object_multi_camera_support_ratio=0.8,
            object_mismatch_residual_m=0.01,
            context_dominance_penalty=0.2,
            silhouette_penalty=0.1,
            final_score=1.2,
            candidate_count=6,
        )
        truth = TruthPairSelectionSummary(
            src_camera_idx=0,
            dst_camera_idx=1,
            mean_valid_ratio=0.8,
            residual_gap=0.1,
            object_warp_valid_ratio_native=0.7,
            object_warp_valid_ratio_ffs=0.75,
            object_residual_mean_native=1.0,
            object_residual_mean_ffs=0.8,
            object_edge_weighted_residual_mean_native=1.3,
            object_edge_weighted_residual_mean_ffs=1.1,
            object_overlap_area=120.0,
            pair_object_visibility_score=0.9,
        )
        products = ProductArtifactSet(output_dir=Path("out"), top_level_paths={"board": "x.png"})
        debug = DebugArtifactSet(enabled=True, debug_dir=Path("out/debug"), paths={"scores": "scores.json"})
        self.assertEqual(angle.to_dict()["candidate_count"], 6)
        self.assertEqual(truth.to_dict()["dst_camera_idx"], 1)
        self.assertTrue(products.to_dict()["top_level_paths"]["board"].endswith("x.png"))
        self.assertTrue(debug.to_dict()["enabled"])
