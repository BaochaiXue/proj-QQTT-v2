from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import numpy as np

from data_process.visualization.io_case import load_case_frame_cloud_with_sources
from data_process.visualization.io_case import load_case_metadata
from data_process.visualization.workflows.masked_pointcloud_compare import (
    load_union_masks_for_camera_clouds,
    run_masked_pointcloud_compare_workflow,
)
from scripts.harness.visual_compare_masked_pointcloud import resolve_mask_roots_for_compare
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class MaskedPointcloudCompareSmokeTest(unittest.TestCase):
    def test_load_union_masks_unions_multiple_object_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            case_dir = tmp_root / "case"
            make_visualization_case(case_dir, frame_num=1)
            mask_root = make_sam31_masks(case_dir, prompt_labels_by_object={1: "sloth", 2: "sloth"})
            metadata = load_case_metadata(case_dir)
            _, _, _, per_camera_clouds = load_case_frame_cloud_with_sources(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="realsense",
                use_float_ffs_depth_when_available=True,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.0,
                depth_max_m=1.5,
            )

            masks, debug = load_union_masks_for_camera_clouds(
                mask_root=mask_root,
                camera_clouds=per_camera_clouds,
                frame_token="0",
                text_prompt="sloth",
            )

            self.assertEqual(sorted(debug[0]["matched_object_ids"]), [1, 2])
            self.assertEqual(sorted(debug[0]["loaded_object_ids"]), [1, 2])
            self.assertEqual(int(np.count_nonzero(masks[0])), 18)

    def test_resolve_mask_roots_same_case_reuses_single_existing_suite(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            case_dir = aligned_root / "both_case"
            make_visualization_case(case_dir, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)
            make_sam31_masks(case_dir)

            result = resolve_mask_roots_for_compare(
                aligned_root=aligned_root,
                case_name="both_case",
                realsense_case=None,
                ffs_case=None,
                output_dir=tmp_root / "output",
                text_prompt="sloth",
                camera_ids=[0, 1, 2],
                frame_idx=0,
                mask_source_mode="reuse_or_generate",
                checkpoint=None,
                source_mode="auto",
            )

            self.assertTrue(result["same_case_mode"])
            self.assertEqual(result["native"]["mask_root"], result["ffs"]["mask_root"])
            self.assertEqual(result["native"]["mask_source"], "reused_existing")
            self.assertEqual(result["ffs"]["mask_source"], "reused_existing")

    def test_require_existing_fails_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)

            with self.assertRaises(FileNotFoundError):
                resolve_mask_roots_for_compare(
                    aligned_root=aligned_root,
                    case_name=None,
                    realsense_case="native_case",
                    ffs_case="ffs_case",
                    output_dir=tmp_root / "output",
                    text_prompt="sloth",
                    camera_ids=[0, 1, 2],
                    frame_idx=0,
                    mask_source_mode="require_existing",
                    checkpoint=None,
                    source_mode="auto",
                )

    def test_reuse_or_generate_calls_qqtt_sam31_helper_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)
            calls: list[tuple[str, str]] = []

            def _fake_run_case_segmentation(**kwargs):
                calls.append((str(kwargs["case_root"]), str(kwargs["output_dir"])))
                output_dir = Path(kwargs["output_dir"]).resolve()
                make_sam31_masks(Path(kwargs["case_root"]), frame_tokens=["0"])
                output_dir.mkdir(parents=True, exist_ok=True)
                (output_dir / "mask").mkdir(parents=True, exist_ok=True)
                for path in (Path(kwargs["case_root"]) / "sam31_masks" / "mask").rglob("*"):
                    if path.is_dir():
                        (output_dir / "mask" / path.relative_to(Path(kwargs["case_root"]) / "sam31_masks" / "mask")).mkdir(parents=True, exist_ok=True)
                for path in (Path(kwargs["case_root"]) / "sam31_masks" / "mask").rglob("*.png"):
                    target = output_dir / "mask" / path.relative_to(Path(kwargs["case_root"]) / "sam31_masks" / "mask")
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
                for path in (Path(kwargs["case_root"]) / "sam31_masks" / "mask").glob("mask_info_*.json"):
                    target = output_dir / "mask" / path.name
                    target.parent.mkdir(parents=True, exist_ok=True)
                    target.write_bytes(path.read_bytes())
                return {"output_dir": str(output_dir)}

            with mock.patch("scripts.harness.sam31_mask_helper.run_case_segmentation", side_effect=_fake_run_case_segmentation):
                result = resolve_mask_roots_for_compare(
                    aligned_root=aligned_root,
                    case_name=None,
                    realsense_case="native_case",
                    ffs_case="ffs_case",
                    output_dir=tmp_root / "output",
                    text_prompt="sloth",
                    camera_ids=[0, 1, 2],
                    frame_idx=0,
                    mask_source_mode="reuse_or_generate",
                    checkpoint=None,
                    source_mode="auto",
                )

            self.assertEqual(len(calls), 2)
            self.assertEqual(result["native"]["mask_source"], "generated_now")
            self.assertEqual(result["ffs"]["mask_source"], "generated_now")
            self.assertTrue((Path(result["native"]["mask_root"]) / "mask").is_dir())
            self.assertTrue((Path(result["ffs"]["mask_root"]) / "mask").is_dir())

    def test_workflow_writes_board_summary_and_respects_mask_counts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)
            native_mask_root = make_sam31_masks(native_case)
            ffs_mask_root = make_sam31_masks(ffs_case)
            output_dir = tmp_root / "masked_output"

            summary = run_masked_pointcloud_compare_workflow(
                aligned_root=aligned_root,
                output_dir=output_dir,
                realsense_case="native_case",
                ffs_case="ffs_case",
                frame_idx=0,
                text_prompt="sloth",
                native_mask_root=native_mask_root,
                ffs_mask_root=ffs_mask_root,
                native_mask_source="reused_existing",
                ffs_mask_source="reused_existing",
                mask_source_mode="reuse_or_generate",
                render_frame_fn=lambda points, colors, **kwargs: np.full((96, 128, 3), 100, dtype=np.uint8),
            )

            self.assertTrue((output_dir / "01_masked_pointcloud_board.png").is_file())
            self.assertTrue((output_dir / "summary.json").is_file())
            self.assertTrue((output_dir / "debug" / "native_unmasked_fused.ply").is_file())
            self.assertTrue((output_dir / "debug" / "native_masked_fused.ply").is_file())
            self.assertTrue((output_dir / "debug" / "ffs_unmasked_fused.ply").is_file())
            self.assertTrue((output_dir / "debug" / "ffs_masked_fused.ply").is_file())
            self.assertEqual(summary["mask_source_mode"], "reuse_or_generate")
            self.assertEqual(summary["shared_view"]["view_name"], "oblique")
            self.assertFalse(summary["empty_mask_fallback_used"])
            for source_name in ("native", "ffs"):
                for camera_entry in summary["mask_sources"][source_name]["per_camera"]:
                    self.assertLessEqual(
                        int(camera_entry["post_mask_point_count"]),
                        int(camera_entry["pre_mask_point_count"]),
                    )

    def test_workflow_uses_unmasked_fallback_when_masks_are_empty(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_root = Path(tmp_dir)
            aligned_root = tmp_root / "data"
            native_case = aligned_root / "native_case"
            ffs_case = aligned_root / "ffs_case"
            make_visualization_case(native_case, frame_num=1)
            make_visualization_case(ffs_case, include_depth_ffs=True, include_depth_ffs_float_m=True, frame_num=1)
            native_mask_root = make_sam31_masks(native_case, prompt_labels_by_object={1: "bear"})
            ffs_mask_root = make_sam31_masks(ffs_case, prompt_labels_by_object={1: "bear"})
            output_dir = tmp_root / "masked_output"

            summary = run_masked_pointcloud_compare_workflow(
                aligned_root=aligned_root,
                output_dir=output_dir,
                realsense_case="native_case",
                ffs_case="ffs_case",
                frame_idx=0,
                text_prompt="sloth",
                native_mask_root=native_mask_root,
                ffs_mask_root=ffs_mask_root,
                native_mask_source="reused_existing",
                ffs_mask_source="reused_existing",
                mask_source_mode="require_existing",
                render_frame_fn=lambda points, colors, **kwargs: np.full((96, 128, 3), 140, dtype=np.uint8),
            )

            self.assertTrue(summary["empty_mask_fallback_used"])
            self.assertEqual(summary["focus_source"], "unmasked_fallback")
            self.assertTrue((output_dir / "01_masked_pointcloud_board.png").is_file())


if __name__ == "__main__":
    unittest.main()
