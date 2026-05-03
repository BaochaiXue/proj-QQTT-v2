from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from data_process.visualization.experiments import sam21_checkpoint_ladder_panel as ladder
from data_process.visualization.experiments import sam21_mask_overlay_panel as mask_overlay
from data_process.visualization.experiments import sam21_pcd_overlay_panel as pcd_overlay
from tests.visualization_test_utils import make_sam31_masks, make_visualization_case


class Sam21CheckpointLadderPanelSmokeTest(unittest.TestCase):
    def test_bbox_xyxy_from_mask_uses_tight_frame0_union(self) -> None:
        mask = np.zeros((8, 10), dtype=bool)
        mask[2:5, 3:7] = True
        self.assertEqual(ladder.bbox_xyxy_from_mask(mask), [3.0, 2.0, 6.0, 4.0])
        self.assertEqual(ladder.bbox_xyxy_from_mask(mask, padding_px=2), [1.0, 0.0, 8.0, 6.0])
        with self.assertRaises(ValueError):
            ladder.bbox_xyxy_from_mask(np.zeros((4, 4), dtype=bool))

    def test_sam2_mask_writer_schema_loads_as_union_mask(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=2)
            mask0 = np.zeros((8, 10), dtype=bool)
            mask1 = np.zeros((8, 10), dtype=bool)
            mask0[1:4, 2:5] = True
            mask1[2:5, 3:6] = True
            mask_root = root / "sam21_masks"
            summary = ladder.write_single_object_masks(
                mask_root=mask_root,
                camera_idx=0,
                object_label="object",
                masks_by_frame_token={"0": mask0, "1": mask1},
                overwrite=False,
            )

            self.assertEqual(summary["saved_frame_count"], 2)
            loaded = ladder.load_union_mask(
                mask_root=mask_root,
                case_dir=case_dir,
                camera_idx=0,
                frame_token="1",
                text_prompt="object",
            )
            self.assertEqual(int(loaded.sum()), int(mask1.sum()))
            self.assertTrue(np.array_equal(loaded, mask1))

    def test_matched_mask_labels_handles_multi_prompt_rope_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
            mask_root = make_sam31_masks(
                case_dir,
                prompt_labels_by_object={
                    0: "white thick twisted rope on top of the blue box",
                },
                camera_ids=[2],
                frame_tokens=["0"],
            )
            labels = ladder.matched_mask_labels(
                mask_root=mask_root,
                camera_idx=2,
                text_prompt="white twisted rope on the blue box, white thick twisted rope on top of the blue box",
            )
            self.assertEqual(labels, ["white thick twisted rope on top of the blue box"])

    def test_timing_aggregate_uses_inference_ms_without_warmup(self) -> None:
        records = [
            {
                "case_key": "a",
                "camera_idx": 0,
                "checkpoint_key": "tiny",
                "warmup_propagate_ms": 999.0,
                "mask_collection_ms": 888.0,
                "inference_ms_per_frame": 10.0,
            },
            {
                "case_key": "a",
                "camera_idx": 1,
                "checkpoint_key": "tiny",
                "warmup_propagate_ms": 999.0,
                "mask_collection_ms": 888.0,
                "inference_ms_per_frame": 20.0,
            },
            {
                "case_key": "a",
                "camera_idx": 0,
                "checkpoint_key": "large",
                "warmup_propagate_ms": 999.0,
                "mask_collection_ms": 888.0,
                "inference_ms_per_frame": 50.0,
            },
        ]
        aggregate = ladder.aggregate_timing_records(records)
        self.assertEqual(aggregate["record_count"], 3)
        self.assertAlmostEqual(
            aggregate["by_checkpoint"]["tiny"]["mean_inference_ms_per_frame"],
            15.0,
        )
        self.assertAlmostEqual(aggregate["by_checkpoint"]["tiny"]["mean_fps"], 1000.0 / 15.0)

    def test_dynamics_case_specs_resolve_expected_cases(self) -> None:
        root = Path("/repo")
        cases = ladder.default_ladder_case_specs_for_set(root=root, case_set=ladder.CASE_SET_DYNAMICS)
        self.assertEqual([case.key for case in cases], ["ffs_dynamics_round1", "ffs_dynamics_round2"])
        self.assertEqual([case.text_prompt for case in cases], ["sloth", "sloth"])
        self.assertEqual(cases[0].output_name, "ffs_dynamics_round1_3x5_time")
        self.assertEqual(
            cases[1].case_dir,
            root / "data/dynamics/ffs_dynamics_round2_20260415",
        )

    def test_stable_manifest_records_mask_init_mode(self) -> None:
        case = ladder.LadderCaseSpec(
            key="dyn",
            label="Dyn",
            output_name="dyn",
            case_dir=Path("/case"),
            text_prompt="sloth",
        )
        checkpoint = ladder.default_ladder_checkpoint_specs()[0]
        manifest = ladder.build_stable_job_manifest(
            case_specs=[case],
            checkpoint_spec=checkpoint,
            checkpoint_cache=Path("/ckpt"),
            output_dir=Path("/out"),
            frames=None,
            sam21_init_mode=ladder.SAM21_INIT_MASK,
            camera_ids=[0],
        )
        self.assertIsNone(manifest["frames"])
        self.assertEqual(manifest["sam21_init_mode"], ladder.SAM21_INIT_MASK)

    def test_stable_manifest_records_external_sam31_mask_root(self) -> None:
        case = ladder.LadderCaseSpec(
            key="dyn",
            label="Dyn",
            output_name="dyn",
            case_dir=Path("/case"),
            text_prompt="sloth",
        )
        checkpoint = ladder.default_ladder_checkpoint_specs()[0]
        manifest = ladder.build_stable_job_manifest(
            case_specs=[case],
            checkpoint_spec=checkpoint,
            checkpoint_cache=Path("/ckpt"),
            output_dir=Path("/out"),
            frames=None,
            sam31_mask_root=Path("/external/sam31_masks"),
            sam21_init_mode=ladder.SAM21_INIT_MASK,
            camera_ids=[0],
        )
        self.assertEqual(manifest["sam31_mask_root"], "/external/sam31_masks")
        self.assertEqual(manifest["jobs"][0]["sam31_mask_root"], "/external/sam31_masks")

    def test_stable_report_describes_no_output_timing_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            report_path = Path(tmp_dir) / "report.md"
            ladder.write_benchmark_report(
                markdown_path=report_path,
                benchmark_payload={
                    "sam21_timing_protocol": "stable_throughput",
                    "stable_warmup_runs": 5,
                    "stable_speed_uses_cudagraph_step_marker": True,
                    "timing_aggregate": {
                        "by_checkpoint": {
                            "tiny": {
                                "mean_inference_ms_per_frame": 8.0,
                                "mean_fps": 125.0,
                                "sample_count": 18,
                            }
                        }
                    },
                    "stable_checkpoint_summaries": [
                        {
                            "checkpoint_key": "tiny",
                            "aggregate_ms_per_frame": 8.0,
                            "aggregate_fps": 125.0,
                            "speed_phase_wall_fps_including_state_setup": 80.0,
                            "total_timed_frames": 540,
                        }
                    ],
                    "gif_summaries": [],
                    "environment": {},
                },
                quality_payload={"cases": {}},
            )
            text = report_path.read_text(encoding="utf-8")
            self.assertIn("no-output propagation timing with per-step cudagraph markers", text)
            self.assertIn("separate mask collection are excluded", text)
            self.assertIn("| tiny | 8.00 | 125.00 | 80.00 | 540 |", text)

    def test_compose_3x5_panel_shape(self) -> None:
        rows = [
            [np.full((12, 16, 3), 30 + row * 20 + col, dtype=np.uint8) for col in range(5)]
            for row in range(3)
        ]
        board = ladder.compose_3x5_panel(
            title_lines=["title", "subtitle"],
            row_headers=["cam0", "cam1", "cam2"],
            column_headers=["sam31", "large", "base+", "small", "tiny"],
            image_rows=rows,
            row_label_width=20,
        )
        self.assertEqual(board.shape, (84 + 38 + 3 * 12, 20 + 5 * 16, 3))

    def test_compose_3x6_panel_shape(self) -> None:
        rows = [
            [np.full((12, 16, 3), 30 + row * 20 + col, dtype=np.uint8) for col in range(6)]
            for row in range(3)
        ]
        board = ladder.compose_panel(
            title_lines=["title", "subtitle"],
            row_headers=["cam0", "cam1", "cam2"],
            column_headers=["sam31", "large", "base+", "small", "tiny", "edgetam"],
            image_rows=rows,
            row_label_width=20,
        )
        self.assertEqual(board.shape, (84 + 38 + 3 * 12, 20 + 6 * 16, 3))

    def test_compose_2x3_pcd_overlay_panel_shape(self) -> None:
        rows = [
            [np.full((12, 16, 3), 30 + row * 20 + col, dtype=np.uint8) for col in range(3)]
            for row in range(2)
        ]
        board = ladder.compose_panel(
            title_lines=["title", "subtitle"],
            row_headers=["small", "edgetam"],
            column_headers=["cam0", "cam1", "cam2"],
            image_rows=rows,
            row_label_width=28,
            expected_rows=2,
        )
        self.assertEqual(board.shape, (84 + 38 + 2 * 12, 28 + 3 * 16, 3))

    def test_edgetam_round1_cli_defaults_to_compiled_mode(self) -> None:
        from scripts.harness.experiments.run_sam21_checkpoint_ladder_3x5_gifs import parse_args

        args = parse_args(["--edgetam-round1-3x6"])
        self.assertEqual(args.edgetam_compile_mode, ladder.EDGETAM_COMPILE_NO_POS_CACHE)

    def test_mask_overlay_stats_and_render_shape(self) -> None:
        reference = np.zeros((8, 10), dtype=bool)
        candidate = np.zeros((8, 10), dtype=bool)
        reference[2:6, 2:6] = True
        candidate[3:7, 3:7] = True
        stats = mask_overlay.compare_masks(reference, candidate)
        self.assertEqual(stats["intersection_pixel_count"], 9)
        self.assertEqual(stats["union_pixel_count"], 23)
        self.assertAlmostEqual(stats["iou"], 9.0 / 23.0)

        color = np.full((8, 10, 3), 80, dtype=np.uint8)
        tile, tile_stats = mask_overlay.build_overlay_tile(
            color_bgr=color,
            reference_mask=reference,
            candidate_mask=candidate,
            variant_label="tiny",
            tile_width=64,
            tile_height=48,
        )
        self.assertEqual(tile.shape, (48, 64, 3))
        self.assertEqual(tile_stats["candidate_only_pixel_count"], 7)

    def test_mask_overlay_black_union_rgb_background(self) -> None:
        reference = np.zeros((8, 10), dtype=bool)
        candidate = np.zeros((8, 10), dtype=bool)
        reference[2:6, 2:6] = True
        candidate[3:7, 3:7] = True
        color = np.full((8, 10, 3), 90, dtype=np.uint8)
        overlay = mask_overlay.render_mask_difference_overlay(
            color,
            reference,
            candidate,
            background_mode="black_union_rgb",
            color_overlap=False,
        )
        self.assertTrue(np.array_equal(overlay[0, 0], np.zeros(3, dtype=np.uint8)))
        self.assertTrue(np.any(overlay[2, 2] > 0))
        self.assertTrue(np.any(overlay[3, 3] > 0))

    def test_pcd_overlay_category_coloring_and_point_iou(self) -> None:
        reference = np.asarray([[True, True], [False, False]], dtype=bool)
        candidate = np.asarray([[True, False], [True, False]], dtype=bool)
        camera_clouds = [
            {
                "camera_idx": 0,
                "points": np.asarray(
                    [
                        [0.0, 0.0, 1.0],
                        [1.0, 0.0, 1.0],
                        [0.0, 1.0, 1.0],
                        [1.0, 1.0, 1.0],
                    ],
                    dtype=np.float32,
                ),
                "colors": np.asarray(
                    [
                        [10, 20, 30],
                        [40, 50, 60],
                        [70, 80, 90],
                        [100, 110, 120],
                    ],
                    dtype=np.uint8,
                ),
                "source_pixel_uv": np.asarray([[0, 0], [1, 0], [0, 1], [1, 1]], dtype=np.int32),
            }
        ]
        points, colors, categories, stats = pcd_overlay.build_category_colored_cloud(
            camera_clouds=camera_clouds,
            reference_masks={0: reference},
            candidate_masks={0: candidate},
        )
        self.assertEqual(len(points), 3)
        self.assertTrue(np.array_equal(colors[0], np.asarray([10, 20, 30], dtype=np.uint8)))
        self.assertTrue(np.array_equal(colors[1], pcd_overlay.SAM31_ONLY_BGR))
        self.assertTrue(np.array_equal(colors[2], pcd_overlay.CANDIDATE_ONLY_BGR))
        self.assertEqual(pcd_overlay.category_counts(categories)["overlap_point_count"], 1)
        self.assertAlmostEqual(stats["point_weighted_iou"], 1.0 / 3.0)

    def test_pcd_overlay_depth_scale_override_is_recorded_for_missing_metadata(self) -> None:
        metadata, used = pcd_overlay.metadata_with_depth_scale_override(
            {"serial_numbers": ["a", "b", "c"], "intrinsics": [1, 2, 3]},
            depth_scale_override_m_per_unit=0.001,
        )
        self.assertTrue(used)
        self.assertEqual(metadata["depth_scale_m_per_unit"], [0.001, 0.001, 0.001])

    def test_pcd_overlay_render_smoke_writes_2x3_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, frame_num=1)
            sam31_root = make_sam31_masks(
                case_dir,
                prompt_labels_by_object={0: "object"},
                frame_tokens=["0"],
            )
            candidate_mask = np.zeros((8, 10), dtype=bool)
            candidate_mask[1:4, 1:4] = True
            output_dir = root / "out"
            small_root = root / "small_masks"
            edgetam_root = root / "edgetam_masks"
            for camera_idx in range(3):
                ladder.write_single_object_masks(
                    mask_root=small_root,
                    camera_idx=camera_idx,
                    object_label="object",
                    masks_by_frame_token={"0": candidate_mask},
                    overwrite=False,
                )
                ladder.write_single_object_masks(
                    mask_root=edgetam_root,
                    camera_idx=camera_idx,
                    object_label="object",
                    masks_by_frame_token={"0": candidate_mask},
                    overwrite=False,
                )
            summary = pcd_overlay.render_fused_pcd_overlay_2x3_gif(
                root=root,
                case_dir=case_dir,
                output_dir=output_dir,
                text_prompt="object",
                sam31_mask_root=sam31_root,
                variant_roots={"small": small_root, "edgetam": edgetam_root},
                frames=1,
                tile_width=48,
                tile_height=32,
                row_label_width=34,
                phystwin_nb_points=1,
                max_points_per_render=None,
            )
            self.assertEqual(summary["frames"], 1)
            self.assertEqual([item["key"] for item in summary["variants"]], ["small", "edgetam"])
            self.assertTrue(Path(summary["gif_path"]).is_file())
            self.assertTrue(Path(summary["first_frame_path"]).is_file())
            self.assertTrue((Path(summary["first_frame_ply_dir"]) / "sloth_base_motion_ffs_small_frame0000_fused_pcd_overlay.ply").is_file())

    def test_pcd_overlay_single_variant_writes_1x3_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
            sam31_root = make_sam31_masks(
                case_dir,
                prompt_labels_by_object={0: "stuffed animal"},
                frame_tokens=["0"],
            )
            candidate_mask = np.zeros((8, 10), dtype=bool)
            candidate_mask[1:4, 1:4] = True
            edgetam_root = root / "edgetam_masks"
            for camera_idx in range(3):
                ladder.write_single_object_masks(
                    mask_root=edgetam_root,
                    camera_idx=camera_idx,
                    object_label="stuffed animal",
                    masks_by_frame_token={"0": candidate_mask},
                    overwrite=False,
                )

            summary = pcd_overlay.render_fused_pcd_overlay_2x3_gif(
                root=root,
                case_dir=case_dir,
                output_dir=root / "out",
                case_key="sloth_set_2_motion_ffs",
                case_label="Sloth Set 2 Motion FFS",
                text_prompt="stuffed animal",
                sam31_mask_root=sam31_root,
                variant_roots={"hf_edgetam_streaming_mask": edgetam_root},
                variants=(("hf_edgetam_streaming_mask", "HF EdgeTAM streaming"),),
                frames=1,
                tile_width=48,
                tile_height=32,
                row_label_width=54,
                phystwin_nb_points=1,
                max_points_per_render=None,
                output_name="custom_pcd_xor",
            )

            self.assertEqual(summary["frames"], 1)
            self.assertEqual([item["key"] for item in summary["variants"]], ["hf_edgetam_streaming_mask"])
            self.assertTrue(Path(summary["gif_path"]).is_file())
            self.assertTrue(Path(summary["first_frame_path"]).is_file())
            self.assertTrue(
                (
                    Path(summary["first_frame_ply_dir"])
                    / "sloth_set_2_motion_ffs_hf_edgetam_streaming_mask_frame0000_fused_pcd_overlay.ply"
                ).is_file()
            )

    def test_hf_edgetam_custom_case_args_and_mask_label(self) -> None:
        from scripts.harness.experiments import run_hf_edgetam_streaming_realcase as hf_stream

        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            sam31_root = root / "sam31_masks"
            args = hf_stream.parse_args(
                [
                    "--case-dir",
                    str(case_dir),
                    "--case-key",
                    "sloth_set_2_motion_ffs",
                    "--case-label",
                    "Sloth Set 2 Motion FFS",
                    "--text-prompt",
                    "stuffed animal",
                    "--sam31-mask-root",
                    str(sam31_root),
                    "--prompt-modes",
                    "mask",
                    "--all-frames",
                ]
            )
            cases = hf_stream._select_cases(args)
            self.assertEqual(len(cases), 1)
            self.assertEqual(cases[0].key, "sloth_set_2_motion_ffs")
            self.assertEqual(cases[0].case_dir, case_dir.resolve())
            self.assertEqual(cases[0].sam31_mask_root, sam31_root.resolve())
            self.assertIsNone(args.frames)
            self.assertEqual(hf_stream._primary_text_prompt_label("stuffed animal"), "stuffed animal")
            self.assertEqual(
                hf_stream._primary_text_prompt_label("white rope, thick rope"),
                "white rope",
            )

            import cv2

            hf_stream.cv2 = cv2
            hf_stream.np = np
            label_mask = np.zeros((8, 10), dtype=bool)
            label_mask[:4, :5] = True
            output = hf_stream._write_single_object_masks(
                mask_root=root / "out_masks",
                case_key="sloth_set_2_motion_ffs",
                prompt_mode="mask",
                object_label="stuffed animal",
                camera_idx=0,
                masks_by_frame={"0": label_mask},
                overwrite=False,
            )
            self.assertEqual(output["object_label"], "stuffed animal")
            info = json.loads(
                (root / "out_masks/sloth_set_2_motion_ffs/mask/mask/mask_info_0.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertEqual(info, {"1": "stuffed animal"})
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
            loaded = ladder.load_union_mask(
                mask_root=root / "out_masks/sloth_set_2_motion_ffs/mask",
                case_dir=case_dir,
                camera_idx=0,
                frame_token="0",
                text_prompt="stuffed animal",
            )
            self.assertEqual(int(loaded.sum()), 20)

    def test_pinhole_renderer_uses_original_camera_z_buffer(self) -> None:
        points = np.asarray(
            [
                [0.0, 0.0, 2.0],
                [0.0, 0.0, 1.0],
                [0.2, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        colors = np.asarray(
            [
                [0, 0, 255],
                [0, 255, 0],
                [255, 0, 0],
            ],
            dtype=np.uint8,
        )
        K = np.asarray([[10.0, 0.0, 5.0], [0.0, 10.0, 5.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        image = ladder.render_pinhole_point_cloud(
            points,
            colors,
            intrinsic_matrix=K,
            extrinsic_matrix=np.eye(4, dtype=np.float32),
            width=12,
            height=12,
            point_radius_px=0,
        )
        self.assertTrue(np.array_equal(image[5, 5], np.asarray([0, 255, 0], dtype=np.uint8)))
        self.assertTrue(np.array_equal(image[5, 7], np.asarray([255, 0, 0], dtype=np.uint8)))

    def test_depth_override_root_is_used_for_frame_cells(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "case"
            make_visualization_case(case_dir, include_depth_ffs_float_m=True, frame_num=1)
            make_sam31_masks(case_dir, prompt_labels_by_object={0: "object"}, frame_tokens=["0"])
            output_dir = root / "out"
            sam2_mask = np.zeros((8, 10), dtype=bool)
            sam2_mask[2:6, 2:7] = True
            for checkpoint_spec in ladder.default_ladder_checkpoint_specs():
                for camera_idx in range(3):
                    ladder.write_single_object_masks(
                        mask_root=output_dir / "masks" / "case" / checkpoint_spec.key,
                        camera_idx=camera_idx,
                        object_label="object",
                        masks_by_frame_token={"0": sam2_mask},
                        overwrite=False,
                    )
            for camera_idx in range(3):
                ladder.write_single_object_masks(
                    mask_root=output_dir / "masks" / "case" / ladder.EDGE_TAM_VARIANT_KEY,
                    camera_idx=camera_idx,
                    object_label="object",
                    masks_by_frame_token={"0": sam2_mask},
                    overwrite=False,
                )

            override = root / "depth_override"
            for camera_idx in range(3):
                depth_dir = override / "depth_ffs_float_m" / str(camera_idx)
                depth_dir.mkdir(parents=True)
                np.save(depth_dir / "0.npy", np.full((8, 10), 0.42, dtype=np.float32))

            case_spec = ladder.LadderCaseSpec(
                key="case",
                label="Case",
                output_name="case",
                case_dir=case_dir,
                text_prompt="object",
            )
            _cells, stats = ladder.build_frame_cells(
                case_spec=case_spec,
                metadata=ladder.load_case_metadata(case_dir),
                output_dir=output_dir,
                checkpoint_specs=ladder.default_ladder_checkpoint_specs(),
                frame_idx=0,
                depth_override_root=override,
                depth_min_m=0.2,
                depth_max_m=1.5,
                max_points_per_camera=None,
                phystwin_radius_m=0.01,
                phystwin_nb_points=1,
                enhanced_component_voxel_size_m=0.01,
                enhanced_keep_near_main_gap_m=0.0,
                variant_keys=(*ladder.DEFAULT_VARIANT_ORDER, ladder.EDGE_TAM_VARIANT_KEY),
                extra_variant_roots={
                    ladder.EDGE_TAM_VARIANT_KEY: output_dir / "masks" / "case" / ladder.EDGE_TAM_VARIANT_KEY,
                },
            )
            self.assertIn(str(override), stats["per_camera"][0]["depth_path"])
            self.assertIn(ladder.EDGE_TAM_VARIANT_KEY, _cells[0])


if __name__ == "__main__":
    unittest.main()
