from __future__ import annotations

import json
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from qqtt.demo import demo33_shape_prior_completion
from qqtt.demo.demo33_shape_prior_warmup import (
    ShapePriorWarmupConfig,
    ShapePriorWarmupResult,
    futurephystwin_single_view_commands,
    load_shape_prior_final_data,
    run_futurephystwin_single_view_route,
    write_futurephystwin_warmup_case,
)


def _tiny_inputs() -> tuple[
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
    dict[int, np.ndarray],
]:
    rgb_by_camera: dict[int, np.ndarray] = {}
    depth_by_camera: dict[int, np.ndarray] = {}
    object_mask_by_camera: dict[int, np.ndarray] = {}
    controller_mask_by_camera: dict[int, np.ndarray] = {}
    intrinsics_by_camera: dict[int, np.ndarray] = {}
    c2w_by_camera: dict[int, np.ndarray] = {}
    for camera_idx in (0, 1, 2):
        rgb = np.zeros((2, 2, 3), dtype=np.uint8)
        rgb[..., 0] = 10 + camera_idx
        rgb[..., 1] = 20 + camera_idx
        rgb[..., 2] = 30 + camera_idx
        rgb_by_camera[camera_idx] = rgb
        depth_by_camera[camera_idx] = np.ones((2, 2), dtype=np.float32)
        object_mask = np.zeros((2, 2), dtype=bool)
        object_mask[0, 0] = True
        object_mask_by_camera[camera_idx] = object_mask
        controller_mask = np.zeros((2, 2), dtype=bool)
        controller_mask[1, 1] = True
        controller_mask_by_camera[camera_idx] = controller_mask
        intrinsics_by_camera[camera_idx] = np.asarray(
            [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )
        c2w = np.eye(4, dtype=np.float32)
        c2w[0, 3] = float(camera_idx)
        c2w_by_camera[camera_idx] = c2w
    return (
        rgb_by_camera,
        depth_by_camera,
        object_mask_by_camera,
        controller_mask_by_camera,
        intrinsics_by_camera,
        c2w_by_camera,
    )


class Demo33ShapePriorWarmupTest(unittest.TestCase):
    def test_writes_minimal_futurephystwin_case(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            config = ShapePriorWarmupConfig(
                enabled=True,
                output_root=Path(tmp_dir),
                run_id="run",
                futurephystwin_root=Path("/future"),
                sam3d_root=Path("/sam3d"),
                object_label="stuffed animal",
                controller_label="towel",
            )
            (
                rgb_by_camera,
                depth_by_camera,
                object_mask_by_camera,
                controller_mask_by_camera,
                intrinsics_by_camera,
                c2w_by_camera,
            ) = _tiny_inputs()

            profile = write_futurephystwin_warmup_case(
                config=config,
                rgb_by_camera=rgb_by_camera,
                depth_by_camera=depth_by_camera,
                object_mask_by_camera=object_mask_by_camera,
                controller_mask_by_camera=controller_mask_by_camera,
                intrinsics_by_camera=intrinsics_by_camera,
                c2w_by_camera=c2w_by_camera,
                camera_ids=(0, 1, 2),
                source_group_id=7,
            )

            case_dir = config.case_dir
            self.assertTrue((case_dir / "color" / "0" / "0.png").is_file())
            self.assertTrue((case_dir / "mask" / "mask_info_0.json").is_file())
            self.assertTrue((case_dir / "mask" / "0" / "0" / "0.png").is_file())
            self.assertTrue((case_dir / "mask" / "0" / "1" / "0.png").is_file())
            self.assertTrue((case_dir / "metadata.json").is_file())
            self.assertTrue((case_dir / "calibrate.pkl").is_file())
            self.assertTrue((case_dir / "pcd" / "0.npz").is_file())
            self.assertTrue((case_dir / "mask" / "processed_masks.pkl").is_file())
            self.assertTrue((case_dir / "track_process_data.pkl").is_file())

            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["source"], "demo3.3_shape_prior_warmup")
            self.assertEqual(metadata["camera_ids"], [0, 1, 2])
            self.assertEqual(metadata["object_label"], "stuffed animal")
            self.assertEqual(metadata["controller_label"], "towel")
            self.assertEqual(metadata["shape_prior_coordinate_frame"], "qqtt_world_c2w")
            self.assertEqual(metadata["shape_prior_units"], "meters")
            self.assertEqual(metadata["shape_prior_ground_policy"], "preserve")
            self.assertEqual(metadata["shape_prior_ground_z"], 0.0)

            with (case_dir / "calibrate.pkl").open("rb") as handle:
                c2ws = pickle.load(handle)
            self.assertEqual(len(c2ws), 3)
            self.assertEqual(np.asarray(c2ws[2]).shape, (4, 4))

            pcd = np.load(case_dir / "pcd" / "0.npz")
            self.assertEqual(pcd["points"].shape, (3, 2, 2, 3))
            self.assertEqual(pcd["colors"].shape, (3, 2, 2, 3))

            with (case_dir / "mask" / "processed_masks.pkl").open("rb") as handle:
                processed_masks = pickle.load(handle)
            self.assertEqual(len(processed_masks), 1)
            self.assertTrue(processed_masks[0][0]["object"][0, 0])
            self.assertTrue(processed_masks[0][1]["controller"][1, 1])

            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_data = pickle.load(handle)
            self.assertEqual(track_data["object_points"].shape, (1, 3, 3))
            self.assertEqual(track_data["object_colors"].shape, (1, 3, 3))
            self.assertEqual(track_data["object_visibilities"].shape, (1, 3))
            self.assertEqual(track_data["object_motions_valid"].shape, (1, 3))
            self.assertLessEqual(track_data["controller_points"].shape[1], 30)
            self.assertEqual(track_data["shape_prior_coordinate_frame"], "qqtt_world_c2w")
            self.assertEqual(track_data["shape_prior_units"], "meters")
            self.assertEqual(track_data["shape_prior_ground_policy"], "preserve")
            self.assertEqual(profile["shape_prior_object_points0"], 3)
            self.assertEqual(profile["shape_prior_coordinate_frame"], "qqtt_world_c2w")
            self.assertEqual(profile["shape_prior_units"], "meters")
            self.assertEqual(profile["shape_prior_ground_policy"], "preserve")
            self.assertFalse(profile["shape_prior_affects_tracker_input"])
            self.assertFalse(profile["shape_prior_affects_live_observation_pcd"])

    def test_futurephystwin_command_order_uses_exact_single_view_route(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            sam3d_root = Path(tmp_dir) / "sam-3d-objects"
            (sam3d_root / "notebook").mkdir(parents=True)
            (sam3d_root / "notebook" / "inference.py").write_text("", encoding="utf-8")
            (sam3d_root / "sam3d_objects").mkdir()
            config = ShapePriorWarmupConfig(
                enabled=True,
                output_root=Path(tmp_dir),
                run_id="run",
                futurephystwin_root=Path("/future"),
                futurephystwin_python="python",
                sam3d_root=sam3d_root,
                object_label="stuffed animal",
                controller_label="towel",
                camera_idx=0,
                force=True,
            )
            commands = futurephystwin_single_view_commands(config)
            self.assertEqual(
                [stage for stage, _command in commands],
                ["image_upscale", "segment_util_image", "shape_prior_sam3d", "align", "data_process_sample"],
            )
            flattened = [" ".join(command) for _stage, command in commands]
            self.assertIn("/future/data_process/image_upscale.py", flattened[0])
            self.assertIn("--mask_path", flattened[0])
            self.assertIn("/future/data_process/segment_util_image.py", flattened[1])
            self.assertIn("--TEXT_PROMPT", flattened[1])
            self.assertIn("/future/data_process_sam3d/shape_prior.py", flattened[2])
            self.assertIn(f"--sam3d_root {sam3d_root.resolve()}", flattened[2])
            self.assertIn("/future/data_process/align.py", flattened[3])
            self.assertIn("--force_rematch", flattened[3])
            self.assertIn("/future/data_process_sam3d/data_process_sample.py", flattened[4])
            self.assertIn("--shape_prior", flattened[4])
            self.assertIn("--ground-policy preserve", flattened[4])
            self.assertIn("--ground-z 0.0", flattened[4])

            fast_config = ShapePriorWarmupConfig(
                enabled=True,
                output_root=Path(tmp_dir),
                run_id="run-fast",
                futurephystwin_root=Path("/future"),
                futurephystwin_python="python",
                sam3d_root=sam3d_root,
                object_label="stuffed animal",
                controller_label="towel",
                camera_idx=0,
                skip_route_visualizations=True,
            )
            fast_commands = futurephystwin_single_view_commands(fast_config)
            fast_flattened = [" ".join(command) for _stage, command in fast_commands]
            self.assertIn("--skip_visualization", fast_flattened[2])
            self.assertIn("--skip_visualization", fast_flattened[3])
            self.assertIn("--skip_visualization", fast_flattened[4])

            calls: list[tuple[list[str], str, bool]] = []

            def fake_runner(command, *, cwd, check):
                calls.append((list(command), str(cwd), bool(check)))

            records = run_futurephystwin_single_view_route(config=config, runner=fake_runner)
            self.assertEqual([record["stage"] for record in records], [stage for stage, _command in commands])
            self.assertEqual([call[1] for call in calls], ["/future"] * 5)
            self.assertTrue(all(call[2] for call in calls))

    def test_load_final_data_builds_structure_points_in_futurephystwin_order(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir()
            object_points = np.asarray([[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]], dtype=np.float32)
            surface_points = np.asarray([[3.0, 0.0, 0.0]], dtype=np.float32)
            interior_points = np.asarray([[4.0, 0.0, 0.0], [5.0, 0.0, 0.0]], dtype=np.float32)
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "object_points": object_points,
                        "surface_points": surface_points,
                        "interior_points": interior_points,
                    },
                    handle,
                )

            result = load_shape_prior_final_data(case_dir)

            expected = np.concatenate([object_points[0], surface_points, interior_points], axis=0)
            np.testing.assert_allclose(result.structure_points, expected)
            self.assertEqual(result.profile["shape_prior_object_points0"], 2)
            self.assertEqual(result.profile["shape_prior_surface_points"], 1)
            self.assertEqual(result.profile["shape_prior_interior_points"], 2)
            self.assertEqual(result.profile["shape_prior_structure_points"], 5)
            self.assertEqual(result.profile["shape_prior_coordinate_validation_status"], "unavailable")

    def test_load_final_data_keeps_valid_world_z_shape_prior_renderable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir()
            object_points = np.asarray(
                [[[0.0, 0.0, 0.08], [0.01, 0.0, 0.11], [0.02, 0.0, 0.14]]],
                dtype=np.float32,
            )
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump({"object_points": object_points}, handle)
            with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source_group_id": 9,
                        "shape_prior_coordinate_frame": "qqtt_world_c2w",
                        "shape_prior_units": "meters",
                        "shape_prior_ground_policy": "preserve",
                        "shape_prior_ground_z": 0.0,
                    },
                    handle,
                )
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "object_points": object_points,
                        "surface_points": np.zeros((0, 3), dtype=np.float32),
                        "interior_points": np.zeros((0, 3), dtype=np.float32),
                    },
                    handle,
                )

            result = load_shape_prior_final_data(case_dir)

            self.assertEqual(result.status, "ready")
            self.assertEqual(result.profile["shape_prior_source_group_id"], 9)
            self.assertEqual(result.profile["shape_prior_coordinate_validation_status"], "valid")
            self.assertEqual(result.profile["shape_prior_coordinate_validation_reason"], "ok")
            self.assertTrue(result.profile["shape_prior_render_layer_enabled"])
            self.assertEqual(len(result.structure_points), 3)
            self.assertGreater(float(np.max(result.structure_points[:, 2])), 0.0)

    def test_load_final_data_rejects_positive_z_ground_clamp(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            case_dir.mkdir()
            source_points = np.asarray(
                [[[0.0, 0.0, 0.05], [0.01, 0.0, 0.10], [0.02, 0.0, 0.15]]],
                dtype=np.float32,
            )
            clamped_points = source_points.copy()
            clamped_points[..., 2] = 0.0
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump({"object_points": source_points}, handle)
            with (case_dir / "metadata.json").open("w", encoding="utf-8") as handle:
                json.dump(
                    {
                        "source_group_id": 10,
                        "shape_prior_coordinate_frame": "qqtt_world_c2w",
                        "shape_prior_units": "meters",
                        "shape_prior_ground_policy": "preserve",
                        "shape_prior_ground_z": 0.0,
                    },
                    handle,
                )
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(
                    {
                        "object_points": clamped_points,
                        "surface_points": np.zeros((0, 3), dtype=np.float32),
                        "interior_points": np.zeros((0, 3), dtype=np.float32),
                    },
                    handle,
                )

            result = load_shape_prior_final_data(case_dir)

            self.assertEqual(result.status, "invalid_coordinate_policy")
            self.assertEqual(result.profile["shape_prior_coordinate_validation_status"], "invalid")
            self.assertIn("positive_z_clamped_to_ground", result.profile["shape_prior_coordinate_validation_reason"])
            self.assertFalse(result.profile["shape_prior_render_layer_enabled"])
            self.assertEqual(len(result.structure_points), 0)
            self.assertEqual(result.profile["shape_prior_raw_structure_points"], 3)

    def test_detached_completion_merges_ready_shape_prior_into_live_profile(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            profile_json = Path(tmp_dir) / "live_profile.json"
            shared_json = Path(tmp_dir) / "shared_profile.json"
            completion_json = Path(tmp_dir) / "live_profile_shape_prior_completion.json"
            shared_json.write_text(
                json.dumps({"summary": {"shape_prior_status": "case_ready"}}),
                encoding="utf-8",
            )
            profile_json.write_text(
                json.dumps(
                    {
                        "summary": {"shape_prior_status": "case_ready"},
                        "shared_runtime_profile": str(shared_json),
                        "cotracker_process_snapshot": {
                            "shape_prior_warmup": {"shape_prior_status": "case_ready"}
                        },
                    }
                ),
                encoding="utf-8",
            )
            completion_profile = {
                "shape_prior_status": "ready",
                "shape_prior_structure_points": 6,
                "shape_prior_coordinate_validation_status": "valid",
                "shape_prior_render_layer_enabled": True,
            }

            merged = demo33_shape_prior_completion.merge_completion_into_live_profile(
                profile_json=profile_json,
                completion_json=completion_json,
                completion_profile=completion_profile,
            )

            self.assertEqual(merged["summary"]["shape_prior_status"], "ready")
            self.assertEqual(merged["summary"]["shape_prior_structure_points"], 6)
            snapshot = merged["cotracker_process_snapshot"]
            self.assertEqual(snapshot["shape_prior_warmup"]["shape_prior_status"], "ready")
            self.assertEqual(snapshot["shape_prior_status"], "ready")
            self.assertTrue(merged["shape_prior_warmup"]["shape_prior_render_layer_enabled"])
            self.assertEqual(merged["shape_prior_status"], "ready")
            shared_payload = json.loads(shared_json.read_text(encoding="utf-8"))
            self.assertEqual(shared_payload["summary"]["shape_prior_status"], "ready")
            self.assertEqual(shared_payload["shape_prior_status"], "ready")

    def test_detached_completion_runs_full_route_and_writes_completion_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            case_dir = root / "out" / "demo33_shape_prior_warmup" / "run" / "case"
            case_dir.mkdir(parents=True)
            profile_json = root / "live_profile.json"
            completion_json = root / "completion.json"
            profile_json.write_text(json.dumps({"summary": {}, "cotracker_process_snapshot": {}}), encoding="utf-8")
            calls: list[ShapePriorWarmupConfig] = []

            def fake_route(*, config, runner):
                _ = runner
                calls.append(config)
                return [{"stage": "image_upscale", "elapsed_ms": 1.0}]

            def fake_load(case_path):
                return ShapePriorWarmupResult(
                    status="ready",
                    case_dir=Path(case_path),
                    object_points0=np.zeros((1, 3), dtype=np.float32),
                    surface_points=np.zeros((0, 3), dtype=np.float32),
                    interior_points=np.zeros((0, 3), dtype=np.float32),
                    structure_points=np.zeros((1, 3), dtype=np.float32),
                    structure_colors_rgb=np.zeros((1, 3), dtype=np.uint8),
                    profile={
                        "shape_prior_status": "ready",
                        "shape_prior_case_dir": str(case_path),
                        "shape_prior_structure_points": 1,
                        "shape_prior_coordinate_validation_status": "valid",
                        "shape_prior_render_layer_enabled": True,
                    },
                )

            original_route = demo33_shape_prior_completion.run_futurephystwin_single_view_route
            original_load = demo33_shape_prior_completion.load_shape_prior_final_data
            try:
                demo33_shape_prior_completion.run_futurephystwin_single_view_route = fake_route
                demo33_shape_prior_completion.load_shape_prior_final_data = fake_load
                args = SimpleNamespace(
                    profile_json=profile_json,
                    completion_json=completion_json,
                    case_dir=case_dir,
                    futurephystwin_root=root / "FuturePhysTwin",
                    futurephystwin_python="python",
                    sam3d_root=root / "sam3d",
                    shape_prior_camera_idx=0,
                    force=False,
                    object_label="stuffed animal",
                    controller_label="towel",
                    ground_policy="preserve",
                    ground_z=0.0,
                    cuda_visible_devices="0",
                    cuda_alloc_conf="expandable_segments:True",
                    skip_route_visualizations=True,
                    wait_for_pid=0,
                    wait_timeout_s=1.0,
                    wait_poll_s=0.01,
                )

                profile = demo33_shape_prior_completion.complete_shape_prior_case(args)
            finally:
                demo33_shape_prior_completion.run_futurephystwin_single_view_route = original_route
                demo33_shape_prior_completion.load_shape_prior_final_data = original_load

            self.assertEqual(profile["shape_prior_status"], "ready")
            self.assertEqual(profile["shape_prior_start_trigger"], "after_teardown_detached")
            self.assertEqual(profile["shape_prior_cuda_visible_devices"], "0")
            self.assertTrue(profile["shape_prior_skip_route_visualizations"])
            self.assertEqual(profile["shape_prior_command_order"], ["image_upscale"])
            self.assertEqual(len(calls), 1)
            self.assertEqual(calls[0].case_dir, case_dir.resolve())
            self.assertTrue(calls[0].skip_route_visualizations)
            completion_payload = json.loads(completion_json.read_text(encoding="utf-8"))
            self.assertEqual(completion_payload["summary"]["shape_prior_status"], "ready")
            live_payload = json.loads(profile_json.read_text(encoding="utf-8"))
            self.assertEqual(live_payload["summary"]["shape_prior_status"], "ready")


if __name__ == "__main__":
    unittest.main()
