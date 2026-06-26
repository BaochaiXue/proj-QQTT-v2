from __future__ import annotations

from contextlib import redirect_stdout
import importlib
import io
import json
import pickle
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np

from demo_v5 import headless_chunk_bridge
from demo_v5.futurephystwin_chunk_writer import (
    FuturePhysTwinChunk,
    build_topology_payload,
    validate_futurephystwin_case,
    write_futurephystwin_chunk_case,
)
from demo_v5.online_case_aggregate import build_aggregate_case_from_chunk_cases
from demo_v5.online_chunk_output import DemoV5OnlineOutputWriter
import demo_v5.realtime_futurephystwin_chunks as demo_v5


REPO_ROOT = Path(__file__).resolve().parents[1]


class FakeProcess:
    def __init__(self, returncode: int | None = 0) -> None:
        self.returncode = returncode
        self.terminated = False
        self.killed = False

    def poll(self) -> int | None:
        return self.returncode

    def wait(self, timeout: float | None = None) -> int | None:
        return self.returncode

    def terminate(self) -> None:
        self.terminated = True
        self.returncode = -15

    def kill(self) -> None:
        self.killed = True
        self.returncode = -9


def _tiny_futurephystwin_chunk(*, chunk_index: int, serial: str = "demo-v5-single-camera") -> FuturePhysTwinChunk:
    frame_count = 2
    height = 4
    width = 5
    rgb_frames = [
        np.full((height, width, 3), 40 + idx, dtype=np.uint8)
        for idx in range(frame_count)
    ]
    object_mask = np.zeros((height, width), dtype=bool)
    object_mask[:2, :3] = True
    controller_mask = np.zeros((height, width), dtype=bool)
    controller_mask[2:, 2:] = True
    processed_masks = [
        [{"object": object_mask.copy(), "controller": controller_mask.copy()}]
        for _ in range(frame_count)
    ]
    object_points = np.array(
        [
            [[0.00, 0.00, 0.10], [0.02, 0.00, 0.10], [0.04, 0.00, 0.10]],
            [[0.00, 0.01, 0.10], [0.02, 0.01, 0.10], [0.04, 0.01, 0.10]],
        ],
        dtype=np.float32,
    )
    controller_points = np.array(
        [
            [[0.10, 0.00, 0.10], [0.12, 0.00, 0.10]],
            [[0.10, 0.01, 0.10], [0.12, 0.01, 0.10]],
        ],
        dtype=np.float32,
    )
    track_process = {
        "object_points": object_points,
        "object_colors": np.ones_like(object_points, dtype=np.float32) * 0.5,
        "object_visibilities": np.ones(object_points.shape[:2], dtype=bool),
        "object_motions_valid": np.ones(object_points.shape[:2], dtype=bool),
        "controller_points": controller_points,
        "controller_mask": np.ones((controller_points.shape[1],), dtype=bool),
    }
    return FuturePhysTwinChunk(
        rgb_frames=rgb_frames,
        processed_masks=processed_masks,
        track_process_data=track_process,
        intrinsics=np.eye(3, dtype=np.float32),
        camera_to_world_c2w=np.eye(4, dtype=np.float32),
        tracks_yx=np.zeros((frame_count, 5, 2), dtype=np.float32),
        tracker_visibility=np.ones((frame_count, 5), dtype=bool),
        queries_txy=np.zeros((5, 3), dtype=np.float32),
        surface_points=np.array([[0.0, 0.0, 0.1]], dtype=np.float32),
        interior_points=np.array([[0.01, 0.0, 0.1]], dtype=np.float32),
        fps=5,
        serial_number=serial,
        depth_backend="native-realsense",
        depth_source_internal="realsense",
        chunk_index=chunk_index,
    )


def _with_anchor_diagnostics(chunk: FuturePhysTwinChunk, *, quality_status: str = "normal") -> FuturePhysTwinChunk:
    track = dict(chunk.track_process_data)
    controller_points = np.asarray(track["controller_points"])
    frame_count, anchor_count = controller_points.shape[:2]
    object_count = int(np.asarray(track["object_points"]).shape[1])
    anchor_query_ids = np.arange(anchor_count, dtype=np.int64) + object_count
    track.update(
        {
            "controller_anchor_query_indices": anchor_query_ids,
            "controller_anchor_active_query_indices": anchor_query_ids,
            "controller_anchor_status": np.asarray(["direct"] * anchor_count, dtype="<U16"),
            "controller_anchor_bundle_query_ids": np.tile(anchor_query_ids[:, None], (1, 3)),
            "controller_anchor_source_query_id": np.tile(anchor_query_ids[None, :], (frame_count, 1)),
            "controller_anchor_observation_mode": np.full((frame_count, anchor_count), "direct_valid", dtype="<U40"),
            "controller_anchor_confidence": np.ones((frame_count, anchor_count), dtype=np.float32),
            "controller_anchor_failure_reason": np.full((frame_count, anchor_count), "none", dtype="<U40"),
            "controller_anchor_bundle_support_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_anchor_bundle_raw_visible_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_anchor_bundle_depth_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_anchor_bundle_processed_mask_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_anchor_bundle_motion_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_anchor_recovery_residual": np.zeros((frame_count, anchor_count), dtype=np.float32),
            "controller_quality_status": str(quality_status),
        }
    )
    return FuturePhysTwinChunk(
        rgb_frames=chunk.rgb_frames,
        processed_masks=chunk.processed_masks,
        track_process_data=track,
        intrinsics=chunk.intrinsics,
        camera_to_world_c2w=chunk.camera_to_world_c2w,
        tracks_yx=chunk.tracks_yx,
        tracker_visibility=chunk.tracker_visibility,
        queries_txy=chunk.queries_txy,
        surface_points=chunk.surface_points,
        interior_points=chunk.interior_points,
        pcd_points=chunk.pcd_points,
        pcd_colors=chunk.pcd_colors,
        fps=chunk.fps,
        serial_number=chunk.serial_number,
        depth_backend=chunk.depth_backend,
        depth_source_internal=chunk.depth_source_internal,
        chunk_index=chunk.chunk_index,
        source_frame_indices=chunk.source_frame_indices,
    )


class DemoV5RealtimePhysTwinTest(unittest.TestCase):
    def test_demo_v5_python_sources_do_not_import_demo_v4_modules(self) -> None:
        for path in sorted((REPO_ROOT / "demo_v5").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("from demo_v4", text)
                self.assertNotIn("import demo_v4", text)

    def test_demo_v5_does_not_keep_shadow_quality_modules(self) -> None:
        shadow_modules = {
            "anchor_recovery.py",
            "contracts.py",
            "controller_selection.py",
            "fps_sampling.py",
            "knn_recovery.py",
            "motion_filter.py",
            "object_sampling.py",
            "session_topology.py",
            "timing.py",
            "topology_assembly.py",
            "topology_warmup.py",
            "tracking_samples.py",
        }

        present = {
            path.name
            for path in (REPO_ROOT / "demo_v5").glob("*.py")
            if path.name in shadow_modules
        }

        self.assertEqual(present, set())

    def test_defaults_route_realtime_to_gpu0_and_point_viewer_to_gpu1(self) -> None:
        args = demo_v5.build_parser().parse_args(["--dry-run"])
        chunk_frame_count = demo_v5.resolve_chunk_frame_count(args)

        self.assertEqual(args.input_source, "fake-live")
        self.assertEqual(args.replay_fps, 5.0)
        self.assertEqual(args.chunk_seconds, 7.0)
        self.assertEqual(chunk_frame_count, 35)
        self.assertEqual(str(args.futurephystwin_base_path), "result/demo_v5/futurephystwin_chunks")
        self.assertEqual(args.case_prefix, "demo_v5")
        self.assertEqual(args.realtime_gpu_mode, None)
        self.assertEqual(args.warmup_gpu_mode, "dual")
        self.assertEqual(demo_v5.resolve_camera_cuda_visible_devices(args), "0")
        self.assertEqual(demo_v5.resolve_shape_prior_worker_cuda_visible_devices(args), "1")
        self.assertEqual(args.shape_prior_worker_mode, "managed")
        self.assertEqual(args.optimization_mode, "disabled")
        self.assertEqual(args.point_viewer_mode, "window")
        self.assertFalse(args.allow_degraded_online)
        self.assertTrue(demo_v5.build_parser().parse_args(["--allow-degraded-online"]).allow_degraded_online)
        self.assertEqual(demo_v5.resolve_point_viewer_cuda_visible_devices(args), "1")
        self.assertEqual(demo_v5.resolve_optimization_cuda_visible_devices(args), "1")
        self.assertEqual(demo_v5.resolve_optimization_device(args), "cuda:0")

        contract = demo_v5._contract(args)

        self.assertEqual(contract["demo_version"], "demo_v5")
        self.assertEqual(contract["optimization_scope"], "disabled")
        self.assertEqual(contract["optimization_segment_len"], 35)
        self.assertEqual(contract["shape_prior_worker_released_before_optimization"], False)
        self.assertEqual(contract["shape_prior_worker_released_before_point_viewer"], True)
        worker_command = contract["shape_prior_worker_command"]
        self.assertIn("--max-observation-to-aligned-p95-m", worker_command)
        self.assertEqual(worker_command[worker_command.index("--max-observation-to-aligned-p95-m") + 1], "0.06")
        self.assertEqual(
            contract["online_dir"],
            "result/demo_v5/futurephystwin_chunks/online_data/demo_v5",
        )
        self.assertTrue(
            str(contract["static_data_path"]).endswith(
                "result/demo_v5/futurephystwin_chunks/data/demo_v5/final_data.pkl"
            )
        )
        point_viewer_command = contract["point_viewer_command"]
        self.assertEqual(point_viewer_command[:6], ["conda", "run", "-n", "demo_2_max", "--no-capture-output", "python"])
        self.assertEqual(point_viewer_command[6], "demo_v5/online_points_viewer.py")
        self.assertEqual(
            point_viewer_command[point_viewer_command.index("--online-dir") + 1],
            "result/demo_v5/futurephystwin_chunks/online_data/demo_v5",
        )
        self.assertEqual(
            point_viewer_command[point_viewer_command.index("--case-dir") + 1],
            "result/demo_v5/futurephystwin_chunks/data/demo_v5",
        )
        self.assertEqual(point_viewer_command[point_viewer_command.index("--fps") + 1], "5.0")
        self.assertEqual(point_viewer_command[point_viewer_command.index("--object-color-mode") + 1], "rainbow")
        opt_command = contract["optimization_command"]
        self.assertEqual(opt_command[1], "train_online_zero_then_first.py")
        self.assertNotIn("--stop_when_finished", opt_command)
        self.assertFalse(contract["optimization_stop_when_finished"])
        self.assertFalse(Path(opt_command[opt_command.index("--base_path") + 1]).is_absolute())
        self.assertEqual(opt_command[opt_command.index("--base_path") + 1], "../result/demo_v5/futurephystwin_chunks/data")
        self.assertEqual(
            opt_command[opt_command.index("--online_dir") + 1],
            "../result/demo_v5/futurephystwin_chunks/online_data/demo_v5",
        )
        self.assertEqual(opt_command[opt_command.index("--segment_len") + 1], "35")
        self.assertEqual(opt_command[opt_command.index("--device") + 1], "cuda:0")

    def test_camera_command_uses_demo_v5_final_data_contract(self) -> None:
        args = demo_v5.build_parser().parse_args([])
        command = demo_v5.build_camera_realtime_command(
            args,
            capture_dir=Path("result/demo_v5/unit_capture"),
            profile_json=Path("result/demo_v5/unit_capture/shape_prior_profile.json"),
            chunk_frame_count=35,
        )

        joined = " ".join(command)
        self.assertIn("demo_v5/realtime_camera_final_data.py", command[1])
        self.assertNotIn("demo_v3_2", joined)
        self.assertNotIn("--depth-backend", command)
        self.assertEqual(command[command.index("--depth-source") + 1], "realsense")
        self.assertEqual(command[command.index("--depth-backend-label") + 1], "native-realsense")
        self.assertIn("--headless-prepared-only", command)
        self.assertIn("--enable-pcd-filter", command)
        self.assertEqual(command[command.index("--pcd-filter-mode") + 1], "sync")
        self.assertEqual(command[command.index("--pcd-filter-preset") + 1], "original")
        self.assertEqual(command[command.index("--table-calibrate") + 1], "table_calibrate.pkl")
        self.assertIn("--enable-table-z-filter", command)
        self.assertEqual(command[command.index("--metadata-demo-version") + 1], "demo_v5")
        self.assertEqual(command[command.index("--metadata-reference-pipeline") + 1], "data_process_sam3d")

        ffs_args = demo_v5.build_parser().parse_args(["--depth-backend", "ir-ffs"])
        ffs_command = demo_v5.build_camera_realtime_command(
            ffs_args,
            capture_dir=Path("result/demo_v5/unit_capture"),
            profile_json=Path("result/demo_v5/unit_capture/shape_prior_profile.json"),
            chunk_frame_count=35,
        )
        self.assertEqual(ffs_command[ffs_command.index("--depth-source") + 1], "ffs")
        self.assertEqual(ffs_command[ffs_command.index("--depth-backend-label") + 1], "ir-ffs")

    def test_chunk_and_aggregate_metadata_record_demo_v5_dataprocess_reference(self) -> None:
        root = Path("result/test_demo_v5_unit_metadata_contract")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            write_futurephystwin_chunk_case(base_path, "demo_v5_meta_chunk_0001", _tiny_futurephystwin_chunk(chunk_index=0))
            write_futurephystwin_chunk_case(base_path, "demo_v5_meta_chunk_0002", _tiny_futurephystwin_chunk(chunk_index=1))
            chunk_metadata = json.loads((base_path / "demo_v5_meta_chunk_0001" / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(chunk_metadata["demo_version"], "demo_v5")
            self.assertEqual(chunk_metadata["runtime_product_name"], "demo_v5_realtime_camera_final_data")
            self.assertEqual(chunk_metadata["reference_pipeline"], "data_process_sam3d")

            aggregate_dir = base_path / "data" / "demo_v5_meta"
            build_aggregate_case_from_chunk_cases(
                [
                    base_path / "demo_v5_meta_chunk_0001",
                    base_path / "demo_v5_meta_chunk_0002",
                ],
                aggregate_dir,
                ready=True,
            )
            aggregate_metadata = json.loads((aggregate_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(aggregate_metadata["demo_version"], "demo_v5")
            self.assertEqual(aggregate_metadata["runtime_product_name"], "demo_v5_realtime_camera_final_data")
            self.assertEqual(aggregate_metadata["reference_pipeline"], "data_process_sam3d")
            self.assertEqual(aggregate_metadata["runtime_contract"], "data_process_sam3d_realtime_final_data_v1")
            with (aggregate_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertEqual(final_data["object_points"].shape[0], 4)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_validate_rejects_object_sample_ids_with_controller_semantics(self) -> None:
        root = Path("result/test_demo_v5_unit_sample_semantics_object")
        shutil.rmtree(root, ignore_errors=True)
        try:
            case_dir = root / "cases" / "demo_v5_semantics_chunk_0001"
            write_futurephystwin_chunk_case(
                root / "cases",
                "demo_v5_semantics_chunk_0001",
                _tiny_futurephystwin_chunk(chunk_index=0),
            )
            with (case_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            final_data["object_sample_query_ids"] = np.asarray(
                final_data["object_sample_query_ids"],
                dtype=np.int64,
            ).copy()
            final_data["object_sample_query_ids"][0] = int(final_data["controller_sample_query_ids"][0])
            topology = build_topology_payload(
                final_data,
                object_sample_query_ids=final_data["object_sample_query_ids"],
                controller_sample_query_ids=final_data["controller_sample_query_ids"],
            )
            for payload in (final_data, track_process):
                for key, value in topology.items():
                    payload[key] = value
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(final_data, handle)
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump(track_process, handle)

            with self.assertRaisesRegex(ValueError, "object_sample_query_ids.*object semantic"):
                validate_futurephystwin_case(case_dir)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_validate_rejects_controller_sample_ids_with_object_semantics(self) -> None:
        root = Path("result/test_demo_v5_unit_sample_semantics_controller")
        shutil.rmtree(root, ignore_errors=True)
        try:
            case_dir = root / "cases" / "demo_v5_semantics_chunk_0001"
            write_futurephystwin_chunk_case(
                root / "cases",
                "demo_v5_semantics_chunk_0001",
                _tiny_futurephystwin_chunk(chunk_index=0),
            )
            with (case_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            final_data["controller_sample_query_ids"] = np.asarray(
                final_data["controller_sample_query_ids"],
                dtype=np.int64,
            ).copy()
            final_data["controller_sample_query_ids"][0] = int(final_data["object_sample_query_ids"][0])
            topology = build_topology_payload(
                final_data,
                object_sample_query_ids=final_data["object_sample_query_ids"],
                controller_sample_query_ids=final_data["controller_sample_query_ids"],
            )
            for payload in (final_data, track_process):
                for key, value in topology.items():
                    payload[key] = value
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(final_data, handle)
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump(track_process, handle)

            with self.assertRaisesRegex(ValueError, "controller_sample_query_ids.*controller semantic"):
                validate_futurephystwin_case(case_dir)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_anchor_diagnostics_are_persisted_to_track_process_online_chunk_and_aggregate(self) -> None:
        root = Path("result/test_demo_v5_unit_anchor_diagnostics")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            chunk = _with_anchor_diagnostics(_tiny_futurephystwin_chunk(chunk_index=0))
            manifest = write_futurephystwin_chunk_case(base_path, "diag_chunk_0001", chunk)
            case_dir = Path(manifest["futurephystwin_case_root"])
            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)

            self.assertEqual(track_process["controller_anchor_confidence"].shape, (2, 2))
            self.assertEqual(track_process["controller_anchor_bundle_query_ids"].shape, (2, 3))
            self.assertEqual(track_process["controller_anchor_bundle_raw_visible_count"].shape, (2, 2))
            self.assertEqual(str(track_process["controller_quality_status"]), "normal")

            writer = DemoV5OnlineOutputWriter(base_path=base_path, case_name="diag", chunk_size=2)
            online_result = writer.commit_case_chunk(case_dir)
            writer.finish()

            with Path(online_result["online_chunk_path"]).open("rb") as handle:
                online_chunk = pickle.load(handle)
            self.assertEqual(online_chunk["controller_anchor_confidence"].shape, (2, 2))
            np.testing.assert_array_equal(
                online_chunk["controller_anchor_bundle_query_ids"],
                track_process["controller_anchor_bundle_query_ids"],
            )

            with (base_path / "data" / "diag" / "track_process_data.pkl").open("rb") as handle:
                aggregate_track = pickle.load(handle)
            self.assertEqual(aggregate_track["controller_anchor_confidence"].shape, (2, 2))
            self.assertEqual(aggregate_track["controller_anchor_bundle_query_ids"].shape, (2, 3))
            self.assertEqual(aggregate_track["controller_anchor_bundle_motion_valid_count"].shape, (2, 2))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_controller_quality_markers_distinguish_ready_degraded_and_invalid(self) -> None:
        root = Path("result/test_demo_v5_unit_quality_markers")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            cases = {}
            for quality in ("normal", "degraded", "invalid"):
                manifest = write_futurephystwin_chunk_case(
                    base_path,
                    f"{quality}_chunk_0001",
                    _with_anchor_diagnostics(_tiny_futurephystwin_chunk(chunk_index=0), quality_status=quality),
                )
                cases[quality] = Path(manifest["futurephystwin_case_root"])
                self.assertEqual(manifest["controller_quality_status"], quality)

            self.assertTrue((cases["normal"] / "READY").is_file())
            self.assertFalse((cases["normal"] / "DEGRADED").exists())
            self.assertFalse((cases["normal"] / "INVALID").exists())
            self.assertTrue((cases["degraded"] / "DEGRADED").is_file())
            self.assertFalse((cases["degraded"] / "READY").exists())
            self.assertTrue((cases["invalid"] / "INVALID").is_file())
            self.assertFalse((cases["invalid"] / "READY").exists())
            validate_futurephystwin_case(cases["normal"], require_ready=True)
            with self.assertRaisesRegex(ValueError, "READY"):
                validate_futurephystwin_case(cases["degraded"], require_ready=True)
            with self.assertRaisesRegex(ValueError, "READY"):
                validate_futurephystwin_case(cases["invalid"], require_ready=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_controller_quality_online_publish_policy_defaults_degraded_to_diagnostic_only(self) -> None:
        self.assertIsNone(
            headless_chunk_bridge._controller_quality_online_publish_skip_reason(
                {"controller_quality_status": "normal"},
                allow_degraded_online=False,
            )
        )
        self.assertEqual(
            headless_chunk_bridge._controller_quality_online_publish_skip_reason(
                {"controller_quality_status": "degraded"},
                allow_degraded_online=False,
            ),
            "controller_quality_degraded",
        )
        self.assertIsNone(
            headless_chunk_bridge._controller_quality_online_publish_skip_reason(
                {"controller_quality_status": "degraded"},
                allow_degraded_online=True,
            )
        )
        self.assertEqual(
            headless_chunk_bridge._controller_quality_online_publish_skip_reason(
                {"controller_quality_status": "invalid"},
                allow_degraded_online=True,
            ),
            "controller_quality_invalid",
        )

    def test_allow_degraded_online_commits_degraded_marker_case(self) -> None:
        root = Path("result/test_demo_v5_unit_allow_degraded_online")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            manifest = write_futurephystwin_chunk_case(
                base_path,
                "degraded_chunk_0001",
                _with_anchor_diagnostics(_tiny_futurephystwin_chunk(chunk_index=0), quality_status="degraded"),
            )
            case_dir = Path(manifest["futurephystwin_case_root"])

            default_writer = DemoV5OnlineOutputWriter(base_path=base_path, case_name="default", chunk_size=2)
            with self.assertRaisesRegex(ValueError, "READY"):
                default_writer.commit_case_chunk(case_dir)

            allowed_writer = DemoV5OnlineOutputWriter(
                base_path=base_path,
                case_name="allowed",
                chunk_size=2,
                allow_degraded=True,
            )
            online_result = allowed_writer.commit_case_chunk(case_dir)
            allowed_writer.finish()

            self.assertTrue(Path(online_result["online_chunk_path"]).is_file())
            self.assertTrue((base_path / "data" / "allowed" / "READY").is_file())
            with (base_path / "data" / "allowed" / "track_process_data.pkl").open("rb") as handle:
                aggregate_track = pickle.load(handle)
            self.assertEqual(str(aggregate_track["controller_quality_status"]), "degraded")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_candidate_motion_validity_survives_anchor_selection_shape_split(self) -> None:
        root = Path("result/test_demo_v5_unit_candidate_motion_validity")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            chunk = _with_anchor_diagnostics(_tiny_futurephystwin_chunk(chunk_index=0))
            track = dict(chunk.track_process_data)
            frame_count, anchor_count = np.asarray(track["controller_points"]).shape[:2]
            candidate_count = 5
            candidate_motion_valid = np.ones((frame_count, candidate_count), dtype=bool)
            candidate_motion_valid[1, -1] = False
            track["controller_mask"] = np.ones((candidate_count,), dtype=bool)
            track["controller_query_indices"] = np.asarray([3, 4, 10, 11, 12], dtype=np.int64)
            track["controller_candidate_motions_valid"] = candidate_motion_valid
            track["controller_motions_valid"] = np.ones((frame_count, anchor_count), dtype=bool)
            chunk = FuturePhysTwinChunk(
                rgb_frames=chunk.rgb_frames,
                processed_masks=chunk.processed_masks,
                track_process_data=track,
                intrinsics=chunk.intrinsics,
                camera_to_world_c2w=chunk.camera_to_world_c2w,
                tracks_yx=chunk.tracks_yx,
                tracker_visibility=chunk.tracker_visibility,
                queries_txy=chunk.queries_txy,
                surface_points=chunk.surface_points,
                interior_points=chunk.interior_points,
                fps=chunk.fps,
                serial_number=chunk.serial_number,
                depth_backend=chunk.depth_backend,
                depth_source_internal=chunk.depth_source_internal,
                chunk_index=chunk.chunk_index,
            )

            manifest = write_futurephystwin_chunk_case(base_path, "candidate_motion_chunk_0001", chunk)

            with (Path(manifest["futurephystwin_case_root"]) / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            self.assertEqual(track_process["controller_motions_valid"].shape, (frame_count, candidate_count))
            np.testing.assert_array_equal(track_process["controller_motions_valid"], candidate_motion_valid)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_source_headless_requires_optimization_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous optimization requires fake-live or live capture"):
            demo_v5.main(["--dry-run", "--optimization-mode", "continuous", "--source-headless-capture", "existing_capture"])

    def test_failed_shape_prior_metadata_raises_without_polling_until_timeout(self) -> None:
        root = Path("result/test_demo_v5_unit_shape_prior_failed")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            capture_dir.mkdir(parents=True)
            (capture_dir / "metadata.json").write_text(
                json.dumps(
                    {
                        "shape_prior_status": "failed",
                        "shape_prior_error": "shape-prior single-view alignment invalid",
                    }
                )
                + "\n",
                encoding="utf-8",
            )

            def fail_if_polled() -> None:
                raise AssertionError("Demo v5 should not keep polling after terminal shape-prior failure")

            with self.assertRaisesRegex(RuntimeError, "shape prior failed.*single-view alignment invalid"):
                headless_chunk_bridge._shape_points_for_chunk(
                    capture_dir,
                    surface_points=None,
                    interior_points=None,
                    require_shape_prior=True,
                    shape_prior_wait_timeout_s=99.0,
                    capture_finished=lambda: False,
                    before_poll=fail_if_polled,
                    poll_interval_s=99.0,
                )
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_continuous_optimization_starts_once_for_whole_online_stream(self) -> None:
        root = Path("result/test_demo_v5_unit_continuous")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"
            popen_calls: list[tuple[list[str], dict[str, object]]] = []

            def fake_popen(command, **kwargs):
                popen_calls.append((list(command), dict(kwargs)))
                return FakeProcess(returncode=0)

            def fake_stream(capture_dir_arg, **kwargs):
                self.assertEqual(Path(capture_dir_arg), capture_dir)
                first = {
                    "case_name": "demo_v5_rt_chunk_0001",
                    "frame_count": 35,
                    "futurephystwin_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                second = {
                    "case_name": "demo_v5_rt_chunk_0002",
                    "frame_count": 35,
                    "futurephystwin_case_root": str(base_path / "demo_v5_rt_chunk_0002"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000001.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 14.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](first)
                kwargs["on_chunk_written"](second)
                return [first, second]

            with mock.patch("demo_v5.realtime_futurephystwin_chunks.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "2",
                                "--optimization-mode",
                                "continuous",
                                "--point-viewer-mode",
                                "disabled",
                                "--optimization-zero-iterations",
                                "1",
                                "--optimization-iterations",
                                "1",
                                "--optimization-start-grace-s",
                                "0",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(popen_calls), 2)
            camera_command, camera_kwargs = popen_calls[0]
            opt_command, opt_kwargs = popen_calls[1]
            self.assertIn("demo_v5/realtime_camera_final_data.py", camera_command[1])
            self.assertNotIn("demo_v3_2", " ".join(camera_command))
            self.assertEqual(camera_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "0")
            self.assertTrue(camera_kwargs["start_new_session"])
            self.assertEqual(opt_command[1], "train_online_zero_then_first.py")
            self.assertEqual(opt_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertTrue(opt_kwargs["start_new_session"])
            self.assertEqual(Path(opt_kwargs["cwd"]), demo_v5._resolved_realtime_phystwin_root(demo_v5.build_parser().parse_args([])))
            self.assertEqual(opt_command[opt_command.index("--case_name") + 1], "demo_v5_rt")
            self.assertEqual(opt_command[opt_command.index("--segment_len") + 1], "35")
            self.assertEqual(opt_command[opt_command.index("--zero_iterations") + 1], "1")
            self.assertEqual(opt_command[opt_command.index("--iterations") + 1], "1")
            self.assertEqual(
                opt_command[opt_command.index("--base_path") + 1],
                "../result/test_demo_v5_unit_continuous/cases/data",
            )
            self.assertEqual(
                opt_command[opt_command.index("--online_dir") + 1],
                "../result/test_demo_v5_unit_continuous/cases/online_data/demo_v5_rt",
            )
            self.assertEqual(
                opt_command[opt_command.index("--static_data_path") + 1],
                "../result/test_demo_v5_unit_continuous/cases/data/demo_v5_rt/final_data.pkl",
            )
            self.assertFalse(Path(opt_command[opt_command.index("--static_data_path") + 1]).is_absolute())

            summary = json.loads(stdout.getvalue())
            self.assertTrue(summary["optimization_started"])
            self.assertEqual(summary["optimization_scope"], "single_continuous_online_case")
            self.assertEqual(summary["optimization_return_code"], 0)
            self.assertEqual(summary["optimization_started_from_chunk"]["case_name"], "demo_v5_rt_chunk_0001")
            self.assertEqual(summary["chunk_count"], 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_managed_worker_is_released_before_gpu1_optimization(self) -> None:
        root = Path("result/test_demo_v5_unit_managed")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"
            worker = FakeProcess(returncode=None)
            camera = FakeProcess(returncode=0)
            optimizer = FakeProcess(returncode=0)
            processes = [worker, camera, optimizer]
            popen_calls: list[tuple[list[str], dict[str, object]]] = []

            def fake_popen(command, **kwargs):
                popen_calls.append((list(command), dict(kwargs)))
                return processes.pop(0)

            def fake_stream(_capture_dir_arg, **kwargs):
                manifest = {
                    "case_name": "demo_v5_rt_chunk_0001",
                    "frame_count": 35,
                    "futurephystwin_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_futurephystwin_chunks.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--optimization-mode",
                                "continuous",
                                "--point-viewer-mode",
                                "disabled",
                                "--optimization-zero-iterations",
                                "1",
                                "--optimization-iterations",
                                "1",
                                "--optimization-start-grace-s",
                                "0",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(popen_calls), 3)
            worker_command, worker_kwargs = popen_calls[0]
            opt_command, opt_kwargs = popen_calls[2]
            self.assertIn("services/shape_prior_remote/server.py", worker_command)
            self.assertEqual(worker_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertTrue(worker_kwargs["start_new_session"])
            self.assertTrue(worker.terminated)
            self.assertEqual(opt_command[1], "train_online_zero_then_first.py")
            self.assertEqual(opt_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertTrue(opt_kwargs["start_new_session"])
            summary = json.loads(stdout.getvalue())
            self.assertTrue(summary["shape_prior_worker_released_before_optimization"])
            self.assertEqual(summary["shape_prior_worker_return_code"], -15)
            self.assertTrue(summary["optimization_started"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_default_point_viewer_starts_after_first_committed_online_chunk(self) -> None:
        root = Path("result/test_demo_v5_unit_point_viewer")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"
            popen_calls: list[tuple[list[str], dict[str, object]]] = []

            def fake_popen(command, **kwargs):
                popen_calls.append((list(command), dict(kwargs)))
                return FakeProcess(returncode=0)

            def fake_stream(_capture_dir_arg, **kwargs):
                manifest = {
                    "case_name": "demo_v5_rt_chunk_0001",
                    "frame_count": 35,
                    "futurephystwin_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_futurephystwin_chunks.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(popen_calls), 2)
            camera_command, _camera_kwargs = popen_calls[0]
            viewer_command, viewer_kwargs = popen_calls[1]
            self.assertIn("demo_v5/realtime_camera_final_data.py", camera_command[1])
            self.assertEqual(viewer_command[:6], ["conda", "run", "-n", "demo_2_max", "--no-capture-output", "python"])
            self.assertEqual(viewer_command[6], "demo_v5/online_points_viewer.py")
            self.assertEqual(viewer_command[viewer_command.index("--online-dir") + 1], str(base_path / "online_data" / "demo_v5_rt"))
            self.assertEqual(viewer_command[viewer_command.index("--case-dir") + 1], str(base_path / "data" / "demo_v5_rt"))
            self.assertEqual(viewer_command[viewer_command.index("--fps") + 1], "5.0")
            self.assertEqual(viewer_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            self.assertEqual(Path(viewer_kwargs["cwd"]), REPO_ROOT)
            self.assertTrue(viewer_kwargs["start_new_session"])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["optimization_mode"], "disabled")
            self.assertFalse(summary["optimization_started"])
            self.assertTrue(summary["point_viewer_started"])
            self.assertEqual(summary["point_viewer_started_from_chunk"]["case_name"], "demo_v5_rt_chunk_0001")
            self.assertEqual(summary["point_viewer_return_code"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_realtime_runner_returns_nonzero_when_controller_quality_is_invalid(self) -> None:
        root = Path("result/test_demo_v5_unit_invalid_quality")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"

            def fake_stream(_capture_dir_arg, **_kwargs):
                return [
                    {
                        "case_name": "demo_v5_rt_chunk_0001",
                        "frame_count": 35,
                        "futurephystwin_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                        "publish_wall_s": 7.0,
                        "backlog_chunks": 0,
                        "shape_prior_complete": True,
                        "controller_quality_status": "invalid",
                        "online_publish_skipped": True,
                    }
                ]

            with mock.patch("demo_v5.realtime_futurephystwin_chunks.subprocess.Popen", return_value=FakeProcess(returncode=0)):
                with mock.patch("demo_v5.realtime_futurephystwin_chunks.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--futurephystwin-base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--point-viewer-mode",
                                "disabled",
                            ]
                        )

            self.assertEqual(exit_code, 1)
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["controller_quality_status"], "invalid")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_online_points_viewer_projects_visible_world_points_to_pixels(self) -> None:
        viewer = importlib.import_module("demo_v5.online_points_viewer")
        points = np.array(
            [
                [0.0, 0.0, 1.0],
                [1.0, 0.0, 1.0],
                [0.0, 1.0, 1.0],
                [0.0, 0.0, -1.0],
            ],
            dtype=np.float64,
        )
        visibility = np.array([True, False, True, True], dtype=bool)
        pixels, point_indices = viewer.project_world_points_to_pixels(
            points,
            intrinsic=np.eye(3, dtype=np.float64),
            camera_to_world=np.eye(4, dtype=np.float64),
            image_size=(4, 4),
            visibility=visibility,
            stride=1,
        )

        np.testing.assert_array_equal(pixels, np.array([[0, 0], [0, 1]], dtype=np.int32))
        np.testing.assert_array_equal(point_indices, np.array([0, 2], dtype=np.int64))


if __name__ == "__main__":
    unittest.main()
