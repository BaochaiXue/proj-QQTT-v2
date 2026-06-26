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

from demo_v5 import realtime_data_process_track
from demo_v5.data_process_chunk_writer import (
    DataProcessChunk,
    build_query_schema_payload,
    validate_data_process_case,
    write_data_process_chunk_case,
)
from demo_v5.chunked_final_data_aggregate import build_aggregate_case_from_chunk_cases
from demo_v5.chunked_final_data_output import ChunkedFinalDataWriter
import demo_v5.realtime_data_process_sam3d as demo_v5


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


def _tiny_data_process_chunk(*, chunk_index: int, serial: str = "demo-v5-single-camera") -> DataProcessChunk:
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
    return DataProcessChunk(
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


def _with_track_diagnostics(chunk: DataProcessChunk, *, quality_status: str = "normal") -> DataProcessChunk:
    track = dict(chunk.track_process_data)
    controller_points = np.asarray(track["controller_points"])
    frame_count, anchor_count = controller_points.shape[:2]
    object_count = int(np.asarray(track["object_points"]).shape[1])
    anchor_query_ids = np.arange(anchor_count, dtype=np.int64) + object_count
    track.update(
        {
            "controller_track_query_indices": anchor_query_ids,
            "controller_track_active_query_indices": anchor_query_ids,
            "controller_track_status": np.asarray(["direct"] * anchor_count, dtype="<U16"),
            "controller_neighbor_query_ids": np.tile(anchor_query_ids[:, None], (1, 3)),
            "controller_source_query_ids": np.tile(anchor_query_ids[None, :], (frame_count, 1)),
            "controller_track_mode": np.full((frame_count, anchor_count), "direct_valid", dtype="<U40"),
            "controller_track_confidence": np.ones((frame_count, anchor_count), dtype=np.float32),
            "controller_filter_reason": np.full((frame_count, anchor_count), "none", dtype="<U40"),
            "controller_neighbor_support_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_neighbor_raw_visible_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_neighbor_depth_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_neighbor_processed_mask_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_neighbor_motion_valid_count": np.full((frame_count, anchor_count), 3, dtype=np.int64),
            "controller_neighbor_fit_residual": np.zeros((frame_count, anchor_count), dtype=np.float32),
            "track_process_status": str(quality_status),
        }
    )
    return DataProcessChunk(
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
        self.assertEqual(str(args.base_path), "result/demo_v5/data_process_sam3d_chunks")
        self.assertEqual(args.case_prefix, "demo_v5")
        self.assertEqual(args.realtime_gpu_mode, None)
        self.assertEqual(args.warmup_gpu_mode, "dual")
        self.assertEqual(demo_v5.resolve_camera_cuda_visible_devices(args), "0")
        self.assertEqual(demo_v5.resolve_shape_prior_worker_cuda_visible_devices(args), "1")
        self.assertEqual(args.shape_prior_worker_mode, "managed")
        self.assertEqual(args.optimization_mode, "disabled")
        self.assertEqual(args.point_viewer_mode, "window")
        self.assertEqual(args.point_viewer_layout, "side-by-side")
        self.assertFalse(args.allow_degraded_online)
        self.assertEqual(args.point_viewer_render_mode, "sam3d-final-data")
        self.assertEqual(
            demo_v5.build_parser().parse_args(["--point-viewer-render-mode", "sam3d-final-data"]).point_viewer_render_mode,
            "sam3d-final-data",
        )
        self.assertTrue(demo_v5.build_parser().parse_args(["--allow-degraded-online"]).allow_degraded_online)
        self.assertEqual(demo_v5.resolve_point_viewer_cuda_visible_devices(args), "1")
        self.assertEqual(demo_v5.resolve_optimization_cuda_visible_devices(args), "1")
        self.assertEqual(demo_v5.resolve_optimization_device(args), "cuda:0")

        contract = demo_v5._contract(args)

        self.assertEqual(contract["demo_version"], "demo_v5")
        self.assertEqual(contract["optimization_scope"], "disabled")
        self.assertEqual(contract["optimization_segment_len"], 35)
        self.assertEqual(contract["shape_prior_worker_released_before_optimization"], False)
        self.assertEqual(contract["shape_prior_worker_released_before_point_viewer"], False)
        self.assertTrue(contract["write_input_rgb_timeline"])
        worker_command = contract["shape_prior_worker_command"]
        self.assertIn("--max-observation-to-aligned-p95-m", worker_command)
        self.assertEqual(worker_command[worker_command.index("--max-observation-to-aligned-p95-m") + 1], "0.06")
        self.assertEqual(
            contract["online_dir"],
            "result/demo_v5/data_process_sam3d_chunks/online_data/demo_v5",
        )
        self.assertTrue(
            str(contract["static_data_path"]).endswith(
                "result/demo_v5/data_process_sam3d_chunks/data/demo_v5/final_data.pkl"
            )
        )
        point_viewer_command = contract["point_viewer_command"]
        self.assertEqual(point_viewer_command[:6], ["conda", "run", "-n", "demo_2_max", "--no-capture-output", "python"])
        self.assertEqual(point_viewer_command[6], "demo_v5/visualize_track.py")
        self.assertEqual(point_viewer_command[point_viewer_command.index("--layout") + 1], "side-by-side")
        self.assertEqual(point_viewer_command[point_viewer_command.index("--render-mode") + 1], "sam3d-final-data")
        self.assertEqual(point_viewer_command[point_viewer_command.index("--capture-dir") + 1], "")
        self.assertEqual(
            point_viewer_command[point_viewer_command.index("--online-dir") + 1],
            "result/demo_v5/data_process_sam3d_chunks/online_data/demo_v5",
        )
        self.assertEqual(
            point_viewer_command[point_viewer_command.index("--case-dir") + 1],
            "result/demo_v5/data_process_sam3d_chunks/data/demo_v5",
        )
        self.assertEqual(point_viewer_command[point_viewer_command.index("--fps") + 1], "5.0")
        self.assertEqual(point_viewer_command[point_viewer_command.index("--object-color-mode") + 1], "rainbow")
        self.assertNotIn("--target-latency-s", point_viewer_command)
        sam3d_viewer_args = demo_v5.build_parser().parse_args(["--point-viewer-render-mode", "sam3d-final-data"])
        sam3d_viewer_command = demo_v5.build_point_viewer_command(sam3d_viewer_args)
        self.assertEqual(sam3d_viewer_command[sam3d_viewer_command.index("--render-mode") + 1], "sam3d-final-data")
        opt_command = contract["optimization_command"]
        self.assertEqual(opt_command[1], "train_online_zero_then_first.py")
        self.assertNotIn("--stop_when_finished", opt_command)
        self.assertFalse(contract["optimization_stop_when_finished"])
        self.assertFalse(Path(opt_command[opt_command.index("--base_path") + 1]).is_absolute())
        self.assertEqual(opt_command[opt_command.index("--base_path") + 1], "../result/demo_v5/data_process_sam3d_chunks/data")
        self.assertEqual(
            opt_command[opt_command.index("--online_dir") + 1],
            "../result/demo_v5/data_process_sam3d_chunks/online_data/demo_v5",
        )
        self.assertEqual(opt_command[opt_command.index("--segment_len") + 1], "35")
        self.assertEqual(opt_command[opt_command.index("--device") + 1], "cuda:0")

    def test_disabled_point_viewer_does_not_force_input_rgb_timeline(self) -> None:
        disabled_args = demo_v5.build_parser().parse_args(["--point-viewer-mode", "disabled"])
        explicit_args = demo_v5.build_parser().parse_args(
            ["--point-viewer-mode", "disabled", "--write-input-rgb-timeline"]
        )

        self.assertFalse(demo_v5.resolve_write_input_rgb_timeline(disabled_args))
        self.assertTrue(demo_v5.resolve_write_input_rgb_timeline(explicit_args))

    def test_camera_command_uses_demo_v5_final_data_contract(self) -> None:
        args = demo_v5.build_parser().parse_args([])
        command = demo_v5.build_camera_realtime_command(
            args,
            capture_dir=Path("result/demo_v5/unit_capture"),
            profile_json=Path("result/demo_v5/unit_capture/shape_prior_profile.json"),
            chunk_frame_count=35,
        )

        joined = " ".join(command)
        self.assertIn("demo_v5/realtime_dense_track.py", command[1])
        self.assertNotIn("demo_v3_2", joined)
        self.assertNotIn("--depth-backend", command)
        self.assertEqual(command[command.index("--depth-source") + 1], "realsense")
        self.assertEqual(command[command.index("--depth-backend-label") + 1], "native-realsense")
        self.assertIn("--headless-prepared-only", command)
        self.assertIn("--write-input-rgb-timeline", command)
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
            write_data_process_chunk_case(base_path, "demo_v5_meta_chunk_0001", _tiny_data_process_chunk(chunk_index=0))
            write_data_process_chunk_case(base_path, "demo_v5_meta_chunk_0002", _tiny_data_process_chunk(chunk_index=1))
            chunk_metadata = json.loads((base_path / "demo_v5_meta_chunk_0001" / "metadata.json").read_text(encoding="utf-8"))

            self.assertEqual(chunk_metadata["demo_version"], "demo_v5")
            self.assertEqual(chunk_metadata["runtime_product_name"], "demo_v5_realtime_dense_track")
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
            self.assertEqual(aggregate_metadata["runtime_product_name"], "demo_v5_realtime_dense_track")
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
            write_data_process_chunk_case(
                root / "cases",
                "demo_v5_semantics_chunk_0001",
                _tiny_data_process_chunk(chunk_index=0),
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
            query_schema = build_query_schema_payload(
                final_data,
                object_sample_query_ids=final_data["object_sample_query_ids"],
                controller_sample_query_ids=final_data["controller_sample_query_ids"],
            )
            for payload in (final_data, track_process):
                for key, value in query_schema.items():
                    payload[key] = value
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(final_data, handle)
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump(track_process, handle)

            with self.assertRaisesRegex(ValueError, "object_sample_query_ids.*object semantic"):
                validate_data_process_case(case_dir)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_validate_rejects_controller_sample_ids_with_object_semantics(self) -> None:
        root = Path("result/test_demo_v5_unit_sample_semantics_controller")
        shutil.rmtree(root, ignore_errors=True)
        try:
            case_dir = root / "cases" / "demo_v5_semantics_chunk_0001"
            write_data_process_chunk_case(
                root / "cases",
                "demo_v5_semantics_chunk_0001",
                _tiny_data_process_chunk(chunk_index=0),
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
            query_schema = build_query_schema_payload(
                final_data,
                object_sample_query_ids=final_data["object_sample_query_ids"],
                controller_sample_query_ids=final_data["controller_sample_query_ids"],
            )
            for payload in (final_data, track_process):
                for key, value in query_schema.items():
                    payload[key] = value
            with (case_dir / "final_data.pkl").open("wb") as handle:
                pickle.dump(final_data, handle)
            with (case_dir / "track_process_data.pkl").open("wb") as handle:
                pickle.dump(track_process, handle)

            with self.assertRaisesRegex(ValueError, "controller_sample_query_ids.*controller semantic"):
                validate_data_process_case(case_dir)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_anchor_diagnostics_are_persisted_to_track_process_online_chunk_and_aggregate(self) -> None:
        root = Path("result/test_demo_v5_unit_anchor_diagnostics")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            chunk = _with_track_diagnostics(_tiny_data_process_chunk(chunk_index=0))
            manifest = write_data_process_chunk_case(base_path, "diag_chunk_0001", chunk)
            case_dir = Path(manifest["data_process_case_root"])
            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)

            self.assertEqual(track_process["controller_track_confidence"].shape, (2, 2))
            self.assertEqual(track_process["controller_neighbor_query_ids"].shape, (2, 3))
            self.assertEqual(track_process["controller_neighbor_raw_visible_count"].shape, (2, 2))
            self.assertEqual(str(track_process["track_process_status"]), "normal")

            writer = ChunkedFinalDataWriter(base_path=base_path, case_name="diag", chunk_size=2)
            online_result = writer.commit_case_chunk(case_dir)
            writer.finish()

            with Path(online_result["online_chunk_path"]).open("rb") as handle:
                online_chunk = pickle.load(handle)
            self.assertEqual(online_chunk["controller_track_confidence"].shape, (2, 2))
            np.testing.assert_array_equal(
                online_chunk["controller_neighbor_query_ids"],
                track_process["controller_neighbor_query_ids"],
            )

            with (base_path / "data" / "diag" / "track_process_data.pkl").open("rb") as handle:
                aggregate_track = pickle.load(handle)
            self.assertEqual(aggregate_track["controller_track_confidence"].shape, (2, 2))
            self.assertEqual(aggregate_track["controller_neighbor_query_ids"].shape, (2, 3))
            self.assertEqual(aggregate_track["controller_neighbor_motion_valid_count"].shape, (2, 2))
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_track_process_markers_distinguish_ready_degraded_and_invalid(self) -> None:
        root = Path("result/test_demo_v5_unit_quality_markers")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            cases = {}
            for quality in ("normal", "degraded", "invalid"):
                manifest = write_data_process_chunk_case(
                    base_path,
                    f"{quality}_chunk_0001",
                    _with_track_diagnostics(_tiny_data_process_chunk(chunk_index=0), quality_status=quality),
                )
                cases[quality] = Path(manifest["data_process_case_root"])
                self.assertEqual(manifest["track_process_status"], quality)

            self.assertTrue((cases["normal"] / "READY").is_file())
            self.assertFalse((cases["normal"] / "DEGRADED").exists())
            self.assertFalse((cases["normal"] / "INVALID").exists())
            self.assertTrue((cases["degraded"] / "DEGRADED").is_file())
            self.assertFalse((cases["degraded"] / "READY").exists())
            self.assertTrue((cases["invalid"] / "INVALID").is_file())
            self.assertFalse((cases["invalid"] / "READY").exists())
            validate_data_process_case(cases["normal"], require_ready=True)
            with self.assertRaisesRegex(ValueError, "READY"):
                validate_data_process_case(cases["degraded"], require_ready=True)
            with self.assertRaisesRegex(ValueError, "READY"):
                validate_data_process_case(cases["invalid"], require_ready=True)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_track_process_online_publish_policy_defaults_degraded_to_diagnostic_only(self) -> None:
        self.assertIsNone(
            realtime_data_process_track._track_process_online_publish_skip_reason(
                {"track_process_status": "normal"},
                allow_degraded_online=False,
            )
        )
        self.assertEqual(
            realtime_data_process_track._track_process_online_publish_skip_reason(
                {"track_process_status": "degraded"},
                allow_degraded_online=False,
            ),
            "track_process_degraded",
        )
        self.assertIsNone(
            realtime_data_process_track._track_process_online_publish_skip_reason(
                {"track_process_status": "degraded"},
                allow_degraded_online=True,
            )
        )
        self.assertEqual(
            realtime_data_process_track._track_process_online_publish_skip_reason(
                {"track_process_status": "invalid"},
                allow_degraded_online=True,
            ),
            "track_process_invalid",
        )

    def test_online_chunk_preserves_realtime_source_indices_and_timestamps(self) -> None:
        root = Path("result/test_demo_v5_unit_online_source_timeline")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            manifest = write_data_process_chunk_case(
                base_path,
                "source_timeline_chunk_0001",
                _tiny_data_process_chunk(chunk_index=0),
            )
            writer = ChunkedFinalDataWriter(base_path=base_path, case_name="source_timeline", chunk_size=2)
            online_result = writer.commit_case_chunk(
                Path(manifest["data_process_case_root"]),
                source_frame_indices=[551, 557],
                source_timestamps_s=[1018.3, 1018.5],
            )
            writer.finish()

            with Path(online_result["online_chunk_path"]).open("rb") as handle:
                online_chunk = pickle.load(handle)
            self.assertEqual(online_chunk["source_frame_indices"], [551, 557])
            self.assertEqual(online_chunk["source_timestamps_s"], [1018.3, 1018.5])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_legacy_controller_anchor_keys_are_normalized_to_track_keys(self) -> None:
        root = Path("result/test_demo_v5_unit_legacy_anchor_schema")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            chunk = _with_track_diagnostics(_tiny_data_process_chunk(chunk_index=0))
            legacy_track = dict(chunk.track_process_data)
            legacy_track["controller_anchor_query_indices"] = legacy_track.pop("controller_track_query_indices")
            legacy_track["controller_anchor_active_query_indices"] = legacy_track.pop("controller_track_active_query_indices")
            legacy_track["controller_anchor_status"] = legacy_track.pop("controller_track_status")
            legacy_track["controller_anchor_bundle_query_ids"] = legacy_track.pop("controller_neighbor_query_ids")
            legacy_track["controller_anchor_source_query_id"] = legacy_track.pop("controller_source_query_ids")
            legacy_track["controller_anchor_observation_mode"] = legacy_track.pop("controller_track_mode")
            legacy_track["controller_anchor_confidence"] = legacy_track.pop("controller_track_confidence")
            legacy_track["controller_anchor_failure_reason"] = legacy_track.pop("controller_filter_reason")
            legacy_track["controller_anchor_bundle_support_count"] = legacy_track.pop("controller_neighbor_support_count")
            legacy_track["controller_anchor_recovery_residual"] = legacy_track.pop("controller_neighbor_fit_residual")
            legacy_track["controller_quality_status"] = legacy_track.pop("track_process_status")
            legacy_chunk = DataProcessChunk(
                rgb_frames=chunk.rgb_frames,
                processed_masks=chunk.processed_masks,
                track_process_data=legacy_track,
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

            manifest = write_data_process_chunk_case(base_path, "legacy_schema_chunk_0001", legacy_chunk)
            case_dir = Path(manifest["data_process_case_root"])
            validate_data_process_case(case_dir, require_ready=True)

            with (case_dir / "track_process_data.pkl").open("rb") as handle:
                track_process = pickle.load(handle)
            self.assertIn("controller_track_query_indices", track_process)
            self.assertIn("controller_neighbor_query_ids", track_process)
            self.assertIn("controller_source_query_ids", track_process)
            self.assertIn("controller_track_confidence", track_process)
            self.assertIn("track_process_status", track_process)
            self.assertNotIn("controller_anchor_query_indices", track_process)
            self.assertNotIn("controller_anchor_bundle_query_ids", track_process)
            self.assertNotIn("controller_quality_status", track_process)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_legacy_demo_v5_modules_are_thin_compatibility_wrappers(self) -> None:
        legacy_writer = importlib.import_module("demo_v5.futurephystwin_chunk_writer")
        legacy_output = importlib.import_module("demo_v5.online_chunk_output")
        legacy_aggregate = importlib.import_module("demo_v5.online_case_aggregate")

        self.assertIs(legacy_writer.FuturePhysTwinChunk, DataProcessChunk)
        self.assertIs(legacy_writer.write_futurephystwin_chunk_case, write_data_process_chunk_case)
        self.assertIs(legacy_writer.validate_futurephystwin_case, validate_data_process_case)
        self.assertIs(legacy_output.DemoV5OnlineOutputWriter, ChunkedFinalDataWriter)
        self.assertIs(legacy_aggregate.OnlineAggregateCaseWriter, legacy_aggregate.FinalDataAggregateWriter)

    def test_allow_degraded_online_commits_degraded_marker_case(self) -> None:
        root = Path("result/test_demo_v5_unit_allow_degraded_online")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            manifest = write_data_process_chunk_case(
                base_path,
                "degraded_chunk_0001",
                _with_track_diagnostics(_tiny_data_process_chunk(chunk_index=0), quality_status="degraded"),
            )
            case_dir = Path(manifest["data_process_case_root"])

            default_writer = ChunkedFinalDataWriter(base_path=base_path, case_name="default", chunk_size=2)
            with self.assertRaisesRegex(ValueError, "READY"):
                default_writer.commit_case_chunk(case_dir)

            allowed_writer = ChunkedFinalDataWriter(
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
            self.assertEqual(str(aggregate_track["track_process_status"]), "degraded")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_candidate_motion_validity_survives_anchor_selection_shape_split(self) -> None:
        root = Path("result/test_demo_v5_unit_candidate_motion_validity")
        shutil.rmtree(root, ignore_errors=True)
        try:
            base_path = root / "cases"
            chunk = _with_track_diagnostics(_tiny_data_process_chunk(chunk_index=0))
            track = dict(chunk.track_process_data)
            frame_count, anchor_count = np.asarray(track["controller_points"]).shape[:2]
            candidate_count = 5
            candidate_motion_valid = np.ones((frame_count, candidate_count), dtype=bool)
            candidate_motion_valid[1, -1] = False
            track["controller_mask"] = np.ones((candidate_count,), dtype=bool)
            track["controller_query_indices"] = np.asarray([3, 4, 10, 11, 12], dtype=np.int64)
            track["controller_candidate_motions_valid"] = candidate_motion_valid
            track["controller_motions_valid"] = np.ones((frame_count, anchor_count), dtype=bool)
            chunk = DataProcessChunk(
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

            manifest = write_data_process_chunk_case(base_path, "candidate_motion_chunk_0001", chunk)

            with (Path(manifest["data_process_case_root"]) / "track_process_data.pkl").open("rb") as handle:
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
                realtime_data_process_track._shape_points_for_chunk(
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

    def test_trim_warmup_delayed_rows_starts_at_realtime_stream(self) -> None:
        rows = [
            {
                "seq": 0,
                "source_frame_index": 0,
                "source_timestamp_s": 1000.0,
                "pipeline_latency_ms": 18234.0,
                "startup_hold_s": 18.2,
            },
            {
                "seq": 1,
                "source_frame_index": 551,
                "source_timestamp_s": 1018.3,
                "pipeline_latency_ms": 248.0,
                "startup_hold_s": 18.2,
            },
            {
                "seq": 2,
                "source_frame_index": 557,
                "source_timestamp_s": 1018.5,
                "pipeline_latency_ms": 221.0,
                "startup_hold_s": 18.2,
            },
        ]

        trimmed, skipped = realtime_data_process_track._trim_warmup_delayed_rows(rows)

        self.assertEqual(skipped, 1)
        self.assertEqual([row["source_frame_index"] for row in trimmed], [551, 557])

    def test_live_start_filter_waits_for_second_row_before_publishing_first_chunk_row(self) -> None:
        state = realtime_data_process_track._WarmupStartFilterState()
        warmup_row = {
            "seq": 0,
            "source_frame_index": 0,
            "source_timestamp_s": 1000.0,
            "pipeline_latency_ms": 18234.0,
            "startup_hold_s": 18.2,
        }
        realtime_row = {
            "seq": 1,
            "source_frame_index": 551,
            "source_timestamp_s": 1018.3,
            "pipeline_latency_ms": 248.0,
            "startup_hold_s": 18.2,
        }
        realtime_next = {
            "seq": 2,
            "source_frame_index": 557,
            "source_timestamp_s": 1018.5,
            "pipeline_latency_ms": 221.0,
            "startup_hold_s": 18.2,
        }

        first = realtime_data_process_track._filter_warmup_start_rows(
            state,
            [warmup_row],
            capture_finished=False,
        )
        second = realtime_data_process_track._filter_warmup_start_rows(
            state,
            [realtime_row],
            capture_finished=False,
        )
        third = realtime_data_process_track._filter_warmup_start_rows(
            state,
            [realtime_next],
            capture_finished=False,
        )

        self.assertEqual(first, [])
        self.assertEqual(second, [])
        self.assertEqual([row["source_frame_index"] for row in third], [551, 557])
        self.assertEqual(state.skipped_rows, 1)

    def test_visualize_track_selects_output_frame_by_source_time_latency(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")

        selected = viewer.select_output_frame_for_input_source_time(
            output_source_times=[18.0, 18.2, 18.4, 25.0, 25.2],
            input_source_time=32.1,
            target_latency_s=7.0,
        )

        self.assertEqual(selected, 3)

    def test_visualize_track_live_output_cursor_does_not_jump_to_new_chunk_tail(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
        cursor = viewer.OutputStreamPlaybackCursor(fps=5.0)
        cursor.output_index = 34
        cursor.last_step_s = 10.0

        first = cursor.advance(latest=69, now_s=10.10, paused=False)
        second = cursor.advance(latest=69, now_s=10.20, paused=False)
        third = cursor.advance(latest=69, now_s=10.40, paused=False)

        self.assertEqual(first, 34)
        self.assertEqual(second, 35)
        self.assertEqual(third, 36)

    def test_visualize_track_live_output_cursor_advances_one_frame_after_ui_stall(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
        cursor = viewer.OutputStreamPlaybackCursor(fps=5.0)
        cursor.output_index = 34
        cursor.last_step_s = 10.0

        selected = cursor.advance(latest=69, now_s=11.0, paused=False)

        self.assertEqual(selected, 35)

    def test_visualize_track_uses_interactive_open3d_backend_for_live_final_data_side_by_side(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")

        live_args = viewer.build_parser().parse_args(
            [
                "--layout",
                "side-by-side",
                "--online-dir",
                "result/demo_v5/unit/online_data/case",
                "--render-mode",
                "sam3d-final-data",
            ]
        )
        video_args = viewer.build_parser().parse_args(
            [
                "--layout",
                "side-by-side",
                "--online-dir",
                "result/demo_v5/unit/online_data/case",
                "--render-mode",
                "sam3d-final-data",
                "--output-video",
                "result/demo_v5/unit/side_by_side.mp4",
            ]
        )
        overlay_args = viewer.build_parser().parse_args(
            [
                "--layout",
                "side-by-side",
                "--online-dir",
                "result/demo_v5/unit/online_data/case",
                "--render-mode",
                "rgb-overlay",
            ]
        )

        self.assertTrue(viewer.use_interactive_side_by_side(live_args))
        self.assertFalse(viewer.use_interactive_side_by_side(video_args))
        self.assertFalse(viewer.use_interactive_side_by_side(overlay_args))

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
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                second = {
                    "case_name": "demo_v5_rt_chunk_0002",
                    "frame_count": 35,
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0002"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000001.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 14.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](first)
                kwargs["on_chunk_written"](second)
                return [first, second]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--base-path",
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
            self.assertIn("demo_v5/realtime_dense_track.py", camera_command[1])
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
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--base-path",
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

    def test_output_only_point_viewer_starts_after_first_committed_online_chunk(self) -> None:
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
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--point-viewer-layout",
                                "output-only",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(len(popen_calls), 2)
            camera_command, _camera_kwargs = popen_calls[0]
            viewer_command, viewer_kwargs = popen_calls[1]
            self.assertIn("demo_v5/realtime_dense_track.py", camera_command[1])
            self.assertEqual(viewer_command[:6], ["conda", "run", "-n", "demo_2_max", "--no-capture-output", "python"])
            self.assertEqual(viewer_command[6], "demo_v5/visualize_track.py")
            self.assertEqual(viewer_command[viewer_command.index("--layout") + 1], "output-only")
            self.assertEqual(viewer_command[viewer_command.index("--render-mode") + 1], "sam3d-final-data")
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
            self.assertEqual(summary["point_viewer_layout"], "output-only")
            self.assertEqual(summary["point_viewer_start_policy"], "after_first_committed_online_chunk")
            self.assertEqual(summary["point_viewer_started_from_chunk"]["case_name"], "demo_v5_rt_chunk_0001")
            self.assertEqual(summary["point_viewer_return_code"], 0)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_side_by_side_point_viewer_starts_immediately_with_capture_dir(self) -> None:
        root = Path("result/test_demo_v5_unit_side_by_side_point_viewer")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"
            popen_calls: list[tuple[list[str], dict[str, object]]] = []
            event_order: list[str] = []

            def fake_popen(command, **kwargs):
                popen_calls.append((list(command), dict(kwargs)))
                if "demo_v5/visualize_track.py" in command:
                    event_order.append("viewer_started")
                elif "demo_v5/realtime_dense_track.py" in command:
                    event_order.append("camera_started")
                return FakeProcess(returncode=0)

            def fake_stream(_capture_dir_arg, **kwargs):
                event_order.append("stream_started")
                manifest = {
                    "case_name": "demo_v5_rt_chunk_0001",
                    "frame_count": 35,
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--base-path",
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
            self.assertEqual(event_order[:3], ["camera_started", "viewer_started", "stream_started"])
            self.assertEqual(len(popen_calls), 2)
            viewer_command, viewer_kwargs = popen_calls[1]
            self.assertEqual(viewer_command[6], "demo_v5/visualize_track.py")
            self.assertEqual(viewer_command[viewer_command.index("--layout") + 1], "side-by-side")
            self.assertEqual(viewer_command[viewer_command.index("--capture-dir") + 1], str(capture_dir))
            self.assertEqual(viewer_command[viewer_command.index("--input-rgb-timeline") + 1], str(capture_dir / "input_frames.jsonl"))
            self.assertEqual(viewer_kwargs["env"]["CUDA_VISIBLE_DEVICES"], "1")
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["point_viewer_layout"], "side-by-side")
            self.assertEqual(summary["point_viewer_start_policy"], "immediate_after_camera_start")
            self.assertEqual(summary["point_viewer_capture_dir"], str(capture_dir))
            self.assertIsNone(summary["point_viewer_started_from_chunk"])
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_output_only_point_viewer_keeps_legacy_chunk_start_policy(self) -> None:
        root = Path("result/test_demo_v5_unit_output_only_point_viewer")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            base_path = root / "cases"
            event_order: list[str] = []

            def fake_popen(command, **_kwargs):
                if "demo_v5/visualize_track.py" in command:
                    event_order.append("viewer_started")
                elif "demo_v5/realtime_dense_track.py" in command:
                    event_order.append("camera_started")
                return FakeProcess(returncode=0)

            def fake_stream(_capture_dir_arg, **kwargs):
                event_order.append("stream_started")
                manifest = {
                    "case_name": "demo_v5_rt_chunk_0001",
                    "frame_count": 35,
                    "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                    "online_chunk_path": str(base_path / "online_data" / "demo_v5_rt" / "chunks" / "chunk_000000.pkl"),
                    "static_data_path": str(base_path / "data" / "demo_v5_rt" / "final_data.pkl"),
                    "publish_wall_s": 7.0,
                    "backlog_chunks": 0,
                    "shape_prior_complete": True,
                }
                kwargs["on_chunk_written"](manifest)
                return [manifest]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", side_effect=fake_popen):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--base-path",
                                str(base_path),
                                "--case-prefix",
                                "demo_v5_rt",
                                "--camera-capture-dir",
                                str(capture_dir),
                                "--max-chunks",
                                "1",
                                "--point-viewer-layout",
                                "output-only",
                            ]
                        )

            self.assertEqual(exit_code, 0)
            self.assertEqual(event_order, ["camera_started", "stream_started", "viewer_started"])
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["point_viewer_layout"], "output-only")
            self.assertEqual(summary["point_viewer_start_policy"], "after_first_committed_online_chunk")
            self.assertEqual(summary["point_viewer_started_from_chunk"]["case_name"], "demo_v5_rt_chunk_0001")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_prepared_only_headless_writer_can_still_write_input_rgb_timeline(self) -> None:
        from qqtt.demo.realtime_masked_edgetam_pcd import (
            CameraIntrinsics,
            FramePacket,
            HeadlessCaptureWriter,
            PipelineTiming,
        )

        root = Path("result/test_demo_v5_unit_input_rgb_timeline")
        shutil.rmtree(root, ignore_errors=True)
        try:
            writer = HeadlessCaptureWriter(
                root,
                metadata={
                    "headless_prepared_only": True,
                    "write_input_rgb_timeline": True,
                },
            )
            packet = FramePacket(
                seq=3,
                color_bgr=np.full((4, 5, 3), (10, 20, 30), dtype=np.uint8),
                depth_source="realsense",
                intrinsics=CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
                depth_scale_m_per_unit=0.001,
                receive_perf_s=12.5,
                timing=PipelineTiming(),
                source_timestamp_s=99.0,
                source_frame_index=123,
                source_step=456,
            )

            writer.write_input_frame(packet)

            rows = [
                json.loads(line)
                for line in (root / "input_frames.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["seq"], 3)
            self.assertEqual(rows[0]["input_rgb_path"], "input_rgb/000003.png")
            self.assertTrue((root / "input_rgb" / "000003.png").is_file())
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_visualize_track_side_by_side_renders_input_with_blank_output_before_chunks(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
        root = Path("result/test_demo_v5_side_by_side_blank")
        shutil.rmtree(root, ignore_errors=True)
        try:
            capture_dir = root / "capture"
            input_rgb_dir = capture_dir / "input_rgb"
            input_rgb_dir.mkdir(parents=True)
            import cv2

            image = np.full((8, 10, 3), (5, 80, 180), dtype=np.uint8)
            cv2.imwrite(str(input_rgb_dir / "000000.png"), image)
            (capture_dir / "input_frames.jsonl").write_text(
                json.dumps({"seq": 0, "input_rgb_path": "input_rgb/000000.png"}) + "\n",
                encoding="utf-8",
            )

            rendered = viewer.render_side_by_side_frame(
                input_frame=viewer.load_latest_input_rgb_frame(capture_dir / "input_frames.jsonl", capture_dir=capture_dir),
                output_frame=None,
                image_size=(10, 8),
                right_blank_label="waiting for first final_data chunk",
            )

            self.assertEqual(rendered.shape, (8, 20, 3))
            np.testing.assert_array_equal(rendered[2, 2], image[2, 2])
            self.assertLess(int(rendered[2, 15].max()), 80)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_realtime_runner_returns_nonzero_when_track_process_is_invalid(self) -> None:
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
                        "data_process_case_root": str(base_path / "demo_v5_rt_chunk_0001"),
                        "publish_wall_s": 7.0,
                        "backlog_chunks": 0,
                        "shape_prior_complete": True,
                        "track_process_status": "invalid",
                        "online_publish_skipped": True,
                    }
                ]

            with mock.patch("demo_v5.realtime_data_process_sam3d.subprocess.Popen", return_value=FakeProcess(returncode=0)):
                with mock.patch("demo_v5.realtime_data_process_sam3d.stream_chunks_from_headless_capture", side_effect=fake_stream):
                    with redirect_stdout(io.StringIO()) as stdout:
                        exit_code = demo_v5.main(
                            [
                                "--shape-prior-worker-mode",
                                "external",
                                "--base-path",
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
            self.assertEqual(summary["track_process_status"], "invalid")
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_visualize_track_projects_visible_world_points_to_pixels(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
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

    def test_visualize_track_uses_sam3d_marker_colors(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
        import matplotlib.pyplot as plt

        camera = viewer.CameraModel(
            intrinsic=np.eye(3, dtype=np.float64),
            camera_to_world=np.eye(4, dtype=np.float64),
            image_size=(60, 60),
            metadata_fps=5.0,
        )
        chunk = {
            "chunk_id": 0,
            "source_frame_indices": [0],
            "query_ids": np.arange(16, dtype=np.int64),
            "object_points": np.array([[[6.0, 6.0, 1.0], [12.0, 12.0, 1.0]]], dtype=np.float32),
            "object_visibilities": np.ones((1, 2), dtype=bool),
            "object_selected_query_ids": np.array([2, 9], dtype=np.int64),
            "controller_points": np.array([[[20.0, 45.0, 1.0]]], dtype=np.float32),
            "controller_source_query_ids": np.array([[7]], dtype=np.int64),
        }

        image = viewer.render_chunk_frame(
            chunk,
            local_frame=0,
            case_dir=Path("/nonexistent"),
            camera=camera,
            cam_idx=0,
            use_background=False,
            show_invisible_object_points=False,
            object_stride=1,
            object_radius=2,
            controller_radius=5,
            object_color_mode="rainbow",
            controller_color=(0, 0, 255),
            fps=5.0,
        )

        object_expected_bgr = (np.asarray(plt.cm.rainbow(0.0)[:3]) * 255.0).astype(np.uint8)[::-1]
        controller_expected_bgr = np.array([0, 0, 255], dtype=np.uint8)
        np.testing.assert_array_equal(image[6, 6], object_expected_bgr)
        np.testing.assert_array_equal(image[45, 20], controller_expected_bgr)
        controller_patch = image[38:53, 13:28]
        self.assertFalse(np.any(np.all(controller_patch >= 235, axis=2)))

    def test_visualize_track_writes_offline_rgb_overlay_video(self) -> None:
        viewer = importlib.import_module("demo_v5.visualize_track")
        root = Path("result/test_demo_v5_online_points_video")
        shutil.rmtree(root, ignore_errors=True)
        try:
            online_dir = root / "online_data" / "case"
            chunks_dir = online_dir / "chunks"
            case_dir = root / "data" / "case"
            chunks_dir.mkdir(parents=True)
            case_dir.mkdir(parents=True)
            (case_dir / "metadata.json").write_text(json.dumps({"WH": [64, 64], "fps": 5.0}), encoding="utf-8")
            chunk = {
                "chunk_id": 0,
                "source_frame_indices": [0, 1],
                "object_points": np.array(
                    [
                        [[10.0, 10.0, 1.0], [20.0, 20.0, 1.0]],
                        [[11.0, 10.0, 1.0], [21.0, 20.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
                "object_visibilities": np.ones((2, 2), dtype=bool),
                "controller_points": np.array(
                    [
                        [[30.0, 40.0, 1.0]],
                        [[31.0, 40.0, 1.0]],
                    ],
                    dtype=np.float32,
                ),
            }
            with (chunks_dir / "chunk_000000.pkl").open("wb") as handle:
                pickle.dump(chunk, handle)
            output_video = root / "offline.mp4"

            exit_code = viewer.main(
                [
                    "--online-dir",
                    str(online_dir),
                    "--case-dir",
                    str(case_dir),
                    "--render-mode",
                    "rgb-overlay",
                    "--output-video",
                    str(output_video),
                    "--fps",
                    "5",
                    "--no-background",
                ]
            )

            self.assertEqual(exit_code, 0)
            self.assertTrue(output_video.is_file())
            import cv2

            cap = cv2.VideoCapture(str(output_video))
            ok, _frame = cap.read()
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            self.assertTrue(ok)
            self.assertEqual(frame_count, 2)
        finally:
            shutil.rmtree(root, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
