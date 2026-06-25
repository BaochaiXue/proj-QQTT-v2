from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
import pickle
from pathlib import Path
import shutil
import unittest
from unittest import mock

import numpy as np

from demo_v5 import headless_chunk_bridge
from demo_v5.futurephystwin_chunk_writer import FuturePhysTwinChunk, write_futurephystwin_chunk_case
from demo_v5.online_case_aggregate import build_aggregate_case_from_chunk_cases
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


class DemoV5RealtimePhysTwinTest(unittest.TestCase):
    def test_demo_v5_python_sources_do_not_import_demo_v4_modules(self) -> None:
        for path in sorted((REPO_ROOT / "demo_v5").glob("*.py")):
            text = path.read_text(encoding="utf-8")
            with self.subTest(path=path.name):
                self.assertNotIn("from demo_v4", text)
                self.assertNotIn("import demo_v4", text)

    def test_defaults_route_realtime_to_gpu0_and_continuous_optimization_to_gpu1(self) -> None:
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
        self.assertEqual(args.optimization_mode, "continuous")
        self.assertEqual(demo_v5.resolve_optimization_cuda_visible_devices(args), "1")
        self.assertEqual(demo_v5.resolve_optimization_device(args), "cuda:0")

        contract = demo_v5._contract(args)

        self.assertEqual(contract["demo_version"], "demo_v5")
        self.assertEqual(contract["optimization_scope"], "single_continuous_online_case")
        self.assertEqual(contract["optimization_segment_len"], 35)
        self.assertEqual(contract["shape_prior_worker_released_before_optimization"], True)
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
            with (aggregate_dir / "final_data.pkl").open("rb") as handle:
                final_data = pickle.load(handle)
            self.assertEqual(final_data["object_points"].shape[0], 4)
        finally:
            shutil.rmtree(root, ignore_errors=True)

    def test_source_headless_requires_optimization_disabled(self) -> None:
        with self.assertRaisesRegex(ValueError, "continuous optimization requires fake-live or live capture"):
            demo_v5.main(["--dry-run", "--source-headless-capture", "existing_capture"])

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


if __name__ == "__main__":
    unittest.main()
