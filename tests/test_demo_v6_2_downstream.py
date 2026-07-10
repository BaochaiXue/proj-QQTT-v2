"""Tests for the Demo v6.1 downstream.mode enum and Phystwin_shen launcher."""

from __future__ import annotations

import copy
import json
import os
import pickle
import signal
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from demo_v6_2 import main as runner
from demo_v6_2 import main_subprocess
from demo_v6_2.online_frame_archive import (
    OnlineFrameArchive,
    OnlineFrameArchiveError,
)
from demo_v6_2.phystwin_strict_product import PreparedPhysTwinFrame
from demo_v6_2.phystwin_shen_launch import (
    PhystwinShenLaunchError,
    PhystwinShenSettings,
    build_full_pipeline_command,
    ensure_port_free,
    launch_phystwin_shen,
    validate_phystwin_shen_repo,
    validate_phystwin_shen_settings,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _settings(repo: Path, base: Path, **overrides) -> PhystwinShenSettings:
    values = dict(
        repo_path=repo,
        pipeline_config=Path("configs/online_full_pipeline.yaml"),
        conda_env="demo_2_max",
        base_path=base,
        cuda_visible_devices="1",
        runtime_config=copy.deepcopy(runner.DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG),
    )
    values.update(overrides)
    return PhystwinShenSettings(**values)


class CameraSerialsConfigTests(unittest.TestCase):
    """camera.camera_serials is a list (multi-camera-extensible schema), but the
    current single-camera runtime requires exactly one entry and fails fast."""

    def test_config_default_is_single_calibration_camera(self) -> None:
        config = runner.load_default_config()
        self.assertEqual(config["camera"]["camera_serials"], ["239222300740"])
        self.assertEqual(runner.DEFAULT_CAMERA_SERIALS, ("239222300740",))

    def test_config_default_forwarded_to_subprocess(self) -> None:
        from demo_v6_2.main_subprocess import build_main_data_processing_command

        args = runner.build_parser().parse_args([])
        self.assertIsNone(args.camera_serials)  # CLI unset -> config list wins
        self.assertEqual(runner.resolve_camera_serials(args), ["239222300740"])
        command = build_main_data_processing_command(
            args,
            capture_dir=Path("capture"),
            profile_json=Path("profile.json"),
            chunk_frame_count=35,
        )
        flag_index = command.index("--serial")
        self.assertEqual(command[flag_index + 1], "239222300740")

    def test_cli_serial_overrides_config(self) -> None:
        args = runner.build_parser().parse_args(["--camera-serial", "112233445566"])
        self.assertEqual(runner.resolve_camera_serials(args), ["112233445566"])

    def test_multiple_serials_fail_fast_with_exact_message(self) -> None:
        args = runner.build_parser().parse_args(
            ["--camera-serial", "112233445566", "--camera-serial", "239222300740"]
        )
        with self.assertRaisesRegex(
            ValueError, "single-camera runtime requires exactly one serial"
        ):
            runner.resolve_camera_serials(args)
        with self.assertRaisesRegex(
            ValueError, "single-camera runtime requires exactly one serial"
        ):
            runner.validate_runtime_args(args, chunk_frame_count=35)

    def test_empty_serial_list_fails_fast(self) -> None:
        args = runner.build_parser().parse_args([])
        args.camera_serials = []
        with self.assertRaisesRegex(
            ValueError, "single-camera runtime requires exactly one serial"
        ):
            runner.resolve_camera_serials(args)

    def test_camera_cli_accepts_serial(self) -> None:
        from demo_v6_2.mdp_cli import build_parser as build_camera_parser

        camera_args = build_camera_parser().parse_args(["--serial", "239222300740"])
        self.assertEqual(camera_args.serial, "239222300740")
        self.assertIsNone(build_camera_parser().parse_args([]).serial)


class DownstreamConfigTests(unittest.TestCase):
    def test_downstream_mode_enum_and_default(self) -> None:
        self.assertEqual(
            runner.DOWNSTREAM_MODES,
            ("disabled", "demo_visualizer", "phystwin_shen"),
        )
        self.assertEqual(runner.DEFAULT_DOWNSTREAM_MODE, "phystwin_shen")
        config = runner.load_default_config()
        self.assertEqual(config["downstream"]["mode"], "phystwin_shen")
        self.assertNotIn("visualizer_mode", config["visualizer"])

    def test_phystwin_shen_config_defaults(self) -> None:
        config = runner.load_default_config()
        section = config["phystwin_shen"]
        self.assertEqual(section["repo_path"], "/home/xinjie/Phystwin_shen")
        self.assertEqual(section["conda_env"], "demo_2_max")
        self.assertEqual(
            section["pipeline_config"], "configs/online_full_pipeline.yaml"
        )
        self.assertEqual(section["common"]["batch_size"], 4)
        self.assertEqual(section["common"]["segment_len"], 30)
        self.assertTrue(section["stage1"]["enabled"])
        self.assertFalse(section["stage2"]["enabled"])
        self.assertEqual(section["train"]["iterations"], 20)
        self.assertFalse(section["train"]["stop_when_finished"])
        self.assertEqual(section["cma_viewer"]["port"], 8765)
        self.assertEqual(section["train_viewer"]["port"], 8766)
        self.assertEqual(str(config["gpu"]["phystwin_shen_cuda_visible_devices"]), "1")

    def test_base_shell_still_launches_wrapper_in_demo_2_max(self) -> None:
        with mock.patch.dict(os.environ, {"CONDA_DEFAULT_ENV": "base"}):
            prefix = runner._python_command_prefix("demo_2_max")
        self.assertEqual(
            prefix,
            [
                "conda",
                "run",
                "-n",
                "demo_2_max",
                "--no-capture-output",
                "python",
            ],
        )

    def test_resolve_downstream_mode_rejects_unknown_value(self) -> None:
        args = runner.build_parser().parse_args([])
        args.downstream_mode = "window"
        with self.assertRaisesRegex(ValueError, "unsupported downstream mode"):
            runner.resolve_downstream_mode(args)

    def test_write_input_rgb_timeline_follows_downstream_mode(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--downstream-mode",
                "demo_visualizer",
                "--visualizer-layout",
                "side-by-side",
            ]
        )
        self.assertTrue(runner.resolve_write_input_rgb_timeline(args))
        args = runner.build_parser().parse_args(["--downstream-mode", "phystwin_shen"])
        self.assertFalse(runner.resolve_write_input_rgb_timeline(args))
        self.assertEqual(runner.visualizer_start_policy(args), "disabled")

    def test_validate_runtime_args_checks_phystwin_repo(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--downstream-mode",
                "phystwin_shen",
                "--phystwin-shen-repo",
                "/nonexistent/phystwin",
            ]
        )
        with self.assertRaises(PhystwinShenLaunchError):
            runner.validate_runtime_args(args, chunk_frame_count=35)

    def test_validate_runtime_args_rejects_nonpositive_max_chunks(self) -> None:
        args = runner.build_parser().parse_args(
            ["--downstream-mode", "disabled", "--max-chunks", "0"]
        )
        with self.assertRaisesRegex(ValueError, "--max-chunks must be positive"):
            runner.validate_runtime_args(args, chunk_frame_count=35)

    def test_validate_runtime_args_requires_asap_for_phystwin(self) -> None:
        args = runner.build_parser().parse_args(
            ["--downstream-mode", "phystwin_shen", "--no-asap-augment"]
        )
        with self.assertRaisesRegex(ValueError, "requires --asap-augment"):
            runner.validate_runtime_args(args, chunk_frame_count=35)

    def test_disabled_dry_run_does_not_require_external_repo(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--downstream-mode",
                "disabled",
                "--phystwin-shen-repo",
                "/nonexistent/phystwin",
            ]
        )
        runner.validate_runtime_args(args, chunk_frame_count=35)
        contract = runner._contract(args)
        self.assertIsNone(contract["phystwin_shen_pipeline_command"])
        self.assertEqual(contract["phystwin_shen_viewer_urls"], {})


class LauncherCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo = root / "Phystwin_shen"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "configs").mkdir()
        (self.repo / "scripts" / "run_online_full_pipeline.py").write_text(
            "import time\ntime.sleep(60)\n",
            encoding="utf-8",
        )
        (self.repo / "configs" / "online_full_pipeline.yaml").write_text(
            "{}\n",
            encoding="utf-8",
        )
        self.base = root / "outputs_v6_1"
        self.base.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_repo(self) -> None:
        expected_config = (
            self.repo / "configs" / "online_full_pipeline.yaml"
        ).resolve()
        self.assertEqual(
            validate_phystwin_shen_repo(
                self.repo,
                Path("configs/online_full_pipeline.yaml"),
            ),
            (self.repo.resolve(), expected_config),
        )
        with self.assertRaisesRegex(PhystwinShenLaunchError, "not a file"):
            validate_phystwin_shen_repo(
                self.repo.parent,
                Path("configs/online_full_pipeline.yaml"),
            )

    def test_local_yaml_builds_one_explicit_full_pipeline_command(self) -> None:
        settings = _settings(self.repo, self.base)
        command = build_full_pipeline_command(
            settings,
            python_prefix=["python"],
        )

        def value(flag: str) -> str:
            return command[command.index(flag) + 1]

        self.assertEqual(command[:2], ["python", "scripts/run_online_full_pipeline.py"])
        self.assertEqual(
            value("--config"),
            str(self.repo / "configs" / "online_full_pipeline.yaml"),
        )
        self.assertEqual(value("--online_dir"), str(self.base / "online_data"))
        self.assertEqual(value("--common_batch_size"), "4")
        self.assertEqual(value("--common_segment_len"), "30")
        self.assertEqual(value("--stage1_enabled"), "true")
        self.assertEqual(value("--stage2_enabled"), "false")
        self.assertEqual(value("--train_stop_when_finished"), "false")
        self.assertEqual(value("--train_train_frame"), "none")
        self.assertEqual(value("--cma_viewer_port"), "8765")
        self.assertEqual(value("--train_viewer_port"), "8766")

    def test_stop_when_finished_is_controlled_by_local_config(self) -> None:
        runtime = copy.deepcopy(runner.DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG)
        runtime["train"]["stop_when_finished"] = True
        settings = _settings(self.repo, self.base, runtime_config=runtime)
        command = build_full_pipeline_command(
            settings,
            python_prefix=["python"],
        )
        index = command.index("--train_stop_when_finished")
        self.assertEqual(command[index + 1], "true")

    def test_duplicate_viewer_endpoint_fails_before_launch(self) -> None:
        runtime = copy.deepcopy(runner.DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG)
        runtime["train_viewer"]["port"] = runtime["cma_viewer"]["port"]
        settings = _settings(self.repo, self.base, runtime_config=runtime)
        with self.assertRaisesRegex(PhystwinShenLaunchError, "cannot share"):
            validate_phystwin_shen_settings(
                settings,
                python_prefix=[sys.executable],
            )

    def test_launch_kills_both_old_viewers_and_starts_one_supervisor(self) -> None:
        ports = (_free_port(), _free_port())
        listeners: list[subprocess.Popen[bytes]] = []
        for port in ports:
            listener = subprocess.Popen(
                [
                    sys.executable,
                    "-c",
                    (
                        "import socket, time; s = socket.socket(); "
                        "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                        f"s.bind(('127.0.0.1', {port})); s.listen(); "
                        "print('ready', flush=True); time.sleep(60)"
                    ),
                ],
                stdout=subprocess.PIPE,
            )
            self.assertIsNotNone(listener.stdout)
            self.assertEqual(listener.stdout.readline().strip(), b"ready")
            listeners.append(listener)

        runtime = copy.deepcopy(runner.DEFAULT_PHYSTWIN_SHEN_RUNTIME_CONFIG)
        runtime["cma_viewer"]["port"] = ports[0]
        runtime["train_viewer"]["port"] = ports[1]
        settings = _settings(self.repo, self.base, runtime_config=runtime)
        launch = None
        try:
            launch = launch_phystwin_shen(
                settings,
                python_prefix=[sys.executable],
                log_dir=self.base / "logs",
                trigger="test",
                wall_time_origin_s=time.monotonic(),
            )
            self.assertIsNone(launch.pipeline_process.poll())
            self.assertEqual(launch.process_group_id, launch.pipeline_process.pid)
            self.assertEqual(
                set(launch.port_takeover),
                {"cma_viewer", "train_viewer"},
            )
            self.assertEqual(
                {item["status"] for item in launch.port_takeover.values()},
                {"killed_occupant"},
            )
            self.assertTrue(all(item.poll() is not None for item in listeners))
        finally:
            if launch is not None:
                runner._stop_phystwin_launch(launch)
            for listener in listeners:
                if listener.poll() is None:
                    listener.kill()
                    listener.wait()
                if listener.stdout is not None:
                    listener.stdout.close()


class PhystwinLifecycleTests(unittest.TestCase):
    def _launch(self, process: mock.Mock) -> SimpleNamespace:
        return SimpleNamespace(
            pipeline_process=process,
            process_group_id=4321,
        )

    def test_normal_completion_waits_and_cleans_process_group(self) -> None:
        process = mock.Mock()
        process.wait.return_value = 0
        launch = self._launch(process)
        with mock.patch.object(runner, "_stop_phystwin_launch") as stop:
            self.assertEqual(runner._wait_for_phystwin_launch(launch), 0)
        process.wait.assert_called_once_with()
        stop.assert_called_once_with(launch)

    def test_nonzero_completion_cleans_group_and_fails(self) -> None:
        process = mock.Mock()
        process.wait.return_value = 7
        launch = self._launch(process)
        with mock.patch.object(runner, "_stop_phystwin_launch") as stop:
            with self.assertRaisesRegex(PhystwinShenLaunchError, "return code 7"):
                runner._wait_for_phystwin_launch(launch)
        stop.assert_called_once_with(launch)

    def test_interrupt_while_waiting_cleans_group(self) -> None:
        process = mock.Mock()
        process.wait.side_effect = KeyboardInterrupt
        launch = self._launch(process)
        with mock.patch.object(runner, "_stop_phystwin_launch") as stop:
            with self.assertRaises(KeyboardInterrupt):
                runner._wait_for_phystwin_launch(launch)
        stop.assert_called_once_with(launch)

    def test_stream_failure_kills_full_pipeline_group_and_camera(self) -> None:
        camera = mock.Mock(pid=1111)
        camera.poll.return_value = None
        pipeline_process = mock.Mock(pid=2222)
        pipeline_process.poll.return_value = None
        launch = SimpleNamespace(
            pipeline_process=pipeline_process,
            process_group_id=2222,
            settings=SimpleNamespace(viewer_urls={}),
        )

        def fail_stream(*args, **kwargs):
            del args
            kwargs["before_poll"]()
            raise RuntimeError("producer failed")

        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(runner, "validate_runtime_args"),
                mock.patch.object(
                    runner,
                    "build_main_data_processing_command",
                    return_value=["camera"],
                ),
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    return_value=camera,
                ),
                mock.patch.object(
                    runner,
                    "launch_phystwin_shen",
                    return_value=launch,
                ),
                mock.patch.object(
                    runner,
                    "stream_chunk_data_from_headless_capture",
                    side_effect=fail_stream,
                ),
                mock.patch.object(
                    runner,
                    "_stop_process",
                    return_value=-signal.SIGTERM,
                ) as stop,
            ):
                with self.assertRaisesRegex(RuntimeError, "producer failed"):
                    runner.main(
                        [
                            "--base-path",
                            tmp,
                            "--downstream-mode",
                            "phystwin_shen",
                            "--no-shape-prior-warmup",
                        ]
                    )

        stop.assert_any_call(pipeline_process, process_group_id=2222)
        stop.assert_any_call(camera)

    def test_camera_failure_during_stop_race_fails_demo(self) -> None:
        camera = mock.Mock(pid=1111)
        camera.poll.return_value = None
        with tempfile.TemporaryDirectory() as tmp:
            with (
                mock.patch.object(runner, "validate_runtime_args"),
                mock.patch.object(
                    runner,
                    "build_main_data_processing_command",
                    return_value=["camera"],
                ),
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    return_value=camera,
                ),
                mock.patch.object(
                    runner,
                    "stream_chunk_data_from_headless_capture",
                    return_value=[],
                ),
                mock.patch.object(
                    runner,
                    "_stop_process",
                    return_value=7,
                ),
            ):
                with self.assertRaisesRegex(
                    RuntimeError,
                    "failed while the Demo was stopping",
                ):
                    runner.main(
                        [
                            "--base-path",
                            tmp,
                            "--downstream-mode",
                            "disabled",
                        ]
                    )


class ProcessGroupCleanupTests(unittest.TestCase):
    def test_normal_group_cleanup_reaps_leader_without_timeout_delay(self) -> None:
        process = subprocess.Popen(
            [sys.executable, "-c", "import time; time.sleep(60)"],
            start_new_session=True,
        )
        started = time.monotonic()
        result = main_subprocess._stop_process(process)
        elapsed = time.monotonic() - started
        self.assertEqual(result, -signal.SIGTERM)
        self.assertLess(elapsed, 2.0)

    def test_kills_descendant_after_supervisor_leader_already_exited(self) -> None:
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import subprocess, sys; "
                    "child = subprocess.Popen([sys.executable, '-c', "
                    "'import time; time.sleep(60)']); "
                    "print(child.pid, flush=True)"
                ),
            ],
            stdout=subprocess.PIPE,
            start_new_session=True,
        )
        self.assertIsNotNone(process.stdout)
        self.assertGreater(int(process.stdout.readline().strip()), 0)
        process.wait(timeout=2.0)
        try:
            result = main_subprocess._stop_process(
                process,
                process_group_id=process.pid,
            )
            self.assertEqual(result, 0)
            self.assertFalse(main_subprocess._process_group_alive(process.pid))
        finally:
            if main_subprocess._process_group_alive(process.pid):
                os.killpg(process.pid, signal.SIGKILL)
            if process.stdout is not None:
                process.stdout.close()

    def test_kills_saved_group_after_supervisor_leader_exits(self) -> None:
        process = mock.Mock(pid=4321, returncode=0)
        process.poll.return_value = 0
        with (
            mock.patch.object(
                main_subprocess,
                "_process_group_alive",
                return_value=True,
            ),
            mock.patch.object(
                main_subprocess,
                "_wait_for_process_group_exit",
                return_value=True,
            ),
            mock.patch.object(main_subprocess.os, "killpg") as killpg,
        ):
            result = main_subprocess._stop_process(
                process,
                process_group_id=4321,
            )
        killpg.assert_called_once_with(4321, signal.SIGTERM)
        self.assertEqual(result, 0)

    def test_escalates_to_sigkill_when_group_ignores_sigterm(self) -> None:
        process = mock.Mock(pid=4321, returncode=0)
        process.poll.return_value = 0
        with (
            mock.patch.object(
                main_subprocess,
                "_process_group_alive",
                return_value=True,
            ),
            mock.patch.object(
                main_subprocess,
                "_wait_for_process_group_exit",
                side_effect=(False, True),
            ),
            mock.patch.object(main_subprocess.os, "killpg") as killpg,
        ):
            main_subprocess._stop_process(process, process_group_id=4321)
        self.assertEqual(
            killpg.call_args_list,
            [
                mock.call(4321, signal.SIGTERM),
                mock.call(4321, signal.SIGKILL),
            ],
        )


class EnsurePortFreeTests(unittest.TestCase):
    def test_free_port_is_noop(self) -> None:
        port = _free_port()
        result = ensure_port_free("127.0.0.1", port)
        self.assertEqual(result["status"], "free")

    def test_kills_current_listener_and_frees_port(self) -> None:
        port = _free_port()
        listener = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import socket, time; s = socket.socket(); "
                    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1); "
                    f"s.bind(('127.0.0.1', {port})); s.listen(); "
                    "print('ready', flush=True); time.sleep(60)"
                ),
            ],
            stdout=subprocess.PIPE,
        )
        try:
            self.assertEqual(listener.stdout.readline().strip(), b"ready")
            result = ensure_port_free("127.0.0.1", port)
            self.assertEqual(result["status"], "killed_occupant")
            self.assertEqual(
                [entry["pid"] for entry in result["killed_pids"]], [listener.pid]
            )
            self.assertIsNotNone(listener.poll())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        finally:
            if listener.poll() is None:
                listener.kill()
                listener.wait()
            if listener.stdout is not None:
                listener.stdout.close()

    def test_does_not_kill_listener_on_unrelated_bind_host(self) -> None:
        port = _free_port()
        listener = subprocess.Popen(
            [
                sys.executable,
                "-c",
                (
                    "import socket, sys, time\n"
                    "s = socket.socket()\n"
                    "s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)\n"
                    "try:\n"
                    f"    s.bind(('127.0.0.2', {port}))\n"
                    "    s.listen()\n"
                    "    print('ready', flush=True)\n"
                    "    time.sleep(60)\n"
                    "except OSError as error:\n"
                    "    print(f'bind-error:{error}', flush=True)\n"
                    "    sys.exit(2)\n"
                ),
            ],
            stdout=subprocess.PIPE,
        )
        try:
            self.assertIsNotNone(listener.stdout)
            line = listener.stdout.readline().strip()
            if line != b"ready":
                self.skipTest(line.decode("utf-8", errors="replace"))
            result = ensure_port_free("127.0.0.1", port)
            self.assertEqual(result["status"], "free")
            self.assertIsNone(listener.poll())
            with socket.socket() as sock:
                sock.bind(("127.0.0.1", port))
        finally:
            if listener.poll() is None:
                listener.kill()
                listener.wait()
            if listener.stdout is not None:
                listener.stdout.close()


class InitializeCaseTests(unittest.TestCase):
    def test_case_dir_seeded_before_first_chunk(self) -> None:
        metadata = {
            "k_color": [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]],
            "camera_to_world_c2w": None,
            "width": 8,
            "height": 6,
            "serial": "seed-serial",
        }
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = OnlineFrameArchive(base_path=base, case_name="demo_v6_2", fps=5)
            archive.initialize_case(metadata, serial_number="fallback")
            online_dir = base / "online_data"
            seeded = json.loads((online_dir / "metadata.json").read_text())
            self.assertEqual(seeded["frame_num"], 0)
            self.assertEqual(seeded["serial_numbers"], ["seed-serial"])
            with (online_dir / "calibrate.pkl").open("rb") as handle:
                c2ws = pickle.load(handle)
            np.testing.assert_array_equal(c2ws[0], np.eye(4))
            enhance = json.loads((online_dir / "enhance_metadata.json").read_text())
            self.assertEqual(enhance["frame_mapping"], [])
            # Idempotent: repeated seeding keeps the committed state.
            archive.initialize_case(metadata, serial_number="fallback")
            seeded = json.loads((online_dir / "metadata.json").read_text())
            self.assertEqual(seeded["frame_num"], 0)


class StreamingArchiveTests(unittest.TestCase):
    """color/depth stream in real time; frame_num stays commit-gated."""

    WIDTH = 8
    HEIGHT = 6

    def _metadata(self) -> dict:
        return {
            "k_color": [[100.0, 0.0, 4.0], [0.0, 100.0, 3.0], [0.0, 0.0, 1.0]],
            "camera_to_world_c2w": None,
            "width": self.WIDTH,
            "height": self.HEIGHT,
            "serial": "stream-serial",
        }

    def _frame(self, seq: int) -> PreparedPhysTwinFrame:
        return PreparedPhysTwinFrame(
            seq=seq,
            rgb_frame=np.full(
                (self.HEIGHT, self.WIDTH, 3), seq % 255, dtype=np.uint8
            ),
            processed_mask_frame={
                "object": np.ones((self.HEIGHT, self.WIDTH), dtype=bool),
                "controller": np.zeros((self.HEIGHT, self.WIDTH), dtype=bool),
            },
            pcd_points=np.zeros((1, self.HEIGHT, self.WIDTH, 3), dtype=np.float32),
            pcd_colors=np.zeros((1, self.HEIGHT, self.WIDTH, 3), dtype=np.uint8),
            tracks_yx=np.zeros((4, 2), dtype=np.float32),
            visibility=np.ones((4,), dtype=bool),
            query_points_yx=np.zeros((4, 2), dtype=np.float32),
            source_timestamp_s=100.0 + seq,
            source_frame_index=seq,
            depth_mm_u16=np.full(
                (self.HEIGHT, self.WIDTH), 1000 + seq, dtype=np.uint16
            ),
        )

    def _archive(self, base: Path) -> OnlineFrameArchive:
        archive = OnlineFrameArchive(base_path=base, case_name="online_data", fps=5)
        archive.initialize_case(self._metadata(), serial_number="fallback")
        return archive

    def test_stream_frame_writes_files_before_any_commit(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = self._archive(base)
            for seq in range(3):
                self.assertEqual(archive.stream_frame(self._frame(seq)), seq)
            online = base / "online_data"
            for index in range(3):
                self.assertTrue((online / "color" / "0" / f"{index}.png").is_file())
                self.assertTrue((online / "depth" / "0" / f"{index}.npy").is_file())
            # No chunk committed yet: frame_num must still read 0.
            self.assertEqual(
                json.loads((online / "metadata.json").read_text())["frame_num"], 0
            )

    def test_archive_chunk_verifies_streamed_frames_and_publish_advances(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = self._archive(base)
            frames = [self._frame(seq) for seq in range(2)]
            for frame in frames:
                archive.stream_frame(frame)
            archive.archive_chunk(
                chunk_id=0,
                metadata=self._metadata(),
                serial_number="fallback",
                frames=frames,
                source_frame_indices=[0, 1],
                source_timestamps_s=None,
                online_start_frame=0,
            )
            archive.publish_metadata()
            online = base / "online_data"
            self.assertEqual(
                json.loads((online / "metadata.json").read_text())["frame_num"], 2
            )
            mapping = json.loads(
                (online / "enhance_metadata.json").read_text()
            )["frame_mapping"]
            self.assertEqual([m["online_frame_index"] for m in mapping], [0, 1])

    def test_archive_chunk_rejects_streamed_identity_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            archive = self._archive(Path(tmp))
            archive.stream_frame(self._frame(5))
            with self.assertRaisesRegex(OnlineFrameArchiveError, "streamed file"):
                archive.archive_chunk(
                    chunk_id=0,
                    metadata=self._metadata(),
                    serial_number="fallback",
                    frames=[self._frame(6)],
                    source_frame_indices=[6],
                    source_timestamps_s=None,
                    online_start_frame=0,
                )

    def test_discard_streamed_tail_removes_only_uncommitted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = self._archive(base)
            frames = [self._frame(seq) for seq in range(3)]
            for frame in frames:
                archive.stream_frame(frame)
            archive.archive_chunk(
                chunk_id=0,
                metadata=self._metadata(),
                serial_number="fallback",
                frames=frames[:2],
                source_frame_indices=[0, 1],
                source_timestamps_s=None,
                online_start_frame=0,
            )
            archive.publish_metadata()
            self.assertEqual(archive.discard_streamed_tail(), 1)
            online = base / "online_data"
            self.assertTrue((online / "color" / "0" / "1.png").is_file())
            self.assertFalse((online / "color" / "0" / "2.png").exists())
            self.assertFalse((online / "depth" / "0" / "2.npy").exists())
            self.assertEqual(archive.frames_streamed, 2)
            self.assertEqual(
                json.loads((online / "metadata.json").read_text())["frame_num"], 2
            )

    def test_batch_path_without_streaming_still_writes_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            archive = self._archive(base)
            frames = [self._frame(seq) for seq in range(2)]
            archive.archive_chunk(
                chunk_id=0,
                metadata=self._metadata(),
                serial_number="fallback",
                frames=frames,
                source_frame_indices=[0, 1],
                source_timestamps_s=None,
                online_start_frame=0,
            )
            online = base / "online_data"
            for index in range(2):
                self.assertTrue((online / "color" / "0" / f"{index}.png").is_file())
                self.assertTrue((online / "depth" / "0" / f"{index}.npy").is_file())
            self.assertEqual(archive.frames_written, 2)
            self.assertEqual(archive.frames_streamed, 2)


if __name__ == "__main__":
    unittest.main()
