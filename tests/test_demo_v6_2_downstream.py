"""Tests for the Demo v6.1 downstream.mode enum and Phystwin_shen launcher."""

from __future__ import annotations

import json
import pickle
import socket
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path

import numpy as np

from demo_v6_2 import main as runner
from demo_v6_2.online_frame_archive import OnlineFrameArchive
from demo_v6_2.phystwin_shen_launch import (
    PhystwinShenLaunchError,
    PhystwinShenSettings,
    build_train_command,
    build_viewer_command,
    ensure_port_free,
    validate_phystwin_shen_repo,
)


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _settings(repo: Path, base: Path, **overrides) -> PhystwinShenSettings:
    values = dict(
        repo_path=repo,
        conda_env="demo_2_max",
        case_name="online_data",
        base_path=base,
        cuda_visible_devices="1",
        viewer_host="127.0.0.1",
        viewer_port=8765,
        viewer_cam_idx=0,
        viewer_point_mode="surface",
        viewer_point_stride=5,
        train_device="cuda:0",
        train_batch_size=1,
        train_segment_len=32,
        train_segment_stride=30,
        train_poll_sec=1.0,
        train_recent_window_count=8,
        train_realtime_vis_every=1,
        train_stop_when_finished=False,
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
        self.assertEqual(section["case_name"], "online_data")
        self.assertEqual(section["host"], "127.0.0.1")
        self.assertEqual(int(section["port"]), 8765)
        self.assertEqual(section["device"], "cuda:0")
        self.assertEqual(int(section["batch_size"]), 5)
        obsolete_keys = {
            "viewer_host",
            "viewer_port",
            "viewer_cam_idx",
            "viewer_point_mode",
            "viewer_point_stride",
            "viewer_image_index_mode",
            "train_device",
            "train_batch_size",
            "train_segment_len",
            "train_segment_stride",
            "train_poll_sec",
            "train_recent_window_count",
            "train_realtime_vis_every",
            "train_stop_when_finished",
        }
        self.assertTrue(obsolete_keys.isdisjoint(section))
        self.assertEqual(str(config["gpu"]["phystwin_shen_cuda_visible_devices"]), "1")

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


class LauncherCommandTests(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        root = Path(self._tmp.name)
        self.repo = root / "Phystwin_shen"
        (self.repo / "scripts").mkdir(parents=True)
        (self.repo / "train_online_warp.py").write_text("# stub\n")
        (self.repo / "scripts" / "html_realtime_viewer.py").write_text("# stub\n")
        self.base = root / "outputs_v6_1"
        self.base.mkdir()

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def test_validate_repo(self) -> None:
        self.assertEqual(validate_phystwin_shen_repo(self.repo), self.repo)
        with self.assertRaisesRegex(PhystwinShenLaunchError, "missing"):
            validate_phystwin_shen_repo(self.repo.parent)

    def test_yaml_defaults_flow_to_phystwin_commands(self) -> None:
        args = runner.build_parser().parse_args(
            [
                "--base-path",
                str(self.base),
                "--phystwin-shen-repo",
                str(self.repo),
            ]
        )
        settings = runner.resolve_phystwin_shen_settings(args)
        viewer = build_viewer_command(settings, python_prefix=["python"])
        train = build_train_command(settings, python_prefix=["python"])

        self.assertEqual(viewer[viewer.index("--host") + 1], "127.0.0.1")
        self.assertEqual(viewer[viewer.index("--port") + 1], "8765")
        self.assertEqual(viewer[viewer.index("--cam_idx") + 1], "0")
        self.assertEqual(viewer[viewer.index("--point_mode") + 1], "surface")
        self.assertEqual(viewer[viewer.index("--point_stride") + 1], "5")
        self.assertEqual(train[train.index("--device") + 1], "cuda:0")
        self.assertEqual(train[train.index("--batch_size") + 1], "5")
        self.assertEqual(train[train.index("--segment_len") + 1], "32")
        self.assertEqual(train[train.index("--segment_stride") + 1], "30")
        self.assertEqual(train[train.index("--poll_sec") + 1], "1.0")
        self.assertEqual(train[train.index("--recent_window_count") + 1], "5")
        self.assertIn("--stop_when_finished", train)

    def test_commands_mirror_manual_launch_script(self) -> None:
        settings = _settings(self.repo, self.base)
        realtime = self.repo / "experiments_online" / "online_data" / "realtime"
        viewer = build_viewer_command(settings, python_prefix=["python"])
        self.assertEqual(
            viewer,
            [
                "python",
                "scripts/html_realtime_viewer.py",
                "--base_path",
                str(self.base),
                "--case_name",
                "online_data",
                "--realtime_dir",
                str(realtime),
                "--rgb_dir",
                str(self.base / "online_data" / "color"),
                "--host",
                "127.0.0.1",
                "--port",
                "8765",
                "--cam_idx",
                "0",
                "--point_mode",
                "surface",
                "--point_stride",
                "5",
            ],
        )
        train = build_train_command(settings, python_prefix=["python"])
        self.assertEqual(
            train,
            [
                "python",
                "train_online_warp.py",
                "--online_dir",
                str(self.base / "online_data"),
                "--experiments_dir",
                str(self.repo / "experiments_online"),
                "--device",
                "cuda:0",
                "--batch_size",
                "1",
                "--segment_len",
                "32",
                "--segment_stride",
                "30",
                "--poll_sec",
                "1.0",
                "--recent_window_count",
                "8",
                "--realtime_vis",
                "--realtime_vis_dir",
                str(realtime),
                "--realtime_vis_every",
                "1",
            ],
        )
        stopping = _settings(self.repo, self.base, train_stop_when_finished=True)
        self.assertIn(
            "--stop_when_finished",
            build_train_command(stopping, python_prefix=["python"]),
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


if __name__ == "__main__":
    unittest.main()
