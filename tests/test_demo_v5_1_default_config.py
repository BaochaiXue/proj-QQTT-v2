from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest import mock

import yaml


EXPECTED_CONFIG_SECTIONS = (
    "paths",
    "input",
    "chunking",
    "camera",
    "gpu",
    "shape_prior",
    "visualizer",
)


class DemoV51DefaultConfigTest(unittest.TestCase):
    def test_default_config_loader_uses_pyyaml_without_fallback_parser(self) -> None:
        from demo_v5_1 import main as runner

        self.assertFalse(hasattr(runner, "_parse_default_config_scalar"))
        self.assertFalse(hasattr(runner, "_parse_default_yaml_subset"))

    def test_camera_env_uses_yaml_sam31_checkpoint_without_file_probe(self) -> None:
        from demo_v5_1 import main as runner

        class CompletedProcess:
            returncode = 0
            pid = None

            def poll(self) -> int:
                return 0

        popen_calls: list[dict[str, object]] = []

        def fake_popen(*args: object, **kwargs: object) -> CompletedProcess:
            popen_calls.append({"args": args, "kwargs": kwargs})
            return CompletedProcess()

        missing_yaml_checkpoint = Path("configured/missing_sam31.pt")
        expected_checkpoint = str(runner.REPO_ROOT / missing_yaml_checkpoint)
        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(
                    runner,
                    "DEFAULT_SAM31_CHECKPOINT_PATH",
                    missing_yaml_checkpoint,
                ),
                mock.patch.object(runner.subprocess, "Popen", fake_popen),
                mock.patch.object(
                    runner,
                    "stream_chunk_data_from_headless_capture",
                    return_value=[],
                ),
                mock.patch.dict(runner.os.environ, {}, clear=True),
            ):
                exit_code = runner.main(
                    [
                        "--base-path",
                        tmpdir,
                        "--case-prefix",
                        "cfg_sam31_env",
                        "--max-chunks",
                        "0",
                        "--no-shape-prior-warmup",
                        "--visualizer-mode",
                        "disabled",
                    ]
                )

        self.assertEqual(0, exit_code)
        self.assertEqual(1, len(popen_calls))
        camera_env = popen_calls[0]["kwargs"]["env"]
        self.assertIsInstance(camera_env, dict)
        self.assertEqual(expected_checkpoint, camera_env[runner.SAM31_CHECKPOINT_ENV])

    def test_default_config_access_uses_single_cfg_helper(self) -> None:
        from demo_v5_1 import main as runner

        self.assertTrue(hasattr(runner, "_cfg"))
        self.assertFalse(hasattr(runner, "_flatten_default_config"))
        self.assertFalse(hasattr(runner, "_apply_default_sam31_checkpoint_env"))
        for name in (
            "_default_value",
            "_default_str",
            "_default_path",
            "_default_int",
            "_default_optional_int",
            "_default_float",
            "_default_int_tuple",
            "_default_str_tuple",
            "_default_str_mapping",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(runner, name))

    def test_legacy_gpu_mode_cli_is_not_accepted(self) -> None:
        from demo_v5_1 import main as runner

        parser = runner.build_parser()
        removed_options = (
            "--gpu-mode",
            "--realtime-gpu-mode",
            "--warmup-gpu-mode",
            "--main-realtime-data-process-cuda-visible-devices",
            "--allow-degraded-online",
            "--" + "-".join(("camera", "cuda", "visible", "devices")),
            "--" + "-".join(("shape", "prior", "cuda", "visible", "devices")),
            "--" + "-".join(("point", "viewer", "mode")),
            "--" + "-".join(("point", "viewer", "layout")),
            "--" + "-".join(("point", "viewer", "cuda", "visible", "devices")),
            "--" + "-".join(("optimization", "mode")),
            "--" + "-".join(("realtime", "phystwin", "root")),
            "--" + "-".join(("future" + "phystwin", "base", "path")),
        )
        for option in removed_options:
            with self.subTest(option=option):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    parser.parse_args([option, "single"])
                self.assertEqual(2, error.exception.code)
        parsed = parser.parse_args([])
        self.assertFalse(hasattr(parsed, "gpu_mode"))
        self.assertFalse(hasattr(parsed, "realtime_gpu_mode"))
        self.assertFalse(hasattr(parsed, "warmup_gpu_mode"))
        self.assertFalse(hasattr(parsed, "allow_degraded_online"))
        self.assertFalse(
            hasattr(parsed, "_".join(("camera", "cuda", "visible", "devices")))
        )
        self.assertFalse(
            hasattr(parsed, "_".join(("shape", "prior", "cuda", "visible", "devices")))
        )
        self.assertEqual(
            runner.DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
            parsed.main_data_processing_cuda_visible_devices,
        )
        self.assertEqual(
            runner.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
            parsed.shape_prior_warmup_cuda_visible_devices,
        )
        self.assertNotIn("allow_degraded_online", runner._contract(parsed))

    def test_track_process_status_does_not_change_main_exit(self) -> None:
        from demo_v5_1 import main as runner

        class CompletedProcess:
            returncode = 0
            pid = None

            def poll(self) -> int:
                return 0

        invalid_manifest = {
            "chunk_name": "demo_v5_1_online_chunk_0000",
            "track_process_status": "invalid",
            "online_publish_skipped": False,
            "publish_wall_s": 1.0,
            "backlog_chunks": 0,
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            with (
                contextlib.redirect_stdout(io.StringIO()),
                mock.patch.object(
                    runner.subprocess,
                    "Popen",
                    return_value=CompletedProcess(),
                ),
                mock.patch.object(
                    runner,
                    "stream_chunk_data_from_headless_capture",
                    return_value=[invalid_manifest],
                ),
                mock.patch.dict(runner.os.environ, {}, clear=True),
            ):
                exit_code = runner.main(
                    [
                        "--base-path",
                        tmpdir,
                        "--case-prefix",
                        "status_warning_only",
                        "--max-chunks",
                        "1",
                        "--no-shape-prior-warmup",
                        "--visualizer-mode",
                        "disabled",
                    ]
                )

            summary_path = Path(tmpdir) / "run_summary.json"
            summary = json.loads(summary_path.read_text(encoding="utf-8"))

        self.assertEqual(0, exit_code)
        self.assertEqual("invalid", summary["track_process_status"])
        self.assertEqual(1, summary["track_process_status_counts"]["invalid"])
        self.assertEqual(0, summary["online_publish_skipped_chunk_count"])
        self.assertEqual("max_chunks_reached", summary["main_data_processing_stop_reason"])

    def test_legacy_camera_runtime_names_are_not_accepted(self) -> None:
        from demo_v5_1 import main as runner

        parser = runner.build_parser()
        legacy_flags = (
            "--" + "-".join(("camera", "device")),
            "--" + "-".join(("camera", "tracker", "device")),
            "--" + "-".join(("camera", "dtype")),
        )
        for option in legacy_flags:
            with self.subTest(option=option):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    parser.parse_args([option, "cuda"])
                self.assertEqual(2, error.exception.code)

        parsed = parser.parse_args(
            [
                "--perception-device",
                "cuda:0",
                "--tracker-device",
                "cuda:1",
                "--inference-dtype",
                "float32",
            ]
        )
        self.assertEqual("cuda:0", parsed.perception_device)
        self.assertEqual("cuda:1", parsed.tracker_device)
        self.assertEqual("float32", parsed.inference_dtype)
        self.assertFalse(hasattr(parsed, "_".join(("camera", "device"))))
        self.assertFalse(hasattr(parsed, "_".join(("camera", "tracker", "device"))))
        self.assertFalse(hasattr(parsed, "_".join(("camera", "dtype"))))

    def test_shape_prior_worker_cli_is_not_accepted(self) -> None:
        from demo_v5_1 import main as runner

        parser = runner.build_parser()
        removed_flags = (
            ("--shape-prior-worker-mode", "managed"),
            ("--shape-prior-worker-conda-env", "demo_2_max"),
            ("--shape-prior-worker-cuda-visible-devices", "1"),
            ("--shape-prior-worker-device", "cuda:0"),
            ("--shape-prior-worker-startup-grace-s", "0"),
            ("--shape-prior-worker-sam3d-root", "sam-3d-objects"),
            ("--shape-prior-worker-config", "pipeline.yaml"),
            ("--shape-prior-worker-preload-models",),
            ("--shape-prior-worker-max-observation-to-aligned-p95-m", "0.06"),
        )
        for args in removed_flags:
            with self.subTest(args=args):
                with (
                    contextlib.redirect_stderr(io.StringIO()),
                    self.assertRaises(SystemExit) as error,
                ):
                    parser.parse_args(list(args))
                self.assertEqual(2, error.exception.code)

        parsed = parser.parse_args([])
        for name in (
            "shape_prior_worker_mode",
            "shape_prior_worker_conda_env",
            "shape_prior_worker_cuda_visible_devices",
            "shape_prior_worker_device",
            "shape_prior_worker_startup_grace_s",
            "shape_prior_worker_sam3d_root",
            "shape_prior_worker_config",
            "shape_prior_worker_preload_models",
            "shape_prior_worker_max_observation_to_aligned_p95_m",
        ):
            with self.subTest(name=name):
                self.assertFalse(hasattr(parsed, name))

    def test_shape_prior_local_warmup_command_lives_under_main_process(self) -> None:
        from demo_v5_1 import main as runner

        with tempfile.TemporaryDirectory() as tmpdir:
            parsed = runner.build_parser().parse_args(
                [
                    "--base-path",
                    tmpdir,
                    "--case-prefix",
                    "local_shape_prior",
                ]
            )
            capture_dir = Path(tmpdir) / "capture"
            command = runner.build_main_data_processing_command(
                parsed,
                capture_dir=capture_dir,
                profile_json=capture_dir / "shape_prior_profile.json",
                chunk_frame_count=1,
            )

        self.assertIn("--shape-prior-warmup-cuda-visible-devices", command)
        self.assertIn(
            runner.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
            command,
        )
        self.assertIn("--shape-prior-controller-name", command)
        self.assertIn(runner.CONFIG_SHAPE_PRIOR_CONTROLLER_NAME, command)
        self.assertIn("--shape-prior-case-root", command)
        self.assertIn(str(Path(tmpdir) / "shape_prior_case"), command)
        self.assertIn("--shape-prior-points-npz", command)
        self.assertIn(str(Path(tmpdir) / "shape_prior" / "points.npz"), command)
        self.assertNotIn("shape_prior_worker.py", " ".join(command))
        self.assertNotIn("--shape-prior-endpoint", command)
        self.assertNotIn("--shape-prior-device", command)

    def test_dry_run_contract_preserves_warmup_and_viewer_defaults(self) -> None:
        from demo_v5_1 import main as runner

        with tempfile.TemporaryDirectory() as tmpdir:
            stdout = io.StringIO()
            with contextlib.redirect_stdout(stdout):
                exit_code = runner.main(
                    [
                        "--dry-run",
                        "--base-path",
                        tmpdir,
                        "--case-prefix",
                        "dry_contract",
                    ]
                )

        contract = json.loads(stdout.getvalue())
        self.assertEqual(0, exit_code)
        self.assertEqual(
            str(Path(tmpdir) / "capture"),
            contract["main_data_processing_capture_dir"],
        )
        self.assertEqual(str(Path(tmpdir) / "online_data"), contract["online_dir"])
        self.assertEqual(
            str(Path(tmpdir) / "data" / "final_data.pkl"),
            contract["static_data_path"],
        )
        self.assertEqual(
            str(Path(tmpdir) / "shape_prior_case"),
            contract["shape_prior_case_root"],
        )
        self.assertEqual(
            str(Path(tmpdir) / "shape_prior" / "points.npz"),
            contract["shape_prior_points_npz"],
        )
        self.assertTrue(contract["shape_prior_warmup"])
        self.assertEqual(
            runner.DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
            contract["main_data_processing_cuda_visible_devices"],
        )
        self.assertEqual(
            runner.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
            contract["shape_prior_warmup_cuda_visible_devices"],
        )
        self.assertNotIn("_".join(("camera", "cuda", "visible", "devices")), contract)
        self.assertNotIn(
            "_".join(("shape", "prior", "cuda", "visible", "devices")),
            contract,
        )
        self.assertNotIn("shape_prior_worker_command", contract)
        self.assertNotIn("shape_prior_worker_mode", contract)
        self.assertNotIn("_".join(("optimization", "mode")), contract)
        self.assertNotIn("_".join(("optimization", "scope")), contract)
        self.assertNotIn("_".join(("realtime", "phystwin", "base", "path")), contract)
        self.assertIn(
            str(Path("demo_v5_1") / "visualize_track.py"),
            contract["visualizer_command"],
        )
        self.assertEqual(
            runner.DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
            contract["visualizer_cuda_visible_devices"],
        )

    def test_runtime_defaults_are_loaded_from_default_yaml(self) -> None:
        from demo_v5_1 import main as runner

        self.assertEqual(
            Path("demo_v5_1/config/default.yaml"),
            runner.DEFAULT_CONFIG_PATH.relative_to(runner.REPO_ROOT),
        )
        self.assertTrue(runner.DEFAULT_CONFIG_PATH.is_file())
        raw_defaults = yaml.safe_load(
            runner.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8")
        )
        self.assertEqual(list(EXPECTED_CONFIG_SECTIONS), list(raw_defaults))

        defaults = runner.load_default_config()
        self.assertEqual(raw_defaults, defaults)
        for section in EXPECTED_CONFIG_SECTIONS:
            with self.subTest(section=section):
                self.assertIsInstance(defaults[section], dict)
        self.assertNotIn("data_process_base_path", defaults)
        self.assertEqual(
            defaults["paths"]["data_process_base_path"],
            runner._cfg("paths", "data_process_base_path"),
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_layout"],
            runner._cfg("visualizer", "visualizer_layout"),
        )
        self.assertNotIn("camera_lossless_input_fps", defaults["camera"])
        self.assertFalse(hasattr(runner, "DEFAULT_CAPTURE_EXTRA_SECONDS"))

        self.assertEqual(
            Path(defaults["paths"]["data_process_base_path"]),
            runner.DEFAULT_DATA_PROCESS_BASE_PATH,
        )
        self.assertNotIn("_".join(("realtime", "phystwin", "root")), defaults["paths"])
        self.assertFalse(
            hasattr(runner, "_".join(("DEFAULT", "REALTIME", "PHYSTWIN", "ROOT")))
        )
        self.assertEqual(defaults["input"]["input_source"], runner.DEFAULT_INPUT_SOURCE)
        self.assertEqual(
            float(defaults["input"]["replay_fps"]), runner.DEFAULT_REPLAY_FPS
        )
        self.assertEqual(
            float(defaults["chunking"]["chunk_seconds"]),
            runner.DEFAULT_CHUNK_SECONDS,
        )
        self.assertEqual(
            float(defaults["chunking"]["chunk_poll_interval_s"]),
            runner.DEFAULT_CHUNK_POLL_INTERVAL_S,
        )
        self.assertEqual(
            float(defaults["input"]["camera_source_replay_fps"]),
            runner.DEFAULT_CAMERA_SOURCE_REPLAY_FPS,
        )
        self.assertEqual(
            int(defaults["camera"]["camera_fps"]), runner.DEFAULT_CAMERA_FPS
        )
        self.assertEqual(
            tuple(defaults["camera"]["camera_fps_choices"]),
            runner.CAMERA_FPS_CHOICES,
        )
        self.assertEqual(
            float(defaults["camera"]["camera_color_exposure"]),
            runner.DEFAULT_CAMERA_COLOR_EXPOSURE,
        )
        self.assertEqual(
            float(defaults["camera"]["camera_color_gain"]),
            runner.DEFAULT_CAMERA_COLOR_GAIN,
        )
        self.assertEqual(defaults["camera"]["case_prefix"], runner.DEFAULT_CASE_PREFIX)
        self.assertEqual(
            defaults["camera"]["depth_backend"], runner.DEFAULT_DEPTH_BACKEND
        )
        self.assertIsNone(runner.DEFAULT_MAX_CHUNKS)
        self.assertEqual(
            int(defaults["shape_prior"]["shape_prior_timeout_ms"]),
            runner.DEFAULT_SHAPE_PRIOR_TIMEOUT_MS,
        )
        self.assertEqual(
            float(defaults["shape_prior"]["shape_prior_chunk_wait_timeout_s"]),
            runner.DEFAULT_SHAPE_PRIOR_CHUNK_WAIT_TIMEOUT_S,
        )
        self.assertEqual(
            defaults["shape_prior"]["shape_prior_controller_name"],
            runner.CONFIG_SHAPE_PRIOR_CONTROLLER_NAME,
        )
        self.assertEqual(
            float(defaults["camera"]["mask_radius_outlier_radius_m"]),
            runner.DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M,
        )
        self.assertEqual(
            int(defaults["camera"]["mask_radius_outlier_nb_points"]),
            runner.DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS,
        )
        self.assertNotIn("gpu_mode", defaults["gpu"])
        self.assertNotIn("realtime_gpu_mode", defaults["gpu"])
        self.assertNotIn("warmup_gpu_mode", defaults["gpu"])
        self.assertNotIn(
            "_".join(("camera", "cuda", "visible", "devices")),
            defaults["gpu"],
        )
        self.assertNotIn(
            "_".join(("shape", "prior", "cuda", "visible", "devices")),
            defaults["gpu"],
        )
        self.assertFalse(hasattr(runner, "DEFAULT_GPU_MODE"))
        self.assertFalse(
            hasattr(
                runner,
                "_".join(("DEFAULT", "CAMERA", "CUDA", "VISIBLE", "DEVICES")),
            )
        )
        self.assertFalse(
            hasattr(
                runner,
                "_".join(("DEFAULT", "SHAPE", "PRIOR", "CUDA", "VISIBLE", "DEVICES")),
            )
        )
        self.assertEqual(
            defaults["gpu"]["main_data_processing_cuda_visible_devices"],
            runner.DEFAULT_MAIN_DATA_PROCESSING_CUDA_VISIBLE_DEVICES,
        )
        self.assertEqual(
            defaults["gpu"]["shape_prior_warmup_cuda_visible_devices"],
            runner.DEFAULT_SHAPE_PRIOR_WARMUP_CUDA_VISIBLE_DEVICES,
        )
        self.assertEqual(
            defaults["gpu"]["visualizer_cuda_visible_devices"],
            runner.DEFAULT_VISUALIZER_CUDA_VISIBLE_DEVICES,
        )
        for legacy_key in (
            "_".join(("camera", "device")),
            "_".join(("camera", "tracker", "device")),
            "_".join(("camera", "dtype")),
        ):
            with self.subTest(legacy_key=legacy_key):
                self.assertNotIn(legacy_key, defaults["camera"])
        for legacy_name in (
            "_".join(("DEFAULT", "CAMERA", "DEVICE")),
            "_".join(("DEFAULT", "CAMERA", "TRACKER", "DEVICE")),
            "_".join(("DEFAULT", "CAMERA", "DTYPE")),
        ):
            with self.subTest(legacy_name=legacy_name):
                self.assertFalse(hasattr(runner, legacy_name))

        self.assertEqual(
            defaults["camera"]["perception_device"],
            runner.DEFAULT_PERCEPTION_DEVICE,
        )
        self.assertEqual(
            defaults["camera"]["tracker_device"], runner.DEFAULT_TRACKER_DEVICE
        )
        self.assertEqual(
            defaults["camera"]["inference_dtype"],
            runner.DEFAULT_INFERENCE_DTYPE,
        )
        for removed_key in (
            "shape_prior_endpoint",
            "shape_prior_worker_mode",
            "shape_prior_worker_conda_env",
            "shape_prior_worker_device",
            "shape_prior_worker_startup_grace_s",
            "shape_prior_worker_max_observation_to_aligned_p95_m",
        ):
            with self.subTest(removed_key=removed_key):
                self.assertNotIn(removed_key, defaults["shape_prior"])
        for removed_name in (
            "DEFAULT_SHAPE_PRIOR_ENDPOINT",
            "DEFAULT_SHAPE_PRIOR_WORKER_MODE",
            "DEFAULT_SHAPE_PRIOR_WORKER_CONDA_ENV",
            "DEFAULT_SHAPE_PRIOR_WORKER_DEVICE",
            "DEFAULT_SHAPE_PRIOR_WORKER_STARTUP_GRACE_S",
            "DEFAULT_SHAPE_PRIOR_WORKER_MAX_OBSERVATION_TO_ALIGNED_P95_M",
        ):
            with self.subTest(removed_name=removed_name):
                self.assertFalse(hasattr(runner, removed_name))
        self.assertNotIn("optimization", defaults)
        for removed_name in (
            "_".join(("DEFAULT", "OPTIMIZATION", "MODE")),
            "_".join(("DEFAULT", "OPTIMIZATION", "CUDA", "VISIBLE", "DEVICES")),
            "_".join(("DEFAULT", "OPTIMIZATION", "DEVICE")),
            "_".join(("DEFAULT", "OPTIMIZATION", "ZERO", "ITERATIONS")),
            "_".join(("DEFAULT", "OPTIMIZATION", "BATCH", "SIZE")),
            "_".join(("DEFAULT", "OPTIMIZATION", "SEGMENT", "STRIDE")),
            "_".join(("DEFAULT", "OPTIMIZATION", "POLL", "SEC")),
            "_".join(("DEFAULT", "OPTIMIZATION", "RECENT", "WINDOW", "COUNT")),
            "_".join(("DEFAULT", "OPTIMIZATION", "SEED")),
            "_".join(("DEFAULT", "OPTIMIZATION", "EXPERIMENTS", "DIR")),
            "_".join(("DEFAULT", "OPTIMIZATION", "ZERO", "EXPERIMENTS", "DIR")),
            "_".join(("DEFAULT", "OPTIMIZATION", "START", "GRACE", "S")),
        ):
            with self.subTest(removed_name=removed_name):
                self.assertFalse(hasattr(runner, removed_name))
        self.assertEqual(
            defaults["visualizer"]["visualizer_mode"],
            runner.DEFAULT_VISUALIZER_MODE,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_conda_env"],
            runner.DEFAULT_VISUALIZER_CONDA_ENV,
        )
        self.assertNotIn("visualizer_cuda_visible_devices", defaults["visualizer"])
        self.assertEqual(
            int(defaults["visualizer"]["visualizer_cam_idx"]),
            runner.DEFAULT_VISUALIZER_CAM_IDX,
        )
        self.assertEqual(
            float(defaults["visualizer"]["visualizer_poll_sec"]),
            runner.DEFAULT_VISUALIZER_POLL_SEC,
        )
        self.assertEqual(
            int(defaults["visualizer"]["visualizer_object_stride"]),
            runner.DEFAULT_VISUALIZER_OBJECT_STRIDE,
        )
        self.assertEqual(
            int(defaults["visualizer"]["visualizer_object_radius"]),
            runner.DEFAULT_VISUALIZER_OBJECT_RADIUS,
        )
        self.assertEqual(
            int(defaults["visualizer"]["visualizer_controller_radius"]),
            runner.DEFAULT_VISUALIZER_CONTROLLER_RADIUS,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_object_color_mode"],
            runner.DEFAULT_VISUALIZER_OBJECT_COLOR_MODE,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_layout_side_by_side"],
            runner.VISUALIZER_LAYOUT_SIDE_BY_SIDE,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_layout_output_only"],
            runner.VISUALIZER_LAYOUT_OUTPUT_ONLY,
        )
        self.assertEqual(
            tuple(defaults["visualizer"]["visualizer_layouts"]),
            runner.VISUALIZER_LAYOUTS,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_layout"],
            runner.DEFAULT_VISUALIZER_LAYOUT,
        )
        self.assertEqual(
            defaults["visualizer"]["visualizer_render_mode"],
            runner.DEFAULT_VISUALIZER_RENDER_MODE,
        )
        self.assertEqual(
            Path(defaults["paths"]["table_calibrate_path"]),
            runner.DEFAULT_TABLE_CALIBRATE_PATH,
        )
        self.assertEqual(
            Path(defaults["paths"]["sam31_checkpoint_path"]),
            runner.DEFAULT_SAM31_CHECKPOINT_PATH,
        )
        self.assertEqual(
            defaults["paths"]["sam31_checkpoint_env"], runner.SAM31_CHECKPOINT_ENV
        )


if __name__ == "__main__":
    unittest.main()
