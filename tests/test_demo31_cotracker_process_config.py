from __future__ import annotations

import inspect
import os
import contextlib
import io
import unittest

from qqtt.demo import demo31_cotracker_process as process_mod


class Demo31CoTrackerProcessConfigTest(unittest.TestCase):
    def test_config_json_roundtrip(self) -> None:
        config = process_mod.CoTrackerProcessConfig(
            camera_ids=(0, 1, 2),
            cotracker_gpu="1",
            query_mode="phystwin_dense",
            query_count_request="auto",
            seed=42,
            sampling_device="cuda",
            init_requires_object_and_controller=True,
            overlay_max_points_per_camera=15,
            overlay_display_scope="controller",
            backend_execution_mode="batch-views",
            update_mode="batch",
            trackon2_checkpoint="/tmp/trackon2.pth",
            trackon2_config="/tmp/trackon2.yaml",
            trackon2_repo_dir="/tmp/track_on",
            litetracker_weights="/tmp/litetracker.pth",
            litetracker_repo_dir="/tmp/lite-tracker",
            litetracker_runtime="onnx-cuda",
            litetracker_onnx_dir="/tmp/litetracker_onnx",
            litetracker_export_onnx=True,
            litetracker_onnx_opset=17,
            litetracker_onnx_optimization_level=5,
            locotrack_repo_dir="/tmp/locotrack/locotrack_pytorch",
            locotrack_checkpoint="/tmp/locotrack_small.ckpt",
            locotrack_model_size="small",
            locotrack_window_frames=12,
            locotrack_resolution=(320, 256),
            locotrack_query_chunk_size=128,
            locotrack_autocast_dtype="fp16",
            tapnet_repo_dir="/tmp/tapnet",
            tapnextpp_checkpoint="/tmp/tapnextpp_ckpt.pt",
            tapnextpp_image_size=(256, 256),
            tapnextpp_autocast_dtype="fp16",
            tapnextpp_use_certainty=True,
            tapnextpp_certainty_radius=6,
            tapnextpp_certainty_threshold=0.4,
            tapnextpp_compile=True,
            tapnextpp_reset_on_reinitialize=False,
            tracker_batch_query_count_policy="min-common",
        )

        restored = process_mod.CoTrackerProcessConfig.from_json(config.to_json())

        self.assertEqual(restored, config)

    def test_subprocess_env_isolates_cotracker_gpu(self) -> None:
        config = process_mod.CoTrackerProcessConfig(cotracker_gpu="1")
        env = process_mod.build_cotracker_process_env(config, base_env={"PATH": os.environ.get("PATH", "")})

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["QQTT_DEMO31_COTRACKER_PROCESS"], "1")
        self.assertEqual(env["QQTT_DEMO31_POINT_TRACKER_PROCESS"], "1")
        self.assertEqual(env["PYTORCH_CUDA_ALLOC_CONF"], "expandable_segments:True")

    def test_subprocess_argv_targets_process_module(self) -> None:
        config = process_mod.CoTrackerProcessConfig(cotracker_gpu="1")
        argv = process_mod.build_cotracker_subprocess_argv(config, python_executable="python")

        self.assertEqual(argv[:3], ["python", "-m", "qqtt.demo.demo31_cotracker_process"])
        self.assertIn("--config-json", argv)

    def test_worker_loop_sets_cuda_visible_before_worker_import(self) -> None:
        source = inspect.getsource(process_mod.run_cotracker_worker_loop)

        env_idx = source.index("configure_cotracker_cuda_environment(config)")
        import_idx = source.index("from qqtt.demo.point_tracker_overlay_worker import")
        self.assertLess(env_idx, import_idx)
        module_source = inspect.getsource(process_mod)
        self.assertNotIn("\nimport torch", module_source)

    def test_print_contract_uses_configured_gpu_without_torch_import(self) -> None:
        config = process_mod.CoTrackerProcessConfig(cotracker_gpu="1")
        old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = process_mod.main(["--config-json", config.to_json(), "--print-contract"])

        self.assertEqual(exit_code, 0)
        self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")
        output = stdout.getvalue()
        self.assertIn("cpu_numpy_latest_wins", output)
        self.assertIn("phystwin_dense", output)
        self.assertIn('"tracking_query_count_requested": "auto"', output)
        self.assertIn('"point_tracker_process": true', output)
        self.assertIn('"tracker_backend": "cotracker3_online"', output)
        self.assertIn('"backend_execution_mode": "batch-views"', output)
        self.assertIn('"overlay_display_scope": "controller"', output)
        self.assertIn('"update_mode": "batch"', output)
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value

    def test_litetracker_config_uses_lazy_query_init_semantics(self) -> None:
        config = process_mod.CoTrackerProcessConfig(
            cotracker_backend="litetracker",
            backend_execution_mode="serial",
            prewarm_backends=False,
        )

        self.assertEqual(config.tracker_family, "litetracker")
        self.assertEqual(config.tracker_prewarm_mode, "lazy_query_init")
        self.assertTrue(config.tracker_query_dependent_init)

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = process_mod.main(["--config-json", config.to_json(), "--print-contract"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn('"tracker_backend": "litetracker"', output)
        self.assertIn('"litetracker_runtime": "pytorch"', output)
        self.assertIn('"tracker_prewarm_mode": "lazy_query_init"', output)
        self.assertIn('"tracker_query_dependent_init": true', output)
        self.assertIn('"ready_state": "ready_to_receive_inputs"', output)

    def test_litetracker_onnx_config_prints_runtime_fields(self) -> None:
        config = process_mod.CoTrackerProcessConfig(
            cotracker_backend="litetracker",
            backend_execution_mode="serial",
            litetracker_runtime="onnx-cuda",
            litetracker_onnx_dir="/tmp/litetracker_onnx",
            litetracker_export_onnx=True,
            litetracker_onnx_opset=17,
            litetracker_onnx_optimization_level=5,
        )

        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = process_mod.main(["--config-json", config.to_json(), "--print-contract"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn('"tracker_backend": "litetracker"', output)
        self.assertIn('"backend_execution_mode": "serial"', output)
        self.assertIn('"litetracker_runtime": "onnx-cuda"', output)
        self.assertIn('"litetracker_onnx_dir": "/tmp/litetracker_onnx"', output)
        self.assertIn('"litetracker_export_onnx": true', output)
        self.assertIn('"litetracker_onnx_opset": 17', output)
        self.assertIn('"litetracker_onnx_optimization_level": 5', output)

    def test_locotrack_config_prints_windowed_contract_fields(self) -> None:
        config = process_mod.CoTrackerProcessConfig(
            cotracker_backend="locotrack",
            backend_execution_mode="batch-views",
            locotrack_repo_dir="/tmp/locotrack/locotrack_pytorch",
            locotrack_checkpoint="/tmp/locotrack_small.ckpt",
            locotrack_model_size="small",
            locotrack_window_frames=8,
            locotrack_resolution=(256, 256),
            locotrack_query_chunk_size=256,
            locotrack_autocast_dtype="bf16",
        )

        self.assertEqual(config.tracker_family, "locotrack")
        self.assertEqual(config.tracker_prewarm_mode, "model_load_only")
        self.assertFalse(config.tracker_query_dependent_init)

        old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = process_mod.main(["--config-json", config.to_json(), "--print-contract"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn('"tracker_backend": "locotrack"', output)
        self.assertIn('"tracker_family": "locotrack"', output)
        self.assertIn('"backend_execution_mode": "batch-views"', output)
        self.assertIn('"locotrack_model_size": "small"', output)
        self.assertIn('"locotrack_window_frames": 8', output)
        self.assertIn('"locotrack_resolution": [', output)
        self.assertIn('"locotrack_query_chunk_size": 256', output)
        self.assertIn('"locotrack_autocast_dtype": "bf16"', output)
        self.assertIn('"tracker_prewarm_mode": "model_load_only"', output)
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value

    def test_tapnextpp_config_prints_stateful_online_contract_fields(self) -> None:
        config = process_mod.CoTrackerProcessConfig(
            cotracker_backend="tapnextpp",
            backend_execution_mode="batch-views",
            tapnet_repo_dir="/tmp/tapnet",
            tapnextpp_checkpoint="/tmp/tapnextpp_ckpt.pt",
            tapnextpp_image_size=(256, 256),
            tapnextpp_autocast_dtype="fp16",
            tapnextpp_use_certainty=False,
            tapnextpp_compile=False,
        )

        self.assertEqual(config.tracker_family, "tapnext")
        self.assertEqual(config.tracker_prewarm_mode, "model_load_only")
        self.assertFalse(config.tracker_query_dependent_init)

        old_value = os.environ.get("CUDA_VISIBLE_DEVICES")
        stdout = io.StringIO()
        with contextlib.redirect_stdout(stdout):
            exit_code = process_mod.main(["--config-json", config.to_json(), "--print-contract"])

        self.assertEqual(exit_code, 0)
        output = stdout.getvalue()
        self.assertIn('"tracker_backend": "tapnextpp"', output)
        self.assertIn('"tracker_family": "tapnext"', output)
        self.assertIn('"backend_execution_mode": "batch-views"', output)
        self.assertIn('"tapnet_repo_dir": "/tmp/tapnet"', output)
        self.assertIn('"tapnextpp_checkpoint": "/tmp/tapnextpp_ckpt.pt"', output)
        self.assertIn('"tapnextpp_image_size": [', output)
        self.assertIn('"tapnextpp_autocast_dtype": "fp16"', output)
        self.assertIn('"tapnextpp_use_certainty": false', output)
        self.assertIn('"tapnextpp_compile": false', output)
        self.assertIn('"tracker_prewarm_mode": "model_load_only"', output)
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value


if __name__ == "__main__":
    unittest.main()
