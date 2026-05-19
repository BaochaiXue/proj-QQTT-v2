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
        )

        restored = process_mod.CoTrackerProcessConfig.from_json(config.to_json())

        self.assertEqual(restored, config)

    def test_subprocess_env_isolates_cotracker_gpu(self) -> None:
        config = process_mod.CoTrackerProcessConfig(cotracker_gpu="1")
        env = process_mod.build_cotracker_process_env(config, base_env={"PATH": os.environ.get("PATH", "")})

        self.assertEqual(env["CUDA_VISIBLE_DEVICES"], "1")
        self.assertEqual(env["QQTT_DEMO31_COTRACKER_PROCESS"], "1")

    def test_subprocess_argv_targets_process_module(self) -> None:
        config = process_mod.CoTrackerProcessConfig(cotracker_gpu="1")
        argv = process_mod.build_cotracker_subprocess_argv(config, python_executable="python")

        self.assertEqual(argv[:3], ["python", "-m", "qqtt.demo.demo31_cotracker_process"])
        self.assertIn("--config-json", argv)

    def test_worker_loop_sets_cuda_visible_before_worker_import(self) -> None:
        source = inspect.getsource(process_mod.run_cotracker_worker_loop)

        env_idx = source.index("configure_cotracker_cuda_environment(config)")
        import_idx = source.index("from qqtt.demo.cotracker3_overlay_worker import")
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
        self.assertIn('"overlay_display_scope": "controller"', output)
        if old_value is None:
            os.environ.pop("CUDA_VISIBLE_DEVICES", None)
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = old_value


if __name__ == "__main__":
    unittest.main()
