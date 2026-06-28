from __future__ import annotations

from pathlib import Path
import unittest

import yaml


EXPECTED_CONFIG_SECTIONS = (
    "paths",
    "input",
    "chunking",
    "camera",
    "gpu",
    "shape_prior",
    "optimization",
    "point_viewer",
)
EXPECTED_DEFAULT_KEYS = (
    "data_process_base_path",
    "realtime_phystwin_root",
    "input_source",
    "replay_fps",
    "chunk_seconds",
    "chunk_poll_interval_s",
    "camera_source_replay_fps",
    "camera_fps",
    "camera_fps_choices",
    "camera_color_exposure",
    "camera_color_gain",
    "case_prefix",
    "depth_backend",
    "max_chunks",
    "shape_prior_endpoint",
    "mask_radius_outlier_radius_m",
    "mask_radius_outlier_nb_points",
    "realtime_gpu_mode",
    "warmup_gpu_mode",
    "gpu_mode",
    "gpu_mode_camera_cuda_visible_devices",
    "gpu_mode_shape_prior_device",
    "camera_device",
    "camera_tracker_device",
    "camera_dtype",
    "shape_prior_worker_mode",
    "shape_prior_worker_conda_env",
    "shape_prior_worker_device",
    "shape_prior_worker_startup_grace_s",
    "shape_prior_worker_max_observation_to_aligned_p95_m",
    "optimization_mode",
    "optimization_cuda_visible_devices",
    "optimization_device",
    "optimization_zero_iterations",
    "optimization_batch_size",
    "optimization_segment_stride",
    "optimization_poll_sec",
    "optimization_recent_window_count",
    "optimization_seed",
    "optimization_experiments_dir",
    "optimization_zero_experiments_dir",
    "optimization_start_grace_s",
    "point_viewer_mode",
    "point_viewer_conda_env",
    "point_viewer_cuda_visible_devices",
    "point_viewer_cam_idx",
    "point_viewer_poll_sec",
    "point_viewer_object_stride",
    "point_viewer_object_radius",
    "point_viewer_controller_radius",
    "point_viewer_object_color_mode",
    "point_viewer_layout_side_by_side",
    "point_viewer_layout_output_only",
    "point_viewer_layouts",
    "point_viewer_layout",
    "point_viewer_render_mode",
    "table_calibrate_path",
    "sam31_checkpoint_path",
    "sam31_checkpoint_env",
)


class DemoV51DefaultConfigTest(unittest.TestCase):
    def test_default_config_loader_uses_pyyaml_without_fallback_parser(self) -> None:
        from demo_v5_1 import realtime_data_process_sam3d as runner

        self.assertFalse(hasattr(runner, "_parse_default_config_scalar"))
        self.assertFalse(hasattr(runner, "_parse_default_yaml_subset"))

    def test_runtime_defaults_are_loaded_from_default_yaml(self) -> None:
        from demo_v5_1 import realtime_data_process_sam3d as runner

        self.assertEqual(Path("demo_v5_1/config/default.yaml"), runner.DEFAULT_CONFIG_PATH.relative_to(runner.REPO_ROOT))
        self.assertTrue(runner.DEFAULT_CONFIG_PATH.is_file())
        raw_defaults = yaml.safe_load(runner.DEFAULT_CONFIG_PATH.read_text(encoding="utf-8"))
        self.assertEqual(list(EXPECTED_CONFIG_SECTIONS), list(raw_defaults))
        for section in EXPECTED_CONFIG_SECTIONS:
            with self.subTest(section=section):
                self.assertIsInstance(raw_defaults[section], dict)
        self.assertNotIn("data_process_base_path", raw_defaults)

        defaults = runner.load_default_config()
        for key in EXPECTED_DEFAULT_KEYS:
            with self.subTest(key=key):
                self.assertIn(key, defaults)
        self.assertNotIn("camera_lossless_input_fps", defaults)
        self.assertNotIn("capture_extra_seconds", defaults)
        self.assertFalse(hasattr(runner, "DEFAULT_CAPTURE_EXTRA_SECONDS"))

        self.assertEqual(Path(defaults["data_process_base_path"]), runner.DEFAULT_DATA_PROCESS_BASE_PATH)
        self.assertEqual(Path(defaults["realtime_phystwin_root"]), runner.DEFAULT_REALTIME_PHYSTWIN_ROOT)
        self.assertEqual(defaults["input_source"], runner.DEFAULT_INPUT_SOURCE)
        self.assertEqual(float(defaults["replay_fps"]), runner.DEFAULT_REPLAY_FPS)
        self.assertEqual(float(defaults["chunk_seconds"]), runner.DEFAULT_CHUNK_SECONDS)
        self.assertEqual(float(defaults["chunk_poll_interval_s"]), runner.DEFAULT_CHUNK_POLL_INTERVAL_S)
        self.assertEqual(float(defaults["camera_source_replay_fps"]), runner.DEFAULT_CAMERA_SOURCE_REPLAY_FPS)
        self.assertEqual(int(defaults["camera_fps"]), runner.DEFAULT_CAMERA_FPS)
        self.assertEqual(tuple(defaults["camera_fps_choices"]), runner.CAMERA_FPS_CHOICES)
        self.assertEqual(float(defaults["camera_color_exposure"]), runner.DEFAULT_CAMERA_COLOR_EXPOSURE)
        self.assertEqual(float(defaults["camera_color_gain"]), runner.DEFAULT_CAMERA_COLOR_GAIN)
        self.assertEqual(defaults["case_prefix"], runner.DEFAULT_CASE_PREFIX)
        self.assertEqual(defaults["depth_backend"], runner.DEFAULT_DEPTH_BACKEND)
        self.assertIsNone(runner.DEFAULT_MAX_CHUNKS)
        self.assertEqual(defaults["shape_prior_endpoint"], runner.DEFAULT_SHAPE_PRIOR_ENDPOINT)
        self.assertEqual(float(defaults["mask_radius_outlier_radius_m"]), runner.DEFAULT_MASK_RADIUS_OUTLIER_RADIUS_M)
        self.assertEqual(int(defaults["mask_radius_outlier_nb_points"]), runner.DEFAULT_MASK_RADIUS_OUTLIER_NB_POINTS)
        self.assertEqual(defaults["realtime_gpu_mode"], runner.DEFAULT_REALTIME_GPU_MODE)
        self.assertEqual(defaults["warmup_gpu_mode"], runner.DEFAULT_WARMUP_GPU_MODE)
        self.assertEqual(defaults["gpu_mode"], runner.DEFAULT_GPU_MODE)
        self.assertEqual(defaults["gpu_mode_camera_cuda_visible_devices"], runner.GPU_MODE_CAMERA_CUDA_VISIBLE_DEVICES)
        self.assertEqual(defaults["gpu_mode_shape_prior_device"], runner.GPU_MODE_SHAPE_PRIOR_DEVICE)
        self.assertEqual(defaults["camera_device"], runner.DEFAULT_CAMERA_DEVICE)
        self.assertEqual(defaults["camera_tracker_device"], runner.DEFAULT_CAMERA_TRACKER_DEVICE)
        self.assertEqual(defaults["camera_dtype"], runner.DEFAULT_CAMERA_DTYPE)
        self.assertEqual(defaults["shape_prior_worker_mode"], runner.DEFAULT_SHAPE_PRIOR_WORKER_MODE)
        self.assertEqual(defaults["shape_prior_worker_conda_env"], runner.DEFAULT_SHAPE_PRIOR_WORKER_CONDA_ENV)
        self.assertEqual(defaults["shape_prior_worker_device"], runner.DEFAULT_SHAPE_PRIOR_WORKER_DEVICE)
        self.assertEqual(float(defaults["shape_prior_worker_startup_grace_s"]), runner.DEFAULT_SHAPE_PRIOR_WORKER_STARTUP_GRACE_S)
        self.assertEqual(float(defaults["shape_prior_worker_max_observation_to_aligned_p95_m"]), runner.DEFAULT_SHAPE_PRIOR_WORKER_MAX_OBSERVATION_TO_ALIGNED_P95_M)
        self.assertEqual(defaults["optimization_mode"], runner.DEFAULT_OPTIMIZATION_MODE)
        self.assertEqual(defaults["optimization_cuda_visible_devices"], runner.DEFAULT_OPTIMIZATION_CUDA_VISIBLE_DEVICES)
        self.assertEqual(defaults["optimization_device"], runner.DEFAULT_OPTIMIZATION_DEVICE)
        self.assertEqual(int(defaults["optimization_zero_iterations"]), runner.DEFAULT_OPTIMIZATION_ZERO_ITERATIONS)
        self.assertEqual(int(defaults["optimization_batch_size"]), runner.DEFAULT_OPTIMIZATION_BATCH_SIZE)
        self.assertEqual(int(defaults["optimization_segment_stride"]), runner.DEFAULT_OPTIMIZATION_SEGMENT_STRIDE)
        self.assertEqual(float(defaults["optimization_poll_sec"]), runner.DEFAULT_OPTIMIZATION_POLL_SEC)
        self.assertEqual(int(defaults["optimization_recent_window_count"]), runner.DEFAULT_OPTIMIZATION_RECENT_WINDOW_COUNT)
        self.assertEqual(int(defaults["optimization_seed"]), runner.DEFAULT_OPTIMIZATION_SEED)
        self.assertEqual(defaults["optimization_experiments_dir"], runner.DEFAULT_OPTIMIZATION_EXPERIMENTS_DIR)
        self.assertEqual(defaults["optimization_zero_experiments_dir"], runner.DEFAULT_OPTIMIZATION_ZERO_EXPERIMENTS_DIR)
        self.assertEqual(float(defaults["optimization_start_grace_s"]), runner.DEFAULT_OPTIMIZATION_START_GRACE_S)
        self.assertEqual(defaults["point_viewer_mode"], runner.DEFAULT_POINT_VIEWER_MODE)
        self.assertEqual(defaults["point_viewer_conda_env"], runner.DEFAULT_POINT_VIEWER_CONDA_ENV)
        self.assertEqual(defaults["point_viewer_cuda_visible_devices"], runner.DEFAULT_POINT_VIEWER_CUDA_VISIBLE_DEVICES)
        self.assertEqual(int(defaults["point_viewer_cam_idx"]), runner.DEFAULT_POINT_VIEWER_CAM_IDX)
        self.assertEqual(float(defaults["point_viewer_poll_sec"]), runner.DEFAULT_POINT_VIEWER_POLL_SEC)
        self.assertEqual(int(defaults["point_viewer_object_stride"]), runner.DEFAULT_POINT_VIEWER_OBJECT_STRIDE)
        self.assertEqual(int(defaults["point_viewer_object_radius"]), runner.DEFAULT_POINT_VIEWER_OBJECT_RADIUS)
        self.assertEqual(int(defaults["point_viewer_controller_radius"]), runner.DEFAULT_POINT_VIEWER_CONTROLLER_RADIUS)
        self.assertEqual(defaults["point_viewer_object_color_mode"], runner.DEFAULT_POINT_VIEWER_OBJECT_COLOR_MODE)
        self.assertEqual(defaults["point_viewer_layout_side_by_side"], runner.POINT_VIEWER_LAYOUT_SIDE_BY_SIDE)
        self.assertEqual(defaults["point_viewer_layout_output_only"], runner.POINT_VIEWER_LAYOUT_OUTPUT_ONLY)
        self.assertEqual(tuple(defaults["point_viewer_layouts"]), runner.POINT_VIEWER_LAYOUTS)
        self.assertEqual(defaults["point_viewer_layout"], runner.DEFAULT_POINT_VIEWER_LAYOUT)
        self.assertEqual(defaults["point_viewer_render_mode"], runner.DEFAULT_POINT_VIEWER_RENDER_MODE)
        self.assertEqual(Path(defaults["table_calibrate_path"]), runner.DEFAULT_TABLE_CALIBRATE_PATH)
        self.assertEqual(Path(defaults["sam31_checkpoint_path"]), runner.DEFAULT_SAM31_CHECKPOINT_PATH)
        self.assertEqual(defaults["sam31_checkpoint_env"], runner.SAM31_CHECKPOINT_ENV)


if __name__ == "__main__":
    unittest.main()
