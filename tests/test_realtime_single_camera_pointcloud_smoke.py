from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from data_process.depth_backends import DEFAULT_FFS_REPO, DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR
from data_process.depth_backends import fast_foundation_stereo as ffs_backend
from data_process.depth_backends.geometry import align_depth_to_color
from scripts.harness import realtime_single_camera_pointcloud as demo


ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class DummyPacket:
    seq: int


class FakeCudaTensor:
    _next_ptr = 1000

    def __init__(self, shape, dtype="float32", contiguous=True):
        self.shape = tuple(shape)
        self.dtype = dtype
        self._contiguous = bool(contiguous)
        self._ptr = FakeCudaTensor._next_ptr
        FakeCudaTensor._next_ptr += 1000

    def to(self, dtype):
        return FakeCudaTensor(self.shape, dtype=dtype, contiguous=True)

    def is_contiguous(self):
        return self._contiguous

    def contiguous(self):
        return FakeCudaTensor(self.shape, dtype=self.dtype, contiguous=True)

    def data_ptr(self):
        return self._ptr


class FakeTorchStream:
    cuda_stream = 7


class FakeTorchCuda:
    @staticmethod
    def current_stream():
        return FakeTorchStream()


class FakeTorch:
    float32 = "float32"
    cuda = FakeTorchCuda()

    @staticmethod
    def empty(shape, *, device=None, dtype=None, pin_memory=False):
        return FakeCudaTensor(shape, dtype=dtype)


class FakeTrt:
    class TensorIOMode:
        OUTPUT = "output"


class FakeTensorRtEngine:
    def get_tensor_dtype(self, name):
        return "float32"


class FakeTensorRtContext:
    def __init__(self):
        self.shape_calls = []
        self.address_calls = []
        self.execute_calls = 0

    def set_input_shape(self, name, shape):
        self.shape_calls.append((name, tuple(shape)))

    def get_tensor_shape(self, name):
        return (1, 1, 2, 3)

    def set_tensor_address(self, name, address):
        self.address_calls.append((name, int(address)))

    def execute_async_v3(self, stream):
        self.execute_calls += 1
        return stream == FakeTorchStream.cuda_stream


class FakeTensorRtRunner:
    def trt_dtype_to_torch(self, dtype):
        return dtype

    def get_io_tensor_names(self, engine, mode):
        self.assert_mode = mode
        return ["disp"]


class RealtimeSingleCameraPointCloudSmokeTest(unittest.TestCase):
    def test_help_exposes_supported_capture_rates_and_profiles(self) -> None:
        result = subprocess.run(
            [sys.executable, "scripts/harness/realtime_single_camera_pointcloud.py", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--fps {5,15,30,60}", result.stdout)
        self.assertIn("--profile {848x480,640x480}", result.stdout)
        self.assertIn("--depth-source {realsense,ffs}", result.stdout)
        self.assertIn("--ffs-repo FFS_REPO", result.stdout)
        self.assertIn("--ffs-trt-model-dir FFS_TRT_MODEL_DIR", result.stdout)
        self.assertIn("--view-mode {camera,orbit}", result.stdout)
        self.assertIn("--render-backend {auto,image,pointcloud}", result.stdout)
        self.assertIn("--backproject-backend {auto,numpy,numba}", result.stdout)
        self.assertIn("--image-splat-px IMAGE_SPLAT_PX", result.stdout)
        self.assertIn("--debug", result.stdout)
        self.assertIn("orbit=200000", result.stdout)
        self.assertIn("orbit=1.0", result.stdout)
        self.assertIn(demo.COORDINATE_FRAME, result.stdout)
        self.assertIn("Use <=0 to disable", result.stdout)
        self.assertIn("far clipping", result.stdout)

    def test_profile_parsing_and_argparse_rejection(self) -> None:
        self.assertEqual(demo.parse_profile("848x480"), (848, 480))
        self.assertEqual(demo.parse_profile("640x480"), (640, 480))
        args = demo.build_parser().parse_args(["--fps", "60"])
        self.assertEqual(args.fps, 60)
        self.assertEqual(args.depth_source, "realsense")
        self.assertEqual(Path(args.ffs_repo), DEFAULT_FFS_REPO)
        self.assertEqual(Path(args.ffs_trt_model_dir), DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR)
        backend_args = demo.build_parser().parse_args(["--backproject-backend", "numpy"])
        self.assertEqual(backend_args.backproject_backend, "numpy")
        with self.assertRaises(ValueError):
            demo.parse_profile("320x240")
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                demo.build_parser().parse_args(["--profile", "320x240"])

    def test_ffs_mode_rejects_missing_tensorrt_artifacts_before_camera_start(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ffs_repo = root / "Fast-FoundationStereo"
            model_dir = root / "engines"
            ffs_repo.mkdir()
            model_dir.mkdir()
            args = demo.build_parser().parse_args(
                [
                    "--depth-source",
                    "ffs",
                    "--ffs-repo",
                    str(ffs_repo),
                    "--ffs-trt-model-dir",
                    str(model_dir),
                ]
            )
            with self.assertRaisesRegex(ValueError, "feature_runner.engine"):
                demo.validate_args(args)

    def test_cached_tensorrt_run_reuses_outputs_and_returns_fresh_dict(self) -> None:
        cached_run = ffs_backend._CachedTensorRTRun(
            torch_module=FakeTorch(),
            trt_module=FakeTrt,
            trt_runner=FakeTensorRtRunner(),
        )
        engine = FakeTensorRtEngine()
        context = FakeTensorRtContext()
        image = FakeCudaTensor((1, 3, 2, 3), dtype="float32")

        first = cached_run.run_trt(engine, context, {"left": image})
        second = cached_run.run_trt(engine, context, {"left": image})
        del first["disp"]
        third = cached_run.run_trt(engine, context, {"left": image})

        self.assertIs(second["disp"], third["disp"])
        self.assertEqual(context.shape_calls, [("left", (1, 3, 2, 3))])
        self.assertEqual(context.execute_calls, 3)
        self.assertIn("disp", third)
        self.assertEqual(context.address_calls, [("left", image.data_ptr()), ("disp", second["disp"].data_ptr())])

    def test_default_depth_max_disables_far_clipping(self) -> None:
        args = demo.build_parser().parse_args([])
        self.assertEqual(args.fps, 60)
        self.assertEqual(args.depth_max_m, 0.0)
        self.assertEqual(args.view_mode, "camera")
        self.assertEqual(demo.resolve_render_backend(args), "image")
        demo.validate_args(args)
        color_bgr = np.array([[[10, 20, 30], [40, 50, 60]]], dtype=np.uint8)
        depth_u16 = np.array([[1000, 6000]], dtype=np.uint16)
        points, _ = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=args.depth_max_m,
            stride=1,
            max_points=0,
        )
        self.assertEqual(points.shape[0], 2)
        np.testing.assert_allclose(points[:, 2], np.array([1.0, 6.0], dtype=np.float32))

    def test_auto_backend_uses_pointcloud_outside_camera_view(self) -> None:
        args = demo.build_parser().parse_args(["--view-mode", "orbit"])
        demo.validate_args(args)
        self.assertEqual(demo.resolve_render_backend(args), "pointcloud")

    def test_backproject_backend_resolution(self) -> None:
        self.assertEqual(
            demo.resolve_backproject_backend("numpy", stride=1, projection_grid_available=True),
            "numpy",
        )
        self.assertEqual(
            demo.resolve_backproject_backend("auto", stride=2, projection_grid_available=True),
            "numpy",
        )
        expected_auto = "numba" if demo.numba_backprojection_available() else "numpy"
        self.assertEqual(
            demo.resolve_backproject_backend("auto", stride=1, projection_grid_available=True),
            expected_auto,
        )
        if demo.numba_backprojection_available():
            self.assertEqual(
                demo.resolve_backproject_backend("numba", stride=1, projection_grid_available=True),
                "numba",
            )
            with self.assertRaises(ValueError):
                demo.resolve_backproject_backend("numba", stride=2, projection_grid_available=True)

    def test_orbit_view_defaults_to_200k_points_unless_explicit(self) -> None:
        camera_args = demo.build_parser().parse_args([])
        orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit"])
        uncapped_orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit", "--max-points", "0"])

        demo.validate_args(camera_args)
        demo.validate_args(orbit_args)
        demo.validate_args(uncapped_orbit_args)

        self.assertEqual(demo.resolve_max_points(camera_args), 0)
        self.assertEqual(demo.resolve_max_points(orbit_args), 200000)
        self.assertEqual(demo.resolve_max_points(uncapped_orbit_args), 0)

        demo.apply_view_defaults(orbit_args)
        self.assertEqual(orbit_args.max_points, 200000)
        self.assertEqual(orbit_args.point_size, 1.0)

    def test_point_size_defaults_are_view_specific_unless_explicit(self) -> None:
        camera_args = demo.build_parser().parse_args([])
        orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit"])
        explicit_orbit_args = demo.build_parser().parse_args(["--view-mode", "orbit", "--point-size", "2"])

        demo.validate_args(camera_args)
        demo.validate_args(orbit_args)
        demo.validate_args(explicit_orbit_args)

        self.assertEqual(demo.resolve_point_size(camera_args), 2.0)
        self.assertEqual(demo.resolve_point_size(orbit_args), 1.0)
        self.assertEqual(demo.resolve_point_size(explicit_orbit_args), 2.0)

    def test_pointcloud_update_readds_only_when_capacity_is_too_small(self) -> None:
        self.assertTrue(
            demo.pointcloud_update_requires_readd(geometry_added=False, current_capacity=0, point_count=10)
        )
        self.assertFalse(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=200000,
                point_count=187949,
            )
        )
        self.assertFalse(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=200000,
                point_count=200000,
            )
        )
        self.assertTrue(
            demo.pointcloud_update_requires_readd(
                geometry_added=True,
                current_capacity=187949,
                point_count=200000,
            )
        )

    def test_image_backend_is_rejected_outside_camera_view(self) -> None:
        args = demo.build_parser().parse_args(["--view-mode", "orbit", "--render-backend", "image"])
        with self.assertRaises(ValueError):
            demo.validate_args(args)

    def test_image_backend_preserves_valid_depth_pixels_in_camera_view(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=2.5,
            splat_px=0,
        )
        self.assertEqual(valid_count, 2)
        np.testing.assert_array_equal(
            image_rgb,
            np.array(
                [
                    [[3, 2, 1], [0, 0, 0]],
                    [[9, 8, 7], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_float_depth_image_backend_preserves_valid_depth_pixels(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_m = np.array([[1.0, 0.0], [np.nan, 2.0]], dtype=np.float32)
        image_rgb, valid_count = demo.build_camera_view_image_from_depth_m(
            color_bgr=color_bgr,
            depth_m=depth_m,
            depth_min_m=0.1,
            depth_max_m=1.5,
            splat_px=0,
        )
        self.assertEqual(valid_count, 1)
        np.testing.assert_array_equal(
            image_rgb,
            np.array(
                [
                    [[3, 2, 1], [0, 0, 0]],
                    [[0, 0, 0], [0, 0, 0]],
                ],
                dtype=np.uint8,
            ),
        )

    def test_float_depth_image_backend_opencv_path_matches_numpy_fallback(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ],
            dtype=np.uint8,
        )
        depth_m = np.array([[0.0, 0.1, np.nan], [np.inf, 1.0, 2.0]], dtype=np.float32)
        default_image, default_count = demo.build_camera_view_image_from_depth_m(
            color_bgr=color_bgr,
            depth_m=depth_m,
            depth_min_m=0.0,
            depth_max_m=1.0,
            splat_px=0,
        )
        original_cv2 = demo.cv2
        try:
            demo.cv2 = None
            fallback_image, fallback_count = demo.build_camera_view_image_from_depth_m(
                color_bgr=color_bgr,
                depth_m=depth_m,
                depth_min_m=0.0,
                depth_max_m=1.0,
                splat_px=0,
            )
        finally:
            demo.cv2 = original_cv2
        self.assertEqual(default_count, fallback_count)
        self.assertEqual(default_count, 2)
        np.testing.assert_array_equal(default_image, fallback_image)

    def test_image_backend_depth_bounds_preserve_raw_threshold_edges(self) -> None:
        self.assertEqual(
            demo._depth_bounds_to_u16(
                depth_scale_m_per_unit=0.001,
                depth_min_m=0.1,
                depth_max_m=0.102,
            ),
            (100, 101),
        )
        color_bgr = np.array(
            [[[10, 20, 30], [11, 21, 31], [12, 22, 32], [13, 23, 33], [14, 24, 34]]],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[99, 100, 101, 102, 103]], dtype=np.uint16)
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=0.102,
            splat_px=0,
        )
        self.assertEqual(valid_count, 2)
        np.testing.assert_array_equal(
            image_rgb,
            np.array([[[0, 0, 0], [31, 21, 11], [32, 22, 12], [0, 0, 0], [0, 0, 0]]], dtype=np.uint8),
        )

    def test_image_backend_numpy_fallback_matches_default_path(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                [[10, 11, 12], [13, 14, 15], [16, 17, 18]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[0, 100, 600], [999, 1000, 2000]], dtype=np.uint16)
        default_image, default_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=1.0,
            splat_px=0,
        )
        original_cv2 = demo.cv2
        try:
            demo.cv2 = None
            fallback_image, fallback_count = demo.build_camera_view_image(
                color_bgr=color_bgr,
                depth_u16=depth_u16,
                depth_scale_m_per_unit=0.001,
                depth_min_m=0.1,
                depth_max_m=1.0,
                splat_px=0,
            )
        finally:
            demo.cv2 = original_cv2
        self.assertEqual(default_count, fallback_count)
        np.testing.assert_array_equal(default_image, fallback_image)

    def test_image_backend_splat_keeps_original_valid_count(self) -> None:
        color_bgr = np.zeros((3, 3, 3), dtype=np.uint8)
        color_bgr[1, 1] = np.array([10, 20, 30], dtype=np.uint8)
        depth_u16 = np.zeros((3, 3), dtype=np.uint16)
        depth_u16[1, 1] = 1000
        image_rgb, valid_count = demo.build_camera_view_image(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=0.0,
            splat_px=1,
        )
        self.assertEqual(valid_count, 1)
        self.assertEqual(int(np.count_nonzero(np.any(image_rgb != 0, axis=2))), 9)
        np.testing.assert_array_equal(image_rgb[0, 0], np.array([30, 20, 10], dtype=np.uint8))

    def test_pointcloud_upload_helpers_keep_float32_and_reuse_color_buffer(self) -> None:
        points = np.arange(12, dtype=np.float32).reshape(4, 3)
        same_points = demo.ensure_float32_c_contiguous(points)
        self.assertIs(same_points, points)

        sliced = np.arange(24, dtype=np.float64).reshape(8, 3)[::2]
        converted = demo.ensure_float32_c_contiguous(sliced)
        self.assertEqual(converted.dtype, np.float32)
        self.assertTrue(converted.flags["C_CONTIGUOUS"])

        color_buffer = demo.ColorFloat32Buffer()
        colors = np.array([[0, 127, 255], [255, 0, 64]], dtype=np.uint8)
        colors_float = color_buffer.convert(colors)
        self.assertEqual(colors_float.dtype, np.float32)
        np.testing.assert_allclose(
            colors_float,
            np.array([[0.0, 127.0 / 255.0, 1.0], [1.0, 0.0, 64.0 / 255.0]], dtype=np.float32),
        )
        self.assertIs(color_buffer.convert(np.zeros_like(colors)), colors_float)

    def test_synthetic_backprojection_returns_expected_xyz_and_rgb(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        points, colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            pixel_grid=demo.build_pixel_grid(width=2, height=2, stride=1),
        )
        np.testing.assert_allclose(
            points,
            np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(colors, np.array([[3, 2, 1], [9, 8, 7], [12, 11, 10]], dtype=np.uint8))

    def test_float_depth_backprojection_returns_expected_xyz_and_rgb(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_m = np.array([[1.0, 0.0], [2.0, 3.0]], dtype=np.float32)
        points, colors = demo.backproject_aligned_rgbd_depth_m(
            color_bgr=color_bgr,
            depth_m=depth_m,
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            pixel_grid=demo.build_pixel_grid(width=2, height=2, stride=1),
        )
        np.testing.assert_allclose(
            points,
            np.array([[0.0, 0.0, 1.0], [0.0, 2.0, 2.0], [3.0, 3.0, 3.0]], dtype=np.float32),
        )
        np.testing.assert_array_equal(colors, np.array([[3, 2, 1], [9, 8, 7], [12, 11, 10]], dtype=np.uint8))

    def test_fast_ir_to_color_alignment_matches_reference(self) -> None:
        depth_ir = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        K = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        expected = align_depth_to_color(depth_ir, K, T, K, output_shape=(2, 2))
        actual = demo.align_ir_depth_to_color_fast(depth_ir, K, T, K, output_shape=(2, 2))
        np.testing.assert_allclose(actual, expected)

    def test_fast_ir_to_color_alignment_accepts_precomputed_ray_grid(self) -> None:
        depth_ir = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        K = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        ray_grid = demo.build_projection_grid_from_matrix(width=2, height=2, K=K)
        without_grid = demo.align_ir_depth_to_color_fast(depth_ir, K, T, K, output_shape=(2, 2))
        with_grid = demo.align_ir_depth_to_color_fast(
            depth_ir,
            K,
            T,
            K,
            output_shape=(2, 2),
            ir_projection_grid=ray_grid,
        )
        np.testing.assert_allclose(with_grid, without_grid)

    def test_ffs_ir_to_color_aligner_matches_reference_and_reuses_output(self) -> None:
        depth_ir = np.array([[1.0, 2.0], [0.0, 3.0]], dtype=np.float32)
        K = np.array([[2.0, 0.0, 0.5], [0.0, 2.0, 0.5], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.eye(4, dtype=np.float32)
        aligner = demo.FfsIrToColorAligner(
            k_ir_left=K,
            t_ir_left_to_color=T,
            k_color=K,
            ir_shape=depth_ir.shape,
            color_shape=(2, 2),
        )
        expected = align_depth_to_color(depth_ir, K, T, K, output_shape=(2, 2))
        actual = aligner.align(depth_ir)
        np.testing.assert_allclose(actual, expected)
        self.assertIs(actual, aligner.output)
        self.assertIs(aligner.align(depth_ir), actual)

    def test_numba_ffs_ir_to_color_aligner_matches_numpy_when_available(self) -> None:
        if not demo.numba_ffs_align_available():
            self.skipTest("numba is not installed")
        depth_ir = np.array(
            [
                [2.0, 1.0, np.nan],
                [np.inf, 0.5, -1.0],
                [1.5, 3.0, 4.0],
            ],
            dtype=np.float32,
        )
        K_ir = np.array([[3.0, 0.0, 1.0], [0.0, 2.5, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
        T = np.array(
            [
                [1.0, 0.0, 0.0, 0.02],
                [0.0, 1.0, 0.0, -0.01],
                [0.0, 0.0, 1.0, 0.03],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )
        K_color = np.array([[2.2, 0.0, 1.1], [0.0, 2.0, 0.8], [0.0, 0.0, 1.0]], dtype=np.float32)
        numba_aligner = demo.FfsIrToColorAligner(
            k_ir_left=K_ir,
            t_ir_left_to_color=T,
            k_color=K_color,
            ir_shape=depth_ir.shape,
            color_shape=(3, 4),
        )
        self.assertEqual(numba_aligner.align_backend, "numba")
        numba_output = numba_aligner.align(depth_ir, invalid_value=-1.0).copy()

        original_numba_align = demo._align_ir_to_color_numba
        try:
            demo._align_ir_to_color_numba = None  # type: ignore[assignment]
            numpy_aligner = demo.FfsIrToColorAligner(
                k_ir_left=K_ir,
                t_ir_left_to_color=T,
                k_color=K_color,
                ir_shape=depth_ir.shape,
                color_shape=(3, 4),
            )
            self.assertEqual(numpy_aligner.align_backend, "numpy")
            numpy_output = numpy_aligner.align(depth_ir, invalid_value=-1.0).copy()
        finally:
            demo._align_ir_to_color_numba = original_numba_align  # type: ignore[assignment]

        np.testing.assert_allclose(numba_output, numpy_output)

    def test_latest_wins_drops_across_depth_and_render_slots(self) -> None:
        depth_slot: demo.LatestSlot[DummyPacket] = demo.LatestSlot()
        render_slot: demo.LatestSlot[DummyPacket] = demo.LatestSlot()
        depth_slot.put(DummyPacket(seq=1))
        depth_slot.put(DummyPacket(seq=2))
        self.assertEqual(depth_slot.dropped_count, 1)
        self.assertEqual(depth_slot.get_latest_after(-1).seq, 2)  # type: ignore[union-attr]
        render_slot.put(DummyPacket(seq=2))
        render_slot.put(DummyPacket(seq=3))
        render_slot.put(DummyPacket(seq=4))
        self.assertEqual(render_slot.dropped_count, 2)
        self.assertEqual(render_slot.get_latest_after(-1).seq, 4)  # type: ignore[union-attr]

    def test_backprojection_max_points_preserves_linspace_valid_sampling(self) -> None:
        color_bgr = np.arange(3 * 4 * 3, dtype=np.uint8).reshape(3, 4, 3)
        depth_u16 = np.arange(1, 13, dtype=np.uint16).reshape(3, 4)
        intrinsics = demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0)
        full_points, full_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.0,
            stride=1,
            max_points=0,
            projection_grid=demo.build_projection_grid(width=4, height=3, stride=1, intrinsics=intrinsics),
        )
        capped_points, capped_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.0,
            stride=1,
            max_points=5,
            projection_grid=demo.build_projection_grid(width=4, height=3, stride=1, intrinsics=intrinsics),
        )
        expected_indices = np.linspace(0, full_points.shape[0] - 1, 5, dtype=np.int64)
        np.testing.assert_allclose(capped_points, full_points[expected_indices])
        np.testing.assert_array_equal(capped_colors, full_colors[expected_indices])

    def test_numba_backprojection_matches_numpy_when_available(self) -> None:
        if not demo.numba_backprojection_available():
            self.skipTest("numba is not installed")
        color_bgr = np.arange(4 * 5 * 3, dtype=np.uint8).reshape(4, 5, 3)
        depth_u16 = np.arange(1, 21, dtype=np.uint16).reshape(4, 5)
        depth_u16[0, 1] = 0
        depth_u16[3, 4] = 1000
        intrinsics = demo.CameraIntrinsics(fx=2.0, fy=3.0, cx=1.0, cy=1.5)
        projection_grid = demo.build_projection_grid(width=5, height=4, stride=1, intrinsics=intrinsics)
        numpy_points, numpy_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.02,
            stride=1,
            max_points=7,
            projection_grid=projection_grid,
            backproject_backend="numpy",
        )
        numba_points, numba_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.001,
            depth_max_m=0.02,
            stride=1,
            max_points=7,
            projection_grid=projection_grid,
            backproject_backend="numba",
        )
        np.testing.assert_allclose(numba_points, numpy_points)
        np.testing.assert_array_equal(numba_colors, numpy_colors)

    def test_direct_script_style_numba_warmup(self) -> None:
        if not demo.numba_backprojection_available():
            self.skipTest("numba is not installed")
        script = (
            "import importlib.util, pathlib, sys; "
            "path = pathlib.Path('scripts/harness/realtime_single_camera_pointcloud.py'); "
            "sys.path.insert(0, str(path.parent)); "
            "spec = importlib.util.spec_from_file_location('realtime_single_camera_pointcloud_direct', path); "
            "module = importlib.util.module_from_spec(spec); "
            "sys.modules[spec.name] = module; "
            "spec.loader.exec_module(module); "
            "module.warm_up_numba_backprojection(); "
            "module.warm_up_numba_ffs_align(); "
            "print(module.resolve_backproject_backend('auto', stride=1, projection_grid_available=True)); "
            "print(module.numba_ffs_align_available())"
        )
        result = subprocess.run(
            [sys.executable, "-c", script],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("numba", result.stdout)
        self.assertIn("True", result.stdout)

    def test_projection_grid_matches_pixel_grid_backprojection(self) -> None:
        color_bgr = np.array(
            [
                [[1, 2, 3], [4, 5, 6]],
                [[7, 8, 9], [10, 11, 12]],
            ],
            dtype=np.uint8,
        )
        depth_u16 = np.array([[1000, 0], [2000, 3000]], dtype=np.uint16)
        intrinsics = demo.CameraIntrinsics(fx=2.0, fy=4.0, cx=0.5, cy=0.25)
        pixel_points, pixel_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            pixel_grid=demo.build_pixel_grid(width=2, height=2, stride=1),
        )
        projection_points, projection_colors = demo.backproject_aligned_rgbd(
            color_bgr=color_bgr,
            depth_u16=depth_u16,
            intrinsics=intrinsics,
            depth_scale_m_per_unit=0.001,
            depth_min_m=0.1,
            depth_max_m=5.0,
            stride=1,
            max_points=0,
            projection_grid=demo.build_projection_grid(width=2, height=2, stride=1, intrinsics=intrinsics),
        )
        np.testing.assert_allclose(projection_points, pixel_points)
        np.testing.assert_array_equal(projection_colors, pixel_colors)

    def test_latest_slot_drops_stale_packets(self) -> None:
        slot: demo.LatestSlot[DummyPacket] = demo.LatestSlot()
        self.assertEqual(slot.dropped_count, 0)
        self.assertEqual(slot.total_dropped_count, 0)
        self.assertEqual(slot.latest_seq(), -1)
        slot.put(DummyPacket(seq=1))
        slot.put(DummyPacket(seq=2))
        self.assertEqual(slot.dropped_count, 1)
        self.assertEqual(slot.total_dropped_count, 1)
        self.assertEqual(slot.latest_seq(), 2)
        packet = slot.get_latest_after(-1)
        self.assertIsNotNone(packet)
        self.assertEqual(packet.seq, 2)
        self.assertIsNone(slot.get_latest_after(2))
        slot.put(DummyPacket(seq=3))
        self.assertEqual(slot.dropped_count, 1)
        self.assertEqual(slot.total_dropped_count, 1)
        self.assertEqual(slot.get_latest_after(2).seq, 3)  # type: ignore[union-attr]

    def test_latest_slot_reset_splits_total_and_steady_state_drops(self) -> None:
        slot: demo.LatestSlot[DummyPacket] = demo.LatestSlot()
        slot.put(DummyPacket(seq=1))
        slot.put(DummyPacket(seq=2))
        self.assertEqual(slot.dropped_count, 1)
        self.assertEqual(slot.total_dropped_count, 1)
        slot.reset_dropped_count()
        self.assertEqual(slot.dropped_count, 0)
        self.assertEqual(slot.total_dropped_count, 1)
        slot.put(DummyPacket(seq=3))
        self.assertEqual(slot.dropped_count, 0)
        self.assertEqual(slot.total_dropped_count, 2)
        slot.put(DummyPacket(seq=4))
        self.assertEqual(slot.dropped_count, 1)
        self.assertEqual(slot.total_dropped_count, 3)

    def test_render_stats_are_deterministic(self) -> None:
        stats = demo.RenderStats(window_s=1.0)
        stats.record_render(render_time_s=0.0, latency_ms=10.0)
        stats.record_render(render_time_s=0.5, latency_ms=20.0)
        stats.record_render(render_time_s=1.0, latency_ms=30.0)
        self.assertAlmostEqual(stats.render_fps, 2.0)
        self.assertAlmostEqual(stats.latest_latency_ms, 30.0)
        self.assertAlmostEqual(stats.mean_latency_ms, 20.0)

    def test_coalesced_post_gate_allows_only_one_pending_callback(self) -> None:
        gate = demo.CoalescedPostGate()
        self.assertFalse(gate.pending)
        self.assertTrue(gate.try_mark_pending())
        self.assertTrue(gate.pending)
        self.assertFalse(gate.try_mark_pending())
        gate.mark_done()
        self.assertFalse(gate.pending)
        self.assertTrue(gate.try_mark_pending())

    def test_drop_stats_snapshot_reports_total_after_warmup_and_window_delta(self) -> None:
        args = demo.build_parser().parse_args(["--debug"])
        viewer = demo.RealtimeSingleCameraPointCloudDemo(args)
        viewer.capture_slot.put(DummyPacket(seq=1))
        viewer.capture_slot.put(DummyPacket(seq=2))
        before_reset = viewer._drop_stats_snapshot(update_window=True)
        self.assertEqual(before_reset.capture_total, 1)
        self.assertEqual(before_reset.capture_after_warmup, 1)
        self.assertEqual(before_reset.capture_delta_last_window, 1)
        viewer._drop_stats_start_s = 10.0
        viewer._maybe_reset_drop_stats(10.0 + demo.DROP_STATS_WARMUP_S + 0.1)
        after_reset = viewer._drop_stats_snapshot(update_window=True)
        self.assertEqual(after_reset.capture_total, 1)
        self.assertEqual(after_reset.capture_after_warmup, 0)
        self.assertEqual(after_reset.capture_delta_last_window, 0)
        viewer.capture_slot.put(DummyPacket(seq=3))
        straddling = viewer._drop_stats_snapshot(update_window=True)
        self.assertEqual(straddling.capture_total, 2)
        self.assertEqual(straddling.capture_after_warmup, 0)
        viewer.capture_slot.put(DummyPacket(seq=4))
        steady = viewer._drop_stats_snapshot(update_window=True)
        self.assertEqual(steady.capture_total, 3)
        self.assertEqual(steady.capture_after_warmup, 1)
        self.assertEqual(steady.capture_delta_last_window, 1)

    def test_depth_to_render_profiler_sum_excludes_camera_wait(self) -> None:
        timing = demo.PipelineTiming(
            wait_ms=33.0,
            align_ms=1.0,
            frame_copy_ms=2.0,
            ffs_ms=6.0,
            ffs_align_ms=1.5,
            image_mask_ms=0.5,
            backproject_ms=3.0,
            open3d_convert_ms=4.0,
            open3d_update_ms=5.0,
            receive_to_render_ms=20.0,
        )
        self.assertAlmostEqual(demo.depth_to_render_ms(timing), 23.0)

    def test_debug_hud_includes_ffs_stage_timing(self) -> None:
        args = demo.build_parser().parse_args(["--debug", "--depth-source", "ffs"])
        viewer = demo.RealtimeSingleCameraPointCloudDemo(args)
        stats = demo.RenderStats()
        timing = demo.PipelineTiming(ffs_ms=9.5, ffs_align_ms=0.7, receive_to_render_ms=20.0)
        packet = demo.ImagePacket(
            seq=1,
            image_rgb_u8=np.zeros((1, 1, 3), dtype=np.uint8),
            valid_count=1,
            depth_source="ffs",
            receive_perf_s=0.0,
            process_done_perf_s=0.0,
            dropped_capture_frames=2,
            dropped_ffs_frames=3,
            timing=timing,
        )
        text = viewer._format_hud(packet=packet, stats=stats, timing=timing)
        self.assertIn("depth source: ffs", text)
        self.assertIn("dropped depth/render packets: 3/0", text)
        self.assertIn("dropped capture frames: 2 (total 2)", text)
        self.assertIn("ffs=9.50", text)
        self.assertIn("ffs_align=0.70", text)

    def test_debug_log_uses_explicit_drop_stat_fields(self) -> None:
        args = demo.build_parser().parse_args(["--debug", "--depth-source", "ffs"])
        viewer = demo.RealtimeSingleCameraPointCloudDemo(args)
        viewer.capture_slot.put(DummyPacket(seq=1))
        viewer.capture_slot.put(DummyPacket(seq=2))
        viewer.depth_slot.put(DummyPacket(seq=1))
        viewer.depth_slot.put(DummyPacket(seq=2))
        viewer.render_slot.put(DummyPacket(seq=1))
        viewer.render_slot.put(DummyPacket(seq=2))
        viewer._last_debug_log_s = -999.0
        packet = demo.ImagePacket(
            seq=2,
            image_rgb_u8=np.zeros((1, 1, 3), dtype=np.uint8),
            valid_count=1,
            depth_source="ffs",
            receive_perf_s=0.0,
            process_done_perf_s=0.0,
            dropped_capture_frames=0,
            dropped_ffs_frames=0,
            timing=demo.PipelineTiming(ffs_ms=9.5, ffs_align_ms=0.7, receive_to_render_ms=20.0),
        )
        with contextlib.redirect_stdout(io.StringIO()) as stdout:
            viewer._maybe_log_debug(packet=packet, stats=demo.RenderStats(), timing=packet.timing, now_s=0.0)
        text = stdout.getvalue()
        self.assertIn("dropped_capture_total=1", text)
        self.assertIn("dropped_capture_after_warmup=1", text)
        self.assertIn("dropped_capture_delta_last_window=1", text)
        self.assertIn("dropped_depth_delta_last_window=1", text)
        self.assertIn("dropped_render_delta_last_window=1", text)

    def test_raw_ffs_depth_aligns_in_render_prep_stage(self) -> None:
        args = demo.build_parser().parse_args(["--depth-source", "ffs"])
        viewer = demo.RealtimeSingleCameraPointCloudDemo(args)
        color_bgr = np.array([[[1, 2, 3], [4, 5, 6]]], dtype=np.uint8)
        K = np.eye(3, dtype=np.float32)
        ray_grid = demo.build_projection_grid_from_matrix(width=2, height=1, K=K)
        calls: list[tuple[tuple[int, int], tuple[int, int]]] = []

        class FakeAligner:
            def align(self, depth_ir_m: np.ndarray) -> np.ndarray:
                calls.append((depth_ir_m.shape, color_bgr.shape[:2]))
                return np.array([[1.0, 0.0]], dtype=np.float32)

        fake_aligner = FakeAligner()
        packet = demo.RawFfsDepthPacket(
            seq=7,
            color_bgr=color_bgr,
            depth_left_m=np.array([[1.0, 2.0]], dtype=np.float32),
            intrinsics=demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            k_ir_left=K,
            t_ir_left_to_color=np.eye(4, dtype=np.float32),
            k_color=K,
            ir_projection_grid=ray_grid,
            ir_to_color_aligner=fake_aligner,  # type: ignore[arg-type]
            receive_perf_s=0.0,
            ffs_done_perf_s=1.0,
            dropped_capture_frames=4,
            dropped_ffs_frames=0,
            timing=demo.PipelineTiming(ffs_ms=10.0),
        )
        viewer._process_image_frame(packet)

        render_packet = viewer.render_slot.get_latest_after(-1)
        self.assertIsInstance(render_packet, demo.ImagePacket)
        assert isinstance(render_packet, demo.ImagePacket)
        self.assertEqual(calls, [((1, 2), (1, 2))])
        self.assertEqual(render_packet.valid_count, 1)
        self.assertEqual(render_packet.depth_source, "ffs")
        self.assertEqual(render_packet.dropped_capture_frames, 4)
        self.assertEqual(render_packet.timing.ffs_ms, 10.0)
        self.assertGreaterEqual(render_packet.timing.ffs_align_ms, 0.0)
        self.assertGreaterEqual(render_packet.timing.image_mask_ms, 0.0)


if __name__ == "__main__":
    unittest.main()
