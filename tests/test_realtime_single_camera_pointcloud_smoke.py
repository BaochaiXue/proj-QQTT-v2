from __future__ import annotations

from dataclasses import dataclass
import contextlib
import io
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import types
import unittest
from unittest import mock

import numpy as np

from data_process.depth_backends import DEFAULT_FFS_REPO, DEFAULT_FFS_TRT_TWO_STAGE_MODEL_DIR
from data_process.depth_backends import fast_foundation_stereo as ffs_backend
from data_process.depth_backends.geometry import align_depth_to_color
from qqtt.demo import pcd_filter_fast
from qqtt.demo import realtime_masked_edgetam_pcd as masked_demo
from qqtt.demo import realtime_single_camera_pointcloud as demo_impl
from services.ffs_remote import ffs_depth_client as ffs_remote_client
from services.ffs_remote import ffs_depth_server as ffs_remote_server
from services.ffs_remote.ffs_depth_client import FfsRemoteDepthClient
from services.ffs_remote.protocol import (
    build_depth_request_parts,
    build_depth_response_parts,
    parse_depth_request_parts,
    parse_depth_response_parts,
)
from scripts.harness.diagnostics.demo import realtime_single_camera_pointcloud as demo


ROOT = Path(__file__).resolve().parents[1]
demo.REPO_ROOT = ROOT
demo_impl.REPO_ROOT = ROOT
masked_demo.REPO_ROOT = ROOT


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
    def test_masked_edgetam_help_exposes_realtime_masked_pcd_contract(self) -> None:
        result = subprocess.run(
            [sys.executable, "-m", "qqtt.demo.realtime_masked_edgetam_pcd", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--depth-source {ffs,ffs_remote,realsense,none}", result.stdout)
        self.assertIn("--ffs-trt-model-dir FFS_TRT_MODEL_DIR", result.stdout)
        self.assertIn("--ffs-remote-endpoint FFS_REMOTE_ENDPOINT", result.stdout)
        self.assertIn("--ffs-remote-return {depth_u16,depth_float_m,masked_uv_depth,masked_xyz}", result.stdout)
        self.assertIn("--ffs-remote-compress {none,zstd,lz4,png}", result.stdout)
        self.assertIn("--enable-remote-ffs-quality", result.stdout)
        self.assertIn("--remote-ffs-quality-return {depth_u16,depth_float_m,masked_uv_depth,masked_xyz}", result.stdout)
        self.assertIn("--init-mode {sam31-first-frame,saved-masks}", result.stdout)
        self.assertIn("--track-mode {controller-object,object-only,controller-only,none}", result.stdout)
        self.assertIn("--pcd-mode {masked,none}", result.stdout)
        self.assertIn("--render-mode {pointcloud,none,panel}", result.stdout)
        self.assertIn("--panel-layout {side-by-side}", result.stdout)
        self.assertIn("--panel-video-output PANEL_VIDEO_OUTPUT", result.stdout)
        self.assertIn("--tracking-background-mask {target-union,rgb}", result.stdout)
        self.assertIn("--demo-preset {none,local-ffs-professor}", result.stdout)
        self.assertIn("--compile-mode {vision-reduce-overhead}", result.stdout)
        self.assertIn("--pcd-color-mode {rgb,class}", result.stdout)
        self.assertIn("--enable-pcd-filter", result.stdout)
        self.assertIn("--pcd-filter-mode {async,sync,none}", result.stdout)
        self.assertIn("--object-filter {none,pt-filter,enhanced-pt,voxel-density}", result.stdout)
        self.assertIn("--controller-filter {none,pt-filter,enhanced-pt,voxel-density}", result.stdout)
        self.assertIn("--view-mode {orbit,camera}", result.stdout)
        self.assertIn("--edgetam-live-session-keep-frames EDGETAM_LIVE_SESSION_KEEP_FRAMES", result.stdout)
        self.assertIn("--filter-every-n FILTER_EVERY_N", result.stdout)
        self.assertIn("--profile-cuda-events", result.stdout)
        self.assertIn("--controller-init-mask CONTROLLER_INIT_MASK", result.stdout)
        self.assertIn("--object-init-mask OBJECT_INIT_MASK", result.stdout)
        self.assertIn("renders only the masked", result.stdout)
        self.assertIn("PCD", result.stdout)

    def test_masked_edgetam_defaults_and_object_id_mapping(self) -> None:
        args = masked_demo.build_parser().parse_args([])
        self.assertEqual(args.depth_source, "ffs")
        self.assertIn("model_20-30-48_iters_4_res_480x864", str(args.ffs_trt_model_dir))
        self.assertEqual(args.compile_mode, "vision-reduce-overhead")
        self.assertEqual(args.init_mode, "sam31-first-frame")
        self.assertEqual(args.track_mode, "controller-object")
        self.assertEqual(args.pcd_mode, "masked")
        self.assertEqual(args.render_mode, "pointcloud")
        self.assertFalse(args.profile_sync)
        self.assertFalse(args.profile_cuda_events)
        self.assertEqual(args.pcd_color_mode, "rgb")
        self.assertEqual(args.view_mode, "orbit")
        self.assertEqual(args.edgetam_live_session_keep_frames, 64)
        self.assertFalse(args.enable_pcd_filter)
        self.assertEqual(args.pcd_filter_mode, "async")
        self.assertEqual(args.object_filter, "none")
        self.assertEqual(args.controller_filter, "none")
        self.assertEqual(args.object_filter_cap, 0)
        self.assertEqual(args.controller_filter_cap, 0)
        self.assertEqual(args.filter_every_n, 3)
        self.assertEqual(args.filter_budget_ms, 12.0)
        self.assertEqual(args.ffs_remote_max_inflight, 1)
        self.assertEqual(args.ffs_remote_timeout_ms, 80)
        self.assertEqual(args.ffs_remote_return, "depth_u16")
        self.assertEqual(args.ffs_remote_compress, "none")
        self.assertFalse(args.enable_remote_ffs_quality)
        self.assertEqual(args.demo_preset, "none")
        self.assertEqual(masked_demo.object_id_labels(), {1: "controller", 2: "object"})
        self.assertEqual(masked_demo.object_id_labels("object-only"), {2: "object"})
        self.assertEqual(masked_demo.object_id_labels("controller-only"), {1: "controller"})
        self.assertEqual(masked_demo.object_id_labels("none"), {})
        object_only_args = masked_demo.build_parser().parse_args(["--track-mode", "object-only"])
        self.assertEqual(masked_demo.active_object_ids(object_only_args), [2])
        controller_only_args = masked_demo.build_parser().parse_args(["--track-mode", "controller-only"])
        self.assertEqual(masked_demo.active_object_ids(controller_only_args), [1])
        capture_only_args = masked_demo.build_parser().parse_args(
            ["--depth-source", "none", "--track-mode", "none", "--pcd-mode", "none", "--render-mode", "none"]
        )
        masked_demo.validate_args(capture_only_args)
        remote_args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "ffs_remote",
                "--ffs-remote-endpoint",
                "tcp://127.0.0.1:7001",
                "--ffs-repo",
                "/missing/ffs",
                "--ffs-trt-model-dir",
                "/missing/engine",
            ]
        )
        masked_demo.validate_args(remote_args)
        sparse_main_args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "ffs_remote",
                "--ffs-remote-endpoint",
                "tcp://127.0.0.1:7001",
                "--ffs-remote-return",
                "masked_uv_depth",
                "--track-mode",
                "object-only",
            ]
        )
        masked_demo.validate_args(sparse_main_args)
        sparse_no_tracking_args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "ffs_remote",
                "--ffs-remote-endpoint",
                "tcp://127.0.0.1:7001",
                "--ffs-remote-return",
                "masked_uv_depth",
                "--track-mode",
                "none",
                "--pcd-mode",
                "none",
                "--render-mode",
                "none",
            ]
        )
        with self.assertRaisesRegex(ValueError, "sparse --depth-source ffs_remote requires EdgeTAM masks"):
            masked_demo.validate_args(sparse_no_tracking_args)
        quality_args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--enable-remote-ffs-quality",
                "--ffs-remote-endpoint",
                "tcp://127.0.0.1:7001",
                "--remote-ffs-quality-return",
                "masked_uv_depth",
                "--track-mode",
                "object-only",
            ]
        )
        masked_demo.validate_args(quality_args)
        missing_remote_args = masked_demo.build_parser().parse_args(["--depth-source", "ffs_remote"])
        with self.assertRaisesRegex(ValueError, "ffs_remote requires --ffs-remote-endpoint"):
            masked_demo.validate_args(missing_remote_args)
        self.assertIn("System warming up", masked_demo.WARMUP_HUD_TEXT)
        self.assertIn("Keep one steady pose", masked_demo.WARMUP_HUD_TEXT)
        with contextlib.redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                masked_demo.build_parser().parse_args(["--compile-mode", "none"])

    def test_pcd_filter_fast_voxel_cap_and_density_filter(self) -> None:
        xyz = np.array(
            [
                [0.00, 0.00, 0.50],
                [0.01, 0.00, 0.50],
                [0.02, 0.00, 0.50],
                [1.00, 1.00, 1.00],
            ],
            dtype=np.float32,
        )
        colors = np.arange(12, dtype=np.uint8).reshape(4, 3)

        capped, capped_colors = pcd_filter_fast.voxel_cap_points(
            xyz,
            colors,
            max_points=2,
            voxel_size_m=0.10,
            rng=np.random.default_rng(0),
        )
        self.assertLessEqual(capped.shape[0], 2)
        self.assertEqual(capped_colors.shape, (capped.shape[0], 3))

        dense_xyz, dense_colors = pcd_filter_fast.voxel_density_filter(
            xyz,
            colors,
            voxel_size_m=0.10,
            min_points_per_voxel=2,
        )
        np.testing.assert_allclose(dense_xyz, xyz[:3])
        np.testing.assert_array_equal(dense_colors, colors[:3])

    def test_async_pcd_filter_worker_latest_output(self) -> None:
        def filter_fn(item: pcd_filter_fast.FilterInput) -> pcd_filter_fast.FilterOutput:
            return pcd_filter_fast.FilterOutput(
                seq=item.seq,
                object_xyz=item.object_xyz,
                object_rgb=item.object_rgb,
                controller_xyz=item.controller_xyz,
                controller_rgb=item.controller_rgb,
                filter_ms=0.1,
                created_perf_s=item.created_perf_s,
            )

        worker = pcd_filter_fast.AsyncPcdFilterWorker(filter_fn)
        worker.start()
        try:
            worker.submit_latest(
                pcd_filter_fast.FilterInput(
                    seq=7,
                    object_xyz=np.zeros((1, 3), dtype=np.float32),
                    object_rgb=np.zeros((1, 3), dtype=np.uint8),
                    controller_xyz=np.zeros((0, 3), dtype=np.float32),
                    controller_rgb=np.zeros((0, 3), dtype=np.uint8),
                )
            )
            deadline = time.perf_counter() + 1.0
            latest = None
            while time.perf_counter() < deadline:
                latest = worker.latest_output()
                if latest is not None:
                    break
                time.sleep(0.01)
            self.assertIsNotNone(latest)
            assert latest is not None
            self.assertEqual(latest.seq, 7)
            self.assertEqual(latest.object_xyz.shape, (1, 3))
        finally:
            worker.stop()

    def test_masked_edgetam_filter_input_caps_before_filter(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--object-filter",
                "none",
                "--controller-filter",
                "none",
                "--object-filter-cap",
                "2",
                "--controller-filter-cap",
                "1",
                "--filter-min-cap",
                "1",
                "--object-filter-voxel-m",
                "0.10",
                "--controller-filter-voxel-m",
                "0.10",
            ]
        )
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        xyz = np.array(
            [
                [0.00, 0.00, 0.50],
                [0.01, 0.00, 0.50],
                [0.20, 0.00, 0.50],
            ],
            dtype=np.float32,
        )
        colors = np.arange(9, dtype=np.uint8).reshape(3, 3)
        item = demo_instance._make_filter_input(
            seq=3,
            object_xyz=xyz,
            object_colors=colors,
            controller_xyz=xyz,
            controller_colors=colors,
        )
        item = pcd_filter_fast.FilterInput(
            seq=item.seq,
            object_xyz=item.object_xyz,
            object_rgb=item.object_rgb,
            controller_xyz=item.controller_xyz[:2],
            controller_rgb=item.controller_rgb[:2],
            object_cap=item.object_cap,
            controller_cap=item.controller_cap,
            object_voxel_size_m=item.object_voxel_size_m,
            controller_voxel_size_m=item.controller_voxel_size_m,
            created_perf_s=item.created_perf_s,
        )
        output = demo_instance._filter_pcd_input(item)
        self.assertLessEqual(output.object_xyz.shape[0], 2)
        self.assertLessEqual(output.controller_xyz.shape[0], 1)
        self.assertEqual(output.stats["object"]["raw_points"], 3)
        self.assertEqual(output.stats["controller"]["raw_points"], 2)

    def test_masked_edgetam_local_ffs_professor_preset_keeps_ffs_semantics(self) -> None:
        args = masked_demo.build_parser().parse_args(["--demo-preset", "local-ffs-professor"])
        masked_demo.apply_demo_preset(args)
        self.assertEqual(args.depth_source, "ffs")
        self.assertEqual(args.compile_mode, "vision-reduce-overhead")
        self.assertEqual(args.pcd_max_points, masked_demo.LOCAL_FFS_PROFESSOR_MAX_POINTS)
        self.assertEqual(args.point_size, masked_demo.LOCAL_FFS_PROFESSOR_POINT_SIZE)
        self.assertEqual(args.latency_target_ms, masked_demo.LOCAL_FFS_PROFESSOR_LATENCY_TARGET_MS)

        explicit_args = masked_demo.build_parser().parse_args(
            [
                "--demo-preset",
                "local-ffs-professor",
                "--pcd-max-points",
                "12000",
                "--point-size",
                "3",
                "--latency-target-ms",
                "160",
            ]
        )
        masked_demo.apply_demo_preset(explicit_args)
        self.assertEqual(explicit_args.pcd_max_points, 12000)
        self.assertEqual(explicit_args.point_size, 3)
        self.assertEqual(explicit_args.latency_target_ms, 160)

        invalid_args = masked_demo.build_parser().parse_args(
            ["--demo-preset", "local-ffs-professor", "--depth-source", "realsense"]
        )
        masked_demo.apply_demo_preset(invalid_args)
        with self.assertRaisesRegex(ValueError, "requires --depth-source ffs"):
            masked_demo.validate_args(invalid_args)

    def test_masked_edgetam_visual_mode_defaults_table_z_filter_to_zero_when_calibrated(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--table-calibrate",
                "table_calibrate.pkl",
                "--render-mode",
                "pointcloud",
                "--demo-visual-mode",
                "tracking",
            ]
        )
        masked_demo.apply_demo_preset(args)

        self.assertTrue(args.enable_table_z_filter)
        self.assertEqual(args.table_z_filter_threshold_m, 0.0)

    def test_masked_edgetam_headless_defaults_table_z_filter_to_zero_when_calibrated(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--table-calibrate",
                "table_calibrate.pkl",
                "--input-source",
                "fake-live",
                "--depth-source",
                "ffs",
                "--render-mode",
                "none",
                "--headless-capture-dir",
                "result/headless_case",
                "--enable-pcd-filter",
                "--pcd-filter-mode",
                "sync",
                "--pcd-filter-preset",
                "enhanced-pt",
            ]
        )
        masked_demo.apply_demo_preset(args)

        self.assertTrue(args.enable_table_z_filter)
        self.assertEqual(args.table_z_filter_threshold_m, 0.0)

    def test_masked_edgetam_visual_mode_table_z_filter_can_be_disabled(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--table-calibrate",
                "table_calibrate.pkl",
                "--render-mode",
                "pointcloud",
                "--demo-visual-mode",
                "pcd",
                "--disable-table-z-filter",
            ]
        )
        masked_demo.apply_demo_preset(args)

        self.assertFalse(args.enable_table_z_filter)
        self.assertEqual(args.table_z_filter_threshold_m, 0.0)

    def test_masked_edgetam_saved_masks_validate_shape(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            mask_path = root / "mask.png"
            Image.fromarray(np.array([[0, 255, 0], [255, 0, 0]], dtype=np.uint8)).save(mask_path)
            mask = masked_demo.load_binary_mask(mask_path, expected_shape=(2, 3))
            self.assertEqual(mask.dtype, np.bool_)
            self.assertEqual(int(np.count_nonzero(mask)), 2)
            with self.assertRaisesRegex(ValueError, "does not match frame shape"):
                masked_demo.load_binary_mask(mask_path, expected_shape=(3, 2))

            args = masked_demo.build_parser().parse_args(
                [
                    "--depth-source",
                    "realsense",
                    "--init-mode",
                    "saved-masks",
                    "--controller-init-mask",
                    str(mask_path),
                    "--object-init-mask",
                    str(mask_path),
                ]
            )
            masked_demo.validate_args(args)

            object_only_args = masked_demo.build_parser().parse_args(
                [
                    "--depth-source",
                    "realsense",
                    "--init-mode",
                    "saved-masks",
                    "--track-mode",
                    "object-only",
                    "--object-init-mask",
                    str(mask_path),
                ]
            )
            masked_demo.validate_args(object_only_args)

            frame = masked_demo.FramePacket(
                seq=0,
                color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
                depth_source="realsense",
                intrinsics=masked_demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
                depth_scale_m_per_unit=0.001,
                receive_perf_s=0.0,
                timing=masked_demo.PipelineTiming(),
            )
            controller_mask, object_mask = masked_demo.resolve_initial_masks(frame, object_only_args)
            self.assertFalse(np.any(controller_mask))
            self.assertEqual(int(np.count_nonzero(object_mask)), 2)

    def test_live_sam31_first_frame_uses_image_one_frame_helper(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--init-mode",
                "sam31-first-frame",
                "--track-mode",
                "object-only",
                "--object-prompt",
                "stuffed animal",
            ]
        )
        captured: dict[str, object] = {}

        def _fake_run_image_segmentation(**kwargs):
            captured.update(kwargs)
            return {
                "masks_by_label": {
                    "stuffed animal": [
                        np.asarray([[False, True, False], [True, True, False]], dtype=bool)
                    ]
                }
            }

        frame = masked_demo.FramePacket(
            seq=0,
            color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_source="realsense",
            intrinsics=masked_demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=0.0,
            timing=masked_demo.PipelineTiming(),
        )

        with mock.patch(
            "scripts.harness.support.sam31_mask_helper.run_image_segmentation",
            side_effect=_fake_run_image_segmentation,
        ) as run_image, mock.patch.object(masked_demo, "release_sam31_runtime_resources") as release:
            controller_mask, object_mask = masked_demo.resolve_initial_masks(frame, args)

        run_image.assert_called_once()
        release.assert_called_once_with("cuda")
        self.assertFalse(captured.get("reuse_model", False))
        self.assertEqual(captured["text_prompt"], "stuffed animal")
        self.assertFalse(np.any(controller_mask))
        self.assertEqual(int(np.count_nonzero(object_mask)), 3)

    def test_sam31_first_frame_can_keep_cached_runtime_for_multicamera_init(self) -> None:
        captured: dict[str, object] = {}

        def _fake_cached_run_image_segmentation(**kwargs):
            captured.update(kwargs)
            return {
                "masks_by_label": {
                    "stuffed animal": [
                        np.asarray([[False, True, False], [True, True, False]], dtype=bool)
                    ]
                }
            }

        args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "realsense",
                "--init-mode",
                "sam31-first-frame",
                "--track-mode",
                "object-only",
                "--object-prompt",
                "stuffed animal",
            ]
        )
        args.sam31_cache_init_model = True
        args.sam31_keep_runtime_until_all_cameras_init = True
        frame = masked_demo.FramePacket(
            seq=0,
            color_bgr=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_source="realsense",
            intrinsics=masked_demo.CameraIntrinsics(fx=1.0, fy=1.0, cx=0.0, cy=0.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=0.0,
            timing=masked_demo.PipelineTiming(),
        )

        with mock.patch(
            "scripts.harness.support.sam31_mask_helper.run_image_segmentation",
            side_effect=_fake_cached_run_image_segmentation,
        ) as run_image, mock.patch.object(masked_demo, "release_sam31_runtime_resources") as release:
            controller_mask, object_mask = masked_demo.resolve_initial_masks(frame, args)

        run_image.assert_called_once()
        release.assert_not_called()
        self.assertTrue(captured["reuse_model"])
        self.assertFalse(np.any(controller_mask))
        self.assertEqual(int(np.count_nonzero(object_mask)), 3)

    def test_ffs_remote_protocol_roundtrip_and_client_depth_decode(self) -> None:
        left = np.arange(6, dtype=np.uint8).reshape(2, 3)
        right = np.arange(10, 16, dtype=np.uint8).reshape(2, 3)
        k_ir = np.eye(3, dtype=np.float32)
        k_color = np.eye(3, dtype=np.float32) * 2.0
        transform = np.eye(4, dtype=np.float32)
        request_parts = build_depth_request_parts(
            frame_id=42,
            ir_left_u8=left,
            ir_right_u8=right,
            color_shape=(2, 3),
            k_ir_left=k_ir,
            k_color=k_color,
            t_ir_left_to_color=transform,
            baseline_m=0.055,
            depth_scale_m_per_unit=0.001,
            return_type="depth_u16",
        )
        parsed = parse_depth_request_parts(request_parts)
        self.assertEqual(parsed.metadata["frame_id"], 42)
        np.testing.assert_array_equal(parsed.ir_left_u8, left)
        np.testing.assert_array_equal(parsed.ir_right_u8, right)

        mask_u8 = np.array([[0, 2, 0], [1, 0, 2]], dtype=np.uint8)
        sparse_request_parts = build_depth_request_parts(
            frame_id=43,
            ir_left_u8=left,
            ir_right_u8=right,
            color_shape=(2, 3),
            k_ir_left=k_ir,
            k_color=k_color,
            t_ir_left_to_color=transform,
            baseline_m=0.055,
            depth_scale_m_per_unit=0.001,
            return_type="masked_uv_depth",
            mask_u8=mask_u8,
            compression="none",
        )
        sparse_request = parse_depth_request_parts(sparse_request_parts)
        self.assertEqual(sparse_request.metadata["return_type"], "masked_uv_depth")
        np.testing.assert_array_equal(sparse_request.mask_u8, mask_u8)

        sparse_response_parts = build_depth_response_parts(
            frame_id=43,
            depth=np.array([[1.0, 0.0, 1.2, 2.0], [0.0, 1.0, 1.4, 1.0]], dtype=np.float32),
            depth_dtype="float32",
            return_type="masked_uv_depth",
            compression="none",
        )
        sparse_response = parse_depth_response_parts(sparse_response_parts)
        self.assertEqual(sparse_response.metadata["return_type"], "masked_uv_depth")
        self.assertEqual(tuple(sparse_response.depth.shape), (2, 4))

        class FakeSocket:
            def __init__(self) -> None:
                self.sent: list[bytes] | None = None

            def send_multipart(self, parts):
                self.sent = list(parts)

            def recv_multipart(self):
                return build_depth_response_parts(
                    frame_id=42,
                    depth=np.array([[0, 1000, 1200], [1300, 0, 1400]], dtype=np.uint16),
                    depth_dtype="uint16",
                    server_ffs_ms=18.0,
                    server_align_ms=6.0,
                    server_total_ms=25.0,
                    depth_scale_m_per_unit=0.001,
                )

        fake_socket = FakeSocket()
        client = FfsRemoteDepthClient(
            endpoint="tcp://example.invalid:7001",
            timeout_ms=80,
            return_type="depth_u16",
            zmq_socket=fake_socket,
        )
        result = client.request_depth_color_m(
            frame_id=42,
            ir_left_u8=left,
            ir_right_u8=right,
            color_shape=(2, 3),
            k_ir_left=k_ir,
            k_color=k_color,
            t_ir_left_to_color=transform,
            baseline_m=0.055,
            depth_scale_m_per_unit=0.001,
        )
        self.assertIsNotNone(fake_socket.sent)
        np.testing.assert_allclose(
            result.depth_color_m,
            np.array([[0.0, 1.0, 1.2], [1.3, 0.0, 1.4]], dtype=np.float32),
            rtol=1e-6,
            atol=1e-6,
        )
        self.assertEqual(result.frame_id, 42)
        self.assertEqual(result.server_ffs_ms, 18.0)
        self.assertEqual(result.server_align_ms, 6.0)
        self.assertEqual(result.server_total_ms, 25.0)

    def test_ffs_remote_client_cli_help_and_echo_benchmark_summary(self) -> None:
        result = subprocess.run(
            [sys.executable, "services/ffs_remote/ffs_depth_client.py", "--help"],
            cwd=ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        self.assertIn("--endpoint ENDPOINT", result.stdout)
        self.assertIn("--echo-benchmark", result.stdout)
        self.assertIn("--real-ir-depth-benchmark", result.stdout)
        self.assertIn("--three-camera-real-ir-depth-benchmark", result.stdout)
        self.assertIn("--serial SERIAL", result.stdout)
        self.assertIn("--serials [SERIALS ...]", result.stdout)
        self.assertIn("--max-cams MAX_CAMS", result.stdout)
        self.assertIn("--inflight INFLIGHT", result.stdout)
        self.assertIn("--drop-stale-replies", result.stdout)
        self.assertIn("--profile PROFILE", result.stdout)
        self.assertIn("--compress {none,zstd,lz4,png}", result.stdout)
        self.assertIn("--mask-fraction MASK_FRACTION", result.stdout)
        self.assertIn("--save-first-depth-preview", result.stdout)

        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0
                self.closed = False

            def request_depth_color_m(self, **kwargs):
                self.calls += 1
                frame_id = int(kwargs["frame_id"])
                return ffs_remote_client.FfsRemoteDepthResult(
                    frame_id=frame_id,
                    depth_color_m=np.zeros(tuple(kwargs["color_shape"]), dtype=np.float32),
                    rtt_ms=4.0 + frame_id,
                    server_ffs_ms=0.0,
                    server_align_ms=0.0,
                    server_total_ms=2.0,
                    request_bytes=100,
                    response_bytes=50,
                )

            def close(self) -> None:
                self.closed = True

        args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--echo-benchmark",
                "--profile",
                "4x3",
                "--fps",
                "20",
                "--duration-s",
                "0.12",
            ]
        )
        fake_client = FakeClient()
        summary = ffs_remote_client.run_echo_benchmark(args, client=fake_client)
        self.assertGreaterEqual(fake_client.calls, 1)
        self.assertGreaterEqual(summary["ok"], 1.0)
        self.assertEqual(summary["failed"], 0.0)
        self.assertFalse(fake_client.closed)

        real_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--real-ir-depth-benchmark",
                "--serial",
                "239222300412",
                "--profile",
                "848x480",
                "--fps",
                "30",
                "--duration-s",
                "1",
                "--compress",
                "lz4",
                "--return-type",
                "depth_u16",
                "--save-first-depth-preview",
            ]
        )
        ffs_remote_client._validate_real_ir_depth_args(real_args)
        self.assertTrue(real_args.real_ir_depth_benchmark)
        self.assertFalse(real_args.echo_benchmark)
        self.assertEqual(real_args.inflight, 1)

        inflight_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--real-ir-depth-benchmark",
                "--return-type",
                "depth_u16",
                "--inflight",
                "4",
                "--drop-stale-replies",
            ]
        )
        ffs_remote_client._validate_real_ir_depth_args(inflight_args)
        self.assertEqual(inflight_args.inflight, 4)
        self.assertTrue(inflight_args.drop_stale_replies)

        sparse_real_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--real-ir-depth-benchmark",
                "--return-type",
                "masked_uv_depth",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires a full-frame return type"):
            ffs_remote_client._validate_real_ir_depth_args(sparse_real_args)

        bad_inflight_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--real-ir-depth-benchmark",
                "--return-type",
                "depth_u16",
                "--inflight",
                "0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "inflight must be positive"):
            ffs_remote_client._validate_real_ir_depth_args(bad_inflight_args)

        three_cam_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--three-camera-real-ir-depth-benchmark",
                "--return-type",
                "depth_u16",
                "--serials",
                "cam0",
                "cam1",
                "cam2",
                "--fps",
                "15",
                "--inflight",
                "6",
            ]
        )
        ffs_remote_client._validate_three_camera_real_ir_depth_args(three_cam_args)
        self.assertTrue(three_cam_args.three_camera_real_ir_depth_benchmark)
        self.assertEqual(three_cam_args.serials, ["cam0", "cam1", "cam2"])
        self.assertEqual(three_cam_args.max_cams, 3)
        self.assertEqual(three_cam_args.inflight, 6)

        sparse_three_cam_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--three-camera-real-ir-depth-benchmark",
                "--return-type",
                "masked_uv_depth",
            ]
        )
        with self.assertRaisesRegex(ValueError, "requires a full-frame return type"):
            ffs_remote_client._validate_three_camera_real_ir_depth_args(sparse_three_cam_args)

        duplicate_serial_args = ffs_remote_client.build_parser().parse_args(
            [
                "--endpoint",
                "tcp://127.0.0.1:7001",
                "--three-camera-real-ir-depth-benchmark",
                "--serials",
                "cam0",
                "cam0",
            ]
        )
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ffs_remote_client._validate_three_camera_real_ir_depth_args(duplicate_serial_args)

    def test_ffs_remote_client_real_ir_benchmark_summary_and_artifacts(self) -> None:
        class FakeClient:
            def __init__(self) -> None:
                self.calls = 0

            def request_depth_color_m(self, **kwargs):
                self.calls += 1
                frame_id = int(kwargs["frame_id"])
                depth = np.array([[0.0, 1.0, 1.2], [1.3, 0.0, 1.4]], dtype=np.float32)
                return ffs_remote_client.FfsRemoteDepthResult(
                    frame_id=frame_id,
                    depth_color_m=depth,
                    rtt_ms=7.0,
                    server_ffs_ms=9.0,
                    server_align_ms=3.0,
                    server_total_ms=14.0,
                    request_bytes=320,
                    response_bytes=160,
                    raw_depth=np.array([[0, 1000, 1200], [1300, 0, 1400]], dtype=np.uint16),
                    metadata={"compression": "lz4"},
                )

        class FakeFrameSource:
            def __init__(self) -> None:
                self.frame_id = 0
                self.serial = "fake-serial"
                self.started = False

            def start(self) -> None:
                self.started = True

            def next_frame(self) -> ffs_remote_client.RealIrDepthFrame:
                frame_id = self.frame_id
                self.frame_id += 1
                k = np.array([[600.0, 0.0, 1.5], [0.0, 600.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32)
                return ffs_remote_client.RealIrDepthFrame(
                    frame_id=frame_id,
                    ir_left_u8=np.full((2, 3), frame_id % 255, dtype=np.uint8),
                    ir_right_u8=np.full((2, 3), (frame_id + 1) % 255, dtype=np.uint8),
                    color_shape=(2, 3),
                    k_ir_left=k,
                    k_color=k,
                    t_ir_left_to_color=np.eye(4, dtype=np.float32),
                    baseline_m=0.055,
                    depth_scale_m_per_unit=0.001,
                )

        with tempfile.TemporaryDirectory() as tmp_dir:
            args = ffs_remote_client.build_parser().parse_args(
                [
                    "--endpoint",
                    "tcp://127.0.0.1:7001",
                    "--real-ir-depth-benchmark",
                    "--profile",
                    "3x2",
                    "--fps",
                    "20",
                    "--duration-s",
                    "0.12",
                    "--compress",
                    "lz4",
                    "--return-type",
                    "depth_u16",
                    "--save-first-depth-preview",
                    "--output-dir",
                    tmp_dir,
                ]
            )
            fake_client = FakeClient()
            fake_source = FakeFrameSource()
            summary = ffs_remote_client.run_real_ir_depth_benchmark(
                args,
                client=fake_client,
                frame_source=fake_source,
            )
            self.assertTrue(fake_source.started)
            self.assertGreaterEqual(fake_client.calls, 1)
            self.assertGreaterEqual(summary["ok"], 1.0)
            self.assertEqual(summary["failed"], 0.0)
            self.assertEqual(summary["input_source"], "real_realsense_ir")
            self.assertEqual(summary["request_compression"], "lz4")
            self.assertEqual(summary["response_compression"], "lz4")
            self.assertGreater(summary["depth_nonzero_count_mean"], 0.0)
            self.assertTrue(Path(str(summary["first_depth_npy_path"])).is_file())
            self.assertEqual(summary["first_depth_npy_path"], summary["first_depth_m_npy_path"])
            self.assertTrue(Path(str(summary["first_depth_m_npy_path"])).is_file())
            self.assertTrue(Path(str(summary["first_depth_u16_npy_path"])).is_file())
            self.assertTrue(Path(str(summary["first_depth_u16_npy_path"])).name.endswith("_depth_u16.npy"))
            self.assertTrue(Path(str(summary["first_depth_preview_path"])).is_file())

    def test_ffs_remote_server_strict_engine_contract_validates_path_tokens(self) -> None:
        args = ffs_remote_server.build_parser().parse_args(
            [
                "--strict-engine-contract",
                "--ffs-trt-model-dir",
                "data/experiments/ffs_trt_4090_848x480_pad864_builderopt5/engines/model_20-30-48_iters_4_res_480x864",
            ]
        )
        ffs_remote_server._validate_engine_contract(args)
        metadata = ffs_remote_server._engine_contract_metadata(args)
        self.assertEqual(metadata["ffs_contract_model"], "20-30-48")
        self.assertEqual(metadata["ffs_contract_valid_iters"], 4)
        self.assertEqual(metadata["ffs_contract_engine_width"], 864)
        self.assertEqual(metadata["ffs_contract_builder_optimization_level"], 5)

        bad_args = ffs_remote_server.build_parser().parse_args(
            ["--strict-engine-contract", "--ffs-trt-model-dir", "engines/model_wrong_iters_2_res_480x848"]
        )
        with self.assertRaisesRegex(ValueError, "strict FFS engine contract failed"):
            ffs_remote_server._validate_engine_contract(bad_args)

    def test_masked_edgetam_releases_sam31_runtime_resources(self) -> None:
        class FakeAutocast:
            def __init__(self) -> None:
                self.exited = False

            def __exit__(self, exc_type, exc, traceback) -> None:
                self.exited = True

        class FakeCuda:
            def __init__(self) -> None:
                self.calls: list[str] = []

            def is_available(self) -> bool:
                return True

            def synchronize(self) -> None:
                self.calls.append("synchronize")

            def empty_cache(self) -> None:
                self.calls.append("empty_cache")

            def ipc_collect(self) -> None:
                self.calls.append("ipc_collect")

        autocast = FakeAutocast()
        helper = types.SimpleNamespace(_CUDA_AUTOCAST_CONTEXT=autocast)
        cuda = FakeCuda()
        fake_torch = types.SimpleNamespace(cuda=cuda)
        original_helper = sys.modules.get("scripts.harness.support.sam31_mask_helper")
        original_torch = sys.modules.get("torch")
        sys.modules["scripts.harness.support.sam31_mask_helper"] = helper
        sys.modules["torch"] = fake_torch
        try:
            masked_demo.release_sam31_runtime_resources("cuda")
        finally:
            if original_helper is None:
                sys.modules.pop("scripts.harness.support.sam31_mask_helper", None)
            else:
                sys.modules["scripts.harness.support.sam31_mask_helper"] = original_helper
            if original_torch is None:
                sys.modules.pop("torch", None)
            else:
                sys.modules["torch"] = original_torch

        self.assertTrue(autocast.exited)
        self.assertIsNone(helper._CUDA_AUTOCAST_CONTEXT)
        self.assertEqual(cuda.calls, ["synchronize", "empty_cache", "ipc_collect"])

    def test_masked_edgetam_backprojects_masked_pixels_only(self) -> None:
        depth_m = np.array([[0.1, 0.5, 1.0], [1.4, 2.0, 0.7]], dtype=np.float32)
        mask = np.array([[True, True, False], [True, True, True]])
        ray_x = np.array([[0.0, 1.0, 2.0], [0.0, 1.0, 2.0]], dtype=np.float32)
        ray_y = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]], dtype=np.float32)

        points = masked_demo.backproject_masked(
            depth_m=depth_m,
            mask=mask,
            ray_x=ray_x,
            ray_y=ray_y,
            depth_min_m=0.2,
            depth_max_m=1.5,
            max_points=0,
        )

        expected = np.array(
            [
                [0.5, 0.0, 0.5],
                [0.0, 1.4, 1.4],
                [1.4, 0.7, 0.7],
            ],
            dtype=np.float32,
        )
        np.testing.assert_allclose(points, expected)

        capped = masked_demo.backproject_masked(
            depth_m=depth_m,
            mask=mask,
            ray_x=ray_x,
            ray_y=ray_y,
            depth_min_m=0.2,
            depth_max_m=1.5,
            max_points=2,
            rng=np.random.default_rng(0),
        )
        self.assertEqual(capped.shape, (2, 3))

    def test_masked_edgetam_rgbd_backprojection_uses_live_rgb_colors(self) -> None:
        color_bgr = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        depth_m = np.array([[0.5, 0.6], [0.7, 0.8]], dtype=np.float32)
        mask = np.array([[False, True], [True, False]])
        ray_x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        ray_y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)

        points, colors = masked_demo.backproject_masked_rgbd(
            color_bgr=color_bgr,
            depth_m=depth_m,
            mask=mask,
            ray_x=ray_x,
            ray_y=ray_y,
            depth_min_m=0.2,
            depth_max_m=1.5,
            max_points=0,
            color_mode="rgb",
            class_rgb=(1, 2, 3),
        )

        np.testing.assert_allclose(points, np.array([[0.6, 0.0, 0.6], [0.0, 0.7, 0.7]], dtype=np.float32))
        np.testing.assert_array_equal(colors, np.array([[60, 50, 40], [90, 80, 70]], dtype=np.uint8))

        _points, class_colors = masked_demo.backproject_masked_rgbd(
            color_bgr=color_bgr,
            depth_m=depth_m,
            mask=mask,
            ray_x=ray_x,
            ray_y=ray_y,
            depth_min_m=0.2,
            depth_max_m=1.5,
            max_points=0,
            color_mode="class",
            class_rgb=(1, 2, 3),
        )
        np.testing.assert_array_equal(class_colors, np.array([[1, 2, 3], [1, 2, 3]], dtype=np.uint8))

    def test_masked_edgetam_sparse_remote_pcd_uses_ffs_payload_and_live_rgb(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--depth-source",
                "ffs_remote",
                "--ffs-remote-endpoint",
                "tcp://127.0.0.1:7001",
                "--ffs-remote-return",
                "masked_uv_depth",
                "--track-mode",
                "object-only",
            ]
        )
        demo_instance = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        color_bgr = np.array(
            [
                [[10, 20, 30], [40, 50, 60]],
                [[70, 80, 90], [100, 110, 120]],
            ],
            dtype=np.uint8,
        )
        payload = np.array(
            [
                [1.0, 0.0, 0.5, masked_demo.OBJECT_ID],
                [0.0, 1.0, 0.6, masked_demo.CONTROLLER_ID],
            ],
            dtype=np.float32,
        )
        ray_x = np.array([[0.0, 1.0], [0.0, 1.0]], dtype=np.float32)
        ray_y = np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float32)
        controller_xyz, controller_colors, object_xyz, object_colors, timing = demo_instance._split_sparse_remote_pcd(
            payload=payload,
            return_type="masked_uv_depth",
            color_bgr=color_bgr,
            ray_x=ray_x,
            ray_y=ray_y,
            rng=np.random.default_rng(0),
        )
        np.testing.assert_allclose(controller_xyz, np.array([[0.0, 0.6, 0.6]], dtype=np.float32))
        np.testing.assert_allclose(object_xyz, np.array([[0.5, 0.0, 0.5]], dtype=np.float32))
        np.testing.assert_array_equal(controller_colors, np.array([[90, 80, 70]], dtype=np.uint8))
        np.testing.assert_array_equal(object_colors, np.array([[60, 50, 40]], dtype=np.uint8))
        self.assertGreaterEqual(timing["pcd_select_ms"], 0.0)

    def test_masked_edgetam_extracts_hf_masks_by_output_object_ids(self) -> None:
        class Output:
            object_ids = np.array([2, 1])

        post_masks = [
            np.array([[[[0.0, 1.0], [0.0, 0.0]]]], dtype=np.float32),
            np.array([[[[1.0, 0.0], [1.0, 0.0]]]], dtype=np.float32),
        ]
        masks = masked_demo.extract_object_masks_from_hf_output(Output(), post_masks)

        self.assertEqual(set(masks), {1, 2})
        np.testing.assert_array_equal(masks[2], np.array([[False, True], [False, False]]))
        np.testing.assert_array_equal(masks[1], np.array([[True, False], [True, False]]))

    def test_help_exposes_supported_capture_rates_and_profiles(self) -> None:
        for command in (
            [sys.executable, "-m", "qqtt.demo.realtime_single_camera_pointcloud", "--help"],
            [sys.executable, "scripts/harness/diagnostics/demo/realtime_single_camera_pointcloud.py", "--help"],
        ):
            with self.subTest(command=command):
                result = subprocess.run(
                    command,
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
        self.assertEqual(demo_impl.REPO_ROOT, ROOT)

    def test_wslg_open3d_wrapper_pins_d3d12_xwayland_defaults(self) -> None:
        for wrapper in (
            ROOT / "scripts" / "harness" / "diagnostics" / "hardware" / "run_wslg_open3d.sh",
        ):
            with self.subTest(wrapper=wrapper):
                text = wrapper.read_text(encoding="utf-8")
                self.assertTrue(wrapper.exists())
                self.assertIn('export WAYLAND_DISPLAY=""', text)
                self.assertIn('EGL_PLATFORM="${EGL_PLATFORM:-x11}"', text)
                self.assertIn('GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"', text)
                self.assertIn('MESA_LOADER_DRIVER_OVERRIDE="${MESA_LOADER_DRIVER_OVERRIDE:-d3d12}"', text)
                self.assertIn('LIBGL_ALWAYS_SOFTWARE="${LIBGL_ALWAYS_SOFTWARE:-0}"', text)
                self.assertIn('QQTT_WSLG_OPEN3D_FAST_EXIT="${QQTT_WSLG_OPEN3D_FAST_EXIT:-1}"', text)
                self.assertIn('MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"', text)
                self.assertIn('exec "$@"', text)

    def test_script_applies_wslg_open3d_defaults_before_import(self) -> None:
        env = {
            "WSL_DISTRO_NAME": "Ubuntu",
            "WAYLAND_DISPLAY": "wayland-0",
            "VK_ICD_FILENAMES": "bad-vulkan.json",
            "MESA_D3D12_DEFAULT_ADAPTER_NAME": "Intel",
        }
        with mock.patch.dict(os.environ, env, clear=False):
            applied = demo.apply_wslg_open3d_env_defaults()
            self.assertEqual(os.environ["WAYLAND_DISPLAY"], "")
            self.assertEqual(os.environ["EGL_PLATFORM"], "x11")
            self.assertEqual(os.environ["GALLIUM_DRIVER"], "d3d12")
            self.assertEqual(os.environ["MESA_LOADER_DRIVER_OVERRIDE"], "d3d12")
            self.assertEqual(os.environ["LIBGL_ALWAYS_SOFTWARE"], "0")
            self.assertEqual(os.environ["QQTT_WSLG_OPEN3D_FAST_EXIT"], "1")
            self.assertEqual(os.environ["MESA_D3D12_DEFAULT_ADAPTER_NAME"], "Intel")
            self.assertNotIn("VK_ICD_FILENAMES", os.environ)
            self.assertEqual(applied["WAYLAND_DISPLAY"], "")
            self.assertEqual(applied["VK_ICD_FILENAMES"], "<unset>")

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

        original_numba_align = demo_impl._align_ir_to_color_numba
        try:
            demo_impl._align_ir_to_color_numba = None  # type: ignore[assignment]
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
            demo_impl._align_ir_to_color_numba = original_numba_align  # type: ignore[assignment]

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
            "path = pathlib.Path('qqtt/demo/realtime_single_camera_pointcloud.py'); "
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
