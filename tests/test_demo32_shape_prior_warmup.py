from __future__ import annotations

import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
from PIL import Image

from qqtt.demo import realtime_masked_edgetam_pcd as masked_demo
from qqtt.demo import single_demo_v3_runtime as runtime
from qqtt.demo import shape_prior_warmup as warmup
from qqtt.demo.single_view_shape_align import (
    ShapeAlignmentConfig,
    align_canonical_shape_to_observation,
)
from qqtt.env.camera.table_calibration import (
    build_table_calibration_metadata,
    write_table_calibration_files,
)
from services.shape_prior_remote.protocol import (
    PROTOCOL_NAME,
    build_error_response_parts,
    build_shape_prior_request_parts,
    build_shape_prior_response_parts,
    parse_shape_prior_request_parts,
    parse_shape_prior_response_parts,
)
from services.shape_prior_remote import server as shape_prior_server


REPO_ROOT = Path(__file__).resolve().parents[1]


def _explicit(argv: list[str]) -> set[str]:
    return {item.split("=", 1)[0] for item in argv if item.startswith("--")}


def _option_value(argv: list[str], option: str) -> str:
    return argv[argv.index(option) + 1]


def _write_valid_table_calibration(path: Path) -> None:
    metadata = build_table_calibration_metadata(
        serial_numbers=["s0"],
        WH=[640, 480],
        fps=30,
        transform_count=1,
        calibration_board={"name": "calibio-12x9-30mm"},
        max_reprojection_error_px=0.5,
        min_corner_fraction=60 / 88,
        min_charuco_corners=60,
        per_camera_reprojection_error=[0.1],
        per_camera_corner_count=[60],
        per_camera_corner_fraction=[60 / 88],
    )
    write_table_calibration_files(path, [np.eye(4, dtype=np.float32)], metadata)


class Sam3dOnlyCliContractTest(unittest.TestCase):
    def _help_text(self, script: str, *, env: dict[str, str] | None = None) -> str:
        result = subprocess.run(
            [sys.executable, script, "--help"],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
        )
        return result.stdout

    def test_shape_prior_help_mentions_only_sam3d_root(self) -> None:
        help_text = self._help_text("data_process_sam3d/shape_prior.py")
        removed_env = "MV" + "SAM3D"
        removed_label = "MV-" + "SAM3D"

        self.assertIn("SAM3D_ROOT", help_text)
        self.assertIn("sam-3d-objects", help_text)
        self.assertNotIn(removed_env, help_text)
        self.assertNotIn(removed_label, help_text)

    def test_data_process_sample_help_has_no_sampling_backend_switch(self) -> None:
        script_text = (REPO_ROOT / "data_process_sam3d" / "data_process_sample.py").read_text()
        removed_backend_option = "shape_prior_" + "sampling_backend"
        removed_marker = "mv" + "sam3d"

        self.assertIn("--shape_prior", script_text)
        self.assertNotIn(removed_backend_option, script_text)
        self.assertNotIn(removed_marker, script_text.lower())

    def test_shape_prior_worker_help_prefers_script_checkout_over_env_root(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stale_root = Path(tmp)
            (stale_root / "qqtt").mkdir()
            (stale_root / "services").mkdir()
            env = dict(os.environ)
            env["QQTT_REPO_ROOT"] = str(stale_root)

            help_text = self._help_text("services/shape_prior_remote/server.py", env=env)

        self.assertIn("Long-lived remote SAM3D shape-prior worker", help_text)
        self.assertIn("--sam3d-root", help_text)

    def test_shape_prior_worker_parser_accepts_preload_and_warmup_flags(self) -> None:
        args = shape_prior_server.build_parser().parse_args(["--preload-models", "--warmup-models"])

        self.assertTrue(args.preload_models)
        self.assertTrue(args.warmup_models)

    def test_shape_prior_worker_warmup_args_imply_preload_args(self) -> None:
        args = shape_prior_server.parse_args(["--warmup-models"])

        self.assertTrue(args.preload_models)
        self.assertTrue(args.warmup_models)


class Demo32ShapePriorWrapperTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.repo_root = Path(self._tmp.name)
        _write_valid_table_calibration(self.repo_root / "table_calibrate.pkl")
        patch = mock.patch.object(runtime, "REPO_ROOT", self.repo_root)
        patch.start()
        self.addCleanup(patch.stop)

    def _parse(self, version: str, argv: list[str]):
        parser = runtime.build_arg_parser(demo_version=version)
        args = parser.parse_args(argv)
        return runtime.apply_preset_defaults(args, explicit_options=_explicit(argv))

    def test_demo32_shape_prior_is_default_on_and_forwarded(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3_2, ["--dry-run"])

        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

        self.assertTrue(args.shape_prior_warmup)
        self.assertEqual(args.shape_prior_start_policy, "async-after-first-mask-depth-pair")
        self.assertEqual(args.shape_prior_execution, "remote-worker")
        self.assertEqual(args.shape_prior_endpoint, "tcp://127.0.0.1:7100")
        self.assertEqual(args.shape_prior_timeout_ms, 180000)
        self.assertEqual(args.shape_prior_device, "cuda:0")
        self.assertTrue(args.shape_prior_skip_route_visualizations)
        self.assertTrue(contract["shape_prior_warmup_enabled"])
        self.assertEqual(contract["shape_prior_status"], "pending")
        self.assertEqual(contract["shape_prior_start_policy"], "async-after-first-mask-depth-pair")
        self.assertEqual(contract["shape_prior_execution"], "remote-worker")
        self.assertEqual(contract["shape_backend"], "sam3d-objects")
        self.assertEqual(contract["shape_prior_timeout_ms"], 180000)
        self.assertEqual(contract["shape_prior_depth_backend"], "native-realsense")
        self.assertEqual(contract["shape_prior_depth_source_internal"], "realsense")
        self.assertIsNone(contract["shape_prior_profile_json"])
        self.assertEqual(contract["profile_summary_fields"]["shape_prior_status"], "pending")
        self.assertEqual(_option_value(delegate, "--depth-source"), "realsense")
        self.assertEqual(_option_value(delegate, "--shape-prior-endpoint"), "tcp://127.0.0.1:7100")
        self.assertEqual(_option_value(delegate, "--shape-prior-timeout-ms"), "180000")
        self.assertIn("--shape-prior-warmup", delegate)
        self.assertIn("--shape-prior-skip-route-visualizations", delegate)
        self.assertNotIn("--shape-prior-profile-json", delegate)

    def test_demo32_shape_prior_profile_json_is_forwarded(self) -> None:
        profile_path = self.repo_root / "shape_profile.json"
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            ["--dry-run", "--shape-prior-profile-json", str(profile_path)],
        )

        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

        self.assertEqual(contract["shape_prior_profile_json"], str(profile_path))
        self.assertEqual(_option_value(delegate, "--shape-prior-profile-json"), str(profile_path))

    def test_demo32_shape_prior_old_strict_pair_policy_remains_explicitly_supported(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            ["--dry-run", "--shape-prior-start-policy", "async-after-first-strict-pair"],
        )

        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

        self.assertEqual(args.shape_prior_start_policy, "async-after-first-strict-pair")
        self.assertEqual(contract["shape_prior_start_policy"], "async-after-first-strict-pair")
        self.assertEqual(_option_value(delegate, "--shape-prior-start-policy"), "async-after-first-strict-pair")

    def test_demo32_shape_prior_can_be_disabled(self) -> None:
        args = self._parse(runtime.DEMO_VERSION_3_2, ["--dry-run", "--no-shape-prior-warmup"])

        runtime.validate_args(args)
        contract = runtime.build_contract(args)
        delegate = runtime.build_live_delegate_argv(args, active_serial="s0")

        self.assertFalse(args.shape_prior_warmup)
        self.assertFalse(contract["shape_prior_warmup_enabled"])
        self.assertEqual(contract["shape_prior_status"], "disabled")
        self.assertIn("--no-shape-prior-warmup", delegate)
        self.assertNotIn("--shape-prior-warmup", delegate)

    def test_shape_prior_options_are_demo32_only(self) -> None:
        for version in (runtime.DEMO_VERSION_3, runtime.DEMO_VERSION_3_1, runtime.DEMO_VERSION_3_3):
            with self.subTest(version=version):
                parser = runtime.build_arg_parser(demo_version=version)
                with contextlib.redirect_stderr(io.StringIO()):
                    with self.assertRaises(SystemExit):
                        parser.parse_args(["--shape-prior-warmup"])

    def test_demo32_native_shape_prior_contract_keeps_depth_backend_source(self) -> None:
        args = self._parse(
            runtime.DEMO_VERSION_3_2,
            ["--dry-run", "--depth-backend", "native-realsense"],
        )

        runtime.validate_args(args)
        contract = runtime.build_contract(args)

        self.assertTrue(contract["shape_prior_warmup_enabled"])
        self.assertEqual(contract["depth_backend"], "native-realsense")
        self.assertEqual(contract["depth_source_internal"], "realsense")
        self.assertEqual(contract["shape_prior_depth_backend"], "native-realsense")
        self.assertEqual(contract["shape_prior_depth_source_internal"], "realsense")


class ShapePriorProtocolAndSnapshotTest(unittest.TestCase):
    def _snapshot(self) -> warmup.ShapePriorSnapshot:
        object_mask = np.zeros((3, 4), dtype=bool)
        object_mask[1, 1:3] = True
        controller_mask = np.zeros_like(object_mask)
        controller_mask[2, 2] = True
        return warmup.ShapePriorSnapshot(
            seq=7,
            source_timestamp_s=1.25,
            input_source="fake-live",
            depth_backend="ir-ffs",
            depth_source_internal="ffs",
            rgb_u8=np.full((3, 4, 3), 120, dtype=np.uint8),
            object_mask=object_mask,
            controller_mask=controller_mask,
            depth_color_m=np.ones((3, 4), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
            table_z_m=0.0,
            table_z_above_direction="negative",
        )

    def test_snapshot_requires_nonempty_object_mask_and_table_transform(self) -> None:
        snapshot = self._snapshot()
        warmup.validate_shape_prior_snapshot(snapshot)

        with self.assertRaisesRegex(ValueError, "object mask"):
            warmup.validate_shape_prior_snapshot(
                warmup.replace_snapshot(snapshot, object_mask=np.zeros_like(snapshot.object_mask))
            )
        with self.assertRaisesRegex(ValueError, "camera_to_world_c2w"):
            warmup.validate_shape_prior_snapshot(
                warmup.replace_snapshot(snapshot, camera_to_world_c2w=None)
            )
        with self.assertRaisesRegex(ValueError, "depth_color_m"):
            warmup.validate_shape_prior_snapshot(
                warmup.replace_snapshot(snapshot, depth_color_m=np.ones((2, 4), dtype=np.float32))
            )

    def test_snapshot_defaults_to_negative_table_z_direction(self) -> None:
        object_mask = np.zeros((2, 2), dtype=bool)
        object_mask[0, 0] = True
        snapshot = warmup.ShapePriorSnapshot(
            seq=0,
            source_timestamp_s=None,
            input_source="fake-live",
            depth_backend="native-realsense",
            depth_source_internal="realsense",
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=object_mask,
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        normalized = warmup.normalize_snapshot(snapshot)

        self.assertEqual(snapshot.table_z_above_direction, "negative")
        self.assertEqual(normalized.table_z_above_direction, "negative")

    def test_protocol_roundtrip_preserves_snapshot_arrays_and_metadata(self) -> None:
        snapshot = self._snapshot()

        parts = build_shape_prior_request_parts(snapshot=snapshot, request_id="req-7")
        request = parse_shape_prior_request_parts(parts)

        self.assertEqual(request.metadata["protocol"], PROTOCOL_NAME)
        self.assertEqual(request.metadata["request_id"], "req-7")
        self.assertEqual(request.metadata["seq"], 7)
        self.assertEqual(request.metadata["depth_backend"], "ir-ffs")
        self.assertEqual(request.metadata["table_z_m"], 0.0)
        self.assertEqual(request.metadata["table_z_above_direction"], "negative")
        np.testing.assert_array_equal(request.rgb_u8, snapshot.rgb_u8)
        np.testing.assert_array_equal(request.object_mask, snapshot.object_mask)
        np.testing.assert_allclose(request.depth_color_m, snapshot.depth_color_m)

        response_parts = build_shape_prior_response_parts(
            request_id="req-7",
            seq=7,
            status="ready",
            points_m=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            colors_rgb_u8=np.full((2, 3), 150, dtype=np.uint8),
            surface_points_m=np.array([[0.0, 0.0, -0.01]], dtype=np.float32),
            interior_points_m=np.array([[0.002, 0.002, -0.02]], dtype=np.float32),
            metadata={"single_view_alignment_ms": 3.5},
        )
        response = parse_shape_prior_response_parts(response_parts)

        self.assertEqual(response.metadata["status"], "ready")
        self.assertEqual(response.metadata["seq"], 7)
        self.assertEqual(response.points_m.shape, (2, 3))
        self.assertEqual(response.colors_rgb_u8.shape, (2, 3))
        np.testing.assert_allclose(response.surface_points_m, [[0.0, 0.0, -0.01]])
        np.testing.assert_allclose(response.interior_points_m, [[0.002, 0.002, -0.02]])
        self.assertEqual(response.metadata["surface_point_count"], 1)
        self.assertEqual(response.metadata["interior_point_count"], 1)
        self.assertEqual(response.metadata["single_view_alignment_ms"], 3.5)

    def test_protocol_fallback_defaults_to_negative_table_z_direction(self) -> None:
        snapshot = SimpleNamespace(
            seq=0,
            source_timestamp_s=None,
            input_source="fake-live",
            depth_backend="native-realsense",
            depth_source_internal="realsense",
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        request = parse_shape_prior_request_parts(
            build_shape_prior_request_parts(snapshot=snapshot, request_id="default-z")
        )

        self.assertEqual(request.metadata["table_z_above_direction"], "negative")

    def test_protocol_error_response_is_parseable_without_point_payload(self) -> None:
        response = parse_shape_prior_response_parts(
            build_error_response_parts(request_id="req-err", seq=4, error="worker unavailable")
        )

        self.assertEqual(response.metadata["status"], "error")
        self.assertIn("worker unavailable", response.metadata["error"])
        self.assertEqual(response.points_m.shape, (0, 3))
        self.assertEqual(response.colors_rgb_u8.shape, (0, 3))

    def test_async_manager_fail_soft_records_error_and_does_not_raise(self) -> None:
        class FailingClient:
            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                raise TimeoutError("no worker")

        manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=FailingClient(),
            start_policy="async-after-first-strict-pair",
        )
        manager.maybe_submit(self._snapshot())

        deadline = time.monotonic() + 2.0
        while time.monotonic() < deadline and manager.status == "pending":
            time.sleep(0.01)

        self.assertEqual(manager.status, "failed")
        self.assertIn("no worker", manager.profile()["shape_prior_error"])
        self.assertIsNone(manager.ready_result())


class ShapePriorWorkerSam3DInputTest(unittest.TestCase):
    def _worker(self) -> shape_prior_server.ShapePriorSam3DWorker:
        return shape_prior_server.ShapePriorSam3DWorker(
            sam3d_root=Path("/does/not/matter"),
            config=None,
            device="cuda:0",
            seed=42,
            max_points=128,
            upscale_category="stuffed animal",
        )

    def test_sam3d_receives_upscaled_object_crop_and_resized_mask(self) -> None:
        rgb = np.zeros((12, 16, 3), dtype=np.uint8)
        rgb[:, :, 0] = np.arange(16, dtype=np.uint8)[None, :]
        rgb[:, :, 1] = np.arange(12, dtype=np.uint8)[:, None]
        object_mask = np.zeros((12, 16), dtype=bool)
        object_mask[3:9, 5:11] = True
        request = shape_prior_server.ShapePriorRequest(
            metadata={"request_id": "req", "seq": 0},
            rgb_u8=rgb,
            object_mask=object_mask,
            controller_mask=np.zeros_like(object_mask),
            depth_color_m=np.ones((12, 16), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )
        calls: dict[str, object] = {}

        class FakeUpscaler:
            def __call__(self, *, prompt: str, image: Image.Image):
                calls["prompt"] = prompt
                calls["crop_size"] = image.size
                upscaled = image.resize((image.width * 4, image.height * 4), Image.Resampling.NEAREST)
                return SimpleNamespace(images=[upscaled])

        class FakePipeline:
            def run(self, image_rgb, mask_u8, **kwargs):
                calls["sam3d_image_shape"] = tuple(image_rgb.shape)
                calls["sam3d_mask_shape"] = tuple(mask_u8.shape)
                calls["sam3d_mask_pixels"] = int(np.count_nonzero(mask_u8))
                calls["sam3d_kwargs"] = dict(kwargs)
                return {"glb": SimpleNamespace(vertices=np.eye(3, dtype=np.float32))}

        worker = self._worker()
        worker._load_upscaler = lambda: FakeUpscaler()  # type: ignore[method-assign]
        worker._load_inference = lambda: SimpleNamespace(_pipeline=FakePipeline())  # type: ignore[method-assign]

        canonical, metadata = worker._canonical_points_from_sam3d(request)

        self.assertEqual(calls["prompt"], "Hand manipulates a stuffed animal.")
        self.assertNotEqual(calls["sam3d_image_shape"], tuple(rgb.shape))
        self.assertEqual(calls["sam3d_image_shape"][:2], calls["sam3d_mask_shape"])
        self.assertEqual(calls["sam3d_image_shape"][0], calls["crop_size"][1] * 4)
        self.assertEqual(calls["sam3d_image_shape"][1], calls["crop_size"][0] * 4)
        self.assertGreater(calls["sam3d_mask_pixels"], 0)
        self.assertEqual(canonical.shape, (3, 3))
        self.assertGreaterEqual(metadata["image_upscale_ms"], 0.0)
        self.assertIn("upscaler_model_load_ms", metadata)
        self.assertGreaterEqual(metadata["mask_refinement_ms"], 0.0)
        self.assertEqual(metadata["sam3d_input_shape"], list(calls["sam3d_image_shape"]))

    def test_preload_models_loads_upscaler_and_sam3d_before_requests(self) -> None:
        worker = self._worker()
        calls: list[str] = []
        worker._load_upscaler = lambda: calls.append("upscaler") or object()  # type: ignore[method-assign]
        worker._load_inference = lambda: calls.append("sam3d") or object()  # type: ignore[method-assign]

        metadata = worker.preload_models()

        self.assertEqual(calls, ["upscaler", "sam3d"])
        self.assertTrue(metadata["worker_preloaded_models"])
        self.assertFalse(metadata["worker_warmed_models"])
        self.assertGreaterEqual(metadata["worker_preload_upscaler_ms"], 0.0)
        self.assertGreaterEqual(metadata["worker_preload_sam3d_ms"], 0.0)

    def test_warmup_models_implies_preload_and_runs_before_ready(self) -> None:
        worker = self._worker()
        calls: list[str] = []
        worker.preload_models = lambda: calls.append("preload") or worker.startup_metadata()  # type: ignore[method-assign]
        worker.run_dummy_warmup = lambda: calls.append("warmup") or worker.startup_metadata()  # type: ignore[method-assign]

        metadata = shape_prior_server._prepare_worker_startup(
            worker,
            preload_models=False,
            warmup_models=True,
        )

        self.assertEqual(calls, ["preload", "warmup"])
        self.assertTrue(metadata["worker_preloaded_models"])
        self.assertTrue(metadata["worker_warmed_models"])

    def test_warmup_failure_fails_startup_before_ready(self) -> None:
        worker = self._worker()
        worker.preload_models = lambda: worker.startup_metadata()  # type: ignore[method-assign]

        def fail_warmup():
            raise RuntimeError("dummy warmup failed")

        worker.run_dummy_warmup = fail_warmup  # type: ignore[method-assign]

        with self.assertRaisesRegex(RuntimeError, "dummy warmup failed"):
            shape_prior_server._prepare_worker_startup(
                worker,
                preload_models=True,
                warmup_models=True,
            )

    def test_dummy_warmup_runs_upscaler_sam3d_and_mesh_conversion(self) -> None:
        worker = self._worker()
        calls: dict[str, object] = {}

        class FakeUpscaler:
            def __call__(self, *, prompt: str, image: Image.Image):
                calls["prompt"] = prompt
                calls["warmup_input_size"] = image.size
                return SimpleNamespace(images=[image.resize((32, 32), Image.Resampling.NEAREST)])

        class FakePipeline:
            def run(self, image_rgb, mask_u8, **kwargs):
                calls["sam3d_image_shape"] = tuple(image_rgb.shape)
                calls["sam3d_mask_pixels"] = int(np.count_nonzero(mask_u8))
                calls["sam3d_kwargs"] = dict(kwargs)
                return {
                    "glb": SimpleNamespace(
                        vertices=np.array(
                            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
                            dtype=np.float32,
                        ),
                        faces=np.array([[0, 1, 2]], dtype=np.int32),
                    )
                }

        worker._load_upscaler = lambda: FakeUpscaler()  # type: ignore[method-assign]
        worker._load_inference = lambda: SimpleNamespace(_pipeline=FakePipeline())  # type: ignore[method-assign]

        metadata = worker.run_dummy_warmup()

        self.assertEqual(calls["prompt"], "Hand manipulates a stuffed animal.")
        self.assertEqual(calls["sam3d_image_shape"], (32, 32, 3))
        self.assertGreater(calls["sam3d_mask_pixels"], 0)
        self.assertTrue(metadata["worker_warmed_models"])
        self.assertGreaterEqual(metadata["worker_dummy_warmup_ms"], 0.0)

    def test_response_metadata_includes_worker_startup_timing(self) -> None:
        worker = self._worker()
        worker._startup_metadata.update(
            {
                "worker_preloaded_models": True,
                "worker_warmed_models": True,
                "worker_preload_upscaler_ms": 11.0,
                "worker_preload_sam3d_ms": 22.0,
                "worker_dummy_warmup_ms": 33.0,
                "worker_ready_ms": 66.0,
            }
        )
        worker.echo_observation = True
        request = shape_prior_server.ShapePriorRequest(
            metadata={"request_id": "req-startup-timing", "seq": 3},
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        response = parse_shape_prior_response_parts(worker.handle(request))

        self.assertEqual(response.metadata["status"], "ready")
        self.assertTrue(response.metadata["worker_preloaded_models"])
        self.assertTrue(response.metadata["worker_warmed_models"])
        self.assertEqual(response.metadata["worker_preload_upscaler_ms"], 11.0)
        self.assertEqual(response.metadata["worker_preload_sam3d_ms"], 22.0)
        self.assertEqual(response.metadata["worker_dummy_warmup_ms"], 33.0)
        self.assertEqual(response.metadata["worker_ready_ms"], 66.0)

    def test_sam3d_scene_geometry_vertices_are_accepted(self) -> None:
        rgb = np.zeros((10, 10, 3), dtype=np.uint8)
        object_mask = np.zeros((10, 10), dtype=bool)
        object_mask[2:8, 2:8] = True
        request = shape_prior_server.ShapePriorRequest(
            metadata={"request_id": "req", "seq": 0},
            rgb_u8=rgb,
            object_mask=object_mask,
            controller_mask=np.zeros_like(object_mask),
            depth_color_m=np.ones((10, 10), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        class FakeUpscaler:
            def __call__(self, *, prompt: str, image: Image.Image):
                return SimpleNamespace(images=[image.resize((16, 16), Image.Resampling.NEAREST)])

        class FakePipeline:
            def run(self, image_rgb, mask_u8, **kwargs):
                return {
                    "glb": SimpleNamespace(
                        geometry={
                            "part_a": SimpleNamespace(
                                vertices=np.array(
                                    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                                    dtype=np.float32,
                                )
                            ),
                            "part_b": SimpleNamespace(
                                vertices=np.array(
                                    [[0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                                    dtype=np.float32,
                                )
                            ),
                        }
                    )
                }

        worker = shape_prior_server.ShapePriorSam3DWorker(
            sam3d_root=Path("/does/not/matter"),
            config=None,
            device="cuda:0",
            seed=42,
            max_points=128,
            upscale_category="stuffed animal",
        )
        worker._load_upscaler = lambda: FakeUpscaler()  # type: ignore[method-assign]
        worker._load_inference = lambda: SimpleNamespace(_pipeline=FakePipeline())  # type: ignore[method-assign]

        canonical, metadata = worker._canonical_points_from_sam3d(request)

        self.assertEqual(canonical.shape, (4, 3))
        self.assertEqual(metadata["sam3d_mesh_source"], "glb")
        self.assertTrue(np.isfinite(canonical).all())

    def test_worker_alignment_uses_request_table_above_direction(self) -> None:
        object_mask = np.zeros((2, 2), dtype=bool)
        object_mask[:, :] = True
        depth = np.full((2, 2), 0.05, dtype=np.float32)
        k_color = np.eye(3, dtype=np.float32)
        c2w = np.diag([1.0, 1.0, -1.0, 1.0]).astype(np.float32)
        request = shape_prior_server.ShapePriorRequest(
            metadata={
                "request_id": "req",
                "seq": 0,
                "table_z_m": 0.0,
                "table_z_above_direction": "negative",
            },
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=object_mask,
            controller_mask=np.zeros_like(object_mask),
            depth_color_m=depth,
            k_color=k_color,
            camera_to_world_c2w=c2w,
        )
        observation = np.array(
            [
                [0.0, 0.0, -0.05],
                [0.05, 0.0, -0.05],
                [0.0, 0.05, -0.05],
                [0.05, 0.05, -0.05],
            ],
            dtype=np.float32,
        )
        worker = shape_prior_server.ShapePriorSam3DWorker(
            sam3d_root=Path("/does/not/matter"),
            config=None,
            device="cuda:0",
            seed=42,
            max_points=128,
            upscale_category="stuffed animal",
        )
        worker._canonical_points_from_sam3d = lambda _request: (  # type: ignore[method-assign]
            observation,
            {
                "sam3d_model_load_ms": 0.0,
                "image_upscale_ms": 0.0,
                "mask_refinement_ms": 0.0,
                "sam3d_inference_ms": 0.0,
                "geometry_export_ms": 0.0,
            },
        )

        response = parse_shape_prior_response_parts(worker.handle(request))

        self.assertEqual(response.metadata["status"], "ready")
        self.assertEqual(response.metadata["alignment"]["ground_z_fraction"], 0.0)

    def test_worker_alignment_config_defaults_to_negative_table_z_direction(self) -> None:
        request = shape_prior_server.ShapePriorRequest(
            metadata={},
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.ones((2, 2), dtype=bool),
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        config = shape_prior_server._alignment_config_from_request(request)

        self.assertEqual(config.above_direction, "negative")

    def test_worker_response_includes_data_process_sam3d_single_view_shape_prior_samples(self) -> None:
        object_mask = np.zeros((8, 8), dtype=bool)
        object_mask[3:5, 3:5] = True
        depth = np.full((8, 8), 0.5, dtype=np.float32)
        depth[3, 3] = 0.46
        depth[4, 4] = 0.54
        request = shape_prior_server.ShapePriorRequest(
            metadata={
                "request_id": "req-single-view",
                "seq": 2,
                "table_z_m": 0.0,
                "table_z_above_direction": "positive",
            },
            rgb_u8=np.zeros((8, 8, 3), dtype=np.uint8),
            object_mask=object_mask,
            controller_mask=np.zeros_like(object_mask),
            depth_color_m=depth,
            k_color=np.array([[10.0, 0.0, 3.5], [0.0, 10.0, 3.5], [0.0, 0.0, 1.0]], dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )

        class FakeUpscaler:
            def __call__(self, *, prompt: str, image: Image.Image):
                return SimpleNamespace(images=[image.resize((16, 16), Image.Resampling.NEAREST)])

        class FakePipeline:
            def run(self, image_rgb, mask_u8, **kwargs):
                return {
                    "glb": SimpleNamespace(
                        vertices=np.array(
                            [
                                [0.0, 0.0, 0.0],
                                [1.0, 0.0, 0.0],
                                [0.0, 1.0, 0.0],
                                [0.0, 0.0, 1.0],
                            ],
                            dtype=np.float32,
                        ),
                        faces=np.array(
                            [
                                [0, 1, 2],
                                [0, 1, 3],
                                [0, 2, 3],
                                [1, 2, 3],
                            ],
                            dtype=np.int32,
                        ),
                    )
                }

        worker = shape_prior_server.ShapePriorSam3DWorker(
            sam3d_root=Path("/does/not/matter"),
            config=None,
            device="cuda:0",
            seed=42,
            max_points=512,
            upscale_category="stuffed animal",
        )
        worker._load_upscaler = lambda: FakeUpscaler()  # type: ignore[method-assign]
        worker._load_inference = lambda: SimpleNamespace(_pipeline=FakePipeline())  # type: ignore[method-assign]

        response = parse_shape_prior_response_parts(worker.handle(request))

        self.assertEqual(response.metadata["status"], "ready")
        self.assertEqual(response.metadata["single_view_shape_prior_sampling_backend"], "sam3d-single-view")
        self.assertFalse(response.metadata["uses_mvsam3d"])
        self.assertEqual(response.metadata["shape_prior_target_surface_points"], 700)
        self.assertEqual(response.metadata["shape_prior_target_interior_points"], 1000)
        self.assertEqual(response.metadata["shape_prior_configured_max_dist_m"], 0.05)
        self.assertEqual(response.metadata["shape_prior_effective_max_dist_m"], 0.05)
        self.assertEqual(response.metadata["shape_prior_distance_policy"], "canonical_single_view_configured")
        self.assertTrue(response.metadata["offline_single_view_parity"])
        self.assertGreater(response.surface_points_m.shape[0], 0)
        self.assertGreater(response.interior_points_m.shape[0], 0)


class RuntimeShapePriorIntegrationTest(unittest.TestCase):
    def test_masked_delegate_shape_prior_warmup_defaults_off_unless_wrapper_enables(self) -> None:
        args = masked_demo.build_parser().parse_args([])

        self.assertFalse(args.shape_prior_warmup)
        self.assertEqual(args.shape_prior_start_policy, "async-after-first-mask-depth-pair")
        self.assertEqual(args.shape_prior_execution, "remote-worker")
        self.assertEqual(args.shape_prior_endpoint, "tcp://127.0.0.1:7100")
        self.assertEqual(args.shape_prior_timeout_ms, 180000)
        self.assertIsNone(args.shape_prior_profile_json)
        self.assertEqual(args.shape_prior_device, "cuda:0")
        self.assertTrue(args.shape_prior_skip_route_visualizations)

    def test_masked_delegate_shape_prior_warmup_can_be_disabled(self) -> None:
        args = masked_demo.build_parser().parse_args(["--no-shape-prior-warmup"])

        self.assertFalse(args.shape_prior_warmup)

    def test_masked_delegate_accepts_shape_prior_options_from_wrapper(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--shape-prior-warmup",
                "--shape-prior-start-policy",
                "blocking-before-first-output",
                "--shape-prior-execution",
                "remote-worker",
                "--shape-prior-endpoint",
                "tcp://127.0.0.1:7100",
                "--shape-prior-timeout-ms",
                "250000",
                "--shape-prior-profile-json",
                "result/shape_profile.json",
                "--shape-prior-device",
                "cuda:0",
                "--shape-prior-skip-route-visualizations",
            ]
        )

        self.assertTrue(args.shape_prior_warmup)
        self.assertEqual(args.shape_prior_start_policy, "blocking-before-first-output")
        self.assertEqual(args.shape_prior_execution, "remote-worker")
        self.assertEqual(args.shape_prior_endpoint, "tcp://127.0.0.1:7100")
        self.assertEqual(args.shape_prior_timeout_ms, 250000)
        self.assertEqual(args.shape_prior_profile_json, Path("result/shape_profile.json"))
        self.assertEqual(args.shape_prior_device, "cuda:0")
        self.assertTrue(args.shape_prior_skip_route_visualizations)

    def test_masked_pcd_packet_carries_optional_shape_prior_reference_layer(self) -> None:
        packet = masked_demo.MaskedPcdPacket(
            seq=3,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            intrinsics=masked_demo.CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=masked_demo.PipelineTiming(),
            shape_prior_points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
            shape_prior_colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
            shape_prior_status="ready",
            shape_prior_profile={"shape_backend": "sam3d-objects"},
        )

        self.assertEqual(packet.shape_prior_point_count, 1)
        self.assertEqual(packet.point_count, 0)
        self.assertEqual(packet.shape_prior_status, "ready")

    def test_headless_metadata_records_shape_prior_status_and_backend(self) -> None:
        args = masked_demo.build_parser().parse_args(
            [
                "--render-mode",
                "none",
                "--track-mode",
                "none",
                "--pcd-mode",
                "none",
                "--depth-source",
                "realsense",
                "--depth-backend-label",
                "native-realsense",
                "--shape-prior-warmup",
                "--shape-prior-start-policy",
                "async-after-first-mask-depth-pair",
            ]
        )
        demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        demo.runtime = SimpleNamespace(
            serial="s0",
            intrinsics=masked_demo.CameraIntrinsics(100.0, 100.0, 2.0, 2.0),
            k_color=np.eye(3, dtype=np.float32),
        )
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=None,
            start_policy="async-after-first-strict-pair",
        )

        metadata = demo._build_headless_capture_metadata()

        self.assertTrue(metadata["shape_prior_enabled"])
        self.assertEqual(metadata["shape_prior_status"], "pending")
        self.assertEqual(metadata["shape_backend"], "sam3d-objects")
        self.assertEqual(metadata["shape_prior_depth_backend"], "native-realsense")
        self.assertEqual(metadata["shape_prior_depth_source_internal"], "realsense")

    def _runtime_for_snapshot(self) -> masked_demo.RealtimeMaskedEdgeTamPcdDemo:
        args = masked_demo.build_parser().parse_args(
            [
                "--render-mode",
                "none",
                "--track-mode",
                "none",
                "--pcd-mode",
                "none",
                "--depth-source",
                "realsense",
                "--depth-backend-label",
                "native-realsense",
                "--shape-prior-warmup",
            ]
        )
        demo = masked_demo.RealtimeMaskedEdgeTamPcdDemo(args)
        demo.runtime = SimpleNamespace(
            serial="s0",
            intrinsics=masked_demo.CameraIntrinsics(100.0, 100.0, 1.0, 1.0),
            k_color=np.eye(3, dtype=np.float32),
        )
        demo.table_c2w = np.eye(4, dtype=np.float32)
        return demo

    def _pcd_result_for_shape_prior(
        self,
        seq: int = 6,
        *,
        object_mask: np.ndarray | None = None,
        depth_m: object = ...,
    ) -> masked_demo.PcdBuildResult:
        color_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        if object_mask is None:
            object_mask = np.array([[True, False], [False, False]], dtype=bool)
        mask_packet = masked_demo.MaskPacket(
            seq=seq,
            color_bgr=color_bgr,
            depth_source="realsense",
            intrinsics=masked_demo.CameraIntrinsics(100.0, 100.0, 1.0, 1.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            timing=masked_demo.PipelineTiming(),
            controller_mask=np.zeros((2, 2), dtype=bool),
            object_mask=object_mask,
            k_color=np.eye(3, dtype=np.float32),
            source_timestamp_s=1.0,
        )
        pcd_packet = masked_demo.MaskedPcdPacket(
            seq=seq,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=masked_demo.PipelineTiming(),
        )
        resolved_depth_m = np.ones((2, 2), dtype=np.float32) if depth_m is ... else depth_m
        return masked_demo.PcdBuildResult(
            packet=pcd_packet,
            depth_m=resolved_depth_m,
            mask_packet=mask_packet,
        )

    def test_shape_prior_snapshot_uses_selected_depth_mask_rgb_and_table_transform(self) -> None:
        demo = self._runtime_for_snapshot()
        color_bgr = np.zeros((2, 3, 3), dtype=np.uint8)
        color_bgr[..., 0] = 10
        color_bgr[..., 1] = 20
        color_bgr[..., 2] = 30
        object_mask = np.zeros((2, 3), dtype=bool)
        object_mask[1, 1] = True
        mask_packet = masked_demo.MaskPacket(
            seq=5,
            color_bgr=color_bgr,
            depth_source="realsense",
            intrinsics=masked_demo.CameraIntrinsics(100.0, 100.0, 1.0, 1.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            timing=masked_demo.PipelineTiming(),
            controller_mask=np.zeros((2, 3), dtype=bool),
            object_mask=object_mask,
            k_color=np.eye(3, dtype=np.float32),
            source_timestamp_s=12.5,
        )
        packet = masked_demo.MaskedPcdPacket(
            seq=5,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=masked_demo.PipelineTiming(),
        )
        result = masked_demo.PcdBuildResult(
            packet=packet,
            depth_m=np.ones((2, 3), dtype=np.float32),
            mask_packet=mask_packet,
        )

        snapshot = demo._shape_prior_snapshot_from_pcd_result(result)

        self.assertIsNotNone(snapshot)
        assert snapshot is not None
        self.assertEqual(snapshot.seq, 5)
        self.assertEqual(snapshot.source_timestamp_s, 12.5)
        self.assertEqual(snapshot.depth_backend, "native-realsense")
        self.assertEqual(snapshot.depth_source_internal, "realsense")
        np.testing.assert_array_equal(snapshot.rgb_u8[0, 0], np.array([30, 20, 10], dtype=np.uint8))
        np.testing.assert_array_equal(snapshot.object_mask, object_mask)
        np.testing.assert_allclose(snapshot.depth_color_m, np.ones((2, 3), dtype=np.float32))
        np.testing.assert_allclose(snapshot.camera_to_world_c2w, np.eye(4, dtype=np.float32))
        self.assertEqual(snapshot.table_z_m, masked_demo.TABLE_Z_M)
        self.assertEqual(snapshot.table_z_above_direction, "negative")

    def test_after_teardown_policy_defers_worker_request_until_teardown(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0

            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                self.calls += 1
                return warmup.ShapePriorResult(
                    seq=snapshot.seq,
                    status="ready",
                    points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                    source_seq=snapshot.seq,
                    source_timestamp_s=snapshot.source_timestamp_s,
                    metadata={"shape_backend": "sam3d-objects"},
                )

        demo = self._runtime_for_snapshot()
        demo.args.shape_prior_start_policy = "after-teardown"
        client = CountingClient()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=client,
            start_policy="after-teardown",
        )
        color_bgr = np.zeros((2, 2, 3), dtype=np.uint8)
        object_mask = np.array([[True, False], [False, False]], dtype=bool)
        mask_packet = masked_demo.MaskPacket(
            seq=6,
            color_bgr=color_bgr,
            depth_source="realsense",
            intrinsics=masked_demo.CameraIntrinsics(100.0, 100.0, 1.0, 1.0),
            depth_scale_m_per_unit=0.001,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            timing=masked_demo.PipelineTiming(),
            controller_mask=np.zeros((2, 2), dtype=bool),
            object_mask=object_mask,
            k_color=np.eye(3, dtype=np.float32),
            source_timestamp_s=1.0,
        )
        pcd_packet = masked_demo.MaskedPcdPacket(
            seq=6,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            intrinsics=mask_packet.intrinsics,
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=masked_demo.PipelineTiming(),
        )
        result = masked_demo.PcdBuildResult(
            packet=pcd_packet,
            depth_m=np.ones((2, 2), dtype=np.float32),
            mask_packet=mask_packet,
        )

        demo._maybe_start_shape_prior_from_pcd_result(result)
        self.assertEqual(client.calls, 0)
        self.assertEqual(demo.shape_prior_manager.status, "pending")

        demo._run_deferred_shape_prior_after_teardown()

        self.assertEqual(client.calls, 1)
        self.assertEqual(demo.shape_prior_manager.status, "ready")

    def test_mask_depth_policy_submits_from_pcd_result_before_strict_pair(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0
                self.seqs: list[int] = []

            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                self.calls += 1
                self.seqs.append(int(snapshot.seq))
                return warmup.ShapePriorResult(
                    seq=snapshot.seq,
                    status="ready",
                    points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                    source_seq=snapshot.seq,
                    source_timestamp_s=snapshot.source_timestamp_s,
                    metadata={"shape_backend": "sam3d-objects"},
                )

        demo = self._runtime_for_snapshot()
        client = CountingClient()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=client,
            start_policy="async-after-first-mask-depth-pair",
        )

        submitted = demo._maybe_start_shape_prior_from_pcd_result(self._pcd_result_for_shape_prior(seq=8))
        demo.shape_prior_manager.wait(timeout_s=1.0)

        self.assertTrue(submitted)
        self.assertEqual(client.calls, 1)
        self.assertEqual(client.seqs, [8])
        self.assertEqual(demo.shape_prior_manager.status, "ready")
        profile = demo.shape_prior_manager.profile()
        self.assertGreaterEqual(profile["shape_prior_submit_ms"], 0.0)
        self.assertEqual(profile["first_mask_depth_pair_ms"], profile["shape_prior_submit_ms"])

    def test_strict_pair_policy_does_not_submit_from_pcd_result(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0

            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                self.calls += 1
                return warmup.ShapePriorResult(
                    seq=snapshot.seq,
                    status="ready",
                    points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                )

        demo = self._runtime_for_snapshot()
        demo.args.shape_prior_start_policy = "async-after-first-strict-pair"
        client = CountingClient()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=client,
            start_policy="async-after-first-strict-pair",
        )

        submitted = demo._maybe_start_shape_prior_from_pcd_result(self._pcd_result_for_shape_prior(seq=9))

        self.assertFalse(submitted)
        self.assertEqual(client.calls, 0)

    def test_mask_depth_policy_waits_for_later_valid_snapshot(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0
                self.seqs: list[int] = []

            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                self.calls += 1
                self.seqs.append(int(snapshot.seq))
                return warmup.ShapePriorResult(
                    seq=snapshot.seq,
                    status="ready",
                    points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                )

        demo = self._runtime_for_snapshot()
        client = CountingClient()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=client,
            start_policy="async-after-first-mask-depth-pair",
        )

        invalid = self._pcd_result_for_shape_prior(seq=1, object_mask=np.zeros((2, 2), dtype=bool))
        valid = self._pcd_result_for_shape_prior(seq=2)

        self.assertFalse(demo._maybe_start_shape_prior_from_pcd_result(invalid))
        self.assertEqual(client.calls, 0)
        self.assertTrue(demo._maybe_start_shape_prior_from_pcd_result(valid))
        demo.shape_prior_manager.wait(timeout_s=1.0)

        self.assertEqual(client.calls, 1)
        self.assertEqual(client.seqs, [2])

    def test_mask_depth_policy_skips_pcd_result_without_dense_depth(self) -> None:
        class CountingClient:
            def __init__(self) -> None:
                self.calls = 0

            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                self.calls += 1
                return warmup.ShapePriorResult(seq=snapshot.seq, status="ready")

        demo = self._runtime_for_snapshot()
        client = CountingClient()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=client,
            start_policy="async-after-first-mask-depth-pair",
        )

        submitted = demo._maybe_start_shape_prior_from_pcd_result(
            self._pcd_result_for_shape_prior(seq=3, depth_m=None)
        )

        self.assertFalse(submitted)
        self.assertEqual(client.calls, 0)

    def test_packet_with_shape_prior_state_attaches_ready_result_without_changing_pcd_counts(self) -> None:
        class ReadyClient:
            def request_shape_prior(self, snapshot: warmup.ShapePriorSnapshot) -> warmup.ShapePriorResult:
                return warmup.ShapePriorResult(
                    seq=snapshot.seq,
                    status="ready",
                    points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                    colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                    source_seq=snapshot.seq,
                    source_timestamp_s=snapshot.source_timestamp_s,
                    metadata={"shape_backend": "sam3d-objects"},
                )

        demo = self._runtime_for_snapshot()
        demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
            enabled=True,
            client=ReadyClient(),
            start_policy="async-after-first-strict-pair",
        )
        snapshot = warmup.ShapePriorSnapshot(
            seq=1,
            source_timestamp_s=0.5,
            input_source="fake-live",
            depth_backend="native-realsense",
            depth_source_internal="realsense",
            rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
            object_mask=np.array([[True, False], [False, False]], dtype=bool),
            controller_mask=np.zeros((2, 2), dtype=bool),
            depth_color_m=np.ones((2, 2), dtype=np.float32),
            k_color=np.eye(3, dtype=np.float32),
            camera_to_world_c2w=np.eye(4, dtype=np.float32),
        )
        demo.shape_prior_manager.maybe_submit(snapshot)
        demo.shape_prior_manager.wait(timeout_s=1.0)
        packet = masked_demo.MaskedPcdPacket(
            seq=2,
            controller_xyz_m=np.empty((0, 3), dtype=np.float32),
            controller_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            object_xyz_m=np.empty((0, 3), dtype=np.float32),
            object_colors_rgb_u8=np.empty((0, 3), dtype=np.uint8),
            intrinsics=masked_demo.CameraIntrinsics(1.0, 1.0, 0.0, 0.0),
            receive_perf_s=1.0,
            process_done_perf_s=2.0,
            dropped_capture_frames=0,
            dropped_seg_frames=0,
            timing=masked_demo.PipelineTiming(),
        )

        attached = demo._packet_with_shape_prior_state(packet)

        self.assertEqual(attached.shape_prior_status, "ready")
        self.assertEqual(attached.shape_prior_point_count, 1)
        self.assertEqual(attached.point_count, 0)

    def test_headless_writer_saves_shape_prior_artifact_and_updates_metadata(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            writer = masked_demo.HeadlessCaptureWriter(
                Path(tmp),
                metadata={
                    "shape_prior_enabled": True,
                    "shape_prior_status": "pending",
                    "shape_backend": "sam3d-objects",
                },
            )
            result = warmup.ShapePriorResult(
                seq=4,
                status="ready",
                points_m=np.array([[0.0, 0.0, 1.0]], dtype=np.float32),
                colors_rgb_u8=np.array([[150, 150, 150]], dtype=np.uint8),
                surface_points_m=np.array([[0.0, 0.0, -0.01]], dtype=np.float32),
                interior_points_m=np.array([[0.01, 0.0, -0.02]], dtype=np.float32),
                source_seq=4,
                source_timestamp_s=2.5,
                metadata={"shape_prior_total_ms": 10.0},
            )

            writer.write_shape_prior_result(result)

            metadata = json.loads((Path(tmp) / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["shape_prior_status"], "ready")
            self.assertEqual(metadata["shape_prior_source_seq"], 4)
            self.assertEqual(metadata["shape_prior_source_time_s"], 2.5)
            self.assertEqual(metadata["shape_prior_path"], "shape_prior/points.npz")
            payload = np.load(Path(tmp) / metadata["shape_prior_path"])
            np.testing.assert_allclose(payload["points_m"], result.points_m)
            np.testing.assert_array_equal(payload["colors_rgb_u8"], result.colors_rgb_u8)
            np.testing.assert_allclose(payload["surface_points_m"], result.surface_points_m)
            np.testing.assert_allclose(payload["interior_points_m"], result.interior_points_m)

    def test_shape_prior_profile_json_records_fail_soft_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            profile_path = Path(tmp) / "shape_profile.json"
            demo = self._runtime_for_snapshot()
            demo.args.shape_prior_profile_json = profile_path
            demo.shape_prior_manager = warmup.ShapePriorWarmupManager(
                enabled=True,
                client=None,
                start_policy="async-after-first-strict-pair",
            )
            snapshot = warmup.ShapePriorSnapshot(
                seq=9,
                source_timestamp_s=3.0,
                input_source="fake-live",
                depth_backend="native-realsense",
                depth_source_internal="realsense",
                rgb_u8=np.zeros((2, 2, 3), dtype=np.uint8),
                object_mask=np.array([[True, False], [False, False]], dtype=bool),
                controller_mask=np.zeros((2, 2), dtype=bool),
                depth_color_m=np.ones((2, 2), dtype=np.float32),
                k_color=np.eye(3, dtype=np.float32),
                camera_to_world_c2w=np.eye(4, dtype=np.float32),
            )

            demo.shape_prior_manager.maybe_submit(snapshot)
            demo._write_shape_prior_profile_json()

            payload = json.loads(profile_path.read_text(encoding="utf-8"))
            self.assertEqual(payload["shape_prior_status"], "failed")
            self.assertEqual(payload["shape_prior_source_seq"], 9)
            self.assertEqual(payload["input_source"], "fake-live")
            self.assertEqual(payload["depth_backend"], "native-realsense")


class SingleViewShapeAlignmentTest(unittest.TestCase):
    def test_alignment_config_defaults_to_negative_table_z_direction(self) -> None:
        self.assertEqual(ShapeAlignmentConfig().above_direction, "negative")

    def test_align_canonical_shape_to_observation_recovers_scale_and_translation(self) -> None:
        canonical = np.array(
            [
                [0.0, 0.0, -0.02],
                [0.02, 0.0, -0.02],
                [0.0, 0.02, -0.03],
                [0.02, 0.02, -0.04],
            ],
            dtype=np.float32,
        )
        observation = canonical * np.float32(2.0) + np.array([0.10, -0.05, -0.01], dtype=np.float32)

        result = align_canonical_shape_to_observation(
            canonical,
            observation,
            config=ShapeAlignmentConfig(max_centroid_drift_m=1e-5),
        )

        self.assertTrue(result.valid, result.validation)
        self.assertAlmostEqual(result.scale, 2.0, places=5)
        np.testing.assert_allclose(result.aligned_points_m, observation, atol=1e-5)
        self.assertLessEqual(result.validation["centroid_drift_m"], 1e-5)

    def test_alignment_invalid_when_ground_fraction_is_too_high(self) -> None:
        canonical = np.array(
            [[0.0, 0.0, 0.0], [0.01, 0.0, 0.0], [0.0, 0.01, 0.0], [0.01, 0.01, 0.0]],
            dtype=np.float32,
        )
        observation = canonical.copy()

        result = align_canonical_shape_to_observation(
            canonical,
            observation,
            config=ShapeAlignmentConfig(max_ground_z_fraction=0.25, ground_z_epsilon_m=0.001),
        )

        self.assertFalse(result.valid)
        self.assertGreater(result.validation["ground_z_fraction"], 0.25)


if __name__ == "__main__":
    unittest.main()
