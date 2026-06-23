from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
import tempfile
import time
import unittest
from unittest import mock

import numpy as np

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
        self.assertEqual(args.shape_prior_start_policy, "async-after-first-strict-pair")
        self.assertEqual(args.shape_prior_execution, "remote-worker")
        self.assertEqual(args.shape_prior_endpoint, "tcp://127.0.0.1:7100")
        self.assertEqual(args.shape_prior_device, "cuda:0")
        self.assertTrue(args.shape_prior_skip_route_visualizations)
        self.assertTrue(contract["shape_prior_warmup_enabled"])
        self.assertEqual(contract["shape_prior_status"], "pending")
        self.assertEqual(contract["shape_prior_start_policy"], "async-after-first-strict-pair")
        self.assertEqual(contract["shape_prior_execution"], "remote-worker")
        self.assertEqual(contract["shape_backend"], "sam3d-objects")
        self.assertEqual(contract["profile_summary_fields"]["shape_prior_status"], "pending")
        self.assertEqual(_option_value(delegate, "--shape-prior-endpoint"), "tcp://127.0.0.1:7100")
        self.assertIn("--shape-prior-warmup", delegate)
        self.assertIn("--shape-prior-skip-route-visualizations", delegate)

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

    def test_protocol_roundtrip_preserves_snapshot_arrays_and_metadata(self) -> None:
        snapshot = self._snapshot()

        parts = build_shape_prior_request_parts(snapshot=snapshot, request_id="req-7")
        request = parse_shape_prior_request_parts(parts)

        self.assertEqual(request.metadata["protocol"], PROTOCOL_NAME)
        self.assertEqual(request.metadata["request_id"], "req-7")
        self.assertEqual(request.metadata["seq"], 7)
        self.assertEqual(request.metadata["depth_backend"], "ir-ffs")
        np.testing.assert_array_equal(request.rgb_u8, snapshot.rgb_u8)
        np.testing.assert_array_equal(request.object_mask, snapshot.object_mask)
        np.testing.assert_allclose(request.depth_color_m, snapshot.depth_color_m)

        response_parts = build_shape_prior_response_parts(
            request_id="req-7",
            seq=7,
            status="ready",
            points_m=np.array([[0.0, 0.0, 0.0], [0.01, 0.0, 0.0]], dtype=np.float32),
            colors_rgb_u8=np.full((2, 3), 150, dtype=np.uint8),
            metadata={"single_view_alignment_ms": 3.5},
        )
        response = parse_shape_prior_response_parts(response_parts)

        self.assertEqual(response.metadata["status"], "ready")
        self.assertEqual(response.metadata["seq"], 7)
        self.assertEqual(response.points_m.shape, (2, 3))
        self.assertEqual(response.colors_rgb_u8.shape, (2, 3))
        self.assertEqual(response.metadata["single_view_alignment_ms"], 3.5)

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


class SingleViewShapeAlignmentTest(unittest.TestCase):
    def test_align_canonical_shape_to_observation_recovers_scale_and_translation(self) -> None:
        canonical = np.array(
            [
                [0.0, 0.0, 0.02],
                [0.02, 0.0, 0.02],
                [0.0, 0.02, 0.03],
                [0.02, 0.02, 0.04],
            ],
            dtype=np.float32,
        )
        observation = canonical * np.float32(2.0) + np.array([0.10, -0.05, 0.01], dtype=np.float32)

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
