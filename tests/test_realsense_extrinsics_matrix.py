from __future__ import annotations

from importlib import import_module
from types import SimpleNamespace
import sys
import unittest

import numpy as np


def _fake_extrinsics() -> SimpleNamespace:
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    return SimpleNamespace(
        rotation=[
            float(rotation[0, 0]),
            float(rotation[1, 0]),
            float(rotation[2, 0]),
            float(rotation[0, 1]),
            float(rotation[1, 1]),
            float(rotation[2, 1]),
            float(rotation[0, 2]),
            float(rotation[1, 2]),
            float(rotation[2, 2]),
        ],
        translation=[0.1, -0.2, 0.3],
    )


def _expected_matrix() -> np.ndarray:
    return np.array(
        [
            [0.0, -1.0, 0.0, 0.1],
            [1.0, 0.0, 0.0, -0.2],
            [0.0, 0.0, 1.0, 0.3],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


class RealsenseExtrinsicsMatrixTests(unittest.TestCase):
    def assert_matrix_matches_realsense_column_major(
        self,
        value: object,
    ) -> None:
        actual = np.asarray(value, dtype=np.float32)
        expected = _expected_matrix()
        self.assertEqual((4, 4), actual.shape)
        np.testing.assert_allclose(actual, expected)
        self.assertFalse(np.allclose(actual[:3, :3], expected[:3, :3].T))
        np.testing.assert_allclose(actual[:3, 3], expected[:3, 3])
        np.testing.assert_allclose(actual[3], expected[3])

    def test_demo_v51_realsense_extrinsics_use_column_major_rotation(self) -> None:
        from demo_v5_1.utils.camera import rs_extrinsics_to_matrix

        self.assert_matrix_matches_realsense_column_major(
            rs_extrinsics_to_matrix(_fake_extrinsics())
        )

    def test_shared_realsense_metadata_extrinsics_use_column_major_rotation(
        self,
    ) -> None:
        sys.modules.setdefault(
            "pyrealsense2",
            SimpleNamespace(option=SimpleNamespace()),
        )
        module = import_module("qqtt.env.camera.realsense.single_realsense")

        self.assert_matrix_matches_realsense_column_major(
            module.extrinsics_to_matrix(_fake_extrinsics())
        )

    def test_d455_probe_extrinsics_use_column_major_rotation(self) -> None:
        module = import_module(
            "scripts.harness.diagnostics.hardware.probe_d455_ir_pair"
        )

        self.assert_matrix_matches_realsense_column_major(
            module.extrinsics_to_matrix(_fake_extrinsics())
        )
