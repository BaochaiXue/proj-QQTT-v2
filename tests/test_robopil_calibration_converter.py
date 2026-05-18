from __future__ import annotations

import json
import pickle
from pathlib import Path
import tempfile
import unittest

import numpy as np

from scripts.convert_robopil_cam_params_to_qqtt_calibrate import (
    convert_cam_params_to_c2ws,
    write_qqtt_calibration_from_robopil,
)


class RobopilCalibrationConverterTest(unittest.TestCase):
    def test_converts_board_to_camera_extrinsic_to_qqtt_c2w(self) -> None:
        board_to_camera_a = np.eye(4)
        board_to_camera_a[:3, 3] = [1.0, 2.0, 3.0]
        board_to_camera_b = np.eye(4)
        board_to_camera_b[:3, 3] = [-0.1, 0.2, 0.3]

        c2ws = convert_cam_params_to_c2ws(
            {
                "cam_a": {"extrinsic": board_to_camera_a},
                "cam_b": {"extrinsic": board_to_camera_b},
            },
            ["cam_b", "cam_a"],
        )

        self.assertEqual(len(c2ws), 2)
        np.testing.assert_allclose(c2ws[0], np.linalg.inv(board_to_camera_b))
        np.testing.assert_allclose(c2ws[1], np.linalg.inv(board_to_camera_a))

    def test_writes_qqtt_calibrate_pkl_and_metadata_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp = Path(tmp_dir)
            cam_params_path = tmp / "cam_params.pkl"
            output_path = tmp / "calibrate.pkl"
            board_to_camera = np.eye(4)
            board_to_camera[:3, 3] = [0.5, 0.0, 1.0]
            with cam_params_path.open("wb") as handle:
                pickle.dump({"239": {"extrinsic": board_to_camera}}, handle)

            _, sidecar_path = write_qqtt_calibration_from_robopil(
                cam_params_path=cam_params_path,
                output_calibrate_path=output_path,
                serials=["239"],
                overwrite=True,
            )

            with output_path.open("rb") as handle:
                c2ws = pickle.load(handle)
            metadata = json.loads(sidecar_path.read_text(encoding="utf-8"))

            self.assertEqual(len(c2ws), 1)
            np.testing.assert_allclose(c2ws[0], np.linalg.inv(board_to_camera))
            self.assertEqual(metadata["serial_numbers"], ["239"])
            self.assertEqual(metadata["transform_convention"], "camera_to_world_c2w")
            self.assertEqual(metadata["compatibility_contract"], "qqtt_calibrate_pkl_c2w_list_v1")
            self.assertEqual(metadata["source_format"], "robopil_cam_params")
            self.assertEqual(metadata["world_frame_convention"], "robopil-rx180")
            self.assertFalse(metadata["distortion_used"])


if __name__ == "__main__":
    unittest.main()
