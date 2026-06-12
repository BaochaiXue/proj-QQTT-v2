from __future__ import annotations

import unittest

import cameras_calibrate
import record_data
import record_data_realtime_align
from qqtt.env.camera.defaults import DEFAULT_NUM_CAM


class SingleCameraDefaultsSmokeTest(unittest.TestCase):
    def test_shared_default_camera_count_is_one(self) -> None:
        self.assertEqual(DEFAULT_NUM_CAM, 1)

    def test_camera_entrypoints_default_to_one_camera(self) -> None:
        for parser in (
            cameras_calibrate.build_parser(),
            record_data.build_parser(),
            record_data_realtime_align.build_parser(),
        ):
            with self.subTest(description=parser.description):
                self.assertEqual(parser.parse_args([]).num_cam, 1)

    def test_entrypoint_descriptions_name_single_camera_branch(self) -> None:
        self.assertIn("single-camera", cameras_calibrate.build_parser().description)
        self.assertIn("single-camera", record_data.build_parser().description)
        self.assertIn("single-camera", record_data_realtime_align.build_parser().description)


if __name__ == "__main__":
    unittest.main()
