from __future__ import annotations

import unittest

import cameras_calibrate
import record_data
import record_data_realtime_align
from qqtt.env.camera.defaults import (
    DEFAULT_COLOR_EXPOSURE_OVERRIDES,
    DEFAULT_COLOR_GAIN_OVERRIDES,
    DEFAULT_EXPOSURE,
    DEFAULT_GAIN,
    resolve_per_camera_control_values,
)


class CameraColorControlsTest(unittest.TestCase):
    def test_current_lab_rig_exposure_overrides_are_balanced(self) -> None:
        self.assertEqual(
            DEFAULT_COLOR_EXPOSURE_OVERRIDES,
            {
                "239222300412": 156.0,
                "239222300781": 70.0,
                "239222303506": 180.0,
            },
        )
        self.assertEqual(DEFAULT_COLOR_GAIN_OVERRIDES, {})

    def test_resolves_per_serial_overrides(self) -> None:
        values = resolve_per_camera_control_values(
            DEFAULT_EXPOSURE,
            overrides=DEFAULT_COLOR_EXPOSURE_OVERRIDES,
            serial_numbers=["239222300412", "239222300781", "239222303506"],
            label="exposure",
        )
        self.assertEqual(values, [156.0, 70.0, 180.0])

    def test_accepts_explicit_per_camera_values(self) -> None:
        values = resolve_per_camera_control_values(
            [100, 110, 120],
            overrides=DEFAULT_COLOR_EXPOSURE_OVERRIDES,
            serial_numbers=["cam0", "cam1", "cam2"],
            label="exposure",
        )
        self.assertEqual(values, [100.0, 110.0, 120.0])

    def test_rejects_mismatched_explicit_values(self) -> None:
        with self.assertRaisesRegex(ValueError, "gain list length"):
            resolve_per_camera_control_values(
                [60.0, 70.0],
                overrides={},
                serial_numbers=["cam0", "cam1", "cam2"],
                label="gain",
            )

    def test_camera_entrypoints_expose_base_color_controls(self) -> None:
        for parser in (
            cameras_calibrate.build_parser(),
            record_data.build_parser(),
            record_data_realtime_align.build_parser(),
        ):
            args = parser.parse_args([])
            self.assertEqual(args.exposure, DEFAULT_EXPOSURE)
            self.assertEqual(args.gain, DEFAULT_GAIN)


if __name__ == "__main__":
    unittest.main()
