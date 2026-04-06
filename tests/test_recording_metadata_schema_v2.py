from __future__ import annotations

import unittest

from qqtt.env.camera.recording_metadata import build_recording_metadata


class RecordingMetadataSchemaV2Test(unittest.TestCase):
    def test_builds_expected_schema(self) -> None:
        stream_metadata = [
            {
                "model_name": "Intel RealSense D455",
                "product_line": "D400",
                "K_color": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
                "K_ir_left": [[2, 0, 0], [0, 2, 0], [0, 0, 1]],
                "K_ir_right": [[3, 0, 0], [0, 3, 0], [0, 0, 1]],
                "T_ir_left_to_right": [[1, 0, 0, -0.095], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "T_ir_left_to_color": [[1, 0, 0, -0.05], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]],
                "ir_baseline_m": 0.095,
                "depth_scale_m_per_unit": 0.001,
                "emitter_actual": 1.0,
                "exposure": 156.0,
                "gain": 60.0,
                "white_balance": 3800.0,
            }
        ]
        metadata = build_recording_metadata(
            serial_numbers=["239222303506"],
            capture_mode="both_eval",
            streams_present=["color", "depth", "ir_left", "ir_right"],
            fps=30,
            WH=(848, 480),
            emitter_request="on",
            stream_metadata=stream_metadata,
        )

        for key in (
            "schema_version",
            "serial_numbers",
            "logical_camera_names",
            "capture_mode",
            "streams_present",
            "camera_model_per_camera",
            "product_line_per_camera",
            "intrinsics",
            "K_color",
            "K_ir_left",
            "K_ir_right",
            "T_ir_left_to_right",
            "T_ir_left_to_color",
            "ir_baseline_m",
            "depth_scale_m_per_unit",
            "depth_encoding",
            "alignment_target",
            "depth_coordinate_frame",
            "emitter_request",
            "emitter_actual",
            "recording",
        ):
            self.assertIn(key, metadata)
        self.assertEqual(metadata["capture_mode"], "both_eval")
        self.assertEqual(metadata["intrinsics"], metadata["K_color"])


if __name__ == "__main__":
    unittest.main()
