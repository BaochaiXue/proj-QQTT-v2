from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from qqtt.env.camera.camera_system import CameraSystem


class _FakeRealsense:
    def __init__(self) -> None:
        self.call_count = 0
        self.stop_called = False

    def get(self):
        self.call_count += 1
        stalled_step = 0
        progressing_step = self.call_count - 1
        return [
            {"timestamp": np.asarray(float(self.call_count)), "step_idx": np.asarray(stalled_step)},
            {"timestamp": np.asarray(float(self.call_count)), "step_idx": np.asarray(progressing_step)},
            {"timestamp": np.asarray(float(self.call_count)), "step_idx": np.asarray(progressing_step)},
        ]

    def stop(self) -> None:
        self.stop_called = True


class CameraSystemPartialStallSmokeTest(unittest.TestCase):
    def test_record_fails_when_one_camera_stops_progressing(self) -> None:
        fake_realsense = _FakeRealsense()
        camera_system = CameraSystem.__new__(CameraSystem)
        camera_system.num_cam = 3
        camera_system.streams_present = []
        camera_system.serial_numbers = ["a", "b", "c"]
        camera_system.listener = None
        camera_system.realsense = fake_realsense
        camera_system.end = False
        camera_system.recording = False
        camera_system.build_recording_metadata = lambda: {"recording": {}}

        clock = {"tick": 0.0}

        def _fake_time() -> float:
            value = clock["tick"]
            clock["tick"] += 1.0
            return value

        with tempfile.TemporaryDirectory() as tmp_dir, patch("qqtt.env.camera.camera_system.time.time", side_effect=_fake_time):
            with self.assertRaisesRegex(RuntimeError, "Recording partially stalled"):
                camera_system.record(str(Path(tmp_dir) / "record_case"), max_frames=30)

        self.assertTrue(fake_realsense.stop_called)


if __name__ == "__main__":
    unittest.main()
