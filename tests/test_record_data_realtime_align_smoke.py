from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import tempfile
import unittest
from unittest import mock

import numpy as np

from data_process.aligned_case_metadata import LEGACY_ALIGNED_METADATA_KEYS
from record_data_realtime_align import run_realtime_export


class FakeClock:
    def __init__(self) -> None:
        self.value = 0.0

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += float(seconds)


class FakeRealtimeCameraSystem:
    def __init__(self, *, clock: FakeClock, **kwargs) -> None:
        self.clock = clock
        self.kwargs = kwargs
        self.serial_numbers = ["cam0", "cam2", "cam1"]
        self.observations = [
            self._observation(step_tuple=(1, 1, 1), timestamps=(1.00, 1.01, 1.02), value=10),
            self._observation(step_tuple=(1, 1, 1), timestamps=(1.00, 1.01, 1.02), value=20),
            self._observation(step_tuple=(2, 2, 2), timestamps=(2.00, 2.20, 2.40), value=30),
            self._observation(step_tuple=(3, 3, 3), timestamps=(3.00, 3.01, 3.02), value=40),
        ]
        self.index = 0
        self.realsense = self
        self.stopped = False
        self.shm_manager = None

    @staticmethod
    def _observation(*, step_tuple: tuple[int, int, int], timestamps: tuple[float, float, float], value: int):
        result = {}
        for camera_idx in range(3):
            result[camera_idx] = {
                "color": np.full((2, 2, 3), value + camera_idx, dtype=np.uint8),
                "depth": np.full((2, 2), 100 + value + camera_idx, dtype=np.uint16),
                "timestamp": timestamps[camera_idx],
                "step_idx": step_tuple[camera_idx],
            }
        return result

    def build_recording_metadata(self):
        return {
            "intrinsics": [
                [[600.0, 0.0, 320.0], [0.0, 600.0, 240.0], [0.0, 0.0, 1.0]],
                [[602.0, 0.0, 322.0], [0.0, 602.0, 242.0], [0.0, 0.0, 1.0]],
                [[601.0, 0.0, 321.0], [0.0, 601.0, 241.0], [0.0, 0.0, 1.0]],
            ],
            "serial_numbers": self.serial_numbers,
            "calibration_reference_serials": ["cam0", "cam1", "cam2"],
            "fps": 30,
            "WH": [2, 2],
        }

    def get_observation(self):
        self.clock.advance(0.11)
        if self.index < len(self.observations):
            observation = self.observations[self.index]
            self.index += 1
            return observation
        return self.observations[-1]

    def stop(self):
        self.stopped = True


class RecordDataRealtimeAlignSmokeTest(unittest.TestCase):
    def test_realtime_export_writes_formal_case_and_external_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            calibrate_path = root / "calibrate.pkl"
            c2ws = []
            for translation_x in (0.0, 1.0, 2.0):
                transform = np.eye(4, dtype=np.float32)
                transform[0, 3] = translation_x
                c2ws.append(transform)
            with calibrate_path.open("wb") as handle:
                pickle.dump(c2ws, handle)

            clock = FakeClock()

            def factory(**kwargs):
                return FakeRealtimeCameraSystem(clock=clock, **kwargs)

            args = argparse.Namespace(
                case_name="native_rt_case",
                output_root=root / "different_types_real_time",
                calibrate_path=calibrate_path,
                width=2,
                height=2,
                fps=30,
                num_cam=3,
                serials=None,
                duration_s=0.5,
                stats_log_interval_s=0.1,
                sync_tolerance_ms=50.0,
                overwrite=False,
            )

            def fake_write_png(path: Path, image: np.ndarray) -> None:
                path.write_bytes(np.asarray(image, dtype=np.uint8).tobytes())

            with mock.patch("record_data_realtime_align._atomic_write_png", side_effect=fake_write_png):
                result = run_realtime_export(args, camera_system_factory=factory, monotonic_fn=clock)

            case_dir = Path(result["case_dir"])
            self.assertEqual(sorted(item.name for item in case_dir.iterdir()), ["calibrate.pkl", "color", "depth", "metadata.json"])
            self.assertFalse((case_dir / "metadata_ext.json").exists())
            self.assertFalse(any((case_dir / "color").glob("*.mp4")))
            self.assertEqual(sorted(item.name for item in (case_dir / "color").iterdir()), ["0", "1", "2"])
            self.assertEqual(sorted(item.name for item in (case_dir / "depth").iterdir()), ["0", "1", "2"])

            metadata = json.loads((case_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(set(metadata), set(LEGACY_ALIGNED_METADATA_KEYS))
            self.assertEqual(metadata["frame_num"], 2)
            self.assertEqual(metadata["start_step"], 0)
            self.assertEqual(metadata["end_step"], 1)
            self.assertEqual(metadata["serial_numbers"], ["cam0", "cam2", "cam1"])

            with (case_dir / "calibrate.pkl").open("rb") as handle:
                normalized_c2ws = pickle.load(handle)
            self.assertEqual([float(item[0][3]) for item in normalized_c2ws], [0.0, 2.0, 1.0])

            for camera_idx in range(3):
                self.assertTrue((case_dir / "color" / str(camera_idx) / "0.png").is_file())
                self.assertTrue((case_dir / "color" / str(camera_idx) / "1.png").is_file())
                self.assertTrue((case_dir / "depth" / str(camera_idx) / "0.npy").is_file())
                self.assertTrue((case_dir / "depth" / str(camera_idx) / "1.npy").is_file())
                self.assertFalse((case_dir / "depth" / str(camera_idx) / "2.npy").exists())

            depth0 = np.load(case_dir / "depth" / "0" / "0.npy")
            depth1 = np.load(case_dir / "depth" / "0" / "1.npy")
            self.assertEqual(int(depth0[0, 0]), 110)
            self.assertEqual(int(depth1[0, 0]), 140)

            stats_jsonl_path = Path(result["stats_jsonl_path"])
            summary_path = Path(result["summary_path"])
            self.assertTrue(stats_jsonl_path.is_file())
            self.assertTrue(summary_path.is_file())
            self.assertEqual(stats_jsonl_path.parent, root / "different_types_real_time" / "_logs")
            self.assertFalse((case_dir / "_logs").exists())

            rows = [json.loads(line) for line in stats_jsonl_path.read_text(encoding="utf-8").splitlines()]
            self.assertGreaterEqual(len(rows), 1)
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
            self.assertEqual(summary["frame_num"], 2)
            self.assertEqual(summary["sync_reject_count"], 1)
            self.assertGreaterEqual(summary["duplicate_skip_count"], 1)
            self.assertGreater(summary["aligned_frame_set_fps_total"], 0.0)


if __name__ == "__main__":
    unittest.main()
