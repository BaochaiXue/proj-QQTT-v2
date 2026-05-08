from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from demo_v0_2 import async_remote_ffs_triplet_client as client


def _write_source_case(root: Path) -> None:
    metadata = {
        "WH": [4, 3],
        "fps": 30,
        "serial_numbers": ["s0", "s1", "s2"],
        "K_ir_left": [np.eye(3, dtype=np.float32).tolist() for _ in range(3)],
        "K_color": [np.eye(3, dtype=np.float32).tolist() for _ in range(3)],
        "T_ir_left_to_color": [np.eye(4, dtype=np.float32).tolist() for _ in range(3)],
        "ir_baseline_m": [0.095, 0.096, 0.097],
    }
    (root / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")
    for cam_idx in range(3):
        for side in ("ir_left", "ir_right"):
            path = root / side / str(cam_idx)
            path.mkdir(parents=True, exist_ok=True)
            for frame_id in (10, 12):
                image = np.full((3, 4), frame_id + cam_idx, dtype=np.uint8)
                Image.fromarray(image).save(path / f"{frame_id}.png")


class DemoV02ClientReplayTest(unittest.TestCase):
    def test_prepare_replay_from_case_cycles_real_ir_triplets(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_case = tmp_path / "source_case"
            replay_dir = tmp_path / "replay"
            source_case.mkdir()
            _write_source_case(source_case)
            args = client.build_parser().parse_args(
                [
                    "--mode",
                    "prepare-replay-from-case",
                    "--source-case",
                    str(source_case),
                    "--replay-dir",
                    str(replay_dir),
                    "--replay-frame-count",
                    "5",
                    "--target-kit-fps",
                    "15",
                ]
            )

            summary = client.prepare_replay_from_case(args)

            self.assertEqual(summary["replay_frame_count"], 5.0)
            metadata = json.loads((replay_dir / "metadata.json").read_text(encoding="utf-8"))
            self.assertEqual(metadata["frame_count"], 5)
            self.assertEqual(metadata["source_unique_frame_count"], 2)
            self.assertEqual(metadata["camera_fps"], 15)
            for cam_idx in range(3):
                self.assertTrue((replay_dir / f"cam{cam_idx}" / "left" / "000004.png").is_file())
                self.assertTrue((replay_dir / f"cam{cam_idx}" / "right" / "000004.png").is_file())

    def test_add_distribution_reports_min_mean_p99_and_max(self) -> None:
        summary: dict[str, float | str] = {}

        client._add_distribution(summary, "latency_ms", [1.0, 2.0, 3.0, 100.0])

        self.assertEqual(summary["latency_ms_min"], 1.0)
        self.assertEqual(summary["latency_ms_max"], 100.0)
        self.assertAlmostEqual(float(summary["latency_ms_mean"]), 26.5)
        self.assertGreater(float(summary["latency_ms_p99"]), 90.0)


if __name__ == "__main__":
    unittest.main()
