from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from scripts.demo_v0_3.prepare_ir_triplet_100kits import main, prepare_ir_triplet_100kits, build_parser


def _write_v02_replay(root: Path, *, frame_count: int, width: int = 4, height: int = 3) -> None:
    cameras = []
    for camera_idx in range(3):
        cameras.append(
            {
                "camera_idx": camera_idx,
                "serial": f"serial{camera_idx}",
                "width": width,
                "height": height,
                "K_ir_left": np.eye(3, dtype=np.float32).tolist(),
                "K_color": np.eye(3, dtype=np.float32).tolist(),
                "T_ir_left_to_color": np.eye(4, dtype=np.float32).tolist(),
                "baseline_m": 0.055,
            }
        )
        for side in ("left", "right"):
            side_dir = root / f"cam{camera_idx}" / side
            side_dir.mkdir(parents=True, exist_ok=True)
            for frame_idx in range(frame_count):
                value = camera_idx * 32 + frame_idx * 4 + (1 if side == "right" else 0)
                image = np.full((height, width), value, dtype=np.uint8)
                Image.fromarray(image).save(side_dir / f"{frame_idx:06d}.png")
    metadata = {
        "mode": "triplet-record",
        "profile": f"{width}x{height}",
        "camera_fps": 30,
        "frame_count": frame_count,
        "cameras": cameras,
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


class DemoV03PrepareIrTripletsSmokeTest(unittest.TestCase):
    def test_prepares_100kit_shape_with_cycle_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "source"
            out = base / "out"
            _write_v02_replay(src, frame_count=2)

            args = build_parser().parse_args(
                [
                    "--src-replay-dir",
                    str(src),
                    "--out-replay-dir",
                    str(out),
                    "--num-kits",
                    "5",
                    "--camera-count",
                    "3",
                    "--width",
                    "4",
                    "--height",
                    "3",
                    "--capture-kit-fps",
                    "15",
                    "--allow-cycle-if-needed",
                    "--write-manifest",
                ]
            )
            manifest = prepare_ir_triplet_100kits(args)

            self.assertEqual(manifest["source_kit_count"], 2)
            self.assertEqual(manifest["output_kit_count"], 5)
            self.assertEqual(manifest["unique_source_kit_count"], 2)
            self.assertTrue(manifest["cycled"])
            self.assertEqual(manifest["source_kit_indices"], [0, 1, 0, 1, 0])
            self.assertTrue((out / "manifest_v03_100kits.json").is_file())
            self.assertTrue((out / "kits.jsonl").is_file())
            self.assertTrue((out / "metadata.json").is_file())

            kits = [
                json.loads(line)
                for line in (out / "kits.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            self.assertEqual(len(kits), 5)
            self.assertEqual(kits[3]["kit_idx"], 3)
            self.assertEqual(kits[3]["source_kit_idx"], 1)
            self.assertEqual(len(kits[3]["cameras"]), 3)
            self.assertTrue((out / kits[3]["cameras"][2]["left_ir_path"]).is_file())
            copied_pngs = list(out.glob("cam*/**/*.png"))
            self.assertEqual(len(copied_pngs), 5 * 3 * 2)

    def test_rejects_short_source_without_cycle_flag(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            src = base / "source"
            out = base / "out"
            _write_v02_replay(src, frame_count=2)

            with self.assertRaises(SystemExit) as raised:
                main(
                    [
                        "--src-replay-dir",
                        str(src),
                        "--out-replay-dir",
                        str(out),
                        "--num-kits",
                        "5",
                        "--camera-count",
                        "3",
                        "--width",
                        "4",
                        "--height",
                        "3",
                    ]
                )
            self.assertEqual(raised.exception.code, 2)


if __name__ == "__main__":
    unittest.main()
