from __future__ import annotations

import tempfile
from pathlib import Path
import unittest

import cv2
import numpy as np

from data_process.visualization.io_artifacts import write_gif, write_image, write_json, write_ply_ascii, write_video


class ArtifactWriterSmokeTest(unittest.TestCase):
    def test_artifact_writers_emit_expected_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            frame_paths = []
            for frame_idx in range(2):
                image = np.full((40, 60, 3), 30 + frame_idx * 40, dtype=np.uint8)
                frame_path = root / f"frame_{frame_idx:03d}.png"
                write_image(frame_path, image)
                frame_paths.append(frame_path)

            write_json(root / "payload.json", {"ok": True})
            write_ply_ascii(
                root / "cloud.ply",
                np.array([[0.0, 0.0, 0.0], [0.1, 0.2, 0.3]], dtype=np.float32),
                np.array([[0, 0, 255], [0, 255, 0]], dtype=np.uint8),
            )
            write_video(root / "preview.mp4", frame_paths, fps=4)
            write_gif(root / "preview.gif", frame_paths, fps=4)

            self.assertTrue((root / "payload.json").exists())
            self.assertTrue((root / "cloud.ply").exists())
            self.assertTrue((root / "preview.mp4").exists())
            self.assertTrue((root / "preview.gif").exists())
            self.assertIsNotNone(cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR))
