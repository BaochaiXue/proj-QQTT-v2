from __future__ import annotations

import json
import pickle
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from data_process.aligned_case_metadata import write_split_aligned_metadata
from data_process.visualization.io_case import load_case_metadata
from data_process.visualization.pointcloud_compare import load_case_frame_cloud


def make_aligned_case(case_dir: Path, *, include_depth_ffs: bool = False) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    for stream in ["color", "depth"] + (["depth_ffs"] if include_depth_ffs else []):
        for cam in range(3):
            (case_dir / stream / str(cam)).mkdir(parents=True, exist_ok=True)

    color = np.zeros((2, 2, 3), dtype=np.uint8)
    color[..., 1] = 255
    depth = np.array([[1000, 1000], [1000, 0]], dtype=np.uint16)
    depth_ffs = np.array([[900, 900], [900, 0]], dtype=np.uint16)

    for cam in range(3):
        cv2.imwrite(str(case_dir / "color" / str(cam) / "0.png"), color)
        np.save(case_dir / "depth" / str(cam) / "0.npy", depth)
        if include_depth_ffs:
            np.save(case_dir / "depth_ffs" / str(cam) / "0.npy", depth_ffs)

    transforms = [
        np.eye(4, dtype=np.float32),
        np.array([[1, 0, 0, 1], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[1, 0, 0, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
    ]
    with (case_dir / "calibrate.pkl").open("wb") as handle:
        pickle.dump(transforms, handle)

    K = [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] for _ in range(3)]
    metadata = {
        "serial_numbers": ["239222300433", "239222300781", "239222303506"],
        "calibration_reference_serials": ["239222300433", "239222300781", "239222303506"],
        "fps": 30,
        "WH": [2, 2],
        "frame_num": 1,
        "start_step": 0,
        "end_step": 0,
        "depth_scale_m_per_unit": [0.001, 0.001, 0.001],
        "intrinsics": K,
        "K_color": K,
    }
    write_split_aligned_metadata(case_dir, metadata)


class PointcloudFusionSmokeTest(unittest.TestCase):
    def test_fuses_three_cameras_into_world_frame(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            case_dir = Path(tmp_dir) / "case"
            make_aligned_case(case_dir, include_depth_ffs=True)

            metadata = load_case_metadata(case_dir)
            points, colors, stats = load_case_frame_cloud(
                case_dir=case_dir,
                metadata=metadata,
                frame_idx=0,
                depth_source="realsense",
                use_float_ffs_depth_when_available=False,
                voxel_size=None,
                max_points_per_camera=None,
                depth_min_m=0.1,
                depth_max_m=2.0,
            )
            self.assertGreater(len(points), 0)
            self.assertEqual(colors.shape[1], 3)
            self.assertEqual(len(stats["per_camera"]), 3)
            self.assertTrue(np.any(points[:, 0] > 0.5))
            self.assertTrue(np.any(points[:, 1] > 0.5))


if __name__ == "__main__":
    unittest.main()
