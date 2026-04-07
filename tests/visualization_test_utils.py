from __future__ import annotations

import json
import pickle
from pathlib import Path

import cv2
import numpy as np


def make_visualization_case(
    case_dir: Path,
    *,
    include_depth_ffs: bool = False,
    include_depth_ffs_float_m: bool = False,
    frame_num: int = 2,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    streams = ["color", "depth"] + (["depth_ffs"] if include_depth_ffs else []) + (["depth_ffs_float_m"] if include_depth_ffs_float_m else [])
    for stream in streams:
        for cam in range(3):
            (case_dir / stream / str(cam)).mkdir(parents=True, exist_ok=True)

    height, width = 8, 10
    yy, xx = np.indices((height, width), dtype=np.float32)
    transforms = [
        np.eye(4, dtype=np.float32),
        np.array([[1, 0, 0, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0.25], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
    ]
    with (case_dir / "calibrate.pkl").open("wb") as handle:
        pickle.dump(transforms, handle)

    K = [[[6.0, 0.0, 4.5], [0.0, 6.0, 3.5], [0.0, 0.0, 1.0]] for _ in range(3)]
    for frame_idx in range(frame_num):
        for cam in range(3):
            color = np.zeros((height, width, 3), dtype=np.uint8)
            color[..., 0] = np.clip(xx * 18 + cam * 20, 0, 255).astype(np.uint8)
            color[..., 1] = np.clip(yy * 22 + frame_idx * 10, 0, 255).astype(np.uint8)
            color[..., 2] = np.clip((xx + yy) * 12 + cam * 15, 0, 255).astype(np.uint8)
            depth_mm = (900 + cam * 60 + frame_idx * 10 + xx * 4 + yy * 3).astype(np.uint16)
            depth_mm[0, 0] = 0
            depth_mm[-1, -1] = 0
            cv2.imwrite(str(case_dir / "color" / str(cam) / f"{frame_idx}.png"), color)
            np.save(case_dir / "depth" / str(cam) / f"{frame_idx}.npy", depth_mm)
            if include_depth_ffs:
                depth_ffs_mm = np.clip(depth_mm.astype(np.int32) - 40 + cam * 5, 0, 65535).astype(np.uint16)
                np.save(case_dir / "depth_ffs" / str(cam) / f"{frame_idx}.npy", depth_ffs_mm)
            if include_depth_ffs_float_m:
                depth_ffs_float = depth_mm.astype(np.float32) * 0.001 - 0.03
                depth_ffs_float[depth_mm == 0] = 0.0
                np.save(case_dir / "depth_ffs_float_m" / str(cam) / f"{frame_idx}.npy", depth_ffs_float.astype(np.float32))

    metadata = {
        "schema_version": "qqtt_aligned_case_v2",
        "serial_numbers": ["239222300433", "239222300781", "239222303506"],
        "calibration_reference_serials": ["239222300433", "239222300781", "239222303506"],
        "frame_num": frame_num,
        "depth_scale_m_per_unit": [0.001, 0.001, 0.001],
        "intrinsics": K,
        "K_color": K,
        "depth_backend_used": "both" if include_depth_ffs else "realsense",
        "depth_source_for_depth_dir": "realsense",
    }
    (case_dir / "metadata.json").write_text(json.dumps(metadata), encoding="utf-8")

