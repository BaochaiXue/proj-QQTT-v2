from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

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


def _look_at_c2w(camera_position: np.ndarray, center: np.ndarray, up: np.ndarray) -> np.ndarray:
    eye = np.asarray(camera_position, dtype=np.float32)
    target = np.asarray(center, dtype=np.float32)
    up_hint = np.asarray(up, dtype=np.float32)
    forward = target - eye
    forward /= max(float(np.linalg.norm(forward)), 1e-6)
    right = np.cross(forward, up_hint)
    right /= max(float(np.linalg.norm(right)), 1e-6)
    true_up = np.cross(right, forward)
    rotation = np.stack([right, true_up, forward], axis=1).astype(np.float32)
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = rotation
    c2w[:3, 3] = eye
    return c2w


def make_object_refinement_raw_scene(tmp_root: Path) -> dict[str, Any]:
    image_dir = tmp_root / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_shape = (480, 640, 3)
    K = np.array([[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]], dtype=np.float32)

    table_xx, table_yy = np.meshgrid(np.linspace(-0.40, 0.40, 46), np.linspace(-0.40, 0.40, 46), indexing="xy")
    table = np.stack([table_xx.reshape(-1), table_yy.reshape(-1), np.zeros(table_xx.size)], axis=1).astype(np.float32)

    torso_xx, torso_yy, torso_zz = np.meshgrid(
        np.linspace(-0.07, 0.07, 12),
        np.linspace(-0.06, 0.06, 12),
        np.linspace(0.05, 0.13, 8),
        indexing="xy",
    )
    torso = np.stack([torso_xx.reshape(-1), torso_yy.reshape(-1), torso_zz.reshape(-1)], axis=1).astype(np.float32)

    head_xx, head_yy, head_zz = np.meshgrid(
        np.linspace(-0.035, 0.035, 8),
        np.linspace(-0.03, 0.03, 8),
        np.linspace(0.17, 0.23, 5),
        indexing="xy",
    )
    head = np.stack([head_xx.reshape(-1), head_yy.reshape(-1), head_zz.reshape(-1)], axis=1).astype(np.float32)

    ear_left = np.stack(
        [
            np.linspace(-0.055, -0.025, 18),
            np.linspace(-0.01, 0.01, 18),
            np.linspace(0.24, 0.275, 18),
        ],
        axis=1,
    ).astype(np.float32)
    ear_right = np.stack(
        [
            np.linspace(0.025, 0.055, 18),
            np.linspace(-0.01, 0.01, 18),
            np.linspace(0.24, 0.275, 18),
        ],
        axis=1,
    ).astype(np.float32)

    object_points = np.concatenate([torso, head, ear_left, ear_right], axis=0)
    all_points = np.concatenate([table, object_points], axis=0)
    table_colors = np.tile(np.array([[170, 150, 120]], dtype=np.uint8), (len(table), 1))
    object_colors = np.tile(np.array([[110, 145, 195]], dtype=np.uint8), (len(object_points), 1))
    all_colors = np.concatenate([table_colors, object_colors], axis=0)

    camera_positions = [
        np.array([0.0, -0.56, 0.20], dtype=np.float32),
        np.array([0.52, 0.02, 0.22], dtype=np.float32),
        np.array([-0.44, 0.18, 0.21], dtype=np.float32),
    ]
    camera_clouds = []
    for camera_idx, camera_position in enumerate(camera_positions):
        image = np.full(image_shape, 150, dtype=np.uint8)
        color_path = image_dir / f"cam{camera_idx}.png"
        cv2.imwrite(str(color_path), image)
        c2w = _look_at_c2w(camera_position, np.array([0.0, 0.0, 0.12], dtype=np.float32), np.array([0.0, 0.0, 1.0], dtype=np.float32))
        camera_clouds.append(
            {
                "camera_idx": camera_idx,
                "serial": f"serial-{camera_idx}",
                "points": all_points.copy(),
                "colors": all_colors.copy(),
                "K_color": K.copy(),
                "c2w": c2w,
                "color_path": str(color_path),
            }
        )

    return {
        "native_points": np.concatenate([item["points"] for item in camera_clouds], axis=0),
        "native_colors": np.concatenate([item["colors"] for item in camera_clouds], axis=0),
        "native_camera_clouds": [{**item} for item in camera_clouds],
        "ffs_points": np.concatenate([item["points"] for item in camera_clouds], axis=0),
        "ffs_colors": np.concatenate([item["colors"] for item in camera_clouds], axis=0),
        "ffs_camera_clouds": [{**item} for item in camera_clouds],
    }
