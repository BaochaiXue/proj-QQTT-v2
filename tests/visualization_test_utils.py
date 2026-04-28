from __future__ import annotations

import json
import pickle
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_process.aligned_case_metadata import write_split_aligned_metadata


def make_visualization_case(
    case_dir: Path,
    *,
    include_depth_ffs: bool = False,
    include_depth_ffs_float_m: bool = False,
    include_depth_original: bool = False,
    include_depth_ffs_original: bool = False,
    include_depth_ffs_float_m_original: bool = False,
    include_depth_ffs_native_like_postprocess: bool = False,
    include_depth_ffs_native_like_postprocess_float_m: bool = False,
    include_ir_pair: bool = False,
    frame_num: int = 2,
    include_sparse_outlier: bool = False,
    depth_backend_used: str | None = None,
    depth_source_for_depth_dir: str | None = None,
) -> None:
    case_dir.mkdir(parents=True, exist_ok=True)
    streams = ["color", "depth"]
    if include_depth_ffs:
        streams.append("depth_ffs")
    if include_depth_ffs_float_m:
        streams.append("depth_ffs_float_m")
    if include_depth_original:
        streams.append("depth_original")
    if include_depth_ffs_original:
        streams.append("depth_ffs_original")
    if include_depth_ffs_float_m_original:
        streams.append("depth_ffs_float_m_original")
    if include_depth_ffs_native_like_postprocess:
        streams.append("depth_ffs_native_like_postprocess")
    if include_depth_ffs_native_like_postprocess_float_m:
        streams.append("depth_ffs_native_like_postprocess_float_m")
    if include_ir_pair:
        streams.extend(["ir_left", "ir_right"])
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
            if include_sparse_outlier and cam == 0:
                depth_mm[1, 1] = 250
                color[0:3, 0:3] = 0
            cv2.imwrite(str(case_dir / "color" / str(cam) / f"{frame_idx}.png"), color)
            np.save(case_dir / "depth" / str(cam) / f"{frame_idx}.npy", depth_mm)
            if include_ir_pair:
                ir_left = np.clip(70 + xx * 8 + yy * 3 + frame_idx * 4 + cam * 5, 0, 255).astype(np.uint8)
                ir_right = np.clip(75 + xx * 6 + yy * 4 + frame_idx * 4 + cam * 4, 0, 255).astype(np.uint8)
                cv2.imwrite(str(case_dir / "ir_left" / str(cam) / f"{frame_idx}.png"), ir_left)
                cv2.imwrite(str(case_dir / "ir_right" / str(cam) / f"{frame_idx}.png"), ir_right)
            if include_depth_ffs:
                depth_ffs_mm = np.clip(depth_mm.astype(np.int32) - 40 + cam * 5, 0, 65535).astype(np.uint16)
                np.save(case_dir / "depth_ffs" / str(cam) / f"{frame_idx}.npy", depth_ffs_mm)
            if include_depth_ffs_float_m:
                depth_ffs_float = depth_mm.astype(np.float32) * 0.001 - 0.03
                depth_ffs_float[depth_mm == 0] = 0.0
                np.save(case_dir / "depth_ffs_float_m" / str(cam) / f"{frame_idx}.npy", depth_ffs_float.astype(np.float32))
            if include_depth_original:
                depth_original_mm = np.clip(depth_mm.astype(np.int32) - 70 + cam * 2, 0, 65535).astype(np.uint16)
                np.save(case_dir / "depth_original" / str(cam) / f"{frame_idx}.npy", depth_original_mm)
            if include_depth_ffs_original:
                depth_ffs_original_mm = np.clip(depth_mm.astype(np.int32) - 55 + cam * 4, 0, 65535).astype(np.uint16)
                np.save(case_dir / "depth_ffs_original" / str(cam) / f"{frame_idx}.npy", depth_ffs_original_mm)
            if include_depth_ffs_float_m_original:
                depth_ffs_float_original = depth_mm.astype(np.float32) * 0.001 - 0.045
                depth_ffs_float_original[depth_mm == 0] = 0.0
                np.save(
                    case_dir / "depth_ffs_float_m_original" / str(cam) / f"{frame_idx}.npy",
                    depth_ffs_float_original.astype(np.float32),
                )
            if include_depth_ffs_native_like_postprocess:
                depth_ffs_native_like_postprocess_mm = np.clip(depth_mm.astype(np.int32) - 20 + cam * 3, 0, 65535).astype(np.uint16)
                np.save(
                    case_dir / "depth_ffs_native_like_postprocess" / str(cam) / f"{frame_idx}.npy",
                    depth_ffs_native_like_postprocess_mm,
                )
            if include_depth_ffs_native_like_postprocess_float_m:
                depth_ffs_native_like_postprocess_float = depth_mm.astype(np.float32) * 0.001 - 0.01
                depth_ffs_native_like_postprocess_float[depth_mm == 0] = 0.0
                np.save(
                    case_dir / "depth_ffs_native_like_postprocess_float_m" / str(cam) / f"{frame_idx}.npy",
                    depth_ffs_native_like_postprocess_float.astype(np.float32),
                )

    metadata = {
        "schema_version": "qqtt_aligned_case_v2",
        "serial_numbers": ["239222300433", "239222300781", "239222303506"],
        "calibration_reference_serials": ["239222300433", "239222300781", "239222303506"],
        "fps": 30,
        "WH": [width, height],
        "frame_num": frame_num,
        "start_step": 0,
        "end_step": frame_num - 1,
        "depth_scale_m_per_unit": [0.001, 0.001, 0.001],
        "intrinsics": K,
        "K_color": K,
        "depth_backend_used": depth_backend_used or ("both" if include_depth_ffs else "realsense"),
        "depth_source_for_depth_dir": depth_source_for_depth_dir or ("realsense"),
    }
    write_split_aligned_metadata(case_dir, metadata)


def make_sam31_masks(
    case_dir: Path,
    *,
    prompt_labels_by_object: dict[int, str] | None = None,
    camera_ids: list[int] | None = None,
    frame_tokens: list[str] | None = None,
) -> Path:
    mask_root = case_dir / "sam31_masks"
    mask_dir = mask_root / "mask"
    mask_dir.mkdir(parents=True, exist_ok=True)
    prompt_labels = prompt_labels_by_object or {1: "sloth", 2: "sloth"}
    camera_ids = [0, 1, 2] if camera_ids is None else [int(item) for item in camera_ids]
    frame_tokens = ["0"] if frame_tokens is None else [str(item) for item in frame_tokens]

    for camera_idx in camera_ids:
        sample_frame = cv2.imread(str(case_dir / "color" / str(camera_idx) / f"{frame_tokens[0]}.png"), cv2.IMREAD_COLOR)
        if sample_frame is None:
            raise FileNotFoundError(f"Missing color frame for mask fixture camera {camera_idx}.")
        height, width = sample_frame.shape[:2]
        with (mask_dir / f"mask_info_{camera_idx}.json").open("w", encoding="utf-8") as handle:
            json.dump({str(obj_id): label for obj_id, label in sorted(prompt_labels.items())}, handle, indent=2)
        for obj_id in prompt_labels:
            for frame_token in frame_tokens:
                object_dir = mask_dir / str(camera_idx) / str(int(obj_id))
                object_dir.mkdir(parents=True, exist_ok=True)
                mask = np.zeros((height, width), dtype=np.uint8)
                if int(obj_id) == 1:
                    mask[1:4, 1:4] = 255
                else:
                    mask[3:6, 5:8] = 255
                cv2.imwrite(str(object_dir / f"{frame_token}.png"), mask)
    summary = {
        "case_root": str(case_dir.resolve()),
        "output_dir": str(mask_root.resolve()),
        "camera_ids": camera_ids,
        "prompt_labels_by_object": {str(key): value for key, value in sorted(prompt_labels.items())},
    }
    with (mask_root / "summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    return mask_root


def make_rerun_compare_cases(
    aligned_root: Path,
    *,
    native_case_name: str = "native_case",
    ffs_case_name: str = "ffs_case",
    frame_num: int = 2,
) -> tuple[Path, Path]:
    native_case_dir = aligned_root / native_case_name
    ffs_case_dir = aligned_root / ffs_case_name
    native_case_dir.mkdir(parents=True, exist_ok=True)
    ffs_case_dir.mkdir(parents=True, exist_ok=True)

    for stream_root, streams in (
        (native_case_dir, ["color", "depth"]),
        (ffs_case_dir, ["color", "ir_left", "ir_right"]),
    ):
        for stream in streams:
            for cam in range(3):
                (stream_root / stream / str(cam)).mkdir(parents=True, exist_ok=True)

    transforms = [
        np.eye(4, dtype=np.float32),
        np.array([[1, 0, 0, 0.25], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
        np.array([[1, 0, 0, 0], [0, 1, 0, 0.25], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32),
    ]
    for case_dir in (native_case_dir, ffs_case_dir):
        with (case_dir / "calibrate.pkl").open("wb") as handle:
            pickle.dump(transforms, handle)

    height, width = 8, 10
    yy, xx = np.indices((height, width), dtype=np.float32)
    k_color = [[[6.0, 0.0, 4.5], [0.0, 6.0, 3.5], [0.0, 0.0, 1.0]] for _ in range(3)]
    t_ir_left_to_color = [
        [[1.0, 0.0, 0.0, 0.01], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
        for _ in range(3)
    ]

    for frame_idx in range(frame_num):
        for cam in range(3):
            color = np.zeros((height, width, 3), dtype=np.uint8)
            color[..., 0] = np.clip(xx * 14 + cam * 25, 0, 255).astype(np.uint8)
            color[..., 1] = np.clip(yy * 20 + frame_idx * 18, 0, 255).astype(np.uint8)
            color[..., 2] = np.clip((xx + yy) * 16 + cam * 10, 0, 255).astype(np.uint8)
            depth_mm = (900 + cam * 40 + frame_idx * 15 + xx * 3 + yy * 2).astype(np.uint16)
            depth_mm[0, 0] = 0
            ir_left = np.clip(80 + xx * 7 + frame_idx * 5 + cam * 4, 0, 255).astype(np.uint8)
            ir_right = np.clip(70 + xx * 6 + frame_idx * 5 + cam * 3, 0, 255).astype(np.uint8)

            cv2.imwrite(str(native_case_dir / "color" / str(cam) / f"{frame_idx}.png"), color)
            np.save(native_case_dir / "depth" / str(cam) / f"{frame_idx}.npy", depth_mm)

            cv2.imwrite(str(ffs_case_dir / "color" / str(cam) / f"{frame_idx}.png"), color)
            cv2.imwrite(str(ffs_case_dir / "ir_left" / str(cam) / f"{frame_idx}.png"), ir_left)
            cv2.imwrite(str(ffs_case_dir / "ir_right" / str(cam) / f"{frame_idx}.png"), ir_right)

    native_metadata = {
        "schema_version": "qqtt_aligned_case_v2",
        "serial_numbers": ["239222300433", "239222300781", "239222303506"],
        "calibration_reference_serials": ["239222300433", "239222300781", "239222303506"],
        "fps": 30,
        "WH": [width, height],
        "frame_num": frame_num,
        "start_step": 0,
        "end_step": frame_num - 1,
        "depth_scale_m_per_unit": [0.001, 0.001, 0.001],
        "intrinsics": k_color,
        "K_color": k_color,
        "depth_backend_used": "realsense",
        "depth_source_for_depth_dir": "realsense",
    }
    write_split_aligned_metadata(native_case_dir, native_metadata)

    ffs_metadata = {
        "schema_version": "qqtt_aligned_case_v2",
        "serial_numbers": ["239222300433", "239222300781", "239222303506"],
        "calibration_reference_serials": ["239222300433", "239222300781", "239222303506"],
        "fps": 30,
        "WH": [width, height],
        "frame_num": frame_num,
        "start_step": 0,
        "end_step": frame_num - 1,
        "intrinsics": k_color,
        "K_color": k_color,
        "K_ir_left": k_color,
        "K_ir_right": k_color,
        "T_ir_left_to_color": t_ir_left_to_color,
        "T_ir_left_to_right": [
            [[1.0, 0.0, 0.0, -0.095], [0.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0], [0.0, 0.0, 0.0, 1.0]]
            for _ in range(3)
        ],
        "ir_baseline_m": [0.095, 0.095, 0.095],
        "depth_backend_used": "ffs",
        "depth_source_for_depth_dir": "ffs",
        "ffs_config": {
            "ffs_repo": "C:/external/fake",
            "model_path": "C:/external/fake/model.pth",
            "scale": 1.0,
            "valid_iters": 8,
            "max_disp": 192,
        },
    }
    write_split_aligned_metadata(ffs_case_dir, ffs_metadata)
    return native_case_dir, ffs_case_dir


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
