from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .depth_diagnostics import label_tile
from .layouts import compose_depth_review_board


def parse_face_patches_json(path: str | Path) -> dict[int, list[dict[str, Any]]]:
    patch_path = Path(path).resolve()
    data = json.loads(patch_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or not data:
        raise ValueError(f"face patch json must contain a non-empty camera mapping: {patch_path}")
    parsed: dict[int, list[dict[str, Any]]] = {}
    for camera_key, patch_payload in data.items():
        camera_idx = int(camera_key)
        camera_patches: list[dict[str, Any]] = []
        if isinstance(patch_payload, dict):
            iterator = patch_payload.items()
            for patch_name, bbox in iterator:
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError(f"Patch {patch_name} for camera {camera_key} must be [x0, y0, x1, y1].")
                x0, y0, x1, y1 = [int(item) for item in bbox]
                if x0 >= x1 or y0 >= y1:
                    raise ValueError(f"Invalid bbox for patch {patch_name} in camera {camera_key}: {bbox}")
                camera_patches.append({"name": str(patch_name), "bbox": (x0, y0, x1, y1)})
        elif isinstance(patch_payload, list):
            for item in patch_payload:
                if not isinstance(item, dict) or "name" not in item or "bbox" not in item:
                    raise ValueError(f"Camera {camera_key} list entries must contain name and bbox.")
                bbox = item["bbox"]
                if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
                    raise ValueError(f"Patch entry for camera {camera_key} must be [x0, y0, x1, y1].")
                x0, y0, x1, y1 = [int(value) for value in bbox]
                if x0 >= x1 or y0 >= y1:
                    raise ValueError(f"Invalid bbox for camera {camera_key}: {bbox}")
                camera_patches.append({"name": str(item["name"]), "bbox": (x0, y0, x1, y1)})
        else:
            raise ValueError(f"Camera {camera_key} patches must be a name->bbox mapping or list of patch dicts.")
        parsed[camera_idx] = camera_patches
    return parsed


def _bbox_indices(bbox: tuple[int, int, int, int], image_shape: tuple[int, int]) -> tuple[int, int, int, int]:
    h, w = image_shape[:2]
    x0, y0, x1, y1 = [int(item) for item in bbox]
    x0 = max(0, min(w - 1, x0))
    y0 = max(0, min(h - 1, y0))
    x1 = max(x0 + 1, min(w, x1))
    y1 = max(y0 + 1, min(h, y1))
    return x0, y0, x1, y1


def _fit_plane(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    pts = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    centroid = pts.mean(axis=0).astype(np.float32)
    centered = pts - centroid[None, :]
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[-1].astype(np.float32)
    normal /= max(1e-6, float(np.linalg.norm(normal)))
    return centroid, normal


def compute_patch_plane_metrics(depth_m: np.ndarray, K_color: np.ndarray, bbox: tuple[int, int, int, int]) -> dict[str, Any]:
    depth = np.asarray(depth_m, dtype=np.float32)
    K = np.asarray(K_color, dtype=np.float32).reshape(3, 3)
    x0, y0, x1, y1 = _bbox_indices(bbox, depth.shape)
    patch_depth = depth[y0:y1, x0:x1]
    yy, xx = np.indices(patch_depth.shape, dtype=np.float32)
    xx = xx + float(x0)
    yy = yy + float(y0)
    valid = np.isfinite(patch_depth) & (patch_depth > 0)
    residual_mm = np.full(patch_depth.shape, np.nan, dtype=np.float32)
    metrics = {
        "valid_depth_ratio": float(np.count_nonzero(valid) / max(1, patch_depth.size)),
        "plane_fit_rmse_mm": 0.0,
        "mad_mm": 0.0,
        "p90_abs_residual_mm": 0.0,
        "point_count": int(np.count_nonzero(valid)),
    }
    if int(np.count_nonzero(valid)) < 3:
        return {**metrics, "residual_mm": residual_mm, "valid_mask": valid}
    z = patch_depth[valid]
    fx = float(K[0, 0])
    fy = float(K[1, 1])
    cx = float(K[0, 2])
    cy = float(K[1, 2])
    x = (xx[valid] - cx) * z / fx
    y = (yy[valid] - cy) * z / fy
    points = np.stack([x, y, z], axis=1).astype(np.float32)
    centroid, normal = _fit_plane(points)
    signed_residual_m = (points - centroid[None, :]) @ normal
    abs_residual_mm = np.abs(signed_residual_m) * 1000.0
    residual_mm[valid] = abs_residual_mm.astype(np.float32)
    metrics["plane_fit_rmse_mm"] = float(np.sqrt(np.mean((signed_residual_m * 1000.0) ** 2)))
    median = float(np.median(abs_residual_mm))
    metrics["mad_mm"] = float(np.median(np.abs(abs_residual_mm - median)))
    metrics["p90_abs_residual_mm"] = float(np.quantile(abs_residual_mm, 0.90))
    return {**metrics, "residual_mm": residual_mm, "valid_mask": valid}


def colorize_patch_residuals(residual_mm: np.ndarray, valid_mask: np.ndarray, *, max_mm: float) -> np.ndarray:
    residual = np.asarray(residual_mm, dtype=np.float32)
    valid = np.asarray(valid_mask, dtype=bool)
    canvas = np.full(residual.shape + (3,), (28, 30, 34), dtype=np.uint8)
    if not np.any(valid):
        return canvas
    normalized = np.clip(residual / max(1e-6, float(max_mm)), 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_INFERNO)
    canvas[valid] = colored[valid]
    return canvas


def draw_face_patch_overlay(image: np.ndarray, bbox: tuple[int, int, int, int], *, label: str) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    x0, y0, x1, y1 = _bbox_indices(bbox, canvas.shape[:2])
    cv2.rectangle(canvas, (x0, y0), (x1 - 1, y1 - 1), (0, 255, 255), 2, cv2.LINE_AA)
    cv2.rectangle(canvas, (x0, max(0, y0 - 24)), (min(canvas.shape[1] - 1, x0 + 180), y0), (0, 0, 0), -1)
    cv2.putText(canvas, label, (x0 + 6, max(16, y0 - 7)), cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def build_face_metric_tile(
    image: np.ndarray,
    *,
    label: str,
    metrics: dict[str, Any] | None = None,
    tile_size: tuple[int, int] = (320, 220),
) -> np.ndarray:
    tile = label_tile(image, label, tile_size)
    if metrics is None:
        return tile
    lines = [
        f"valid={metrics['valid_depth_ratio']:.3f}",
        f"rmse={metrics['plane_fit_rmse_mm']:.2f} mm",
        f"mad={metrics['mad_mm']:.2f} mm",
        f"p90={metrics['p90_abs_residual_mm']:.2f} mm",
    ]
    y = tile.shape[0] - 70
    cv2.rectangle(tile, (8, y - 18), (tile.shape[1] - 8, tile.shape[0] - 8), (0, 0, 0), -1)
    for idx, line in enumerate(lines):
        cv2.putText(tile, line, (16, y + idx * 16), cv2.FONT_HERSHEY_SIMPLEX, 0.46, (235, 235, 235), 1, cv2.LINE_AA)
    return tile


def compose_face_quality_board(
    *,
    title_lines: list[str],
    patch_rows: list[list[np.ndarray]],
    metric_lines: list[str] | None = None,
) -> np.ndarray:
    return compose_depth_review_board(
        title_lines=title_lines,
        metric_lines=[] if metric_lines is None else metric_lines,
        rows=patch_rows,
    )
