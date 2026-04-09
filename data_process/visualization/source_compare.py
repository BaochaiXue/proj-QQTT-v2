from __future__ import annotations

from pathlib import Path
from typing import Any
import warnings

import cv2
import numpy as np

from .renderers import rasterize_point_cloud_view, render_point_cloud


SOURCE_CAMERA_COLORS_BGR = {
    0: (80, 165, 255),
    1: (110, 220, 120),
    2: (255, 220, 80),
}
SOURCE_CAMERA_LABELS = {
    0: "Cam0",
    1: "Cam1",
    2: "Cam2",
}


def source_color_bgr(camera_idx: int) -> tuple[int, int, int]:
    return SOURCE_CAMERA_COLORS_BGR.get(int(camera_idx), (220, 220, 220))


def _constant_color_array(length: int, color_bgr: tuple[int, int, int]) -> np.ndarray:
    if int(length) <= 0:
        return np.empty((0, 3), dtype=np.uint8)
    return np.tile(np.asarray(color_bgr, dtype=np.uint8).reshape(1, 3), (int(length), 1))


def build_source_legend_image(
    *,
    width: int = 320,
    height: int = 140,
) -> np.ndarray:
    canvas = np.full((int(height), int(width), 3), (18, 18, 22), dtype=np.uint8)
    cv2.putText(canvas, "Source Attribution", (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (255, 255, 255), 2, cv2.LINE_AA)
    for row_idx, camera_idx in enumerate((0, 1, 2)):
        y = 54 + row_idx * 24
        color = source_color_bgr(camera_idx)
        cv2.rectangle(canvas, (16, y - 10), (42, y + 10), color, -1)
        cv2.putText(
            canvas,
            f"{SOURCE_CAMERA_LABELS[camera_idx]}",
            (54, y + 6),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(canvas, "Semi-transparent source overlay", (14, height - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.54, (210, 210, 210), 1, cv2.LINE_AA)
    return canvas


def write_source_legend_image(path: Path) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), build_source_legend_image())
    return str(path)


def overlay_source_legend(image: np.ndarray) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    legend = build_source_legend_image(width=260, height=122)
    if legend.shape[1] > canvas.shape[1] - 12 or legend.shape[0] > canvas.shape[0] - 12:
        scale = min(
            max(0.25, float(canvas.shape[1] - 20) / float(max(1, legend.shape[1]))),
            max(0.25, float(canvas.shape[0] - 20) / float(max(1, legend.shape[0]))),
            1.0,
        )
        new_w = max(80, int(round(legend.shape[1] * scale)))
        new_h = max(48, int(round(legend.shape[0] * scale)))
        legend = cv2.resize(legend, (new_w, new_h), interpolation=cv2.INTER_AREA)
    x0 = 16
    y0 = canvas.shape[0] - legend.shape[0] - 18
    canvas[y0:y0 + legend.shape[0], x0:x0 + legend.shape[1]] = legend
    cv2.rectangle(canvas, (x0 - 1, y0 - 1), (x0 + legend.shape[1], y0 + legend.shape[0]), (255, 255, 255), 1, cv2.LINE_AA)
    return canvas


def render_source_attribution_overlay(
    camera_clouds: list[dict[str, Any]],
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
    alpha: float = 0.32,
    background_bgr: tuple[int, int, int] = (28, 30, 34),
) -> tuple[np.ndarray, dict[str, Any]]:
    rasters: list[dict[str, Any]] = []
    metrics_per_camera: list[dict[str, Any]] = []
    for camera_cloud in sorted(camera_clouds, key=lambda item: int(item["camera_idx"])):
        camera_idx = int(camera_cloud["camera_idx"])
        const_colors = _constant_color_array(len(camera_cloud["points"]), source_color_bgr(camera_idx))
        raster = rasterize_point_cloud_view(
            np.asarray(camera_cloud["points"], dtype=np.float32),
            const_colors,
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        rasters.append(
            {
                "camera_idx": camera_idx,
                "serial": camera_cloud["serial"],
                "depth": np.asarray(raster["depth"], dtype=np.float32),
                "valid": np.asarray(raster["valid"], dtype=bool),
                "color_bgr": source_color_bgr(camera_idx),
            }
        )
        metrics_per_camera.append(
            {
                "camera_idx": camera_idx,
                "serial": camera_cloud["serial"],
                "point_count": int(len(camera_cloud["points"])),
                "visible_pixel_count": int(np.count_nonzero(raster["valid"])),
            }
        )

    canvas = np.full((height, width, 3), background_bgr, dtype=np.float32)
    if rasters:
        depth_stack = np.stack(
            [np.where(item["valid"], item["depth"], -np.inf).astype(np.float32) for item in rasters],
            axis=0,
        )
        order = np.argsort(-depth_stack, axis=0, kind="stable")
        valid_stack = np.stack([item["valid"] for item in rasters], axis=0)
        for layer_idx in range(len(rasters)):
            source_order = order[layer_idx]
            for source_idx, raster in enumerate(rasters):
                mask = (source_order == source_idx) & valid_stack[source_idx]
                if not np.any(mask):
                    continue
                color = np.asarray(raster["color_bgr"], dtype=np.float32).reshape(1, 3)
                canvas[mask] = canvas[mask] * (1.0 - float(alpha)) + color * float(alpha)
    overlay = np.clip(canvas, 0.0, 255.0).astype(np.uint8)
    overlay = overlay_source_legend(overlay)
    return overlay, {
        "alpha": float(alpha),
        "per_camera": metrics_per_camera,
    }


def render_source_split_images(
    camera_clouds: list[dict[str, Any]],
    *,
    view_config: dict[str, Any],
    scalar_bounds: dict[str, tuple[float, float]],
    renderer: str,
    width: int,
    height: int,
    point_radius_px: int,
    supersample_scale: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    split_images: list[np.ndarray] = []
    metrics: list[dict[str, Any]] = []
    clouds_by_idx = {int(item["camera_idx"]): item for item in camera_clouds}
    for camera_idx in (0, 1, 2):
        camera_cloud = clouds_by_idx.get(camera_idx)
        if camera_cloud is None or len(camera_cloud["points"]) == 0:
            split_images.append(np.full((height, width, 3), (20, 20, 24), dtype=np.uint8))
            metrics.append({"camera_idx": int(camera_idx), "point_count": 0, "renderer_used": "none"})
            continue
        image, renderer_used = render_point_cloud(
            np.asarray(camera_cloud["points"], dtype=np.float32),
            _constant_color_array(len(camera_cloud["points"]), source_color_bgr(camera_idx)),
            renderer=renderer,
            view_config=view_config,
            render_mode="color_by_rgb",
            scalar_bounds=scalar_bounds,
            width=width,
            height=height,
            point_radius_px=point_radius_px,
            supersample_scale=supersample_scale,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        split_images.append(overlay_source_legend(image))
        metrics.append(
            {
                "camera_idx": int(camera_idx),
                "serial": camera_cloud["serial"],
                "point_count": int(len(camera_cloud["points"])),
                "renderer_used": renderer_used,
            }
        )
    return split_images, metrics


def _colorize_residual_map(
    residual_map: np.ndarray,
    valid_mask: np.ndarray,
    *,
    max_value: float,
) -> np.ndarray:
    canvas = np.full(residual_map.shape + (3,), (22, 24, 28), dtype=np.uint8)
    if np.any(valid_mask):
        normalized = np.clip(residual_map / max(1e-6, float(max_value)), 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[valid_mask] = colored[valid_mask]
    return canvas


def overlay_mismatch_legend(
    image: np.ndarray,
    *,
    residual_max_m: float,
) -> np.ndarray:
    canvas = np.asarray(image, dtype=np.uint8).copy()
    x0 = 16
    y0 = canvas.shape[0] - 90
    cv2.rectangle(canvas, (x0 - 8, y0 - 26), (x0 + 250, y0 + 56), (18, 18, 22), -1)
    cv2.putText(canvas, "Mismatch Residual", (x0, y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas, f"0.000 m -> {residual_max_m:.3f} m", (x0, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.56, (235, 235, 235), 1, cv2.LINE_AA)
    cv2.putText(canvas, "low = stable overlap | high = mismatch", (x0, y0 + 44), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (215, 215, 215), 1, cv2.LINE_AA)
    return canvas


def compute_mismatch_residual(
    camera_clouds: list[dict[str, Any]],
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> dict[str, Any]:
    if not camera_clouds:
        empty = np.zeros((height, width), dtype=np.float32)
        return {
            "residual_map": empty,
            "valid_mask": np.zeros((height, width), dtype=bool),
            "overlap_mask": np.zeros((height, width), dtype=bool),
            "per_camera": [],
            "summary": {
                "overlap_pixel_count": 0,
                "residual_mean_m": 0.0,
                "residual_p90_m": 0.0,
                "residual_max_m": 0.0,
            },
        }

    depth_maps = []
    valid_maps = []
    source_meta = []
    for camera_cloud in sorted(camera_clouds, key=lambda item: int(item["camera_idx"])):
        raster = rasterize_point_cloud_view(
            np.asarray(camera_cloud["points"], dtype=np.float32),
            _constant_color_array(len(camera_cloud["points"]), source_color_bgr(int(camera_cloud["camera_idx"]))),
            view_config=view_config,
            width=width,
            height=height,
            projection_mode=projection_mode,
            ortho_scale=ortho_scale,
        )
        depth_maps.append(np.where(raster["valid"], raster["depth"], np.nan).astype(np.float32))
        valid_maps.append(np.asarray(raster["valid"], dtype=bool))
        source_meta.append({"camera_idx": int(camera_cloud["camera_idx"]), "serial": camera_cloud["serial"]})

    depth_stack = np.stack(depth_maps, axis=0)
    valid_stack = np.stack(valid_maps, axis=0)
    overlap_mask = np.count_nonzero(valid_stack, axis=0) >= 2
    residual_map = np.zeros((height, width), dtype=np.float32)
    if np.any(overlap_mask):
        residual_map[overlap_mask] = np.nanmax(depth_stack[:, overlap_mask], axis=0) - np.nanmin(depth_stack[:, overlap_mask], axis=0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        consensus_depth = np.nanmedian(depth_stack, axis=0)
    per_camera_metrics: list[dict[str, Any]] = []
    for source_idx, meta in enumerate(source_meta):
        mask = overlap_mask & valid_stack[source_idx]
        residual_values = np.abs(depth_stack[source_idx] - consensus_depth)
        residual_values = residual_values[mask]
        if len(residual_values) == 0:
            per_camera_metrics.append(
                {
                    "camera_idx": meta["camera_idx"],
                    "serial": meta["serial"],
                    "overlap_pixel_count": 0,
                    "residual_mean_m": 0.0,
                    "residual_p90_m": 0.0,
                    "residual_max_m": 0.0,
                }
            )
            continue
        per_camera_metrics.append(
            {
                "camera_idx": meta["camera_idx"],
                "serial": meta["serial"],
                "overlap_pixel_count": int(len(residual_values)),
                "residual_mean_m": float(np.mean(residual_values)),
                "residual_p90_m": float(np.quantile(residual_values, 0.90)),
                "residual_max_m": float(np.max(residual_values)),
            }
        )

    overlap_values = residual_map[overlap_mask]
    summary = {
        "overlap_pixel_count": int(len(overlap_values)),
        "residual_mean_m": float(np.mean(overlap_values)) if len(overlap_values) > 0 else 0.0,
        "residual_p90_m": float(np.quantile(overlap_values, 0.90)) if len(overlap_values) > 0 else 0.0,
        "residual_max_m": float(np.max(overlap_values)) if len(overlap_values) > 0 else 0.0,
    }
    return {
        "residual_map": residual_map,
        "valid_mask": overlap_mask,
        "overlap_mask": overlap_mask,
        "per_camera": per_camera_metrics,
        "summary": summary,
    }


def render_mismatch_residual(
    camera_clouds: list[dict[str, Any]],
    *,
    view_config: dict[str, Any],
    width: int,
    height: int,
    projection_mode: str,
    ortho_scale: float | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    residual = compute_mismatch_residual(
        camera_clouds,
        view_config=view_config,
        width=width,
        height=height,
        projection_mode=projection_mode,
        ortho_scale=ortho_scale,
    )
    residual_max = max(0.01, float(residual["summary"]["residual_p90_m"]) * 1.2, float(residual["summary"]["residual_max_m"]) * 0.8)
    image = _colorize_residual_map(residual["residual_map"], residual["valid_mask"], max_value=residual_max)
    image = overlay_mismatch_legend(image, residual_max_m=residual_max)
    return image, residual
