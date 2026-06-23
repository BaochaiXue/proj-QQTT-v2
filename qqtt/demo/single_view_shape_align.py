from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass(frozen=True)
class ShapeAlignmentConfig:
    max_centroid_drift_m: float = 0.05
    min_z_extent_ratio: float = 0.25
    max_z_extent_ratio: float = 4.0
    max_ground_z_fraction: float = 0.35
    ground_z_epsilon_m: float = 0.003
    table_z_m: float = 0.0
    above_direction: str = "positive"


@dataclass(frozen=True)
class ShapeAlignmentResult:
    aligned_points_m: np.ndarray
    scale: float
    rotation: np.ndarray
    translation: np.ndarray
    valid: bool
    validation: dict[str, Any] = field(default_factory=dict)


def _points_array(name: str, points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"{name} must be an Nx3 array, got {arr.shape}")
    finite = np.isfinite(arr).all(axis=1)
    return np.ascontiguousarray(arr[finite], dtype=np.float32)


def _robust_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    center = np.median(points, axis=0)
    distances = np.linalg.norm(points - center, axis=1)
    return float(np.percentile(distances, 90))


def _z_extent(points: np.ndarray) -> float:
    if len(points) == 0:
        return 0.0
    return float(np.nanmax(points[:, 2]) - np.nanmin(points[:, 2]))


def _umeyama_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src64 = np.asarray(src, dtype=np.float64)
    dst64 = np.asarray(dst, dtype=np.float64)
    src_mean = src64.mean(axis=0)
    dst_mean = dst64.mean(axis=0)
    src_centered = src64 - src_mean
    dst_centered = dst64 - dst_mean
    variance = float(np.mean(np.sum(src_centered * src_centered, axis=1)))
    if variance <= 1e-12:
        raise ValueError("canonical shape has zero variance")
    covariance = (dst_centered.T @ src_centered) / float(len(src64))
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / variance)
    translation = dst_mean - scale * (rotation @ src_mean)
    return scale, rotation.astype(np.float32), translation.astype(np.float32)


def _centroid_scale_similarity(src: np.ndarray, dst: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    src_center = np.median(src, axis=0).astype(np.float32)
    dst_center = np.median(dst, axis=0).astype(np.float32)
    src_extent = _robust_extent(src)
    dst_extent = _robust_extent(dst)
    if src_extent <= 1e-9:
        raise ValueError("canonical shape has zero extent")
    scale = float(dst_extent / src_extent) if dst_extent > 1e-9 else 1.0
    rotation = np.eye(3, dtype=np.float32)
    translation = np.asarray(dst_center - np.float32(scale) * src_center, dtype=np.float32)
    return scale, rotation, translation


def _apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.float32(scale) * (np.asarray(points, dtype=np.float32) @ np.asarray(rotation, dtype=np.float32).T)
    aligned = aligned + np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray(aligned, dtype=np.float32)


def _validate_alignment(
    aligned: np.ndarray,
    observation: np.ndarray,
    *,
    config: ShapeAlignmentConfig,
) -> tuple[bool, dict[str, Any]]:
    aligned_centroid = aligned.mean(axis=0) if len(aligned) else np.zeros((3,), dtype=np.float32)
    observation_centroid = observation.mean(axis=0) if len(observation) else np.zeros((3,), dtype=np.float32)
    centroid_drift_m = float(np.linalg.norm(aligned_centroid - observation_centroid))
    aligned_z_extent = _z_extent(aligned)
    observation_z_extent = _z_extent(observation)
    if observation_z_extent <= 1e-9:
        z_extent_ratio = 1.0 if aligned_z_extent <= float(config.ground_z_epsilon_m) else float("inf")
    else:
        z_extent_ratio = float(aligned_z_extent / observation_z_extent)
    if str(config.above_direction) == "negative":
        ground = aligned[:, 2] >= float(config.table_z_m) - float(config.ground_z_epsilon_m)
    else:
        ground = aligned[:, 2] <= float(config.table_z_m) + float(config.ground_z_epsilon_m)
    ground_z_fraction = float(np.count_nonzero(ground) / max(1, len(aligned)))
    valid = (
        centroid_drift_m <= float(config.max_centroid_drift_m)
        and float(config.min_z_extent_ratio) <= z_extent_ratio <= float(config.max_z_extent_ratio)
        and ground_z_fraction <= float(config.max_ground_z_fraction)
    )
    return bool(valid), {
        "centroid_drift_m": centroid_drift_m,
        "aligned_z_extent_m": float(aligned_z_extent),
        "observation_z_extent_m": float(observation_z_extent),
        "z_extent_ratio": float(z_extent_ratio),
        "ground_z_fraction": ground_z_fraction,
        "max_centroid_drift_m": float(config.max_centroid_drift_m),
        "min_z_extent_ratio": float(config.min_z_extent_ratio),
        "max_z_extent_ratio": float(config.max_z_extent_ratio),
        "max_ground_z_fraction": float(config.max_ground_z_fraction),
    }


def align_canonical_shape_to_observation(
    canonical_points_m: np.ndarray,
    observation_points_m: np.ndarray,
    *,
    config: ShapeAlignmentConfig | None = None,
) -> ShapeAlignmentResult:
    cfg = config or ShapeAlignmentConfig()
    canonical = _points_array("canonical_points_m", canonical_points_m)
    observation = _points_array("observation_points_m", observation_points_m)
    if len(canonical) < 3:
        raise ValueError("canonical shape requires at least 3 finite points")
    if len(observation) < 3:
        raise ValueError("observation PCD requires at least 3 finite points")

    if len(canonical) == len(observation):
        scale, rotation, translation = _umeyama_similarity(canonical, observation)
    else:
        scale, rotation, translation = _centroid_scale_similarity(canonical, observation)
    aligned = _apply_similarity(canonical, scale, rotation, translation)
    valid, validation = _validate_alignment(aligned, observation, config=cfg)
    return ShapeAlignmentResult(
        aligned_points_m=aligned,
        scale=float(scale),
        rotation=np.ascontiguousarray(rotation, dtype=np.float32),
        translation=np.ascontiguousarray(translation, dtype=np.float32),
        valid=bool(valid),
        validation=validation,
    )
