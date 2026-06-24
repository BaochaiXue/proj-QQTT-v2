from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations, product
from typing import Any

import numpy as np
from scipy.spatial import cKDTree


@dataclass(frozen=True)
class ShapeAlignmentConfig:
    max_centroid_drift_m: float = 0.05
    min_z_extent_ratio: float = 0.25
    max_z_extent_ratio: float = 4.0
    max_ground_z_fraction: float = 0.35
    ground_z_epsilon_m: float = 0.003
    table_z_m: float = 0.0
    above_direction: str = "negative"
    max_observation_to_aligned_p95_m: float = 0.05
    score_sample_count: int = 6000
    icp_iterations: int = 4


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


def _mean_center(points: np.ndarray) -> np.ndarray:
    return np.asarray(np.mean(points, axis=0), dtype=np.float32)


def _rms_radius(points: np.ndarray, center: np.ndarray) -> float:
    centered = np.asarray(points, dtype=np.float64) - np.asarray(center, dtype=np.float64).reshape(1, 3)
    radius = float(np.sqrt(np.mean(np.sum(centered * centered, axis=1))))
    return radius


def _rms_scale_similarity(src: np.ndarray, dst: np.ndarray, rotation: np.ndarray) -> tuple[float, np.ndarray]:
    src_center = _mean_center(src)
    dst_center = _mean_center(dst)
    src_radius = _rms_radius(src, src_center)
    dst_radius = _rms_radius(dst, dst_center)
    if src_radius <= 1e-9:
        raise ValueError("canonical shape has zero extent")
    scale = float(dst_radius / src_radius) if dst_radius > 1e-9 else 1.0
    translation = np.asarray(
        dst_center - np.float32(scale) * (np.asarray(rotation, dtype=np.float32) @ src_center),
        dtype=np.float32,
    )
    return scale, translation


def _principal_axes(points: np.ndarray) -> np.ndarray:
    centered = np.asarray(points, dtype=np.float64) - np.mean(points, axis=0, dtype=np.float64).reshape(1, 3)
    covariance = (centered.T @ centered) / max(1, len(centered) - 1)
    _, vectors = np.linalg.eigh(covariance)
    axes = vectors[:, ::-1]
    if np.linalg.det(axes) < 0.0:
        axes[:, -1] *= -1.0
    return np.ascontiguousarray(axes, dtype=np.float32)


def _proper_axis_hypotheses() -> list[np.ndarray]:
    matrices: list[np.ndarray] = []
    for perm in permutations(range(3)):
        permutation = np.eye(3, dtype=np.float32)[:, perm]
        for signs in product((-1.0, 1.0), repeat=3):
            signed = permutation @ np.diag(np.asarray(signs, dtype=np.float32))
            if np.linalg.det(signed) > 0.0:
                matrices.append(np.ascontiguousarray(signed, dtype=np.float32))
    return matrices


_AXIS_HYPOTHESES = _proper_axis_hypotheses()


def _apply_similarity(points: np.ndarray, scale: float, rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    aligned = np.float32(scale) * (np.asarray(points, dtype=np.float32) @ np.asarray(rotation, dtype=np.float32).T)
    aligned = aligned + np.asarray(translation, dtype=np.float32).reshape(1, 3)
    return np.ascontiguousarray(aligned, dtype=np.float32)


def _deterministic_sample(points: np.ndarray, max_points: int) -> np.ndarray:
    count = int(max_points)
    if count <= 0 or len(points) <= count:
        return np.ascontiguousarray(points, dtype=np.float32)
    indices = np.linspace(0, len(points) - 1, count, dtype=np.int64)
    return np.ascontiguousarray(points[indices], dtype=np.float32)


def _nearest_metrics(
    aligned: np.ndarray,
    observation: np.ndarray,
    *,
    sample_count: int,
) -> dict[str, float]:
    aligned_sample = _deterministic_sample(aligned, int(sample_count))
    observation_sample = _deterministic_sample(observation, int(sample_count))
    observation_tree = cKDTree(observation_sample)
    aligned_tree = cKDTree(aligned_sample)
    aligned_to_observation, _ = observation_tree.query(aligned_sample, k=1, workers=-1)
    observation_to_aligned, _ = aligned_tree.query(observation_sample, k=1, workers=-1)
    return {
        "aligned_to_observation_mean_m": float(np.mean(aligned_to_observation)),
        "aligned_to_observation_p95_m": float(np.percentile(aligned_to_observation, 95)),
        "observation_to_aligned_mean_m": float(np.mean(observation_to_aligned)),
        "observation_to_aligned_p95_m": float(np.percentile(observation_to_aligned, 95)),
        "symmetric_chamfer_mean_m": float(
            0.5 * (np.mean(aligned_to_observation) + np.mean(observation_to_aligned))
        ),
    }


def _alignment_score(metrics: dict[str, float]) -> float:
    return float(
        metrics["symmetric_chamfer_mean_m"]
        + 0.25 * metrics["aligned_to_observation_p95_m"]
        + 0.75 * metrics["observation_to_aligned_p95_m"]
    )


def _score_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    sample_count: int,
) -> tuple[float, dict[str, float]]:
    aligned = _apply_similarity(src, scale, rotation, translation)
    metrics = _nearest_metrics(aligned, dst, sample_count=sample_count)
    return _alignment_score(metrics), metrics


def _icp_refine_similarity(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
    sample_count: int,
    iterations: int,
) -> tuple[float, np.ndarray, np.ndarray, float, dict[str, float]]:
    src_sample = _deterministic_sample(src, int(sample_count))
    dst_sample = _deterministic_sample(dst, int(sample_count))
    dst_tree = cKDTree(dst_sample)
    best_scale = float(scale)
    best_rotation = np.ascontiguousarray(rotation, dtype=np.float32)
    best_translation = np.ascontiguousarray(translation, dtype=np.float32)
    best_score, best_metrics = _score_similarity(
        src,
        dst,
        scale=best_scale,
        rotation=best_rotation,
        translation=best_translation,
        sample_count=sample_count,
    )
    for _ in range(max(0, int(iterations))):
        aligned_sample = _apply_similarity(src_sample, best_scale, best_rotation, best_translation)
        distances, indices = dst_tree.query(aligned_sample, k=1, workers=-1)
        if len(distances) < 3:
            break
        distance_gate = float(np.percentile(distances, 80))
        keep = distances <= max(distance_gate, 1e-9)
        if np.count_nonzero(keep) < 3:
            keep = np.ones_like(distances, dtype=bool)
        candidate_scale, candidate_rotation, candidate_translation = _umeyama_similarity(
            src_sample[keep],
            dst_sample[indices[keep]],
        )
        candidate_score, candidate_metrics = _score_similarity(
            src,
            dst,
            scale=candidate_scale,
            rotation=candidate_rotation,
            translation=candidate_translation,
            sample_count=sample_count,
        )
        if candidate_score + 1e-9 >= best_score:
            break
        best_scale = float(candidate_scale)
        best_rotation = np.ascontiguousarray(candidate_rotation, dtype=np.float32)
        best_translation = np.ascontiguousarray(candidate_translation, dtype=np.float32)
        best_score = float(candidate_score)
        best_metrics = candidate_metrics
    return best_scale, best_rotation, best_translation, best_score, best_metrics


def _candidate_similarities(src: np.ndarray, dst: np.ndarray) -> list[tuple[str, float, np.ndarray, np.ndarray]]:
    candidates: list[tuple[str, float, np.ndarray, np.ndarray]] = []
    identity_scale, identity_rotation, identity_translation = _centroid_scale_similarity(src, dst)
    candidates.append(("centroid_scale_identity", identity_scale, identity_rotation, identity_translation))

    src_axes = _principal_axes(src)
    dst_axes = _principal_axes(dst)
    seen: set[tuple[float, ...]] = set()
    for idx, axis_hypothesis in enumerate(_AXIS_HYPOTHESES):
        rotation = np.ascontiguousarray(dst_axes @ axis_hypothesis @ src_axes.T, dtype=np.float32)
        if np.linalg.det(rotation) < 0.0:
            continue
        key = tuple(np.round(rotation.reshape(-1), 6).tolist())
        if key in seen:
            continue
        seen.add(key)
        scale, translation = _rms_scale_similarity(src, dst, rotation)
        candidates.append((f"pca_axis_hypothesis_{idx:02d}", scale, rotation, translation))

    if len(src) == len(dst):
        try:
            scale, rotation, translation = _umeyama_similarity(src, dst)
            candidates.append(("ordered_umeyama_hypothesis", scale, rotation, translation))
        except ValueError:
            pass
    return candidates


def _validate_alignment(
    aligned: np.ndarray,
    observation: np.ndarray,
    *,
    config: ShapeAlignmentConfig,
    nearest_metrics: dict[str, float] | None = None,
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
    metrics = (
        _nearest_metrics(aligned, observation, sample_count=int(config.score_sample_count))
        if nearest_metrics is None
        else dict(nearest_metrics)
    )
    coverage_valid = metrics["observation_to_aligned_p95_m"] <= float(config.max_observation_to_aligned_p95_m)
    valid = bool(valid and coverage_valid)
    payload = {
        "centroid_drift_m": centroid_drift_m,
        "aligned_z_extent_m": float(aligned_z_extent),
        "observation_z_extent_m": float(observation_z_extent),
        "z_extent_ratio": float(z_extent_ratio),
        "ground_z_fraction": ground_z_fraction,
        "max_centroid_drift_m": float(config.max_centroid_drift_m),
        "min_z_extent_ratio": float(config.min_z_extent_ratio),
        "max_z_extent_ratio": float(config.max_z_extent_ratio),
        "max_ground_z_fraction": float(config.max_ground_z_fraction),
        "max_observation_to_aligned_p95_m": float(config.max_observation_to_aligned_p95_m),
        "coverage_valid": bool(coverage_valid),
    }
    payload.update(metrics)
    return bool(valid), payload


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

    sample_count = int(max(3, int(cfg.score_sample_count)))
    best: tuple[str, float, np.ndarray, np.ndarray, float, dict[str, float]] | None = None
    for name, candidate_scale, candidate_rotation, candidate_translation in _candidate_similarities(
        canonical,
        observation,
    ):
        refined_scale, refined_rotation, refined_translation, score, metrics = _icp_refine_similarity(
            canonical,
            observation,
            scale=candidate_scale,
            rotation=candidate_rotation,
            translation=candidate_translation,
            sample_count=sample_count,
            iterations=int(cfg.icp_iterations),
        )
        if best is None or score < best[4]:
            best = (name, refined_scale, refined_rotation, refined_translation, score, metrics)

    if best is None:
        raise ValueError("could not generate a valid shape alignment candidate")

    candidate_name, scale, rotation, translation, score, nearest_metrics = best
    aligned = _apply_similarity(canonical, scale, rotation, translation)
    valid, validation = _validate_alignment(
        aligned,
        observation,
        config=cfg,
        nearest_metrics=nearest_metrics,
    )
    validation["alignment_candidate"] = candidate_name
    validation["alignment_score"] = float(score)
    validation["alignment_method"] = "pca_hypotheses_symmetric_nn_icp"
    return ShapeAlignmentResult(
        aligned_points_m=aligned,
        scale=float(scale),
        rotation=np.ascontiguousarray(rotation, dtype=np.float32),
        translation=np.ascontiguousarray(translation, dtype=np.float32),
        valid=bool(valid),
        validation=validation,
    )
