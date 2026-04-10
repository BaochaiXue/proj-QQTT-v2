from __future__ import annotations

from typing import Any

import cv2
import numpy as np

from .fast_foundation_stereo import compute_disparity_audit_stats


def derive_ir_right_to_color(
    T_ir_left_to_right: np.ndarray,
    T_ir_left_to_color: np.ndarray,
) -> np.ndarray:
    left_to_right = np.asarray(T_ir_left_to_right, dtype=np.float32).reshape(4, 4)
    left_to_color = np.asarray(T_ir_left_to_color, dtype=np.float32).reshape(4, 4)
    right_to_left = np.linalg.inv(left_to_right)
    return (left_to_color @ right_to_left).astype(np.float32)


def colorize_signed_disparity(disparity_raw: np.ndarray) -> np.ndarray:
    disparity = np.asarray(disparity_raw, dtype=np.float32)
    finite = np.isfinite(disparity)
    canvas = np.full(disparity.shape + (3,), (32, 32, 36), dtype=np.uint8)
    if not np.any(finite):
        return canvas
    max_abs = max(1e-3, float(np.quantile(np.abs(disparity[finite]), 0.95)))
    normalized = np.clip((disparity / max_abs + 1.0) * 0.5, 0.0, 1.0)
    colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
    canvas[finite] = colored[finite]
    return canvas


def summarize_left_right_audit(
    *,
    normal_run: dict[str, Any],
    swapped_run: dict[str, Any],
    normal_face_metrics: list[dict[str, Any]] | None = None,
    swapped_face_metrics: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    normal_stats = dict(normal_run.get("audit_stats") or compute_disparity_audit_stats(normal_run["disparity"]))
    swapped_stats = dict(swapped_run.get("audit_stats") or compute_disparity_audit_stats(swapped_run["disparity"]))
    normal_valid_depth_ratio = float(np.count_nonzero(np.asarray(normal_run["depth_ir_left_m"], dtype=np.float32) > 0) / max(1, np.asarray(normal_run["depth_ir_left_m"]).size))
    swapped_valid_depth_ratio = float(np.count_nonzero(np.asarray(swapped_run["depth_ir_left_m"], dtype=np.float32) > 0) / max(1, np.asarray(swapped_run["depth_ir_left_m"]).size))

    def _aggregate_face_metrics(items: list[dict[str, Any]] | None) -> dict[str, float]:
        if not items:
            return {
                "patch_count": 0.0,
                "plane_fit_rmse_mm_mean": 0.0,
                "mad_mm_mean": 0.0,
                "p90_mm_mean": 0.0,
                "valid_ratio_mean": 0.0,
            }
        return {
            "patch_count": float(len(items)),
            "plane_fit_rmse_mm_mean": float(np.mean([float(item["plane_fit_rmse_mm"]) for item in items])),
            "mad_mm_mean": float(np.mean([float(item["mad_mm"]) for item in items])),
            "p90_mm_mean": float(np.mean([float(item["p90_abs_residual_mm"]) for item in items])),
            "valid_ratio_mean": float(np.mean([float(item["valid_depth_ratio"]) for item in items])),
        }

    normal_face = _aggregate_face_metrics(normal_face_metrics)
    swapped_face = _aggregate_face_metrics(swapped_face_metrics)
    normal_score = (
        2.0 * float(normal_stats["positive_fraction_of_finite"])
        + 1.0 * normal_valid_depth_ratio
        + 0.8 * normal_face["valid_ratio_mean"]
        - 0.002 * normal_face["plane_fit_rmse_mm_mean"]
        - 0.002 * normal_face["mad_mm_mean"]
        - 0.001 * normal_face["p90_mm_mean"]
    )
    swapped_score = (
        2.0 * float(swapped_stats["positive_fraction_of_finite"])
        + 1.0 * swapped_valid_depth_ratio
        + 0.8 * swapped_face["valid_ratio_mean"]
        - 0.002 * swapped_face["plane_fit_rmse_mm_mean"]
        - 0.002 * swapped_face["mad_mm_mean"]
        - 0.001 * swapped_face["p90_mm_mean"]
    )
    plausible_order = "normal" if normal_score >= swapped_score else "swapped"
    return {
        "normal": {
            "audit_stats": normal_stats,
            "valid_depth_ratio": normal_valid_depth_ratio,
            "face_metrics_summary": normal_face,
            "plausibility_score": float(normal_score),
        },
        "swapped": {
            "audit_stats": swapped_stats,
            "valid_depth_ratio": swapped_valid_depth_ratio,
            "face_metrics_summary": swapped_face,
            "plausibility_score": float(swapped_score),
        },
        "plausible_ordering": plausible_order,
        "score_margin": float(abs(normal_score - swapped_score)),
    }
