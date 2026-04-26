from __future__ import annotations

import numpy as np


def _ratio(count: int, total: int) -> float:
    return float(int(count) / max(1, int(total)))


def _confidence_to_uint8(confidence: np.ndarray, *, valid_depth: np.ndarray) -> np.ndarray:
    confidence_float = np.asarray(confidence, dtype=np.float32)
    confidence_clean = np.nan_to_num(confidence_float, nan=0.0, posinf=0.0, neginf=0.0)
    confidence_uint8 = np.rint(np.clip(confidence_clean, 0.0, 1.0) * 255.0).astype(np.uint8)
    confidence_uint8[~np.asarray(valid_depth, dtype=bool)] = 0
    return confidence_uint8


def build_confidence_filtered_depth_uint16(
    *,
    depth_m: np.ndarray,
    confidence: np.ndarray | None,
    confidence_threshold: float | None,
    depth_scale_m_per_unit: float,
    depth_min_m: float,
    depth_max_m: float,
    object_mask: np.ndarray | None = None,
) -> dict[str, np.ndarray | dict[str, float]]:
    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"Expected depth_m shaped [H, W], got {depth.shape}.")
    scale = float(depth_scale_m_per_unit)
    if scale <= 0.0:
        raise ValueError(f"depth_scale_m_per_unit must be > 0, got {depth_scale_m_per_unit}.")
    min_m = float(depth_min_m)
    max_m = float(depth_max_m)
    if min_m > max_m:
        raise ValueError(f"depth_min_m must be <= depth_max_m, got {min_m} > {max_m}.")

    valid_depth = np.isfinite(depth) & (depth > 0.0) & (depth >= min_m) & (depth <= max_m)
    if object_mask is not None:
        mask = np.asarray(object_mask, dtype=bool)
        if mask.shape != depth.shape:
            raise ValueError(f"object_mask must match depth_m shape. Got {mask.shape} vs {depth.shape}.")
        valid_depth &= mask

    if confidence is None:
        confidence_uint8 = np.zeros(depth.shape, dtype=np.uint8)
        confidence_uint8[valid_depth] = 255
        valid = valid_depth
        low_confidence_reject_count = 0
    else:
        confidence_float = np.asarray(confidence, dtype=np.float32)
        if confidence_float.shape != depth.shape:
            raise ValueError(f"confidence must match depth_m shape. Got {confidence_float.shape} vs {depth.shape}.")
        confidence_uint8 = _confidence_to_uint8(confidence_float, valid_depth=valid_depth)
        if confidence_threshold is None:
            valid = valid_depth
            low_confidence_reject_count = 0
        else:
            threshold = float(confidence_threshold)
            confidence_keep = np.isfinite(confidence_float) & (confidence_float >= threshold)
            valid = valid_depth & confidence_keep
            low_confidence_reject_count = int(np.count_nonzero(valid_depth & ~confidence_keep))

    depth_uint16 = np.zeros(depth.shape, dtype=np.uint16)
    if np.any(valid):
        depth_encoded = np.rint(depth[valid] / scale)
        depth_uint16[valid] = np.clip(depth_encoded, 1, np.iinfo(np.uint16).max).astype(np.uint16)

    valid_mask_uint8 = np.asarray(valid, dtype=np.uint8)
    pixel_count = int(depth.size)
    valid_before_count = int(np.count_nonzero(valid_depth))
    valid_after_count = int(np.count_nonzero(valid))
    stats: dict[str, float] = {
        "pixel_count": float(pixel_count),
        "valid_depth_ratio_before_confidence": _ratio(valid_before_count, pixel_count),
        "valid_ratio_after_confidence": _ratio(valid_after_count, pixel_count),
        "hole_ratio_after_confidence": 1.0 - _ratio(valid_after_count, pixel_count),
        "low_confidence_reject_ratio": _ratio(low_confidence_reject_count, pixel_count),
        "depth_min_m": min_m,
        "depth_max_m": max_m,
        "confidence_threshold": 0.0 if confidence_threshold is None else float(confidence_threshold),
    }
    return {
        "depth_uint16": depth_uint16,
        "valid_mask_uint8": valid_mask_uint8,
        "confidence_uint8": confidence_uint8,
        "stats": stats,
    }
