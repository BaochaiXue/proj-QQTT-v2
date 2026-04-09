from __future__ import annotations

from typing import Any

import numpy as np


def format_angle_token(angle_deg: float) -> str:
    sign = "p" if angle_deg >= 0 else "m"
    scaled = int(round(abs(float(angle_deg)) * 10.0))
    return f"{sign}{scaled:04d}"


def source_histogram(source_camera_idx: np.ndarray) -> dict[str, int]:
    values = np.asarray(source_camera_idx, dtype=np.int16).reshape(-1)
    if len(values) == 0:
        return {}
    unique_values, counts = np.unique(values, return_counts=True)
    return {str(int(camera_idx)): int(count) for camera_idx, count in zip(unique_values, counts, strict=False)}


def aggregate_step_metric_series(step_metrics: list[dict[str, Any]], *, key: str) -> float:
    values = [float(item[key]) for item in step_metrics if key in item]
    return float(np.mean(values)) if values else 0.0
