from __future__ import annotations

import math
from typing import Any, Mapping


def resolve_simulation_timing(
    fps: float,
    *,
    max_integration_dt_s: float,
) -> dict[str, float | int]:
    """Resolve an exact observation interval with a bounded integration step."""
    fps_value = float(fps)
    max_dt = float(max_integration_dt_s)
    if not math.isfinite(fps_value) or fps_value <= 0.0:
        raise ValueError(f"fps must be positive, got {fps!r}")
    if not math.isfinite(max_dt) or max_dt <= 0.0:
        raise ValueError(
            "max_integration_dt_s must be positive, "
            f"got {max_integration_dt_s!r}"
        )
    frame_dt = 1.0 / fps_value
    num_substeps = max(1, int(math.ceil(frame_dt / max_dt)))
    integration_dt = frame_dt / float(num_substeps)
    return {
        "fps": fps_value,
        "frame_dt_s": frame_dt,
        "integration_dt_s": integration_dt,
        "num_substeps": num_substeps,
    }


def resolve_manifest_timing(
    manifest: Mapping[str, Any],
    *,
    max_integration_dt_s: float,
) -> dict[str, float | int]:
    timing = resolve_simulation_timing(
        float(manifest["fps"]),
        max_integration_dt_s=max_integration_dt_s,
    )
    declared = manifest.get("frame_dt_s")
    if declared is not None and not math.isclose(
        float(declared),
        float(timing["frame_dt_s"]),
        rel_tol=0.0,
        abs_tol=1e-9,
    ):
        raise ValueError(
            "manifest frame_dt_s does not match fps: "
            f"fps={timing['fps']}, frame_dt_s={declared}"
        )
    return timing


__all__ = ["resolve_manifest_timing", "resolve_simulation_timing"]
