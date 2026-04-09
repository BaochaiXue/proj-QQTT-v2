from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .io_artifacts import write_image
from .layouts import compose_hero_compare


def select_hero_step_index(orbit_steps: list[dict[str, Any]]) -> int:
    if not orbit_steps:
        return 0
    return min(
        range(len(orbit_steps)),
        key=lambda idx: (
            abs(float(orbit_steps[idx].get("angle_deg", 0.0))),
            int(orbit_steps[idx].get("step_idx", idx)),
        ),
    )


def write_hero_compare_image(
    *,
    output_dir: Path,
    mode_name: str,
    case_label: str,
    frame_idx: int,
    angle_deg: float,
    projection_mode: str,
    scene_crop_mode: str,
    native_image: np.ndarray,
    ffs_image: np.ndarray,
    overview_inset: np.ndarray | None,
    warning_text: str | None = None,
) -> Path | None:
    if mode_name not in ("geom", "rgb"):
        return None
    hero_board = compose_hero_compare(
        title_lines=[
            f"{case_label} | Native vs FFS",
            f"frame={frame_idx} | orbit={angle_deg:+.1f} deg | proj={projection_mode} | crop={scene_crop_mode}",
        ],
        native_image=native_image,
        ffs_image=ffs_image,
        overview_inset=overview_inset,
        warning_text=warning_text,
    )
    hero_path = Path(output_dir) / f"hero_compare_{mode_name}.png"
    write_image(hero_path, hero_board)
    return hero_path
