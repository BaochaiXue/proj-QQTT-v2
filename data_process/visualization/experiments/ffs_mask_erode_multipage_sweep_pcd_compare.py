from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import numpy as np

from .ffs_confidence_panels import DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT
from .ffs_mask_erode_sweep_pcd_compare import (
    DEFAULT_ROW_LABEL_WIDTH,
    build_default_mask_erode_multipage_specs,
    run_ffs_mask_erode_sweep_pcd_workflow,
)
from .native_ffs_fused_pcd_compare import DEFAULT_PHYSTWIN_NB_POINTS, DEFAULT_PHYSTWIN_RADIUS_M


def run_ffs_mask_erode_multipage_sweep_pcd_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    frame_idx: int = 0,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    point_size: float = 2.0,
    look_distance: float = 1.0,
    tile_width: int = 480,
    tile_height: int = 360,
    row_label_width: int = DEFAULT_ROW_LABEL_WIDTH,
    max_points_per_camera: int | None = 80_000,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    use_object_mask: bool = True,
    phystwin_like_postprocess: bool = True,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    use_float_ffs_depth_when_available: bool = True,
    round_specs: list[dict[str, Any]] | None = None,
    render_frame_fn: Callable[..., np.ndarray] | None = None,
    board_page_specs: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    page_specs = build_default_mask_erode_multipage_specs() if board_page_specs is None else board_page_specs
    return run_ffs_mask_erode_sweep_pcd_workflow(
        aligned_root=Path(aligned_root),
        output_root=Path(output_root),
        frame_idx=int(frame_idx),
        erode_pixels=[1],
        depth_min_m=float(depth_min_m),
        depth_max_m=float(depth_max_m),
        point_size=float(point_size),
        look_distance=float(look_distance),
        tile_width=int(tile_width),
        tile_height=int(tile_height),
        row_label_width=int(row_label_width),
        max_points_per_camera=max_points_per_camera,
        text_prompt=str(text_prompt),
        use_object_mask=bool(use_object_mask),
        phystwin_like_postprocess=bool(phystwin_like_postprocess),
        phystwin_radius_m=float(phystwin_radius_m),
        phystwin_nb_points=int(phystwin_nb_points),
        use_float_ffs_depth_when_available=bool(use_float_ffs_depth_when_available),
        round_specs=round_specs,
        render_frame_fn=render_frame_fn,
        board_page_specs=page_specs,
    )
