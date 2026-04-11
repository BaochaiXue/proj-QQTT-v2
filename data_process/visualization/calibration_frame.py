from __future__ import annotations

from typing import Any

from .types import DisplayFrameContract

CALIBRATION_WORLD_FRAME_KIND = "charuco_board_world_c2w"
SEMANTIC_WORLD_FRAME_KIND = "semantic_world"
OVERVIEW_DISPLAY_FRAME_KIND = "calibration_world_topdown_display"
SEMANTIC_OVERVIEW_DISPLAY_FRAME_KIND = "semantic_world_topdown_display"


def build_visualization_frame_contract(
    *,
    display_frame: str = "calibration_world",
    uses_semantic_world: bool,
    semantic_world_frame_kind: str | None = None,
    overview_display_frame_kind: str = OVERVIEW_DISPLAY_FRAME_KIND,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return DisplayFrameContract(
        display_frame=display_frame,
        calibration_world_frame_kind=CALIBRATION_WORLD_FRAME_KIND,
        uses_semantic_world=bool(uses_semantic_world),
        semantic_world_frame_kind=semantic_world_frame_kind,
        overview_display_frame_kind=overview_display_frame_kind,
        notes=[] if notes is None else list(notes),
    ).to_dict()
