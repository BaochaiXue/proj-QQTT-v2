from __future__ import annotations


# Generic fused point-cloud workflows in this repo target tabletop captures by default.
# Keep only positive metric depth and clip farther background geometry aggressively.
DEFAULT_POINTCLOUD_DEPTH_MIN_M = 0.0
DEFAULT_POINTCLOUD_DEPTH_MAX_M = 1.5
