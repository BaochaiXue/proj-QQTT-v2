#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


TRACK_MODE_OBJECT_ONLY = "object-only"
TRACK_MODE_CONTROLLER_OBJECT = "controller-object"
TRACK_MODES = (TRACK_MODE_OBJECT_ONLY, TRACK_MODE_CONTROLLER_OBJECT)

DEPTH_SOURCE_FFS = "ffs"
DEPTH_SOURCE_FFS_REMOTE = "ffs_remote"
DEPTH_SOURCE_REALSENSE = "realsense"
DEPTH_SOURCES = (DEPTH_SOURCE_FFS, DEPTH_SOURCE_FFS_REMOTE, DEPTH_SOURCE_REALSENSE)

POSTPROCESS_NONE = "none"
POSTPROCESS_PT_FILTER = "pt-filter"
POSTPROCESS_ENHANCED_PT = "enhanced-pt"
POSTPROCESS_MODES = (POSTPROCESS_NONE, POSTPROCESS_PT_FILTER, POSTPROCESS_ENHANCED_PT)

DEFAULT_CAMERA_IDS = (0, 1, 2)
DEFAULT_OBJECT_LABEL = "object"
DEFAULT_CONTROLLER_LABEL = "controller"
OBJECT_ID = 2
CONTROLLER_ID = 1


@dataclass(frozen=True)
class SemanticLayerSpec:
    obj_id: int
    label: str
    default_postprocess: str


@dataclass(frozen=True)
class CameraLayerCloud:
    camera_idx: int
    label: str
    points_m: np.ndarray
    colors_rgb: np.ndarray


@dataclass(frozen=True)
class FusedLayerCloud:
    label: str
    postprocess_mode: str
    points_m: np.ndarray
    colors_rgb: np.ndarray
    per_camera: tuple[dict[str, int], ...]

    @property
    def point_count(self) -> int:
        return int(self.points_m.shape[0])


def _normalize_label(label: str) -> str:
    return str(label).strip().lower().replace("_", " ").replace("-", " ")


def is_controller_label(label: str) -> bool:
    normalized = _normalize_label(label)
    return normalized in {"controller", "hand", "hands", "left hand", "right hand", "hand a", "hand b"}


def resolve_postprocess_mode(
    label: str,
    *,
    object_postprocess: str = POSTPROCESS_ENHANCED_PT,
    controller_postprocess: str = POSTPROCESS_PT_FILTER,
) -> str:
    if object_postprocess not in POSTPROCESS_MODES:
        raise ValueError(f"Unsupported object postprocess mode: {object_postprocess}")
    if controller_postprocess not in POSTPROCESS_MODES:
        raise ValueError(f"Unsupported controller postprocess mode: {controller_postprocess}")
    if is_controller_label(label):
        return controller_postprocess
    return object_postprocess


def semantic_layers_for_track_mode(
    track_mode: str,
    *,
    object_label: str = DEFAULT_OBJECT_LABEL,
    controller_label: str = DEFAULT_CONTROLLER_LABEL,
    object_postprocess: str = POSTPROCESS_ENHANCED_PT,
    controller_postprocess: str = POSTPROCESS_PT_FILTER,
) -> tuple[SemanticLayerSpec, ...]:
    if track_mode not in TRACK_MODES:
        raise ValueError(f"Unsupported track mode: {track_mode}")
    layers: list[SemanticLayerSpec] = []
    if track_mode == TRACK_MODE_CONTROLLER_OBJECT:
        layers.append(
            SemanticLayerSpec(
                obj_id=CONTROLLER_ID,
                label=str(controller_label),
                default_postprocess=resolve_postprocess_mode(
                    controller_label,
                    object_postprocess=object_postprocess,
                    controller_postprocess=controller_postprocess,
                ),
            )
        )
    layers.append(
        SemanticLayerSpec(
            obj_id=OBJECT_ID,
            label=str(object_label),
            default_postprocess=resolve_postprocess_mode(
                object_label,
                object_postprocess=object_postprocess,
                controller_postprocess=controller_postprocess,
            ),
        )
    )
    return tuple(layers)


def _as_points(points: np.ndarray) -> np.ndarray:
    arr = np.asarray(points, dtype=np.float32)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.float32)
    return arr.reshape(-1, 3)


def _as_colors(colors: np.ndarray) -> np.ndarray:
    arr = np.asarray(colors, dtype=np.uint8)
    if arr.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return arr.reshape(-1, 3)


def fuse_semantic_camera_clouds(
    camera_clouds: Sequence[CameraLayerCloud],
    layers: Sequence[SemanticLayerSpec],
) -> dict[str, FusedLayerCloud]:
    """Fuse cam0/cam1/cam2 clouds per semantic label without mixing labels."""

    clouds_by_label: dict[str, list[CameraLayerCloud]] = {layer.label: [] for layer in layers}
    postprocess_by_label = {layer.label: layer.default_postprocess for layer in layers}
    for cloud in camera_clouds:
        if cloud.label not in clouds_by_label:
            continue
        clouds_by_label[cloud.label].append(cloud)

    fused: dict[str, FusedLayerCloud] = {}
    for label, clouds in clouds_by_label.items():
        point_sets: list[np.ndarray] = []
        color_sets: list[np.ndarray] = []
        per_camera: list[dict[str, int]] = []
        for cloud in clouds:
            points = _as_points(cloud.points_m)
            colors = _as_colors(cloud.colors_rgb)
            if len(colors) != len(points):
                raise ValueError(
                    f"Point/color count mismatch for {label} cam{cloud.camera_idx}: "
                    f"{len(points)} points vs {len(colors)} colors"
                )
            point_sets.append(points)
            color_sets.append(colors)
            per_camera.append(
                {
                    "camera_idx": int(cloud.camera_idx),
                    "point_count": int(len(points)),
                }
            )

        if point_sets:
            fused_points = np.concatenate(point_sets, axis=0)
            fused_colors = np.concatenate(color_sets, axis=0)
        else:
            fused_points = np.empty((0, 3), dtype=np.float32)
            fused_colors = np.empty((0, 3), dtype=np.uint8)

        fused[label] = FusedLayerCloud(
            label=label,
            postprocess_mode=postprocess_by_label[label],
            points_m=fused_points,
            colors_rgb=fused_colors,
            per_camera=tuple(per_camera),
        )
    return fused


def apply_semantic_postprocess(
    layer: FusedLayerCloud,
    *,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Apply the configured semantic PCD cleanup to one fused layer."""

    points = _as_points(layer.points_m)
    colors = _as_colors(layer.colors_rgb)
    if layer.postprocess_mode == POSTPROCESS_NONE:
        return points, colors, {
            "enabled": False,
            "mode": POSTPROCESS_NONE,
            "input_point_count": int(len(points)),
            "output_point_count": int(len(points)),
        }
    if layer.postprocess_mode == POSTPROCESS_PT_FILTER:
        from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
            _apply_phystwin_like_radius_postprocess,
        )

        filtered_points, filtered_colors, stats = _apply_phystwin_like_radius_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
        )
        stats["mode"] = POSTPROCESS_PT_FILTER
        return filtered_points, filtered_colors, stats
    if layer.postprocess_mode == POSTPROCESS_ENHANCED_PT:
        from data_process.visualization.experiments.ffs_confidence_filter_pcd_compare import (
            _apply_enhanced_phystwin_like_postprocess,
        )

        filtered_points, filtered_colors, stats = _apply_enhanced_phystwin_like_postprocess(
            points=points,
            colors=colors,
            enabled=True,
            radius_m=float(phystwin_radius_m),
            nb_points=int(phystwin_nb_points),
            component_voxel_size_m=float(enhanced_component_voxel_size_m),
            keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
        )
        stats["mode"] = POSTPROCESS_ENHANCED_PT
        return filtered_points, filtered_colors, stats
    raise ValueError(f"Unsupported postprocess mode: {layer.postprocess_mode}")


def parse_camera_ids(value: str) -> tuple[int, ...]:
    ids = tuple(int(part.strip()) for part in str(value).split(",") if part.strip())
    if len(ids) != 3:
        raise argparse.ArgumentTypeError("Demo 2.1 expects exactly three camera ids, e.g. 0,1,2")
    if len(set(ids)) != len(ids):
        raise argparse.ArgumentTypeError(f"Camera ids must be unique: {ids}")
    return ids


def build_contract(args: argparse.Namespace) -> dict[str, Any]:
    layers = semantic_layers_for_track_mode(
        args.track_mode,
        object_label=args.object_prompt,
        controller_label=args.controller_prompt,
        object_postprocess=args.object_postprocess,
        controller_postprocess=args.controller_postprocess,
    )
    return {
        "demo": "demo_2_1_three_view_fused_masked_pcd",
        "camera_ids": list(args.camera_ids),
        "track_mode": args.track_mode,
        "frame_by_frame_streaming": True,
        "offline_video_input_used": False,
        "edge_backend": "HF EdgeTAMVideo",
        "compile_mode": args.compile_mode,
        "dtype": args.dtype,
        "depth_source": args.depth_source,
        "official_quality_depth": args.depth_source in {DEPTH_SOURCE_FFS, DEPTH_SOURCE_FFS_REMOTE},
        "native_realsense_depth_role": "fallback/debug only",
        "ffs_contract": {
            "checkpoint": "20-30-48",
            "valid_iters": 4,
            "capture_resolution": "848x480",
            "engine_input": "864x480",
            "builderOptimizationLevel": 5,
        },
        "fusion": {
            "mode": "semantic_layers",
            "labels_are_filtered_separately": True,
            "do_not_filter_object_controller_union": True,
        },
        "semantic_layers": [
            {
                "obj_id": layer.obj_id,
                "label": layer.label,
                "postprocess": layer.default_postprocess,
            }
            for layer in layers
        ],
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Demo 2.1 three-view masked and fused PCD contract. The first implementation "
            "slice locks semantic fusion and postprocess policy before wiring the live hardware loop."
        )
    )
    parser.add_argument("--camera-ids", type=parse_camera_ids, default=DEFAULT_CAMERA_IDS)
    parser.add_argument("--track-mode", choices=TRACK_MODES, default=TRACK_MODE_OBJECT_ONLY)
    parser.add_argument("--object-prompt", default="stuffed animal")
    parser.add_argument("--controller-prompt", default=DEFAULT_CONTROLLER_LABEL)
    parser.add_argument("--depth-source", choices=DEPTH_SOURCES, default=DEPTH_SOURCE_FFS)
    parser.add_argument("--compile-mode", choices=("vision-reduce-overhead",), default="vision-reduce-overhead")
    parser.add_argument("--dtype", choices=("bfloat16", "float16"), default="bfloat16")
    parser.add_argument("--object-postprocess", choices=POSTPROCESS_MODES, default=POSTPROCESS_ENHANCED_PT)
    parser.add_argument("--controller-postprocess", choices=POSTPROCESS_MODES, default=POSTPROCESS_PT_FILTER)
    parser.add_argument("--phystwin-radius-m", type=float, default=0.01)
    parser.add_argument("--phystwin-nb-points", type=int, default=12)
    parser.add_argument("--enhanced-component-voxel-size-m", type=float, default=0.006)
    parser.add_argument("--enhanced-keep-near-main-gap-m", type=float, default=0.035)
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the Demo 2.1 runtime contract and exit without opening cameras.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    contract = build_contract(args)
    if args.dry_run:
        print(json.dumps(contract, indent=2, sort_keys=True))
        return 0
    parser.error(
        "Demo 2.1 live three-camera loop is not wired in this implementation slice yet. "
        "Use --dry-run to inspect the locked fusion/filter contract."
    )
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
