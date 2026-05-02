from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from ..io_artifacts import write_image, write_json, write_ply_ascii
from ..io_case import load_case_frame_camera_clouds, load_case_metadata
from .ffs_confidence_filter_pcd_compare import _apply_enhanced_phystwin_like_postprocess_with_trace
from .native_ffs_fused_pcd_compare import DEFAULT_PHYSTWIN_NB_POINTS, DEFAULT_PHYSTWIN_RADIUS_M
from .sam21_checkpoint_ladder_panel import (
    EDGE_TAM_VARIANT_KEY,
    _build_original_camera_view_specs,
    _format_point_count,
    compose_panel,
    label_tile,
    load_union_mask,
    render_pinhole_point_cloud,
)
from .sam21_mask_overlay_panel import (
    SLOTH_BASE_MOTION_CASE_DIR,
    SLOTH_BASE_MOTION_CASE_KEY,
    SLOTH_BASE_MOTION_CASE_LABEL,
    SLOTH_BASE_MOTION_DOC_JSON,
    SLOTH_BASE_MOTION_OUTPUT_DIR,
    compare_masks,
    sorted_frame_tokens,
)


DEFAULT_OUTPUT_DIR = Path("result/sloth_base_motion_ffs_fused_pcd_overlay_2x3")
DEFAULT_OUTPUT_NAME = "sloth_base_motion_ffs_fused_pcd_overlay_2x3_small_edgetam_compiled"
DEFAULT_DOC_MD = Path("docs/generated/sloth_base_motion_ffs_fused_pcd_overlay_2x3_benchmark.md")
DEFAULT_DOC_JSON = Path("docs/generated/sloth_base_motion_ffs_fused_pcd_overlay_2x3_results.json")
DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT = 0.001
DEFAULT_CAMERA_IDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("small", "SAM2.1 Small"),
    (EDGE_TAM_VARIANT_KEY, "EdgeTAM compiled"),
)
DEFAULT_SAM31_MASK_ROOT = SLOTH_BASE_MOTION_OUTPUT_DIR / "sam31_masks"
DEFAULT_VARIANT_ROOTS: dict[str, Path] = {
    "small": SLOTH_BASE_MOTION_OUTPUT_DIR / "masks" / SLOTH_BASE_MOTION_CASE_KEY / "small",
    EDGE_TAM_VARIANT_KEY: SLOTH_BASE_MOTION_OUTPUT_DIR / "masks" / SLOTH_BASE_MOTION_CASE_KEY / EDGE_TAM_VARIANT_KEY,
}

SAM31_ONLY_BGR = np.asarray((45, 45, 255), dtype=np.uint8)
CANDIDATE_ONLY_BGR = np.asarray((255, 230, 35), dtype=np.uint8)
CATEGORY_OVERLAP = 0
CATEGORY_SAM31_ONLY = 1
CATEGORY_CANDIDATE_ONLY = 2


def _resolve_path(root: Path, value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def metadata_with_depth_scale_override(
    metadata: Mapping[str, Any],
    *,
    depth_scale_override_m_per_unit: float,
) -> tuple[dict[str, Any], bool]:
    payload = dict(metadata)
    num_cameras = len(payload.get("serial_numbers", []))
    if num_cameras <= 0:
        raise ValueError("metadata must contain at least one serial number.")
    existing = payload.get("depth_scale_m_per_unit")
    if existing is None:
        payload["depth_scale_m_per_unit"] = [float(depth_scale_override_m_per_unit) for _ in range(num_cameras)]
        return payload, True
    if not isinstance(existing, list):
        existing = [existing for _ in range(num_cameras)]
    if len(existing) != num_cameras or any(item is None for item in existing):
        payload["depth_scale_m_per_unit"] = [float(depth_scale_override_m_per_unit) for _ in range(num_cameras)]
        return payload, True
    payload["depth_scale_m_per_unit"] = [float(item) for item in existing]
    return payload, False


def _camera_stats_with_view_sources(
    camera_clouds: Sequence[dict[str, Any]],
    camera_stats: dict[str, Any],
) -> dict[str, Any]:
    stats = dict(camera_stats)
    stats["original_view_source_cameras"] = [
        {
            "camera_idx": int(camera_cloud["camera_idx"]),
            "serial": str(camera_cloud["serial"]),
            "K_color": np.asarray(camera_cloud["K_color"], dtype=np.float32).tolist(),
            "c2w": np.asarray(camera_cloud["c2w"], dtype=np.float32).tolist(),
            "color_path": str(camera_cloud["color_path"]),
        }
        for camera_cloud in camera_clouds
    ]
    return stats


def _point_mask_values(mask: np.ndarray, source_pixel_uv: np.ndarray) -> np.ndarray:
    pixel_uv = np.asarray(source_pixel_uv, dtype=np.int32).reshape(-1, 2)
    if len(pixel_uv) == 0:
        return np.zeros((0,), dtype=bool)
    height, width = np.asarray(mask).shape[:2]
    x = np.clip(pixel_uv[:, 0], 0, width - 1)
    y = np.clip(pixel_uv[:, 1], 0, height - 1)
    return np.asarray(mask, dtype=bool)[y, x]


def build_category_colored_cloud(
    *,
    camera_clouds: Sequence[dict[str, Any]],
    reference_masks: Mapping[int, np.ndarray],
    candidate_masks: Mapping[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    point_sets: list[np.ndarray] = []
    color_sets: list[np.ndarray] = []
    category_sets: list[np.ndarray] = []
    per_camera: list[dict[str, Any]] = []

    for camera_cloud in camera_clouds:
        camera_idx = int(camera_cloud["camera_idx"])
        points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        reference = _point_mask_values(reference_masks[camera_idx], np.asarray(camera_cloud["source_pixel_uv"]))
        candidate = _point_mask_values(candidate_masks[camera_idx], np.asarray(camera_cloud["source_pixel_uv"]))
        union = reference | candidate
        overlap = reference & candidate
        reference_only = reference & ~candidate
        candidate_only = candidate & ~reference

        selected_points = points[union]
        selected_colors = colors[union].copy()
        selected_categories = np.full((len(selected_points),), CATEGORY_OVERLAP, dtype=np.uint8)
        selected_reference_only = reference_only[union]
        selected_candidate_only = candidate_only[union]
        selected_categories[selected_reference_only] = CATEGORY_SAM31_ONLY
        selected_categories[selected_candidate_only] = CATEGORY_CANDIDATE_ONLY
        selected_colors[selected_reference_only] = SAM31_ONLY_BGR
        selected_colors[selected_candidate_only] = CANDIDATE_ONLY_BGR

        point_sets.append(selected_points)
        color_sets.append(selected_colors)
        category_sets.append(selected_categories)
        union_count = int(np.count_nonzero(union))
        overlap_count = int(np.count_nonzero(overlap))
        per_camera.append(
            {
                "camera_idx": camera_idx,
                "reference_point_count": int(np.count_nonzero(reference)),
                "candidate_point_count": int(np.count_nonzero(candidate)),
                "overlap_point_count": overlap_count,
                "sam31_only_point_count": int(np.count_nonzero(reference_only)),
                "candidate_only_point_count": int(np.count_nonzero(candidate_only)),
                "union_point_count": union_count,
                "point_weighted_iou": float(overlap_count / union_count) if union_count else 1.0,
            }
        )

    if point_sets:
        points_out = np.concatenate(point_sets, axis=0)
        colors_out = np.concatenate(color_sets, axis=0)
        categories_out = np.concatenate(category_sets, axis=0)
    else:
        points_out = np.empty((0, 3), dtype=np.float32)
        colors_out = np.empty((0, 3), dtype=np.uint8)
        categories_out = np.empty((0,), dtype=np.uint8)

    counts = category_counts(categories_out)
    return points_out, colors_out, categories_out, {
        "per_camera": per_camera,
        "overlap_point_count": int(counts["overlap_point_count"]),
        "sam31_only_point_count": int(counts["sam31_only_point_count"]),
        "candidate_only_point_count": int(counts["candidate_only_point_count"]),
        "union_point_count": int(counts["total_point_count"]),
        "point_weighted_iou": float(
            counts["overlap_point_count"] / counts["total_point_count"]
        )
        if int(counts["total_point_count"])
        else 1.0,
    }


def category_counts(categories: np.ndarray) -> dict[str, int]:
    category_array = np.asarray(categories, dtype=np.uint8).reshape(-1)
    overlap = int(np.count_nonzero(category_array == CATEGORY_OVERLAP))
    sam31_only = int(np.count_nonzero(category_array == CATEGORY_SAM31_ONLY))
    candidate_only = int(np.count_nonzero(category_array == CATEGORY_CANDIDATE_ONLY))
    return {
        "overlap_point_count": overlap,
        "sam31_only_point_count": sam31_only,
        "candidate_only_point_count": candidate_only,
        "total_point_count": int(len(category_array)),
    }


def _postprocess_category_cloud(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    categories: np.ndarray,
    phystwin_radius_m: float,
    phystwin_nb_points: int,
    enhanced_component_voxel_size_m: float,
    enhanced_keep_near_main_gap_m: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], dict[str, int]]:
    filtered_points, filtered_colors, postprocess_stats, trace = _apply_enhanced_phystwin_like_postprocess_with_trace(
        points=points,
        colors=colors,
        enabled=True,
        radius_m=float(phystwin_radius_m),
        nb_points=int(phystwin_nb_points),
        component_voxel_size_m=float(enhanced_component_voxel_size_m),
        keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
    )
    kept_mask = np.asarray(trace.get("kept_mask", np.ones((len(categories),), dtype=bool)), dtype=bool)
    filtered_categories = np.asarray(categories, dtype=np.uint8).reshape(-1)[kept_mask]
    counts = category_counts(filtered_categories)
    return filtered_points, filtered_colors, filtered_categories, postprocess_stats, counts


def _load_variant_masks(
    *,
    case_dir: Path,
    mask_root: Path,
    camera_ids: Sequence[int],
    frame_token: str,
    text_prompt: str,
) -> dict[int, np.ndarray]:
    return {
        int(camera_idx): load_union_mask(
            mask_root=mask_root,
            case_dir=case_dir,
            camera_idx=int(camera_idx),
            frame_token=str(frame_token),
            text_prompt=str(text_prompt),
        )
        for camera_idx in camera_ids
    }


def _aggregate_2d_stats(stats_by_camera: Mapping[int, dict[str, Any]]) -> dict[str, Any]:
    values = list(stats_by_camera.values())
    intersection = int(sum(int(item["intersection_pixel_count"]) for item in values))
    union = int(sum(int(item["union_pixel_count"]) for item in values))
    reference = int(sum(int(item["reference_pixel_count"]) for item in values))
    candidate = int(sum(int(item["candidate_pixel_count"]) for item in values))
    return {
        "reference_pixel_count": reference,
        "candidate_pixel_count": candidate,
        "intersection_pixel_count": intersection,
        "union_pixel_count": union,
        "reference_only_pixel_count": int(sum(int(item["reference_only_pixel_count"]) for item in values)),
        "candidate_only_pixel_count": int(sum(int(item["candidate_only_pixel_count"]) for item in values)),
        "iou": float(intersection / union) if union else 1.0,
        "area_ratio_candidate_over_reference": float(candidate / reference) if reference else 0.0,
    }


def _timing_by_variant(timing_records: Sequence[Mapping[str, Any]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for record in timing_records:
        key = str(record.get("checkpoint_key", ""))
        if key:
            grouped.setdefault(key, []).append(record)
    result: dict[str, dict[str, float]] = {}
    for key, records in grouped.items():
        ms_values = [float(item["inference_ms_per_frame"]) for item in records if "inference_ms_per_frame" in item]
        fps_values = [float(item["fps"]) for item in records if "fps" in item]
        if ms_values:
            result[key] = {
                "mean_inference_ms_per_frame": float(np.mean(ms_values)),
                "mean_fps": float(np.mean(fps_values)) if fps_values else float(1000.0 / np.mean(ms_values)),
                "record_count": int(len(ms_values)),
            }
    return result


def _cell_label(
    *,
    variant_label: str,
    variant_key: str,
    timing_summary: Mapping[str, Mapping[str, float]],
    point_iou: float,
    point_count: int,
) -> str:
    timing = timing_summary.get(str(variant_key), {})
    ms = float(timing.get("mean_inference_ms_per_frame", 0.0))
    if ms > 0.0:
        return f"{variant_label} | {ms:.1f}ms/f | pIoU {float(point_iou):.3f} | {_format_point_count(point_count)} pts"
    return f"{variant_label} | pIoU {float(point_iou):.3f} | {_format_point_count(point_count)} pts"


def _render_overlay_tile(
    *,
    points: np.ndarray,
    colors: np.ndarray,
    view_spec: Mapping[str, Any],
    label: str,
    tile_width: int,
    tile_height: int,
    max_points_per_render: int | None,
) -> np.ndarray:
    point_array = np.asarray(points, dtype=np.float32).reshape(-1, 3)
    color_array = np.asarray(colors, dtype=np.uint8).reshape(-1, 3)
    if max_points_per_render is not None and len(point_array) > int(max_points_per_render):
        indices = np.linspace(0, len(point_array) - 1, int(max_points_per_render), dtype=np.int64)
        point_array = point_array[indices]
        color_array = color_array[indices]
    rendered = render_pinhole_point_cloud(
        point_array,
        color_array,
        intrinsic_matrix=np.asarray(view_spec["intrinsic_matrix"], dtype=np.float32),
        extrinsic_matrix=np.asarray(view_spec["extrinsic_matrix"], dtype=np.float32),
        width=int(tile_width),
        height=int(tile_height),
        point_radius_px=1,
        background_bgr=(6, 6, 7),
    )
    return label_tile(rendered, label=label, tile_width=int(tile_width), tile_height=int(tile_height))


def render_fused_pcd_overlay_2x3_gif(
    *,
    root: str | Path,
    case_dir: str | Path = SLOTH_BASE_MOTION_CASE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    case_key: str = SLOTH_BASE_MOTION_CASE_KEY,
    case_label: str = SLOTH_BASE_MOTION_CASE_LABEL,
    text_prompt: str = "sloth,stuffed animal",
    sam31_mask_root: str | Path = DEFAULT_SAM31_MASK_ROOT,
    variant_roots: Mapping[str, str | Path] | None = None,
    variants: Sequence[tuple[str, str]] = DEFAULT_VARIANTS,
    timing_records: Sequence[Mapping[str, Any]] = (),
    frames: int | None = None,
    gif_fps: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
    row_label_width: int = 140,
    depth_scale_override_m_per_unit: float = DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    max_points_per_camera: int | None = None,
    max_points_per_render: int | None = 100_000,
    phystwin_radius_m: float = DEFAULT_PHYSTWIN_RADIUS_M,
    phystwin_nb_points: int = DEFAULT_PHYSTWIN_NB_POINTS,
    enhanced_component_voxel_size_m: float = 0.01,
    enhanced_keep_near_main_gap_m: float = 0.0,
    output_name: str = DEFAULT_OUTPUT_NAME,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    case_path = _resolve_path(root_path, case_dir)
    output_path = _resolve_path(root_path, output_dir)
    sam31_root = _resolve_path(root_path, sam31_mask_root)
    selected_variants = tuple((str(key), str(label)) for key, label in variants)
    roots = {key: _resolve_path(root_path, value) for key, value in DEFAULT_VARIANT_ROOTS.items()}
    if variant_roots:
        roots.update({str(key): _resolve_path(root_path, value) for key, value in variant_roots.items()})
    for variant_key, _label in selected_variants:
        if variant_key not in roots:
            raise KeyError(f"No mask root configured for variant {variant_key!r}")

    metadata_raw = load_case_metadata(case_path)
    metadata, depth_scale_override_used = metadata_with_depth_scale_override(
        metadata_raw,
        depth_scale_override_m_per_unit=float(depth_scale_override_m_per_unit),
    )
    frame_tokens = sorted_frame_tokens(case_path, camera_idx=0, frames=frames)
    available_frames = len(frame_tokens)
    camera_ids = DEFAULT_CAMERA_IDS
    gif_dir = output_path / "gifs"
    first_dir = output_path / "first_frames"
    ply_dir = output_path / "first_frame_ply"
    gif_dir.mkdir(parents=True, exist_ok=True)
    first_dir.mkdir(parents=True, exist_ok=True)
    ply_dir.mkdir(parents=True, exist_ok=True)
    gif_path = gif_dir / f"{output_name}.gif"
    first_frame_path = first_dir / f"{output_name}_first.png"
    timing_summary = _timing_by_variant(timing_records)

    per_frame: list[dict[str, Any]] = []
    aggregate_values: dict[str, dict[str, list[float]]] = {
        key: {
            "mask_iou": [],
            "raw_point_iou": [],
            "postprocess_point_iou": [],
            "output_point_count": [],
        }
        for key, _label in selected_variants
    }
    view_specs: dict[int, dict[str, Any]] | None = None

    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(gif_fps)), loop=0) as writer:
        for frame_idx, frame_token in enumerate(frame_tokens):
            camera_clouds, camera_stats = load_case_frame_camera_clouds(
                case_dir=case_path,
                metadata=metadata,
                frame_idx=int(frame_token),
                depth_source="ffs",
                use_float_ffs_depth_when_available=True,
                max_points_per_camera=max_points_per_camera,
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
            )
            camera_clouds = [
                camera_cloud
                for camera_cloud in camera_clouds
                if int(camera_cloud["camera_idx"]) in set(int(item) for item in camera_ids)
            ]
            if view_specs is None:
                view_specs = _build_original_camera_view_specs(
                    _camera_stats_with_view_sources(camera_clouds, camera_stats),
                    tile_width=int(tile_width),
                    tile_height=int(tile_height),
                )
            reference_masks = _load_variant_masks(
                case_dir=case_path,
                mask_root=sam31_root,
                camera_ids=camera_ids,
                frame_token=str(frame_token),
                text_prompt=str(text_prompt),
            )

            image_rows: list[list[np.ndarray]] = []
            frame_variants: list[dict[str, Any]] = []
            first_frame_plys: list[tuple[str, np.ndarray, np.ndarray]] = []
            for variant_key, variant_label in selected_variants:
                candidate_masks = _load_variant_masks(
                    case_dir=case_path,
                    mask_root=roots[variant_key],
                    camera_ids=camera_ids,
                    frame_token=str(frame_token),
                    text_prompt=str(text_prompt),
                )
                mask_stats_by_camera = {
                    int(camera_idx): compare_masks(reference_masks[int(camera_idx)], candidate_masks[int(camera_idx)])
                    for camera_idx in camera_ids
                }
                aggregate_2d = _aggregate_2d_stats(mask_stats_by_camera)
                raw_points, raw_colors, raw_categories, raw_point_stats = build_category_colored_cloud(
                    camera_clouds=camera_clouds,
                    reference_masks=reference_masks,
                    candidate_masks=candidate_masks,
                )
                points, colors, categories, postprocess_stats, post_counts = _postprocess_category_cloud(
                    points=raw_points,
                    colors=raw_colors,
                    categories=raw_categories,
                    phystwin_radius_m=float(phystwin_radius_m),
                    phystwin_nb_points=int(phystwin_nb_points),
                    enhanced_component_voxel_size_m=float(enhanced_component_voxel_size_m),
                    enhanced_keep_near_main_gap_m=float(enhanced_keep_near_main_gap_m),
                )
                post_point_iou = (
                    float(post_counts["overlap_point_count"] / post_counts["total_point_count"])
                    if int(post_counts["total_point_count"])
                    else 1.0
                )
                aggregate_values[variant_key]["mask_iou"].append(float(aggregate_2d["iou"]))
                aggregate_values[variant_key]["raw_point_iou"].append(float(raw_point_stats["point_weighted_iou"]))
                aggregate_values[variant_key]["postprocess_point_iou"].append(float(post_point_iou))
                aggregate_values[variant_key]["output_point_count"].append(float(post_counts["total_point_count"]))
                row_tiles: list[np.ndarray] = []
                label = _cell_label(
                    variant_label=variant_label,
                    variant_key=variant_key,
                    timing_summary=timing_summary,
                    point_iou=post_point_iou,
                    point_count=int(post_counts["total_point_count"]),
                )
                for camera_idx in camera_ids:
                    row_tiles.append(
                        _render_overlay_tile(
                            points=points,
                            colors=colors,
                            view_spec=view_specs[int(camera_idx)],
                            label=label,
                            tile_width=int(tile_width),
                            tile_height=int(tile_height),
                            max_points_per_render=max_points_per_render,
                        )
                    )
                image_rows.append(row_tiles)
                if frame_idx == 0:
                    first_frame_plys.append((variant_key, points, colors))
                frame_variants.append(
                    {
                        "variant_key": variant_key,
                        "variant_label": variant_label,
                        "mask_iou_by_camera": {
                            str(camera_idx): mask_stats_by_camera[int(camera_idx)]
                            for camera_idx in camera_ids
                        },
                        "mask_iou_aggregate": aggregate_2d,
                        "raw_point_metrics": raw_point_stats,
                        "postprocess_point_metrics": {
                            **post_counts,
                            "point_weighted_iou": post_point_iou,
                        },
                        "postprocess_stats": postprocess_stats,
                    }
                )

            board = compose_panel(
                title_lines=[
                    f"{case_label} | fused PCD overlay vs SAM3.1 | frame {frame_idx + 1}/{available_frames}",
                    "rows=model, columns=original camera pinhole view | red=SAM3.1 only | cyan=candidate only | overlap=RGB",
                ],
                row_headers=[label for _key, label in selected_variants],
                column_headers=[f"cam{camera_idx}" for camera_idx in camera_ids],
                image_rows=image_rows,
                row_label_width=int(row_label_width),
                expected_rows=len(selected_variants),
            )
            if frame_idx == 0:
                write_image(first_frame_path, board)
                for variant_key, points, colors in first_frame_plys:
                    write_ply_ascii(
                        ply_dir / f"{case_key}_{variant_key}_frame0000_fused_pcd_overlay.ply",
                        points,
                        colors,
                    )
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            per_frame.append({"frame_idx": int(frame_idx), "frame_token": str(frame_token), "variants": frame_variants})
            if frame_idx == 0 or frame_idx + 1 == available_frames or (frame_idx + 1) % 10 == 0:
                print(f"[pcd-overlay] rendered {case_key} frame {frame_idx + 1}/{available_frames}", flush=True)

    aggregate: dict[str, dict[str, Any]] = {}
    for variant_key, values_by_metric in aggregate_values.items():
        aggregate[variant_key] = {}
        for metric_key, values in values_by_metric.items():
            aggregate[variant_key][metric_key] = {
                "mean": float(np.mean(values)) if values else 0.0,
                "min": float(np.min(values)) if values else 0.0,
                "max": float(np.max(values)) if values else 0.0,
                "sample_count": int(len(values)),
            }

    return {
        "case_key": str(case_key),
        "case_label": str(case_label),
        "case_dir": str(case_path),
        "text_prompt": str(text_prompt),
        "frames": int(available_frames),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "first_frame_ply_dir": str(ply_dir),
        "sam31_mask_root": str(sam31_root),
        "variant_roots": {key: str(value) for key, value in roots.items()},
        "variants": [{"key": key, "label": label} for key, label in selected_variants],
        "camera_ids": [int(item) for item in camera_ids],
        "view_mode": "fused_pcd_original_camera_pinhole",
        "overlay_legend": {
            "overlap": "candidate and SAM3.1 overlap keeps original RGB point color",
            "red": "SAM3.1 only",
            "cyan": "candidate only",
            "background": "dark pinhole render background",
        },
        "depth": {
            "source": "depth",
            "depth_source_argument": "ffs",
            "depth_scale_override_m_per_unit": float(depth_scale_override_m_per_unit),
            "depth_scale_override_used": bool(depth_scale_override_used),
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
        },
        "postprocess": {
            "mode": "enhanced_phystwin_like_radius_then_component_filter",
            "radius_m": float(phystwin_radius_m),
            "nb_points": int(phystwin_nb_points),
            "component_voxel_size_m": float(enhanced_component_voxel_size_m),
            "keep_near_main_gap_m": float(enhanced_keep_near_main_gap_m),
        },
        "timing_summary": timing_summary,
        "timing_records": [dict(item) for item in timing_records],
        "aggregate": aggregate,
        "frames_detail": per_frame,
    }


def write_pcd_overlay_report(markdown_path: str | Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# sloth_base_motion_ffs Fused PCD Overlay 2x3",
        "",
        "## Output",
        "",
        f"- GIF: `{summary.get('gif_path')}`",
        f"- first frame: `{summary.get('first_frame_path')}`",
        f"- first-frame PLY dir: `{summary.get('first_frame_ply_dir')}`",
        f"- case: `{summary.get('case_key')}`",
        f"- frames: `{summary.get('frames')}`",
        "",
        "## Render Contract",
        "",
        "- rows: SAM2.1 Small, EdgeTAM compiled",
        "- columns: cam0/cam1/cam2 original camera pinhole views",
        "- fused PCD: three masked camera RGB point clouds fused before rendering",
        "- overlap: RGB point color",
        "- red: SAM3.1-only points",
        "- cyan: candidate-only points",
        "",
        "## Depth",
        "",
        f"- source: `{summary.get('depth', {}).get('source')}`",
        f"- depth scale override: `{summary.get('depth', {}).get('depth_scale_override_m_per_unit')}`",
        f"- override used: `{summary.get('depth', {}).get('depth_scale_override_used')}`",
        "",
        "## Aggregate Metrics",
        "",
        "| variant | mean 2D IoU | min 2D IoU | mean raw pIoU | mean post pIoU | mean output pts | mean ms/f | mean FPS |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    timing = summary.get("timing_summary", {})
    for variant in summary.get("variants", []):
        key = str(variant["key"])
        aggregate = summary.get("aggregate", {}).get(key, {})
        timing_stats = timing.get(key, {})
        lines.append(
            f"| {key} | "
            f"{float(aggregate.get('mask_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('mask_iou', {}).get('min', 0.0)):.4f} | "
            f"{float(aggregate.get('raw_point_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('postprocess_point_iou', {}).get('mean', 0.0)):.4f} | "
            f"{float(aggregate.get('output_point_count', {}).get('mean', 0.0)):.0f} | "
            f"{float(timing_stats.get('mean_inference_ms_per_frame', 0.0)):.2f} | "
            f"{float(timing_stats.get('mean_fps', 0.0)):.2f} |"
        )
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def load_sloth_existing_timing_records(results_json: str | Path) -> list[dict[str, Any]]:
    path = Path(results_json)
    payload = json.loads(path.read_text(encoding="utf-8"))
    wanted = {"small", EDGE_TAM_VARIANT_KEY}
    records: list[dict[str, Any]] = []
    for record in list(payload.get("sam21_timing_records", [])) + list(payload.get("edgetam_timing_records", [])):
        if str(record.get("checkpoint_key", "")) in wanted:
            records.append(dict(record))
    return records


def run_sloth_base_motion_fused_pcd_overlay_workflow(
    *,
    root: str | Path,
    case_dir: str | Path = SLOTH_BASE_MOTION_CASE_DIR,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    existing_results_json: str | Path = SLOTH_BASE_MOTION_DOC_JSON,
    frames: int | None = None,
    gif_fps: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
    row_label_width: int = 140,
    depth_scale_override_m_per_unit: float = DEFAULT_DEPTH_SCALE_OVERRIDE_M_PER_UNIT,
    max_points_per_camera: int | None = None,
    max_points_per_render: int | None = 100_000,
    doc_md: str | Path = DEFAULT_DOC_MD,
    doc_json: str | Path = DEFAULT_DOC_JSON,
) -> dict[str, Any]:
    root_path = Path(root).resolve()
    existing_json = _resolve_path(root_path, existing_results_json)
    timing_records = load_sloth_existing_timing_records(existing_json)
    summary = render_fused_pcd_overlay_2x3_gif(
        root=root_path,
        case_dir=case_dir,
        output_dir=output_dir,
        timing_records=timing_records,
        frames=frames,
        gif_fps=int(gif_fps),
        tile_width=int(tile_width),
        tile_height=int(tile_height),
        row_label_width=int(row_label_width),
        depth_scale_override_m_per_unit=float(depth_scale_override_m_per_unit),
        max_points_per_camera=max_points_per_camera,
        max_points_per_render=max_points_per_render,
    )
    json_path = _resolve_path(root_path, doc_json)
    md_path = _resolve_path(root_path, doc_md)
    write_json(json_path, summary)
    write_pcd_overlay_report(md_path, summary)
    write_json(
        _resolve_path(root_path, output_dir) / "summary.json",
        {**summary, "docs": {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}},
    )
    summary["docs"] = {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}
    return summary
