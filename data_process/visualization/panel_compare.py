from __future__ import annotations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .depth_diagnostics import (
    absolute_depth_difference_heatmap,
    annotate_rois,
    build_rgb_depth_edge_overlay,
    clamp_roi,
    colorize_depth_map,
    compute_depth_panel_metrics,
    default_rois,
    estimate_depth_difference_scale,
    label_tile,
    load_color_frame,
    load_depth_frame,
    make_roi_tile,
    normalize_roi_entries,
    resolve_camera_ids,
    shaded_depth_map,
    valid_mask_comparison,
)
from .io_artifacts import write_json
from .layouts import compose_depth_review_board, overlay_scalar_colorbar
from .pointcloud_compare import (
    get_frame_count,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
    write_video,
)


def _aggregate_camera_metrics(frame_metrics: list[dict[str, Any]]) -> dict[str, float | None]:
    if not frame_metrics:
        return {
            "native_valid_ratio_mean": None,
            "ffs_valid_ratio_mean": None,
            "median_abs_diff_m_mean": None,
            "p90_abs_diff_m_mean": None,
        }

    def _mean_or_none(key: str) -> float | None:
        values = [float(item[key]) for item in frame_metrics if item.get(key) is not None]
        if not values:
            return None
        return float(np.mean(values))

    return {
        "native_valid_ratio_mean": _mean_or_none("native_valid_ratio"),
        "ffs_valid_ratio_mean": _mean_or_none("ffs_valid_ratio"),
        "median_abs_diff_m_mean": _mean_or_none("median_abs_diff_m"),
        "p90_abs_diff_m_mean": _mean_or_none("p90_abs_diff_m"),
    }


def run_depth_panel_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    camera_ids: list[int] | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
    rois: list[dict[str, Any] | tuple[int, int, int, int]] | None = None,
    write_mp4: bool = False,
    fps: int = 10,
    use_float_ffs_depth_when_available: bool = True,
    ffs_native_like_postprocess: bool = False,
    preset: str | None = None,
    show_edge_overlay: bool = False,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    native_case_dir, ffs_case_dir, same_case_mode = resolve_case_dirs(
        aligned_root=aligned_root,
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
    )
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)
    selected_cameras = resolve_camera_ids(native_metadata, camera_ids)
    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for panel visualization.")

    output_dir.mkdir(parents=True, exist_ok=True)
    review_quality = preset == "review_quality"
    main_tile_size = (520, 320) if review_quality else (380, 240)
    roi_tile_size = (760, 400) if review_quality else (560, 300)
    summary = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "camera_ids": selected_cameras,
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "ffs_native_like_postprocess_enabled": bool(ffs_native_like_postprocess),
        "preset": preset,
        "show_edge_overlay": bool(show_edge_overlay),
        "per_camera": {},
    }

    for camera_idx in selected_cameras:
        camera_dir = output_dir / f"camera_{camera_idx}"
        frames_dir = camera_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        resolved_rois = None
        frame_metrics: list[dict[str, Any]] = []

        for panel_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
            native_rgb = load_color_frame(native_case_dir, camera_idx, native_frame_idx)
            ffs_rgb = native_rgb if same_case_mode else load_color_frame(ffs_case_dir, camera_idx, ffs_frame_idx)
            _, native_depth_m, native_info = load_depth_frame(
                case_dir=native_case_dir,
                metadata=native_metadata,
                camera_idx=camera_idx,
                frame_idx=native_frame_idx,
                depth_source="realsense",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                ffs_native_like_postprocess=False,
            )
            _, ffs_depth_m, ffs_info = load_depth_frame(
                case_dir=ffs_case_dir,
                metadata=ffs_metadata,
                camera_idx=camera_idx,
                frame_idx=ffs_frame_idx,
                depth_source="ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
                ffs_native_like_postprocess=ffs_native_like_postprocess,
            )

            camera_rois = rois or default_rois(native_rgb.shape[:2])
            camera_rois = normalize_roi_entries(camera_rois, image_shape=native_rgb.shape[:2])
            resolved_rois = camera_rois

            diff_scale_m = estimate_depth_difference_scale(native_depth_m, ffs_depth_m)
            native_depth_vis = colorize_depth_map(native_depth_m, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            ffs_depth_vis = colorize_depth_map(ffs_depth_m, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            diff_heatmap_raw = absolute_depth_difference_heatmap(native_depth_m, ffs_depth_m, max_diff_m=diff_scale_m)
            valid_mask_vis = valid_mask_comparison(native_depth_m, ffs_depth_m)
            native_shaded = shaded_depth_map(native_depth_m, native_metadata["K_color"][camera_idx])
            ffs_shaded = shaded_depth_map(ffs_depth_m, ffs_metadata["K_color"][camera_idx])
            native_edge_overlay = build_rgb_depth_edge_overlay(native_rgb, native_depth_m)
            ffs_edge_overlay = build_rgb_depth_edge_overlay(ffs_rgb, ffs_depth_m)

            frame_metric = compute_depth_panel_metrics(
                native_depth_m,
                ffs_depth_m,
                roi_entries=camera_rois,
            )
            frame_metric.update(
                {
                    "panel_frame_idx": int(panel_idx),
                    "native_frame_idx": int(native_frame_idx),
                    "ffs_frame_idx": int(ffs_frame_idx),
                    "native_depth_dir_used": native_info["depth_dir_used"],
                    "ffs_depth_dir_used": ffs_info["depth_dir_used"],
                    "ffs_native_like_postprocess_enabled": bool(ffs_info.get("ffs_native_like_postprocess_enabled", False)),
                    "ffs_native_like_postprocess_applied": bool(ffs_info.get("ffs_native_like_postprocess_applied", False)),
                    "ffs_native_like_postprocess_origin": ffs_info.get("ffs_native_like_postprocess_origin", "none"),
                    "depth_range_m": [float(depth_min_m), float(depth_max_m)],
                    "diff_range_m": [0.0, float(diff_scale_m)],
                }
            )
            frame_metrics.append(frame_metric)

            native_rgb_overview = annotate_rois(native_rgb, camera_rois)
            ffs_rgb_overview = annotate_rois(ffs_rgb, camera_rois)
            native_depth_overview = annotate_rois(native_depth_vis, camera_rois)
            ffs_depth_overview = annotate_rois(ffs_depth_vis, camera_rois)
            native_depth_overview = overlay_scalar_colorbar(
                native_depth_overview,
                label="m",
                min_text=f"{depth_min_m:.2f}",
                max_text=f"{depth_max_m:.2f}",
                colormap=cv2.COLORMAP_TURBO,
            )
            ffs_depth_overview = overlay_scalar_colorbar(
                ffs_depth_overview,
                label="m",
                min_text=f"{depth_min_m:.2f}",
                max_text=f"{depth_max_m:.2f}",
                colormap=cv2.COLORMAP_TURBO,
            )
            diff_heatmap = overlay_scalar_colorbar(
                diff_heatmap_raw,
                label="|Δ| m",
                min_text="0.000",
                max_text=f"{diff_scale_m:.3f}",
                colormap=cv2.COLORMAP_INFERNO,
            )
            roi_tiles = [
                make_roi_tile(
                    native_rgb,
                    native_depth_vis,
                    ffs_depth_vis,
                    diff_heatmap_raw,
                    roi,
                    tile_size=roi_tile_size,
                )
                for roi in camera_rois[:2]
            ]
            while len(roi_tiles) < 2:
                roi_tiles.append(label_tile(native_rgb_overview, "ROI N/A", roi_tile_size))

            rows = [
                [
                    label_tile(native_rgb_overview, "Native RGB", main_tile_size),
                    label_tile(ffs_rgb_overview, "FFS RGB", main_tile_size),
                ],
                [
                    label_tile(native_depth_overview, f"Native Depth ({native_info['depth_dir_used']})", main_tile_size),
                    label_tile(ffs_depth_overview, f"FFS Depth ({ffs_info['depth_dir_used']})", main_tile_size),
                ],
                [
                    label_tile(native_shaded, "Native Surface Shading", main_tile_size),
                    label_tile(ffs_shaded, "FFS Surface Shading", main_tile_size),
                ],
                [
                    label_tile(diff_heatmap, "|Native - FFS|", main_tile_size),
                    label_tile(valid_mask_vis, "Valid Mask Compare", main_tile_size),
                ],
            ]
            if show_edge_overlay:
                rows.append(
                    [
                        label_tile(native_edge_overlay, "Native RGB vs Depth Edges", main_tile_size),
                        label_tile(ffs_edge_overlay, "FFS RGB vs Depth Edges", main_tile_size),
                    ]
                )

            metric_lines = [
                (
                    f"Depth range: [{depth_min_m:.2f}, {depth_max_m:.2f}] m | "
                    f"Native valid: {frame_metric['native_valid_ratio'] * 100.0:.1f}% | "
                    f"FFS valid: {frame_metric['ffs_valid_ratio'] * 100.0:.1f}% | "
                    f"Median |Δ|: {0.0 if frame_metric['median_abs_diff_m'] is None else frame_metric['median_abs_diff_m'] * 1000.0:.1f} mm"
                ),
            ]
            if frame_metric["roi_metrics"]:
                roi_summary_parts: list[str] = []
                for roi_metric in frame_metric["roi_metrics"][:2]:
                    if roi_metric["median_abs_diff_m"] is None:
                        metric_text = "n/a"
                    else:
                        metric_text = f"{float(roi_metric['median_abs_diff_m']) * 1000.0:.1f} mm"
                    roi_summary_parts.append(f"{roi_metric['name']}: {metric_text}")
                roi_summary = " | ".join(roi_summary_parts)
                metric_lines.append(f"ROI median |Δ|: {roi_summary}")

            rows.append(
                [
                    label_tile(roi_tiles[0], f"{camera_rois[0]['name']} Detail", roi_tile_size),
                    label_tile(roi_tiles[1], f"{camera_rois[1]['name']} Detail" if len(camera_rois) > 1 else "ROI 2 Detail", roi_tile_size),
                ]
            )
            title_lines = [
                f"Per-Camera Depth Review | Cam{camera_idx} | native={native_case_dir.name} | ffs={ffs_case_dir.name}",
                f"frame native={native_frame_idx} / ffs={ffs_frame_idx} | preset={preset or 'standard'} | compare={'same-case' if same_case_mode else 'two-case'}",
            ]
            panel = compose_depth_review_board(
                title_lines=title_lines,
                metric_lines=metric_lines,
                rows=rows,
            )
            frame_path = frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(frame_path), panel)
            frame_paths.append(frame_path)

        if write_mp4:
            write_video(camera_dir / "panels.mp4", frame_paths, fps)
        summary["per_camera"][str(camera_idx)] = {
            "frames_written": len(frame_paths),
            "rois": resolved_rois or [],
            "frame_metrics": frame_metrics,
            "aggregate_metrics": _aggregate_camera_metrics(frame_metrics),
        }

    write_json(output_dir / "summary.json", summary)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
