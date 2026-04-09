from __future__ import annotations
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .depth_diagnostics import (
    absolute_depth_difference_heatmap,
    annotate_rois,
    clamp_roi,
    colorize_depth_map,
    default_rois,
    label_tile,
    load_color_frame,
    load_depth_frame,
    make_roi_tile,
    resolve_camera_ids,
    shaded_depth_map,
    valid_mask_comparison,
)
from .io_artifacts import write_json
from .pointcloud_compare import (
    get_frame_count,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
    write_video,
)


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
    rois: list[tuple[int, int, int, int]] | None = None,
    write_mp4: bool = False,
    fps: int = 10,
    use_float_ffs_depth_when_available: bool = True,
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
    tile_size = (320, 220)
    summary = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "camera_ids": selected_cameras,
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "per_camera": {},
    }

    for camera_idx in selected_cameras:
        camera_dir = output_dir / f"camera_{camera_idx}"
        frames_dir = camera_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        resolved_rois = None

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
            )
            _, ffs_depth_m, ffs_info = load_depth_frame(
                case_dir=ffs_case_dir,
                metadata=ffs_metadata,
                camera_idx=camera_idx,
                frame_idx=ffs_frame_idx,
                depth_source="ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            )

            camera_rois = rois or default_rois(native_rgb.shape[:2])
            camera_rois = [clamp_roi(roi, native_rgb.shape[:2]) for roi in camera_rois]
            resolved_rois = camera_rois

            native_depth_vis = colorize_depth_map(native_depth_m, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            ffs_depth_vis = colorize_depth_map(ffs_depth_m, depth_min_m=depth_min_m, depth_max_m=depth_max_m)
            diff_heatmap = absolute_depth_difference_heatmap(native_depth_m, ffs_depth_m)
            valid_mask_vis = valid_mask_comparison(native_depth_m, ffs_depth_m)
            native_shaded = shaded_depth_map(native_depth_m, native_metadata["K_color"][camera_idx])
            ffs_shaded = shaded_depth_map(ffs_depth_m, ffs_metadata["K_color"][camera_idx])

            native_rgb_overview = annotate_rois(native_rgb, camera_rois)
            ffs_rgb_overview = annotate_rois(ffs_rgb, camera_rois)
            roi_tiles = [
                make_roi_tile(
                    native_rgb,
                    native_depth_vis,
                    ffs_depth_vis,
                    diff_heatmap,
                    roi,
                    tile_size=tile_size,
                )
                for roi in camera_rois[:2]
            ]
            while len(roi_tiles) < 2:
                roi_tiles.append(label_tile(native_rgb_overview, "ROI N/A", tile_size))

            tiles = [
                label_tile(native_rgb_overview, "Native RGB", tile_size),
                label_tile(ffs_rgb_overview, "FFS RGB", tile_size),
                label_tile(native_depth_vis, f"Native Depth ({native_info['depth_dir_used']})", tile_size),
                label_tile(ffs_depth_vis, f"FFS Depth ({ffs_info['depth_dir_used']})", tile_size),
                label_tile(diff_heatmap, "|Native - FFS|", tile_size),
                label_tile(valid_mask_vis, "Valid Mask Compare", tile_size),
                label_tile(native_shaded, "Native Surface Shading", tile_size),
                label_tile(ffs_shaded, "FFS Surface Shading", tile_size),
                label_tile(roi_tiles[0], "ROI 1 Detail", tile_size),
                label_tile(roi_tiles[1], "ROI 2 Detail", tile_size),
            ]
            panel = np.vstack(
                [
                    np.hstack(tiles[0:2]),
                    np.hstack(tiles[2:4]),
                    np.hstack(tiles[4:6]),
                    np.hstack(tiles[6:8]),
                    np.hstack(tiles[8:10]),
                ]
            )
            frame_path = frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(frame_path), panel)
            frame_paths.append(frame_path)

        if write_mp4:
            write_video(camera_dir / "panels.mp4", frame_paths, fps)
        summary["per_camera"][str(camera_idx)] = {
            "frames_written": len(frame_paths),
            "rois": resolved_rois or [],
        }

    write_json(output_dir / "summary.json", summary)
    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
