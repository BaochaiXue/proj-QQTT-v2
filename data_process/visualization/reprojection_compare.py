from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .depth_diagnostics import (
    compute_photometric_residual,
    get_case_camera_transform,
    label_tile,
    load_color_frame,
    load_depth_frame,
    resolve_camera_ids,
    warp_rgb_between_cameras,
)
from .pointcloud_compare import (
    get_case_intrinsics,
    get_frame_count,
    load_case_metadata,
    resolve_case_dirs,
    select_frame_indices,
    write_video,
)


def parse_camera_pair(spec: str) -> tuple[int, int]:
    parts = [part.strip() for part in spec.split(",")]
    if len(parts) != 2:
        raise ValueError(f"Camera pair must be src,dst: {spec}")
    src, dst = [int(part) for part in parts]
    if src == dst:
        raise ValueError(f"Camera pair must use different ids: {spec}")
    return src, dst


def build_camera_pairs(camera_ids: list[int], explicit_pairs: list[tuple[int, int]] | None) -> list[tuple[int, int]]:
    if explicit_pairs:
        return explicit_pairs
    return [(src, dst) for src in camera_ids for dst in camera_ids if src != dst]


def run_reprojection_compare_workflow(
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
    camera_pairs: list[tuple[int, int]] | None = None,
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
    selected_camera_ids = resolve_camera_ids(native_metadata, camera_ids)
    selected_pairs = build_camera_pairs(selected_camera_ids, camera_pairs)
    if not selected_pairs:
        raise ValueError("No camera pairs selected for reprojection comparison.")

    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for reprojection comparison.")

    native_intrinsics = get_case_intrinsics(native_metadata)
    ffs_intrinsics = get_case_intrinsics(ffs_metadata)
    native_c2w = get_case_camera_transform(case_dir=native_case_dir, metadata=native_metadata)
    ffs_c2w = get_case_camera_transform(case_dir=ffs_case_dir, metadata=ffs_metadata)

    output_dir.mkdir(parents=True, exist_ok=True)
    summary_metrics: dict[str, Any] = {
        "same_case_mode": same_case_mode,
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_pairs": frame_pairs,
        "camera_pairs": selected_pairs,
        "per_pair": {},
    }

    tile_size = (320, 220)
    for src_idx, dst_idx in selected_pairs:
        pair_key = f"{src_idx}_to_{dst_idx}"
        pair_dir = output_dir / f"pair_{pair_key}"
        frames_dir = pair_dir / "frames"
        frames_dir.mkdir(parents=True, exist_ok=True)
        frame_paths: list[Path] = []
        pair_metrics = []

        for panel_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
            native_src_rgb = load_color_frame(native_case_dir, src_idx, native_frame_idx)
            native_dst_rgb = load_color_frame(native_case_dir, dst_idx, native_frame_idx)
            _, native_src_depth_m, native_depth_info = load_depth_frame(
                case_dir=native_case_dir,
                metadata=native_metadata,
                camera_idx=src_idx,
                frame_idx=native_frame_idx,
                depth_source="realsense",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            )

            ffs_src_rgb = native_src_rgb if same_case_mode else load_color_frame(ffs_case_dir, src_idx, ffs_frame_idx)
            ffs_dst_rgb = native_dst_rgb if same_case_mode else load_color_frame(ffs_case_dir, dst_idx, ffs_frame_idx)
            _, ffs_src_depth_m, ffs_depth_info = load_depth_frame(
                case_dir=ffs_case_dir,
                metadata=ffs_metadata,
                camera_idx=src_idx,
                frame_idx=ffs_frame_idx,
                depth_source="ffs",
                use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            )

            native_warped_rgb, native_valid, _ = warp_rgb_between_cameras(
                source_rgb=native_src_rgb,
                source_depth_m=native_src_depth_m,
                source_K=native_intrinsics[src_idx],
                source_c2w=native_c2w[src_idx],
                target_K=native_intrinsics[dst_idx],
                target_c2w=native_c2w[dst_idx],
                output_shape=native_dst_rgb.shape[:2],
            )
            ffs_warped_rgb, ffs_valid, _ = warp_rgb_between_cameras(
                source_rgb=ffs_src_rgb,
                source_depth_m=ffs_src_depth_m,
                source_K=ffs_intrinsics[src_idx],
                source_c2w=ffs_c2w[src_idx],
                target_K=ffs_intrinsics[dst_idx],
                target_c2w=ffs_c2w[dst_idx],
                output_shape=ffs_dst_rgb.shape[:2],
            )

            native_heatmap, native_stats = compute_photometric_residual(native_warped_rgb, native_dst_rgb, native_valid)
            ffs_heatmap, ffs_stats = compute_photometric_residual(ffs_warped_rgb, ffs_dst_rgb, ffs_valid)

            panel_tiles = [
                label_tile(native_src_rgb, f"Native Src C{src_idx}", tile_size),
                label_tile(native_dst_rgb, f"Native Tgt C{dst_idx}", tile_size),
                label_tile(native_warped_rgb, f"Native Warp ({native_depth_info['depth_dir_used']})", tile_size),
                label_tile(native_heatmap, "Native Residual", tile_size),
                label_tile(ffs_src_rgb, f"FFS Src C{src_idx}", tile_size),
                label_tile(ffs_dst_rgb, f"FFS Tgt C{dst_idx}", tile_size),
                label_tile(ffs_warped_rgb, f"FFS Warp ({ffs_depth_info['depth_dir_used']})", tile_size),
                label_tile(ffs_heatmap, "FFS Residual", tile_size),
            ]
            panel = np.vstack([
                np.hstack(panel_tiles[0:4]),
                np.hstack(panel_tiles[4:8]),
            ])
            frame_path = frames_dir / f"{panel_idx:06d}.png"
            cv2.imwrite(str(frame_path), panel)
            frame_paths.append(frame_path)
            pair_metrics.append(
                {
                    "panel_frame_idx": panel_idx,
                    "native_frame_idx": native_frame_idx,
                    "ffs_frame_idx": ffs_frame_idx,
                    "src_camera_idx": src_idx,
                    "dst_camera_idx": dst_idx,
                    "native": native_stats,
                    "ffs": ffs_stats,
                }
            )

        if write_mp4:
            write_video(pair_dir / "reprojection.mp4", frame_paths, fps)

        summary_metrics["per_pair"][pair_key] = {
            "frames_written": len(frame_paths),
            "metrics": pair_metrics,
        }

    (output_dir / "summary_metrics.json").write_text(json.dumps(summary_metrics, indent=2), encoding="utf-8")
    return {
        "output_dir": str(output_dir),
        "summary_metrics": summary_metrics,
    }

