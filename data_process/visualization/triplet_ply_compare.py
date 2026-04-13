from __future__ import annotations

from pathlib import Path
from typing import Any

from .io_artifacts import write_json, write_ply_ascii
from .io_case import (
    get_frame_count,
    load_case_frame_cloud_with_sources,
    load_case_metadata,
    resolve_case_dirs,
)


RAW_FFS_DEPTH_DIRS = ("depth_ffs_float_m", "depth_ffs")
TRIPLET_POINTCLOUD_CONTRACT = {
    "native": "aligned native depth -> K_color deprojection -> c2w calibration-world transform -> fused across 3 cameras",
    "ffs_raw": "aligned FFS raw depth -> K_color deprojection -> c2w calibration-world transform -> fused across 3 cameras",
    "ffs_postprocess": "aligned FFS depth -> native-like postprocess from auxiliary stream or on_the_fly fallback -> K_color deprojection -> c2w calibration-world transform -> fused across 3 cameras",
}


def _select_single_frame_index(*, native_count: int, ffs_count: int, frame_idx: int) -> tuple[int, int]:
    max_index = min(int(native_count), int(ffs_count)) - 1
    selected = int(frame_idx)
    if selected < 0 or selected > max_index:
        raise ValueError(
            f"frame_idx={selected} is out of range for native_count={native_count}, ffs_count={ffs_count}. "
            f"Expected 0 <= frame_idx <= {max_index}."
        )
    return selected, selected


def _case_has_ffs_raw_depth(case_dir: Path) -> bool:
    return any((case_dir / directory_name).is_dir() for directory_name in RAW_FFS_DEPTH_DIRS)


def _aggregate_postprocess_origin(per_camera: list[dict[str, Any]]) -> str:
    origins = {
        str(item.get("ffs_native_like_postprocess_origin", "none"))
        for item in per_camera
        if str(item.get("ffs_native_like_postprocess_origin", "none")) != "none"
    }
    if not origins:
        return "none"
    if len(origins) == 1:
        return next(iter(origins))
    return "mixed"


def _variant_summary(
    *,
    stats: dict[str, Any],
    per_camera_clouds: list[dict[str, Any]],
    fused_ply_path: Path,
) -> dict[str, Any]:
    per_camera_summary = []
    for camera_stats, camera_cloud in zip(stats["per_camera"], per_camera_clouds, strict=False):
        per_camera_summary.append(
            {
                "camera_idx": int(camera_stats["camera_idx"]),
                "serial": str(camera_stats["serial"]),
                "point_count": int(len(camera_cloud["points"])),
                "valid_depth_pixels": int(camera_stats["valid_depth_pixels"]),
                "depth_dir_used": str(camera_stats["depth_dir_used"]),
                "source_depth_dir_used": str(camera_stats["source_depth_dir_used"]),
                "used_float_depth": bool(camera_stats["used_float_depth"]),
                "depth_path": str(camera_stats["depth_path"]),
                "ffs_native_like_postprocess_enabled": bool(camera_stats["ffs_native_like_postprocess_enabled"]),
                "ffs_native_like_postprocess_applied": bool(camera_stats["ffs_native_like_postprocess_applied"]),
                "ffs_native_like_postprocess_origin": str(camera_stats["ffs_native_like_postprocess_origin"]),
            }
        )

    return {
        "fused_ply": str(fused_ply_path.resolve()),
        "fused_point_count": int(stats["fused_point_count"]),
        "depth_dirs_used": sorted({str(item["depth_dir_used"]) for item in per_camera_summary}),
        "source_depth_dirs_used": sorted({str(item["source_depth_dir_used"]) for item in per_camera_summary}),
        "used_float_depth": bool(all(bool(item["used_float_depth"]) for item in per_camera_summary)) if per_camera_summary else False,
        "ffs_native_like_postprocess_enabled": bool(any(bool(item["ffs_native_like_postprocess_enabled"]) for item in per_camera_summary)),
        "ffs_native_like_postprocess_applied": bool(any(bool(item["ffs_native_like_postprocess_applied"]) for item in per_camera_summary)),
        "ffs_native_like_postprocess_origin": _aggregate_postprocess_origin(per_camera_summary),
        "per_camera": per_camera_summary,
    }


def _load_triplet_variant(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    depth_source: str,
    use_float_ffs_depth_when_available: bool,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
    ffs_native_like_postprocess: bool,
) -> tuple[Any, Any, dict[str, Any], list[dict[str, Any]]]:
    return load_case_frame_cloud_with_sources(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=frame_idx,
        depth_source=depth_source,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        ffs_native_like_postprocess=ffs_native_like_postprocess,
    )


def run_triplet_ply_compare_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = 0.1,
    depth_max_m: float = 3.0,
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

    if not _case_has_ffs_raw_depth(ffs_case_dir):
        raise ValueError(
            "Triplet PLY compare requires an aligned FFS case containing depth_ffs/ or depth_ffs_float_m/."
        )
    if len(native_metadata["serial_numbers"]) != len(ffs_metadata["serial_numbers"]):
        raise ValueError("Native and FFS cases must have the same number of cameras for triplet comparison.")

    native_frame_idx, ffs_frame_idx = _select_single_frame_index(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_idx=frame_idx,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = output_dir / "ply_fullscene"
    ply_dir.mkdir(parents=True, exist_ok=True)

    variants: dict[str, dict[str, Any]] = {}
    for variant_name, case_dir, metadata, variant_frame_idx, depth_source, enable_postprocess in (
        ("native", native_case_dir, native_metadata, native_frame_idx, "realsense", False),
        ("ffs_raw", ffs_case_dir, ffs_metadata, ffs_frame_idx, "ffs", False),
        ("ffs_postprocess", ffs_case_dir, ffs_metadata, ffs_frame_idx, "ffs", True),
    ):
        points, colors, stats, per_camera_clouds = _load_triplet_variant(
            case_dir=case_dir,
            metadata=metadata,
            frame_idx=variant_frame_idx,
            depth_source=depth_source,
            use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
            voxel_size=voxel_size,
            max_points_per_camera=max_points_per_camera,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            ffs_native_like_postprocess=enable_postprocess,
        )
        ply_path = ply_dir / f"{variant_name}_frame_{variant_frame_idx:04d}_fused_fullscene.ply"
        write_ply_ascii(ply_path, points, colors)
        variants[variant_name] = _variant_summary(
            stats=stats,
            per_camera_clouds=per_camera_clouds,
            fused_ply_path=ply_path,
        )

    summary = {
        "aligned_root": str(aligned_root),
        "output_dir": str(output_dir),
        "same_case_mode": bool(same_case_mode),
        "case_name": case_name,
        "native_case_name": str(native_case_dir.name),
        "ffs_case_name": str(ffs_case_dir.name),
        "native_case_dir": str(native_case_dir),
        "ffs_case_dir": str(ffs_case_dir),
        "frame_idx": int(frame_idx),
        "native_frame_idx": int(native_frame_idx),
        "ffs_frame_idx": int(ffs_frame_idx),
        "use_float_ffs_depth_when_available": bool(use_float_ffs_depth_when_available),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "pointcloud_contract": dict(TRIPLET_POINTCLOUD_CONTRACT),
        "variants": variants,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
