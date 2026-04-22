from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from data_process.depth_backends import (
    FastFoundationStereoRunner,
    align_depth_to_color,
    apply_remove_invisible_mask,
    disparity_to_metric_depth,
)

from .calibration_io import load_calibration_transforms
from .io_artifacts import write_json, write_ply_ascii
from .io_case import (
    depth_to_camera_points,
    get_case_intrinsics,
    get_frame_count,
    load_case_frame_cloud,
    load_case_metadata,
    select_frame_indices,
    transform_points,
    voxel_downsample,
)
from .pointcloud_defaults import DEFAULT_POINTCLOUD_DEPTH_MAX_M, DEFAULT_POINTCLOUD_DEPTH_MIN_M


RERUN_OUTPUT_MODES = ("viewer_and_rrd", "viewer_only", "rrd_only")
VIEWER_LAYOUT_MODES = ("default", "horizontal_triple")


def _import_rerun():
    try:
        import rerun as rr
    except ImportError as exc:
        raise ImportError(
            "visual_compare_rerun requires rerun-sdk. Install it with "
            "`python -m pip install rerun-sdk` or add it to the active conda env."
        ) from exc
    return rr


def _to_rerun_rgb(colors_bgr: np.ndarray) -> np.ndarray:
    colors = np.asarray(colors_bgr, dtype=np.uint8)
    if colors.size == 0:
        return np.empty((0, 3), dtype=np.uint8)
    return colors[:, ::-1].copy()


def _set_rerun_frame_time(rr: Any, frame_idx: int) -> None:
    if hasattr(rr, "set_time_sequence"):
        rr.set_time_sequence("frame", int(frame_idx))
        return
    rr.set_time("frame", sequence=int(frame_idx))


def _build_viewer_blueprint(rr: Any, *, viewer_layout: str) -> Any | None:
    if viewer_layout == "default":
        return None
    if viewer_layout != "horizontal_triple":
        raise ValueError(f"Unsupported viewer_layout: {viewer_layout}")

    rrb = rr.blueprint
    return rrb.Blueprint(
        rrb.Horizontal(
            rrb.Spatial3DView(origin="/", contents=["native"], name="Native"),
            rrb.Spatial3DView(origin="/", contents=["ffs_remove_1"], name="FFS remove_invisible=1"),
            rrb.Spatial3DView(origin="/", contents=["ffs_remove_0"], name="FFS remove_invisible=0"),
            name="Point Cloud Compare 1x3",
        )
    )


def _resolve_ffs_runtime_config(
    *,
    metadata: dict[str, Any],
    ffs_repo: str | Path | None,
    ffs_model_path: str | Path | None,
) -> dict[str, Any]:
    config = metadata.get("ffs_config", {})
    repo_value = str(ffs_repo) if ffs_repo is not None else config.get("ffs_repo")
    model_value = str(ffs_model_path) if ffs_model_path is not None else config.get("model_path")
    if not repo_value or not model_value:
        raise ValueError(
            "FFS repo/model path are required. Provide --ffs_repo and --ffs_model_path, "
            "or keep ffs_config in the aligned FFS case metadata."
        )
    return {
        "ffs_repo": repo_value,
        "model_path": model_value,
        "scale": float(config.get("scale", 1.0)),
        "valid_iters": int(config.get("valid_iters", 8)),
        "max_disp": int(config.get("max_disp", 192)),
    }


def _fuse_point_sets(
    *,
    point_sets: list[np.ndarray],
    color_sets: list[np.ndarray],
    voxel_size: float | None,
) -> tuple[np.ndarray, np.ndarray]:
    if point_sets:
        points = np.concatenate(point_sets, axis=0) if len(point_sets) > 1 else point_sets[0]
        colors = np.concatenate(color_sets, axis=0) if len(color_sets) > 1 else color_sets[0]
    else:
        points = np.empty((0, 3), dtype=np.float32)
        colors = np.empty((0, 3), dtype=np.uint8)
    return voxel_downsample(points, colors, voxel_size)


def _load_native_variant(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    points, colors, stats = load_case_frame_cloud(
        case_dir=case_dir,
        metadata=metadata,
        frame_idx=frame_idx,
        depth_source="realsense",
        use_float_ffs_depth_when_available=False,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
    )
    return points, colors, {
        "fused_point_count": int(stats["fused_point_count"]),
        "per_camera": stats["per_camera"],
        "remove_invisible_pixel_count": 0,
        "remove_invisible_ratio": 0.0,
    }


def _load_ffs_variants(
    *,
    case_dir: Path,
    metadata: dict[str, Any],
    frame_idx: int,
    runner: Any,
    voxel_size: float | None,
    max_points_per_camera: int | None,
    depth_min_m: float,
    depth_max_m: float,
) -> dict[str, dict[str, Any]]:
    serials = metadata["serial_numbers"]
    calibration_reference_serials = metadata.get("calibration_reference_serials", serials)
    c2w_list = load_calibration_transforms(
        case_dir / "calibrate.pkl",
        serial_numbers=serials,
        calibration_reference_serials=calibration_reference_serials,
    )
    k_color_list = get_case_intrinsics(metadata)
    k_ir_left_list = [np.asarray(matrix, dtype=np.float32) for matrix in metadata["K_ir_left"]]
    t_ir_left_to_color_list = [np.asarray(matrix, dtype=np.float32) for matrix in metadata["T_ir_left_to_color"]]
    baselines = [float(value) for value in metadata["ir_baseline_m"]]

    variants: dict[str, dict[str, Any]] = {
        "ffs_remove_1": {
            "point_sets": [],
            "color_sets": [],
            "per_camera": [],
            "remove_invisible_pixel_count": 0,
            "pixel_count": 0,
        },
        "ffs_remove_0": {
            "point_sets": [],
            "color_sets": [],
            "per_camera": [],
            "remove_invisible_pixel_count": 0,
            "pixel_count": 0,
        },
    }

    for camera_idx, serial in enumerate(serials):
        color_path = case_dir / "color" / str(camera_idx) / f"{frame_idx}.png"
        left_path = case_dir / "ir_left" / str(camera_idx) / f"{frame_idx}.png"
        right_path = case_dir / "ir_right" / str(camera_idx) / f"{frame_idx}.png"

        color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        left_image = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
        right_image = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
        if color_image is None:
            raise FileNotFoundError(f"Missing color frame: {color_path}")
        if left_image is None or right_image is None:
            raise FileNotFoundError(f"Missing aligned IR stereo pair for camera={camera_idx}, frame={frame_idx}.")

        run_output = runner.run_pair(
            left_image,
            right_image,
            K_ir_left=k_ir_left_list[camera_idx],
            baseline_m=baselines[camera_idx],
            audit_mode=True,
        )
        disparity_remove_0 = np.asarray(run_output["disparity"], dtype=np.float32)
        disparity_remove_1, remove_stats = apply_remove_invisible_mask(
            np.asarray(run_output.get("disparity_raw", disparity_remove_0), dtype=np.float32)
        )
        k_ir_left_used = np.asarray(run_output["K_ir_left_used"], dtype=np.float32)
        baseline_m = float(run_output["baseline_m"])

        for variant_name, disparity_variant, remove_pixel_count, remove_ratio in (
            (
                "ffs_remove_1",
                disparity_remove_1,
                int(remove_stats["remove_invisible_pixel_count"]),
                float(remove_stats["remove_invisible_ratio"]),
            ),
            (
                "ffs_remove_0",
                disparity_remove_0,
                0,
                0.0,
            ),
        ):
            depth_ir_left_m = disparity_to_metric_depth(
                disparity_variant,
                fx_ir=float(k_ir_left_used[0, 0]),
                baseline_m=baseline_m,
            )
            depth_color_m = align_depth_to_color(
                depth_ir_left_m,
                k_ir_left_used,
                t_ir_left_to_color_list[camera_idx],
                k_color_list[camera_idx],
                output_shape=(int(color_image.shape[0]), int(color_image.shape[1])),
            )
            camera_points, camera_colors, _, camera_stats = depth_to_camera_points(
                depth_color_m,
                k_color_list[camera_idx],
                depth_min_m=depth_min_m,
                depth_max_m=depth_max_m,
                color_image=color_image,
                max_points_per_camera=max_points_per_camera,
            )
            world_points = transform_points(camera_points, c2w_list[camera_idx])
            variants[variant_name]["point_sets"].append(world_points)
            variants[variant_name]["color_sets"].append(camera_colors)
            variants[variant_name]["per_camera"].append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "pixel_count": int(remove_stats["pixel_count"]),
                    "valid_depth_pixels": int(camera_stats["valid_depth_pixels"]),
                    "points_after_sampling": int(camera_stats["points_after_sampling"]),
                    "remove_invisible_pixel_count": int(remove_pixel_count),
                    "remove_invisible_ratio": float(remove_ratio),
                }
            )
            variants[variant_name]["remove_invisible_pixel_count"] += int(remove_pixel_count)
            variants[variant_name]["pixel_count"] += int(remove_stats["pixel_count"])

    result: dict[str, dict[str, Any]] = {}
    for variant_name, payload in variants.items():
        points, colors = _fuse_point_sets(
            point_sets=payload["point_sets"],
            color_sets=payload["color_sets"],
            voxel_size=voxel_size,
        )
        result[variant_name] = {
            "points": points,
            "colors": colors,
            "summary": {
                "fused_point_count": int(len(points)),
                "per_camera": payload["per_camera"],
                "remove_invisible_pixel_count": int(payload["remove_invisible_pixel_count"]),
                "remove_invisible_ratio": float(
                    payload["remove_invisible_pixel_count"] / max(1, payload["pixel_count"])
                ),
            },
        }
    return result


def run_rerun_compare_workflow(
    *,
    aligned_root: Path,
    realsense_case: str,
    ffs_case: str,
    output_dir: Path,
    frame_start: int | None = None,
    frame_end: int | None = None,
    frame_stride: int = 1,
    ffs_repo: str | Path | None = None,
    ffs_model_path: str | Path | None = None,
    rerun_output: str = "viewer_and_rrd",
    viewer_layout: str = "default",
    voxel_size: float | None = None,
    max_points_per_camera: int | None = None,
    depth_min_m: float = DEFAULT_POINTCLOUD_DEPTH_MIN_M,
    depth_max_m: float = DEFAULT_POINTCLOUD_DEPTH_MAX_M,
    runner_factory: Callable[..., Any] = FastFoundationStereoRunner,
    rerun_module: Any | None = None,
) -> dict[str, Any]:
    if rerun_output not in RERUN_OUTPUT_MODES:
        raise ValueError(f"Unsupported rerun_output: {rerun_output}")
    if viewer_layout not in VIEWER_LAYOUT_MODES:
        raise ValueError(f"Unsupported viewer_layout: {viewer_layout}")

    native_case_dir = aligned_root / realsense_case
    ffs_case_dir = aligned_root / ffs_case
    native_metadata = load_case_metadata(native_case_dir)
    ffs_metadata = load_case_metadata(ffs_case_dir)
    frame_pairs = select_frame_indices(
        native_count=get_frame_count(native_metadata),
        ffs_count=get_frame_count(ffs_metadata),
        frame_start=frame_start,
        frame_end=frame_end,
        frame_stride=frame_stride,
    )
    if not frame_pairs:
        raise ValueError("No frame pairs selected for rerun comparison.")

    output_dir.mkdir(parents=True, exist_ok=True)
    ply_dir = output_dir / "ply_fullscene"
    ply_dir.mkdir(parents=True, exist_ok=True)

    ffs_runtime = _resolve_ffs_runtime_config(
        metadata=ffs_metadata,
        ffs_repo=ffs_repo,
        ffs_model_path=ffs_model_path,
    )
    runner = runner_factory(
        ffs_repo=ffs_runtime["ffs_repo"],
        model_path=ffs_runtime["model_path"],
        scale=float(ffs_runtime["scale"]),
        valid_iters=int(ffs_runtime["valid_iters"]),
        max_disp=int(ffs_runtime["max_disp"]),
    )

    rr = rerun_module if rerun_module is not None else _import_rerun()
    spawn_viewer = rerun_output in ("viewer_and_rrd", "viewer_only")
    rrd_path = output_dir / "pointcloud_compare.rrd"
    rr.init("qqtt_pointcloud_compare", spawn=spawn_viewer)
    blueprint = _build_viewer_blueprint(rr, viewer_layout=viewer_layout)
    if blueprint is not None and spawn_viewer and hasattr(rr, "send_blueprint"):
        rr.send_blueprint(blueprint)
    if rerun_output in ("viewer_and_rrd", "rrd_only"):
        if blueprint is not None:
            rr.save(str(rrd_path), default_blueprint=blueprint)
        else:
            rr.save(str(rrd_path))

    summary_frames = []
    for time_idx, (native_frame_idx, ffs_frame_idx) in enumerate(frame_pairs):
        native_points, native_colors, native_summary = _load_native_variant(
            case_dir=native_case_dir,
            metadata=native_metadata,
            frame_idx=native_frame_idx,
            voxel_size=voxel_size,
            max_points_per_camera=max_points_per_camera,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )
        ffs_variants = _load_ffs_variants(
            case_dir=ffs_case_dir,
            metadata=ffs_metadata,
            frame_idx=ffs_frame_idx,
            runner=runner,
            voxel_size=voxel_size,
            max_points_per_camera=max_points_per_camera,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
        )

        frame_summary = {
            "panel_frame_idx": int(time_idx),
            "native_frame_idx": int(native_frame_idx),
            "ffs_frame_idx": int(ffs_frame_idx),
            "variants": {},
        }

        _set_rerun_frame_time(rr, int(native_frame_idx))
        for variant_name, points, colors, variant_summary in (
            ("native", native_points, native_colors, native_summary),
            (
                "ffs_remove_1",
                ffs_variants["ffs_remove_1"]["points"],
                ffs_variants["ffs_remove_1"]["colors"],
                ffs_variants["ffs_remove_1"]["summary"],
            ),
            (
                "ffs_remove_0",
                ffs_variants["ffs_remove_0"]["points"],
                ffs_variants["ffs_remove_0"]["colors"],
                ffs_variants["ffs_remove_0"]["summary"],
            ),
        ):
            rr.log(variant_name, rr.Points3D(points, colors=_to_rerun_rgb(colors)))
            ply_frame_idx = int(native_frame_idx if variant_name == "native" else ffs_frame_idx)
            ply_path = ply_dir / f"{variant_name}_frame_{ply_frame_idx:04d}_fused_fullscene.ply"
            write_ply_ascii(ply_path, points, colors)
            frame_summary["variants"][variant_name] = {
                **variant_summary,
                "fused_ply": str(ply_path.resolve()),
            }
        summary_frames.append(frame_summary)

    summary = {
        "realsense_case": realsense_case,
        "ffs_case": ffs_case,
        "aligned_root": str(aligned_root.resolve()),
        "output_dir": str(output_dir.resolve()),
        "rerun_output": rerun_output,
        "viewer_layout": viewer_layout,
        "rrd_path": str(rrd_path.resolve()) if rerun_output in ("viewer_and_rrd", "rrd_only") else None,
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
        "ffs_runner": {
            "ffs_repo": str(Path(str(ffs_runtime["ffs_repo"])).resolve()),
            "model_path": str(Path(str(ffs_runtime["model_path"])).resolve()),
            "scale": float(ffs_runtime["scale"]),
            "valid_iters": int(ffs_runtime["valid_iters"]),
            "max_disp": int(ffs_runtime["max_disp"]),
        },
        "pointcloud_contract": {
            "native": "aligned native depth -> K_color deprojection -> c2w world transform -> fused across 3 cameras",
            "ffs_remove_1": "FFS disparity -> remove_invisible mask applied -> IR-left metric depth -> color reprojection -> K_color deprojection -> c2w world transform -> fused across 3 cameras",
            "ffs_remove_0": "FFS disparity -> no remove_invisible mask -> IR-left metric depth -> color reprojection -> K_color deprojection -> c2w world transform -> fused across 3 cameras",
        },
        "frames": summary_frames,
    }
    write_json(output_dir / "summary.json", summary)
    return summary
