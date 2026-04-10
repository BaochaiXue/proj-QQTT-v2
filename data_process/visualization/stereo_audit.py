from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np

from data_process.depth_backends import (
    FastFoundationStereoRunner,
    align_depth_to_color,
    derive_ir_right_to_color,
    summarize_left_right_audit,
)

from .calibration_frame import build_visualization_frame_contract
from .depth_diagnostics import colorize_depth_map, load_depth_frame
from .face_quality import (
    build_face_metric_tile,
    colorize_patch_residuals,
    compose_face_quality_board,
    compute_patch_plane_metrics,
    draw_face_patch_overlay,
    parse_face_patches_json,
)
from .io_artifacts import write_image, write_json
from .layouts import compose_depth_review_board, compose_registration_matrix_board
from .object_compare import build_object_first_layers
from .object_roi import estimate_table_color_bgr
from .pointcloud_compare import estimate_ortho_scale, get_frame_count, load_case_metadata, resolve_case_dirs
from .professor_triptych import _build_turntable_scene
from .source_compare import SOURCE_CAMERA_COLORS_BGR, build_source_legend_image, render_source_attribution_overlay
from .io_case import depth_to_camera_points, transform_points


def _resolve_frame_idx(*, native_metadata: dict[str, Any], ffs_metadata: dict[str, Any], frame_idx: int) -> int:
    max_index = min(int(get_frame_count(native_metadata)), int(get_frame_count(ffs_metadata))) - 1
    if int(frame_idx) < 0 or int(frame_idx) > max_index:
        raise ValueError(f"frame_idx={frame_idx} is out of range. Expected 0 <= frame_idx <= {max_index}.")
    return int(frame_idx)


def _require_ffs_ir_geometry(case_dir: Path, metadata: dict[str, Any], camera_idx: int) -> None:
    if not ((case_dir / "ir_left").is_dir() and (case_dir / "ir_right").is_dir()):
        raise RuntimeError(f"Aligned case does not contain ir_left/ir_right required for FFS audit: {case_dir}")
    required_keys = ("K_ir_left", "K_ir_right", "T_ir_left_to_right", "T_ir_left_to_color", "ir_baseline_m", "K_color")
    for key in required_keys:
        value = metadata.get(key)
        if not isinstance(value, list) or value[int(camera_idx)] is None:
            raise RuntimeError(f"Aligned case is missing {key} for camera {camera_idx}: {case_dir}")


def _load_aligned_ir_pair(case_dir: Path, *, camera_idx: int, frame_idx: int) -> tuple[np.ndarray, np.ndarray]:
    left_path = case_dir / "ir_left" / str(camera_idx) / f"{frame_idx}.png"
    right_path = case_dir / "ir_right" / str(camera_idx) / f"{frame_idx}.png"
    left = cv2.imread(str(left_path), cv2.IMREAD_UNCHANGED)
    right = cv2.imread(str(right_path), cv2.IMREAD_UNCHANGED)
    if left is None or right is None:
        raise RuntimeError(f"Failed to load aligned IR stereo pair camera={camera_idx} frame={frame_idx} from {case_dir}")
    return left, right


def _make_disparity_tile(disparity_raw: np.ndarray, *, label: str) -> np.ndarray:
    disparity = np.asarray(disparity_raw, dtype=np.float32)
    finite = np.isfinite(disparity)
    canvas = np.full(disparity.shape + (3,), (28, 30, 34), dtype=np.uint8)
    if np.any(finite):
        max_abs = max(1e-3, float(np.quantile(np.abs(disparity[finite]), 0.95)))
        normalized = np.clip((disparity / max_abs + 1.0) * 0.5, 0.0, 1.0)
        colored = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO)
        canvas[finite] = colored[finite]
    return build_face_metric_tile(canvas, label=label, metrics=None, tile_size=(360, 240))


def _compute_face_patch_metrics_for_depth(
    *,
    depth_m: np.ndarray,
    K_color: np.ndarray,
    patches: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for patch in patches:
        patch_metrics = compute_patch_plane_metrics(depth_m, K_color, tuple(patch["bbox"]))
        metrics.append(
            {
                "name": patch["name"],
                "valid_depth_ratio": patch_metrics["valid_depth_ratio"],
                "plane_fit_rmse_mm": patch_metrics["plane_fit_rmse_mm"],
                "mad_mm": patch_metrics["mad_mm"],
                "p90_abs_residual_mm": patch_metrics["p90_abs_residual_mm"],
            }
        )
    return metrics


def _crop_with_padding(image: np.ndarray, bbox: tuple[int, int, int, int], *, padding: int = 14) -> np.ndarray:
    h, w = image.shape[:2]
    x0, y0, x1, y1 = [int(item) for item in bbox]
    x0 = max(0, x0 - padding)
    y0 = max(0, y0 - padding)
    x1 = min(w, x1 + padding)
    y1 = min(h, y1 + padding)
    return np.asarray(image, dtype=np.uint8)[y0:y1, x0:x1].copy()


def run_ffs_left_right_audit_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    camera_idx: int = 0,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 8,
    max_disp: int = 192,
    face_patches_json: str | Path | None = None,
    runner_factory=FastFoundationStereoRunner,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_dir = Path(output_dir).resolve()
    if case_name is not None:
        _, ffs_case_dir, _ = resolve_case_dirs(
            aligned_root=aligned_root,
            case_name=case_name,
            realsense_case=None,
            ffs_case=None,
        )
        native_case_dir = ffs_case_dir
    else:
        if ffs_case is None:
            raise ValueError("Either case_name or ffs_case is required.")
        ffs_case_dir = (aligned_root / ffs_case).resolve()
        native_case_dir = ffs_case_dir
    ffs_metadata = load_case_metadata(ffs_case_dir)
    native_metadata = load_case_metadata(native_case_dir)
    selected_frame_idx = _resolve_frame_idx(native_metadata=native_metadata, ffs_metadata=ffs_metadata, frame_idx=frame_idx)
    _require_ffs_ir_geometry(ffs_case_dir, ffs_metadata, int(camera_idx))

    left_image, right_image = _load_aligned_ir_pair(ffs_case_dir, camera_idx=int(camera_idx), frame_idx=selected_frame_idx)
    K_ir_left = np.asarray(ffs_metadata["K_ir_left"][camera_idx], dtype=np.float32)
    K_ir_right = np.asarray(ffs_metadata["K_ir_right"][camera_idx], dtype=np.float32)
    K_color = np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32)
    T_ir_left_to_right = np.asarray(ffs_metadata["T_ir_left_to_right"][camera_idx], dtype=np.float32)
    T_ir_left_to_color = np.asarray(ffs_metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
    T_ir_right_to_color = derive_ir_right_to_color(T_ir_left_to_right, T_ir_left_to_color)
    baseline_m = float(ffs_metadata["ir_baseline_m"][camera_idx])

    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
    )
    normal_run = runner.run_pair(left_image, right_image, K_ir_left=K_ir_left, baseline_m=baseline_m, audit_mode=True)
    swapped_run = runner.run_pair(right_image, left_image, K_ir_left=K_ir_right, baseline_m=baseline_m, audit_mode=True)

    output_shape = (int(ffs_metadata["WH"][1]), int(ffs_metadata["WH"][0]))
    normal_depth_color = align_depth_to_color(
        np.asarray(normal_run["depth_ir_left_m"], dtype=np.float32),
        np.asarray(normal_run["K_ir_left_used"], dtype=np.float32),
        T_ir_left_to_color,
        K_color,
        output_shape=output_shape,
    )
    swapped_depth_color = align_depth_to_color(
        np.asarray(swapped_run["depth_ir_left_m"], dtype=np.float32),
        np.asarray(swapped_run["K_ir_left_used"], dtype=np.float32),
        T_ir_right_to_color,
        K_color,
        output_shape=output_shape,
    )

    patches_by_camera = {} if face_patches_json is None else parse_face_patches_json(face_patches_json)
    camera_patches = patches_by_camera.get(int(camera_idx), [])
    normal_face_metrics = _compute_face_patch_metrics_for_depth(depth_m=normal_depth_color, K_color=K_color, patches=camera_patches)
    swapped_face_metrics = _compute_face_patch_metrics_for_depth(depth_m=swapped_depth_color, K_color=K_color, patches=camera_patches)
    audit_summary = summarize_left_right_audit(
        normal_run=normal_run,
        swapped_run=swapped_run,
        normal_face_metrics=normal_face_metrics,
        swapped_face_metrics=swapped_face_metrics,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    left_tile = build_face_metric_tile(cv2.cvtColor(left_image, cv2.COLOR_GRAY2BGR), label="IR Left", metrics=None, tile_size=(360, 240))
    right_tile = build_face_metric_tile(cv2.cvtColor(right_image, cv2.COLOR_GRAY2BGR), label="IR Right", metrics=None, tile_size=(360, 240))
    normal_disp_tile = _make_disparity_tile(np.asarray(normal_run["disparity_raw"], dtype=np.float32), label="Normal raw disparity")
    normal_depth_tile = build_face_metric_tile(
        colorize_depth_map(normal_depth_color, depth_min_m=0.2, depth_max_m=1.5),
        label="Normal aligned depth",
        metrics=None,
        tile_size=(360, 240),
    )
    swapped_disp_tile = _make_disparity_tile(np.asarray(swapped_run["disparity_raw"], dtype=np.float32), label="Swapped raw disparity")
    swapped_depth_tile = build_face_metric_tile(
        colorize_depth_map(swapped_depth_color, depth_min_m=0.2, depth_max_m=1.5),
        label="Swapped aligned depth",
        metrics=None,
        tile_size=(360, 240),
    )
    board = compose_depth_review_board(
        title_lines=[
            f"{ffs_case_dir.name} | frame_idx={selected_frame_idx} | cam{camera_idx} | left/right audit",
            f"plausible ordering={audit_summary['plausible_ordering']} | score_margin={audit_summary['score_margin']:.3f}",
        ],
        metric_lines=[
            f"normal: pos={audit_summary['normal']['audit_stats']['positive_fraction_of_finite']:.3f} | valid={audit_summary['normal']['valid_depth_ratio']:.3f} | face-rmse={audit_summary['normal']['face_metrics_summary']['plane_fit_rmse_mm_mean']:.2f} mm",
            f"swapped: pos={audit_summary['swapped']['audit_stats']['positive_fraction_of_finite']:.3f} | valid={audit_summary['swapped']['valid_depth_ratio']:.3f} | face-rmse={audit_summary['swapped']['face_metrics_summary']['plane_fit_rmse_mm_mean']:.2f} mm",
        ],
        rows=[
            [left_tile, right_tile, normal_disp_tile],
            [normal_depth_tile, swapped_disp_tile, swapped_depth_tile],
        ],
    )
    write_image(output_dir / "left_right_audit_board.png", board)
    write_json(
        output_dir / "left_right_audit.json",
        {
            "case_dir": str(ffs_case_dir),
            "frame_idx": int(selected_frame_idx),
            "camera_idx": int(camera_idx),
            "normal": {
                "audit_stats": audit_summary["normal"]["audit_stats"],
                "valid_depth_ratio": audit_summary["normal"]["valid_depth_ratio"],
                "face_patch_metrics": normal_face_metrics,
                "plausibility_score": audit_summary["normal"]["plausibility_score"],
            },
            "swapped": {
                "audit_stats": audit_summary["swapped"]["audit_stats"],
                "valid_depth_ratio": audit_summary["swapped"]["valid_depth_ratio"],
                "face_patch_metrics": swapped_face_metrics,
                "plausibility_score": audit_summary["swapped"]["plausibility_score"],
            },
            "plausible_ordering": audit_summary["plausible_ordering"],
            "score_margin": audit_summary["score_margin"],
        },
    )
    return {
        "output_dir": str(output_dir),
        "summary": audit_summary,
    }


def run_face_smoothness_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    face_patches_json: str | Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 8,
    max_disp: int = 192,
    runner_factory=FastFoundationStereoRunner,
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
    selected_frame_idx = _resolve_frame_idx(native_metadata=native_metadata, ffs_metadata=ffs_metadata, frame_idx=frame_idx)
    patches_by_camera = parse_face_patches_json(face_patches_json)
    cameras_with_patches = sorted(patches_by_camera.keys())
    if not cameras_with_patches:
        raise ValueError("face patch json does not contain any patches.")
    for camera_idx in cameras_with_patches:
        _require_ffs_ir_geometry(ffs_case_dir, ffs_metadata, int(camera_idx))

    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
    )
    swapped_depth_by_camera: dict[int, np.ndarray] = {}
    for camera_idx in cameras_with_patches:
        left_image, right_image = _load_aligned_ir_pair(ffs_case_dir, camera_idx=int(camera_idx), frame_idx=selected_frame_idx)
        K_ir_right = np.asarray(ffs_metadata["K_ir_right"][camera_idx], dtype=np.float32)
        K_color = np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32)
        T_ir_left_to_right = np.asarray(ffs_metadata["T_ir_left_to_right"][camera_idx], dtype=np.float32)
        T_ir_left_to_color = np.asarray(ffs_metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
        T_ir_right_to_color = derive_ir_right_to_color(T_ir_left_to_right, T_ir_left_to_color)
        baseline_m = float(ffs_metadata["ir_baseline_m"][camera_idx])
        swapped_run = runner.run_pair(right_image, left_image, K_ir_left=K_ir_right, baseline_m=baseline_m, audit_mode=True)
        swapped_depth_by_camera[int(camera_idx)] = align_depth_to_color(
            np.asarray(swapped_run["depth_ir_left_m"], dtype=np.float32),
            np.asarray(swapped_run["K_ir_left_used"], dtype=np.float32),
            T_ir_right_to_color,
            K_color,
            output_shape=(int(ffs_metadata["WH"][1]), int(ffs_metadata["WH"][0])),
        )

    patch_rows: list[list[np.ndarray]] = []
    patch_summaries: list[dict[str, Any]] = []
    for camera_idx in cameras_with_patches:
        native_rgb = cv2.imread(str(native_case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"), cv2.IMREAD_COLOR)
        if native_rgb is None:
            raise RuntimeError(f"Missing native RGB for camera {camera_idx} frame {selected_frame_idx}")
        _, native_depth, _ = load_depth_frame(
            case_dir=native_case_dir,
            metadata=native_metadata,
            camera_idx=camera_idx,
            frame_idx=selected_frame_idx,
            depth_source="realsense",
            use_float_ffs_depth_when_available=True,
        )
        _, ffs_depth, _ = load_depth_frame(
            case_dir=ffs_case_dir,
            metadata=ffs_metadata,
            camera_idx=camera_idx,
            frame_idx=selected_frame_idx,
            depth_source="ffs",
            use_float_ffs_depth_when_available=True,
        )
        K_color_native = np.asarray(native_metadata["K_color"][camera_idx], dtype=np.float32)
        K_color_ffs = np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32)
        swapped_depth = swapped_depth_by_camera[int(camera_idx)]
        for patch in patches_by_camera[camera_idx]:
            bbox = tuple(patch["bbox"])
            rgb_overlay = draw_face_patch_overlay(native_rgb, bbox, label=f"Cam{camera_idx} | {patch['name']}")
            native_metrics = compute_patch_plane_metrics(native_depth, K_color_native, bbox)
            ffs_metrics = compute_patch_plane_metrics(ffs_depth, K_color_ffs, bbox)
            swapped_metrics = compute_patch_plane_metrics(swapped_depth, K_color_ffs, bbox)
            max_mm = max(
                2.0,
                native_metrics["p90_abs_residual_mm"],
                ffs_metrics["p90_abs_residual_mm"],
                swapped_metrics["p90_abs_residual_mm"],
            )
            native_tile = build_face_metric_tile(
                colorize_patch_residuals(native_metrics["residual_mm"], native_metrics["valid_mask"], max_mm=max_mm),
                label="Native residual",
                metrics=native_metrics,
            )
            ffs_tile = build_face_metric_tile(
                colorize_patch_residuals(ffs_metrics["residual_mm"], ffs_metrics["valid_mask"], max_mm=max_mm),
                label="FFS residual",
                metrics=ffs_metrics,
            )
            swapped_tile = build_face_metric_tile(
                colorize_patch_residuals(swapped_metrics["residual_mm"], swapped_metrics["valid_mask"], max_mm=max_mm),
                label="FFS-swapped residual",
                metrics=swapped_metrics,
            )
            patch_rows.append(
                [
                    build_face_metric_tile(_crop_with_padding(rgb_overlay, bbox), label=f"RGB | {patch['name']}", metrics=None),
                    native_tile,
                    ffs_tile,
                    swapped_tile,
                ]
            )
            patch_summaries.append(
                {
                    "camera_idx": int(camera_idx),
                    "patch_name": str(patch["name"]),
                    "native": {
                        "valid_depth_ratio": native_metrics["valid_depth_ratio"],
                        "plane_fit_rmse_mm": native_metrics["plane_fit_rmse_mm"],
                        "mad_mm": native_metrics["mad_mm"],
                        "p90_abs_residual_mm": native_metrics["p90_abs_residual_mm"],
                    },
                    "ffs": {
                        "valid_depth_ratio": ffs_metrics["valid_depth_ratio"],
                        "plane_fit_rmse_mm": ffs_metrics["plane_fit_rmse_mm"],
                        "mad_mm": ffs_metrics["mad_mm"],
                        "p90_abs_residual_mm": ffs_metrics["p90_abs_residual_mm"],
                    },
                    "ffs_swapped": {
                        "valid_depth_ratio": swapped_metrics["valid_depth_ratio"],
                        "plane_fit_rmse_mm": swapped_metrics["plane_fit_rmse_mm"],
                        "mad_mm": swapped_metrics["mad_mm"],
                        "p90_abs_residual_mm": swapped_metrics["p90_abs_residual_mm"],
                    },
                }
            )

    board = compose_face_quality_board(
        title_lines=[
            f"{native_case_dir.name} vs {ffs_case_dir.name} | frame_idx={selected_frame_idx} | face smoothness",
            "Rows = fixed face patches | Columns = RGB patch / Native / FFS / FFS-swapped",
        ],
        metric_lines=[
            "Better on a face patch means: higher valid ratio, lower plane-fit RMSE, lower MAD, lower p90 residual.",
        ],
        patch_rows=patch_rows,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_image(output_dir / "face_quality_board.png", board)
    return {
        "output_dir": str(output_dir),
        "patch_summaries": patch_summaries,
        "same_case_mode": bool(same_case_mode),
    }


def _crop_camera_clouds_to_bounds(
    camera_clouds: list[dict[str, Any]],
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> list[dict[str, Any]]:
    cropped: list[dict[str, Any]] = []
    lower = np.asarray(bounds_min, dtype=np.float32)
    upper = np.asarray(bounds_max, dtype=np.float32)
    for camera_cloud in camera_clouds:
        points = np.asarray(camera_cloud["points"], dtype=np.float32).reshape(-1, 3)
        colors = np.asarray(camera_cloud["colors"], dtype=np.uint8).reshape(-1, 3)
        source_camera_idx = np.asarray(
            camera_cloud.get("source_camera_idx", np.full((len(points),), int(camera_cloud["camera_idx"]), dtype=np.int16)),
            dtype=np.int16,
        ).reshape(-1)
        source_serial = np.asarray(
            camera_cloud.get("source_serial", np.full((len(points),), camera_cloud["serial"], dtype=object)),
            dtype=object,
        ).reshape(-1)
        valid = np.all(points >= lower[None, :], axis=1) & np.all(points <= upper[None, :], axis=1)
        cropped.append(
            {
                **camera_cloud,
                "points": points[valid],
                "colors": colors[valid],
                "source_camera_idx": source_camera_idx[valid],
                "source_serial": source_serial[valid],
            }
        )
    return cropped


def _compute_point_bounds(point_sets: list[np.ndarray], *, fallback_center: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    clouds = [np.asarray(item, dtype=np.float32).reshape(-1, 3) for item in point_sets if len(item) > 0]
    if not clouds:
        center = np.asarray(fallback_center, dtype=np.float32)
        return center - 0.25, center + 0.25
    stacked = np.concatenate(clouds, axis=0)
    return stacked.min(axis=0).astype(np.float32), stacked.max(axis=0).astype(np.float32)


def _build_swapped_ffs_camera_clouds(
    *,
    selection: dict[str, Any],
    ffs_repo: str | Path,
    model_path: str | Path,
    depth_min_m: float,
    depth_max_m: float,
    scale: float,
    valid_iters: int,
    max_disp: int,
    runner_factory=FastFoundationStereoRunner,
) -> list[dict[str, Any]]:
    ffs_case_dir = Path(selection["ffs_case_dir"])
    ffs_metadata = selection["ffs_metadata"]
    c2w_list = selection["native_c2w"]
    frame_idx = int(selection["ffs_frame_idx"])
    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
    )

    per_camera_clouds: list[dict[str, Any]] = []
    for camera_idx in selection["camera_ids"]:
        _require_ffs_ir_geometry(ffs_case_dir, ffs_metadata, int(camera_idx))
        left_image, right_image = _load_aligned_ir_pair(ffs_case_dir, camera_idx=int(camera_idx), frame_idx=frame_idx)
        K_ir_right = np.asarray(ffs_metadata["K_ir_right"][camera_idx], dtype=np.float32)
        K_color = np.asarray(ffs_metadata["K_color"][camera_idx], dtype=np.float32)
        T_ir_left_to_right = np.asarray(ffs_metadata["T_ir_left_to_right"][camera_idx], dtype=np.float32)
        T_ir_left_to_color = np.asarray(ffs_metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
        T_ir_right_to_color = derive_ir_right_to_color(T_ir_left_to_right, T_ir_left_to_color)
        baseline_m = float(ffs_metadata["ir_baseline_m"][camera_idx])
        swapped_run = runner.run_pair(
            right_image,
            left_image,
            K_ir_left=K_ir_right,
            baseline_m=baseline_m,
            audit_mode=False,
        )
        output_shape = (int(ffs_metadata["WH"][1]), int(ffs_metadata["WH"][0]))
        swapped_depth_color = align_depth_to_color(
            np.asarray(swapped_run["depth_ir_left_m"], dtype=np.float32),
            np.asarray(swapped_run["K_ir_left_used"], dtype=np.float32),
            T_ir_right_to_color,
            K_color,
            output_shape=output_shape,
        )
        color_path = ffs_case_dir / "color" / str(camera_idx) / f"{frame_idx}.png"
        color_image = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
        if color_image is None:
            raise FileNotFoundError(f"Missing aligned RGB frame for swapped FFS point cloud: {color_path}")
        camera_points, camera_colors, _ = depth_to_camera_points(
            swapped_depth_color,
            K_color,
            depth_min_m=depth_min_m,
            depth_max_m=depth_max_m,
            color_image=color_image,
            pixel_roi=None,
            max_points_per_camera=None,
        )
        world_points = transform_points(camera_points, np.asarray(c2w_list[camera_idx], dtype=np.float32))
        serial = str(ffs_metadata["serial_numbers"][camera_idx])
        per_camera_clouds.append(
            {
                "camera_idx": int(camera_idx),
                "serial": serial,
                "depth_dir_used": "ffs_swapped_runtime",
                "used_float_depth": True,
                "K_color": K_color,
                "c2w": np.asarray(c2w_list[camera_idx], dtype=np.float32),
                "color_path": str(color_path),
                "points": world_points,
                "colors": camera_colors,
                "source_camera_idx": np.full((len(world_points),), int(camera_idx), dtype=np.int16),
                "source_serial": np.full((len(world_points),), serial, dtype=object),
            }
        )
    return per_camera_clouds


def _build_registration_view_specs(
    *,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    focus_point: np.ndarray,
    closeup_scale: float = 1.0,
) -> list[dict[str, Any]]:
    center = np.asarray(focus_point, dtype=np.float32)
    extents = np.maximum(np.asarray(bounds_max, dtype=np.float32) - np.asarray(bounds_min, dtype=np.float32), 1e-6)
    radius = max(0.20, float(np.linalg.norm(extents)) * float(closeup_scale))
    return [
        {
            "label": "Oblique",
            "projection_mode": "perspective",
            "view_config": {
                "view_name": "oblique",
                "label": "Oblique",
                "center": center,
                "camera_position": center + np.array([1.30, -1.15, 0.90], dtype=np.float32) * radius,
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        },
        {
            "label": "Top",
            "projection_mode": "orthographic",
            "view_config": {
                "view_name": "top",
                "label": "Top",
                "center": center,
                "camera_position": center + np.array([0.0, 0.0, 2.10], dtype=np.float32) * radius,
                "up": np.array([0.0, -1.0, 0.0], dtype=np.float32),
            },
        },
        {
            "label": "Front",
            "projection_mode": "orthographic",
            "view_config": {
                "view_name": "front",
                "label": "Front",
                "center": center,
                "camera_position": center + np.array([2.10, 0.0, 0.0], dtype=np.float32) * radius,
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        },
        {
            "label": "Side",
            "projection_mode": "orthographic",
            "view_config": {
                "view_name": "side",
                "label": "Side",
                "center": center,
                "camera_position": center + np.array([0.0, -2.10, 0.0], dtype=np.float32) * radius,
                "up": np.array([0.0, 0.0, 1.0], dtype=np.float32),
            },
        },
    ]


def _render_registration_row(
    camera_clouds: list[dict[str, Any]],
    *,
    view_specs: list[dict[str, Any]],
    panel_width: int,
    panel_height: int,
    alpha: float,
) -> tuple[list[np.ndarray], list[dict[str, Any]]]:
    row_metrics: list[dict[str, Any]] = []
    images: list[np.ndarray] = []
    for view_spec in view_specs:
        ortho_scale = view_spec.get("ortho_scale")
        image, render_metrics = render_source_attribution_overlay(
            camera_clouds,
            view_config=view_spec["view_config"],
            width=panel_width,
            height=panel_height,
            projection_mode=view_spec["projection_mode"],
            ortho_scale=ortho_scale,
            alpha=alpha,
            show_legend=False,
        )
        images.append(image)
        row_metrics.append(
            {
                "view": str(view_spec["label"]),
                "projection_mode": str(view_spec["projection_mode"]),
                "ortho_scale": None if ortho_scale is None else float(ortho_scale),
                **render_metrics,
            }
        )
    return images, row_metrics


def _build_registration_board_rows(
    *,
    scene: dict[str, Any],
    refinement: dict[str, Any],
    swapped_camera_clouds: list[dict[str, Any]],
    context_max_points_per_camera: int | None,
    object_height_min: float,
    object_height_max: float,
) -> dict[str, Any]:
    crop_bounds = scene["crop_bounds"]
    cropped_swapped_camera_clouds = _crop_camera_clouds_to_bounds(
        swapped_camera_clouds,
        bounds_min=np.asarray(crop_bounds["min"], dtype=np.float32),
        bounds_max=np.asarray(crop_bounds["max"], dtype=np.float32),
    )
    if len(scene["native_points"]) > 0:
        table_color_bgr = estimate_table_color_bgr(
            scene["native_points"],
            scene["native_colors"],
            plane_point=np.asarray(scene["plane_point"], dtype=np.float32),
            plane_normal=np.asarray(scene["plane_normal"], dtype=np.float32),
        )
    elif len(scene["ffs_points"]) > 0:
        table_color_bgr = estimate_table_color_bgr(
            scene["ffs_points"],
            scene["ffs_colors"],
            plane_point=np.asarray(scene["plane_point"], dtype=np.float32),
            plane_normal=np.asarray(scene["plane_normal"], dtype=np.float32),
        )
    else:
        table_color_bgr = np.array([128.0, 128.0, 128.0], dtype=np.float32)
    swapped_layers = build_object_first_layers(
        cropped_swapped_camera_clouds,
        object_roi_min=np.asarray(scene["object_roi_bounds"]["min"], dtype=np.float32),
        object_roi_max=np.asarray(scene["object_roi_bounds"]["max"], dtype=np.float32),
        plane_point=np.asarray(scene["plane_point"], dtype=np.float32),
        plane_normal=np.asarray(scene["plane_normal"], dtype=np.float32),
        table_color_bgr=table_color_bgr,
        object_height_min=float(object_height_min),
        object_height_max=float(object_height_max),
        context_max_points_per_camera=context_max_points_per_camera,
        pixel_mask_by_camera=refinement.get("final_ffs_masks"),
    )
    return {
        "native_object_camera_clouds": scene["native_object_camera_clouds"],
        "ffs_object_camera_clouds": scene["ffs_object_camera_clouds"],
        "swapped_object_camera_clouds": swapped_layers["object_camera_clouds"],
        "swapped_object_points": swapped_layers["object_points"],
        "swapped_object_colors": swapped_layers["object_colors"],
        "swapped_metrics": swapped_layers["per_camera_metrics"],
    }


def run_stereo_order_registration_workflow(
    *,
    aligned_root: Path,
    output_dir: Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    case_name: str | None = None,
    realsense_case: str | None = None,
    ffs_case: str | None = None,
    frame_idx: int = 0,
    scene_crop_mode: str = "auto_object_bbox",
    focus_mode: str = "table",
    crop_margin_xy: float = 0.12,
    crop_min_z: float = -0.15,
    crop_max_z: float = 0.35,
    object_height_min: float = 0.02,
    object_height_max: float = 0.60,
    object_component_mode: str = "graph_union",
    object_component_topk: int = 2,
    roi_x_min: float | None = None,
    roi_x_max: float | None = None,
    roi_y_min: float | None = None,
    roi_y_max: float | None = None,
    roi_z_min: float | None = None,
    roi_z_max: float | None = None,
    manual_image_roi_json: str | Path | None = None,
    voxel_size: float | None = None,
    max_points_per_camera: int | None = 50000,
    depth_min_m: float = 0.2,
    depth_max_m: float = 1.5,
    use_float_ffs_depth_when_available: bool = True,
    camera_ids: list[int] | None = None,
    panel_width: int = 380,
    panel_height: int = 300,
    alpha: float = 0.34,
    scale: float = 1.0,
    valid_iters: int = 8,
    max_disp: int = 192,
    write_debug: bool = False,
    write_closeup: bool = False,
    runner_factory=FastFoundationStereoRunner,
) -> dict[str, Any]:
    output_dir = Path(output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    turntable_state = _build_turntable_scene(
        aligned_root=Path(aligned_root).resolve(),
        case_name=case_name,
        realsense_case=realsense_case,
        ffs_case=ffs_case,
        frame_idx=frame_idx,
        camera_ids=camera_ids,
        voxel_size=voxel_size,
        max_points_per_camera=max_points_per_camera,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        use_float_ffs_depth_when_available=use_float_ffs_depth_when_available,
        focus_mode=focus_mode,
        scene_crop_mode=scene_crop_mode,
        crop_margin_xy=crop_margin_xy,
        crop_min_z=crop_min_z,
        crop_max_z=crop_max_z,
        object_height_min=object_height_min,
        object_height_max=object_height_max,
        object_component_mode=object_component_mode,
        object_component_topk=object_component_topk,
        roi_x_min=roi_x_min,
        roi_x_max=roi_x_max,
        roi_y_min=roi_y_min,
        roi_y_max=roi_y_max,
        roi_z_min=roi_z_min,
        roi_z_max=roi_z_max,
        manual_image_roi_json=manual_image_roi_json,
    )
    selection = turntable_state["selection"]
    scene = turntable_state["scene"]
    refinement = turntable_state["refinement"]
    swapped_camera_clouds = _build_swapped_ffs_camera_clouds(
        selection=selection,
        ffs_repo=ffs_repo,
        model_path=model_path,
        depth_min_m=depth_min_m,
        depth_max_m=depth_max_m,
        scale=scale,
        valid_iters=valid_iters,
        max_disp=max_disp,
        runner_factory=runner_factory,
    )
    row_state = _build_registration_board_rows(
        scene=scene,
        refinement=refinement,
        swapped_camera_clouds=swapped_camera_clouds,
        context_max_points_per_camera=max_points_per_camera,
        object_height_min=object_height_min,
        object_height_max=object_height_max,
    )
    focus_point = np.asarray(scene["focus_point"], dtype=np.float32)
    all_object_points = [
        np.asarray(scene["native_object_points"], dtype=np.float32),
        np.asarray(scene["ffs_object_points"], dtype=np.float32),
        np.asarray(row_state["swapped_object_points"], dtype=np.float32),
    ]
    object_bounds_min, object_bounds_max = _compute_point_bounds(all_object_points, fallback_center=focus_point)

    def _render_board(closeup_scale: float) -> tuple[np.ndarray, dict[str, Any]]:
        view_specs = _build_registration_view_specs(
            bounds_min=object_bounds_min,
            bounds_max=object_bounds_max,
            focus_point=focus_point,
            closeup_scale=closeup_scale,
        )
        shared_point_sets = [
            np.asarray(scene["native_object_points"], dtype=np.float32),
            np.asarray(scene["ffs_object_points"], dtype=np.float32),
            np.asarray(row_state["swapped_object_points"], dtype=np.float32),
        ]
        for view_spec in view_specs:
            if view_spec["projection_mode"] == "orthographic":
                view_spec["ortho_scale"] = estimate_ortho_scale(
                    shared_point_sets,
                    view_config=view_spec["view_config"],
                    margin=1.10,
                )
            else:
                view_spec["ortho_scale"] = None
        native_images, native_metrics = _render_registration_row(
            row_state["native_object_camera_clouds"],
            view_specs=view_specs,
            panel_width=panel_width,
            panel_height=panel_height,
            alpha=alpha,
        )
        current_images, current_metrics = _render_registration_row(
            row_state["ffs_object_camera_clouds"],
            view_specs=view_specs,
            panel_width=panel_width,
            panel_height=panel_height,
            alpha=alpha,
        )
        swapped_images, swapped_metrics = _render_registration_row(
            row_state["swapped_object_camera_clouds"],
            view_specs=view_specs,
            panel_width=panel_width,
            panel_height=panel_height,
            alpha=alpha,
        )
        case_label = (
            selection["native_case_dir"].name
            if selection["same_case_mode"]
            else f"{selection['native_case_dir'].name} vs {selection['ffs_case_dir'].name}"
        )
        title_lines = [
            f"{case_label} | frame_idx={selection['native_frame_idx']} | stereo-order registration",
            f"shared object ROI | colors encode source camera | crop={scene_crop_mode}",
        ]
        board = compose_registration_matrix_board(
            title_lines=title_lines,
            row_headers=["Native", "FFS-current", "FFS-swapped"],
            column_headers=[item["label"] for item in view_specs],
            image_rows=[native_images, current_images, swapped_images],
            legend_image=build_source_legend_image(),
        )
        metrics = {
            "views": [
                {
                    "label": str(item["label"]),
                    "projection_mode": str(item["projection_mode"]),
                    "ortho_scale": None if item.get("ortho_scale") is None else float(item["ortho_scale"]),
                    "camera_position": np.asarray(item["view_config"]["camera_position"], dtype=np.float32).tolist(),
                    "center": np.asarray(item["view_config"]["center"], dtype=np.float32).tolist(),
                    "up": np.asarray(item["view_config"]["up"], dtype=np.float32).tolist(),
                }
                for item in view_specs
            ],
            "rows": {
                "native": native_metrics,
                "ffs_current": current_metrics,
                "ffs_swapped": swapped_metrics,
            },
        }
        return board, metrics

    board, board_metrics = _render_board(closeup_scale=1.0)
    board_path = output_dir / "01_stereo_order_registration_board.png"
    write_image(board_path, board)

    closeup_path = None
    closeup_metrics = None
    if write_closeup:
        closeup_board, closeup_metrics = _render_board(closeup_scale=0.72)
        closeup_path = output_dir / "02_stereo_order_closeup_board.png"
        write_image(closeup_path, closeup_board)

    summary = {
        "same_case_mode": bool(selection["same_case_mode"]),
        "native_case_dir": str(selection["native_case_dir"]),
        "ffs_case_dir": str(selection["ffs_case_dir"]),
        "frame_idx": int(selection["native_frame_idx"]),
        "scene_crop_mode": scene_crop_mode,
        "camera_ids": [int(item) for item in selection["camera_ids"]],
        "visualization_frame_contract": build_visualization_frame_contract(
            uses_semantic_world=False,
            semantic_world_frame_kind=None,
            notes=[
                "All rows are rendered in the same raw calibration-world frame.",
                "This board is point-cloud-only and colors points by source camera.",
            ],
        ),
        "source_color_map_bgr": {str(key): list(value) for key, value in SOURCE_CAMERA_COLORS_BGR.items()},
        "object_roi_bounds": {
            "min": np.asarray(scene["object_roi_bounds"]["min"], dtype=np.float32).tolist(),
            "max": np.asarray(scene["object_roi_bounds"]["max"], dtype=np.float32).tolist(),
        },
        "crop_bounds": {
            "min": np.asarray(scene["crop_bounds"]["min"], dtype=np.float32).tolist(),
            "max": np.asarray(scene["crop_bounds"]["max"], dtype=np.float32).tolist(),
        },
        "render_settings": {
            "panel_width": int(panel_width),
            "panel_height": int(panel_height),
            "alpha": float(alpha),
            "depth_min_m": float(depth_min_m),
            "depth_max_m": float(depth_max_m),
            "scale": float(scale),
            "valid_iters": int(valid_iters),
            "max_disp": int(max_disp),
        },
        "row_point_counts": {
            "native_object_points": int(len(scene["native_object_points"])),
            "ffs_current_object_points": int(len(scene["ffs_object_points"])),
            "ffs_swapped_object_points": int(len(row_state["swapped_object_points"])),
        },
        "board_views": board_metrics["views"],
        "board_row_metrics": board_metrics["rows"],
        "top_level_output": str(board_path),
        "closeup_output": None if closeup_path is None else str(closeup_path),
        "debug_written": bool(write_debug),
    }
    write_json(output_dir / "match_board_summary.json", summary)

    if write_debug:
        debug_dir = output_dir / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        write_json(
            debug_dir / "registration_board_debug.json",
            {
                "swapped_object_metrics": row_state["swapped_metrics"],
                "refinement_valid": bool(refinement["pass2_refinement_valid"]),
                "final_compare_source": "pass2" if refinement["pass2_refinement_valid"] else "pass1",
                "closeup_metrics": closeup_metrics,
                "board_metrics": board_metrics,
            },
        )

    return {
        "output_dir": str(output_dir),
        "summary": summary,
    }
