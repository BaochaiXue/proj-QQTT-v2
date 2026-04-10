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
from .layouts import compose_depth_review_board
from .pointcloud_compare import get_frame_count, load_case_metadata, resolve_case_dirs


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
