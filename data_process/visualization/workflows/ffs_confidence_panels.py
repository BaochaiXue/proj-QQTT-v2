from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import cv2
import numpy as np

from data_process.depth_backends import (
    FastFoundationStereoRunner,
    align_depth_to_color,
    align_ir_scalar_to_color,
)
from ..depth_colormap import colorize_depth_meters
from ..depth_diagnostics import label_tile
from ..io_artifacts import write_image, write_json
from ..io_case import load_case_metadata, resolve_case_dir
from ..layouts import compose_registration_matrix_board, overlay_scalar_colorbar
from .masked_camera_view_compare import _mask_rgb_image
from .masked_pointcloud_compare import load_union_masks_for_camera_clouds


DEFAULT_STATIC_CONFIDENCE_CASE_REFS: tuple[str, ...] = (
    "static/ffs_30_static_round1_20260410_235202",
    "static/ffs_30_static_round2_20260414",
    "static/ffs_30_static_round3_20260414",
)
DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT = "stuffed animal"
DEFAULT_CONFIDENCE_METRICS: tuple[str, ...] = ("margin", "max_softmax")


def _round_id_from_case_ref(case_ref: str) -> str:
    stem = Path(case_ref).name
    if "round1" in stem:
        return "round1"
    if "round2" in stem:
        return "round2"
    if "round3" in stem:
        return "round3"
    raise ValueError(f"Could not infer round id from case_ref={case_ref!r}.")


def _round_label_from_round_id(round_id: str) -> str:
    return round_id.replace("round", "Round ")


def build_static_confidence_round_specs(
    *,
    aligned_root: Path,
    case_refs: tuple[str, ...] = DEFAULT_STATIC_CONFIDENCE_CASE_REFS,
) -> list[dict[str, Any]]:
    aligned_root = Path(aligned_root).resolve()
    specs: list[dict[str, Any]] = []
    for case_ref in case_refs:
        round_id = _round_id_from_case_ref(case_ref)
        mask_root = (
            aligned_root
            / "static"
            / f"masked_pointcloud_compare_{round_id}_frame_0000_stuffed_animal"
            / "_generated_masks"
            / "ffs"
            / "sam31_masks"
        )
        specs.append(
            {
                "round_id": round_id,
                "round_label": _round_label_from_round_id(round_id),
                "case_ref": str(case_ref),
                "mask_root": mask_root.resolve(),
            }
        )
    return specs


def _resolve_metric_names(metrics: str) -> tuple[str, ...]:
    normalized = str(metrics).strip().lower()
    if normalized == "both":
        return DEFAULT_CONFIDENCE_METRICS
    if normalized in DEFAULT_CONFIDENCE_METRICS:
        return (normalized,)
    raise ValueError(f"Unsupported confidence metrics selection: {metrics!r}")


def _load_color_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Missing color image: {path}")
    return image


def _load_ir_image(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise FileNotFoundError(f"Missing IR image: {path}")
    return image


def _mask_image(image: np.ndarray, *, mask: np.ndarray) -> np.ndarray:
    masked = np.zeros_like(np.asarray(image, dtype=np.uint8))
    bool_mask = np.asarray(mask, dtype=bool)
    masked[bool_mask] = np.asarray(image, dtype=np.uint8)[bool_mask]
    return masked


def _colorize_confidence_map(confidence: np.ndarray, *, valid_mask: np.ndarray | None = None) -> np.ndarray:
    confidence = np.asarray(confidence, dtype=np.float32)
    normalized = np.clip(confidence, 0.0, 1.0)
    colorized = cv2.applyColorMap((normalized * 255.0).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
    if valid_mask is not None:
        colorized = np.asarray(colorized, dtype=np.uint8).copy()
        colorized[~np.asarray(valid_mask, dtype=bool)] = 0
    return colorized


def _metric_label(metric_name: str) -> str:
    if metric_name == "margin":
        return "Confidence (margin)"
    if metric_name == "max_softmax":
        return "Confidence (max_softmax)"
    raise ValueError(f"Unsupported metric_name={metric_name!r}.")


def build_confidence_board(
    *,
    round_label: str,
    frame_idx: int,
    metric_name: str,
    model_config: dict[str, Any],
    column_headers: list[str],
    rgb_images: list[np.ndarray],
    depth_images: list[np.ndarray],
    confidence_images: list[np.ndarray],
) -> np.ndarray:
    if len(rgb_images) != 3 or len(depth_images) != 3 or len(confidence_images) != 3:
        raise ValueError("Confidence board requires exactly 3 images per row.")

    board_rows = [
        [label_tile(image, "RGB", (image.shape[1], image.shape[0])) for image in rgb_images],
        [
            label_tile(
                overlay_scalar_colorbar(
                    image,
                    label="m",
                    min_text=f"{float(model_config['depth_min_m']):.1f}",
                    max_text=f"{float(model_config['depth_max_m']):.1f}",
                    colormap=cv2.COLORMAP_TURBO,
                ),
                "FFS Depth",
                (image.shape[1], image.shape[0]),
            )
            for image in depth_images
        ],
        [
            label_tile(
                overlay_scalar_colorbar(
                    image,
                    label="conf",
                    min_text="0.0",
                    max_text="1.0",
                    colormap=cv2.COLORMAP_VIRIDIS,
                ),
                _metric_label(metric_name),
                (image.shape[1], image.shape[0]),
            )
            for image in confidence_images
        ],
    ]
    return compose_registration_matrix_board(
        title_lines=[
            f"FFS Static Confidence | {round_label} | frame {int(frame_idx)}",
            (
                f"metric={metric_name} | scale={float(model_config['scale']):.2f} | "
                f"iters={int(model_config['valid_iters'])} | disp={int(model_config['max_disp'])}"
            ),
        ],
        row_headers=["RGB", "FFS Depth", "Confidence"],
        column_headers=column_headers,
        image_rows=board_rows,
    )


def run_ffs_static_confidence_panels_workflow(
    *,
    aligned_root: Path,
    output_root: Path,
    ffs_repo: str | Path,
    model_path: str | Path,
    scale: float = 1.0,
    valid_iters: int = 8,
    max_disp: int = 192,
    depth_min_m: float = 0.0,
    depth_max_m: float = 1.5,
    metrics: str = "both",
    frame_idx: int = 0,
    text_prompt: str = DEFAULT_STATIC_CONFIDENCE_MASK_PROMPT,
    round_specs: list[dict[str, Any]] | None = None,
    runner_factory: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    aligned_root = Path(aligned_root).resolve()
    output_root = Path(output_root).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    round_specs = (
        build_static_confidence_round_specs(aligned_root=aligned_root)
        if round_specs is None
        else list(round_specs)
    )
    metric_names = _resolve_metric_names(metrics)
    model_config = {
        "ffs_repo": str(Path(ffs_repo).resolve()),
        "model_path": str(Path(model_path).resolve()),
        "scale": float(scale),
        "valid_iters": int(valid_iters),
        "max_disp": int(max_disp),
        "depth_min_m": float(depth_min_m),
        "depth_max_m": float(depth_max_m),
    }
    runner_factory = FastFoundationStereoRunner if runner_factory is None else runner_factory
    runner = runner_factory(
        ffs_repo=ffs_repo,
        model_path=model_path,
        scale=float(scale),
        valid_iters=int(valid_iters),
        max_disp=int(max_disp),
    )

    rounds_summary: list[dict[str, Any]] = []
    for round_spec in round_specs:
        case_dir = resolve_case_dir(aligned_root=aligned_root, case_ref=str(round_spec["case_ref"]))
        metadata = load_case_metadata(case_dir)
        selected_frame_idx = int(frame_idx)
        if selected_frame_idx < 0 or selected_frame_idx >= int(metadata["frame_num"]):
            raise ValueError(
                f"frame_idx={selected_frame_idx} is out of range for case {case_dir.name} "
                f"with frame_num={int(metadata['frame_num'])}."
            )

        camera_ids = list(range(len(metadata["serial_numbers"])))
        if len(camera_ids) != 3:
            raise ValueError(f"Static confidence workflow expects exactly 3 cameras. Got {camera_ids}.")

        round_output_dir = output_root / str(round_spec["round_id"])
        round_output_dir.mkdir(parents=True, exist_ok=True)
        mask_root = Path(round_spec["mask_root"]).resolve()
        if not mask_root.is_dir():
            raise FileNotFoundError(f"Missing static mask root for {round_spec['round_id']}: {mask_root}")

        minimal_clouds = [
            {
                "camera_idx": int(camera_idx),
                "serial": str(metadata["serial_numbers"][camera_idx]),
                "color_path": str(case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"),
            }
            for camera_idx in camera_ids
        ]
        mask_by_camera, mask_debug = load_union_masks_for_camera_clouds(
            mask_root=mask_root,
            camera_clouds=minimal_clouds,
            frame_token=str(selected_frame_idx),
            text_prompt=text_prompt,
        )

        column_headers: list[str] = []
        rgb_by_camera: list[np.ndarray] = []
        depth_vis_by_camera: list[np.ndarray] = []
        confidence_vis_by_metric: dict[str, list[np.ndarray]] = {metric_name: [] for metric_name in metric_names}
        per_camera_summary: list[dict[str, Any]] = []

        for camera_idx in camera_ids:
            serial = str(metadata["serial_numbers"][camera_idx])
            color_path = case_dir / "color" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_left_path = case_dir / "ir_left" / str(camera_idx) / f"{selected_frame_idx}.png"
            ir_right_path = case_dir / "ir_right" / str(camera_idx) / f"{selected_frame_idx}.png"
            color_image = _load_color_image(color_path)
            ir_left = _load_ir_image(ir_left_path)
            ir_right = _load_ir_image(ir_right_path)

            run_output = runner.run_pair_with_confidence(
                ir_left,
                ir_right,
                K_ir_left=np.asarray(metadata["K_ir_left"][camera_idx], dtype=np.float32),
                baseline_m=float(metadata["ir_baseline_m"][camera_idx]),
                audit_mode=False,
            )
            depth_ir_left_m = np.asarray(run_output["depth_ir_left_m"], dtype=np.float32)
            K_ir_left_used = np.asarray(run_output["K_ir_left_used"], dtype=np.float32)
            T_ir_left_to_color = np.asarray(metadata["T_ir_left_to_color"][camera_idx], dtype=np.float32)
            K_color = np.asarray(metadata["K_color"][camera_idx], dtype=np.float32)
            output_shape = (int(color_image.shape[0]), int(color_image.shape[1]))
            depth_color_m = align_depth_to_color(
                depth_ir_left_m,
                K_ir_left_used,
                T_ir_left_to_color,
                K_color,
                output_shape=output_shape,
                invalid_value=0.0,
            )
            mask = np.asarray(mask_by_camera[int(camera_idx)], dtype=bool)
            rgb_by_camera.append(_mask_rgb_image(color_path, mask=mask))
            depth_color_vis = colorize_depth_meters(
                depth_color_m,
                depth_min_m=float(depth_min_m),
                depth_max_m=float(depth_max_m),
            )
            depth_vis_by_camera.append(_mask_image(depth_color_vis, mask=mask))
            depth_valid_mask = np.isfinite(depth_color_m) & (depth_color_m > 0)

            confidence_summary: dict[str, Any] = {}
            for metric_name in metric_names:
                confidence_ir = np.asarray(
                    run_output[f"confidence_{metric_name}_ir_left"],
                    dtype=np.float32,
                )
                confidence_color = align_ir_scalar_to_color(
                    depth_ir_left_m,
                    confidence_ir,
                    K_ir_left_used,
                    T_ir_left_to_color,
                    K_color,
                    output_shape=output_shape,
                    invalid_value=0.0,
                )
                confidence_color_vis = _colorize_confidence_map(confidence_color, valid_mask=depth_valid_mask)
                confidence_vis_by_metric[metric_name].append(_mask_image(confidence_color_vis, mask=mask))
                confidence_summary[metric_name] = {
                    "aligned_valid_pixel_count": int(np.count_nonzero(depth_valid_mask)),
                    "masked_valid_pixel_count": int(np.count_nonzero(mask & depth_valid_mask)),
                    "min": float(np.min(confidence_color[depth_valid_mask])) if np.any(depth_valid_mask) else 0.0,
                    "max": float(np.max(confidence_color[depth_valid_mask])) if np.any(depth_valid_mask) else 0.0,
                }

            per_camera_summary.append(
                {
                    "camera_idx": int(camera_idx),
                    "serial": serial,
                    "mask_pixel_count": int(np.count_nonzero(mask)),
                    "color_path": str(color_path.resolve()),
                    "ir_left_path": str(ir_left_path.resolve()),
                    "ir_right_path": str(ir_right_path.resolve()),
                    "depth_valid_pixel_count_aligned": int(np.count_nonzero((np.isfinite(depth_color_m)) & (depth_color_m > 0))),
                    "depth_valid_pixel_count_masked": int(np.count_nonzero(mask & np.isfinite(depth_color_m) & (depth_color_m > 0))),
                    "confidence": confidence_summary,
                }
            )
            column_headers.append(f"Cam{int(camera_idx)} | {serial}")

        board_paths: dict[str, str] = {}
        for metric_name in metric_names:
            board = build_confidence_board(
                round_label=str(round_spec["round_label"]),
                frame_idx=selected_frame_idx,
                metric_name=metric_name,
                model_config=model_config,
                column_headers=column_headers,
                rgb_images=rgb_by_camera,
                depth_images=depth_vis_by_camera,
                confidence_images=confidence_vis_by_metric[metric_name],
            )
            board_path = round_output_dir / f"{metric_name}_board.png"
            write_image(board_path, board)
            board_paths[metric_name] = str(board_path.resolve())

        round_summary = {
            "round_id": str(round_spec["round_id"]),
            "round_label": str(round_spec["round_label"]),
            "case_ref": str(round_spec["case_ref"]),
            "case_dir": str(case_dir.resolve()),
            "frame_idx": int(selected_frame_idx),
            "mask_root": str(mask_root.resolve()),
            "text_prompt": str(text_prompt),
            "metrics": list(metric_names),
            "model_config": dict(model_config),
            "row_headers": ["RGB", "FFS Depth", "Confidence"],
            "column_headers": column_headers,
            "board_paths": board_paths,
            "per_camera": per_camera_summary,
            "mask_debug": {str(key): value for key, value in mask_debug.items()},
            "output_dir": str(round_output_dir.resolve()),
        }
        write_json(round_output_dir / "summary.json", round_summary)
        rounds_summary.append(round_summary)

    manifest = {
        "output_dir": str(output_root.resolve()),
        "frame_idx": int(frame_idx),
        "text_prompt": str(text_prompt),
        "metrics": list(metric_names),
        "model_config": dict(model_config),
        "rounds": rounds_summary,
    }
    write_json(output_root / "summary.json", manifest)
    return manifest
