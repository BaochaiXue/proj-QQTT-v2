from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import cv2
import imageio.v2 as imageio
import numpy as np

from .sam21_checkpoint_ladder_panel import (
    EDGE_TAM_VARIANT_KEY,
    compose_panel,
    label_tile,
    load_union_mask,
    write_image,
    write_json,
)


DEFAULT_CASE_KEY = "ffs_dynamics_round1"
DEFAULT_CASE_LABEL = "FFS Dynamics Round1"
DEFAULT_CASE_DIR = Path("data/dynamics/ffs_dynamics_round1_20260414")
SLOTH_BASE_MOTION_CASE_KEY = "sloth_base_motion_ffs"
SLOTH_BASE_MOTION_CASE_LABEL = "sloth_base_motion_ffs"
SLOTH_BASE_MOTION_CASE_DIR = Path("data/different_types/sloth_base_motion_ffs")
SLOTH_BASE_MOTION_OUTPUT_DIR = Path("result/sloth_base_motion_ffs_mask_overlay_3x3")
DEFAULT_OUTPUT_DIR = Path(
    "result/"
    "sam21_dynamics_checkpoint_ladder_3x5_time_gifs_"
    "ffs203048_iter4_trt_level5_maskinit_stable_throughput"
)
DEFAULT_DOC_MD = Path("docs/generated/sam21_edgetam_mask_overlay_3x3_benchmark.md")
DEFAULT_DOC_JSON = Path("docs/generated/sam21_edgetam_mask_overlay_3x3_results.json")
SLOTH_BASE_MOTION_DOC_MD = Path("docs/generated/sloth_base_motion_ffs_mask_overlay_3x3_benchmark.md")
SLOTH_BASE_MOTION_DOC_JSON = Path("docs/generated/sloth_base_motion_ffs_mask_overlay_3x3_results.json")
DEFAULT_VARIANTS: tuple[tuple[str, str], ...] = (
    ("small", "SAM2.1 Small"),
    ("tiny", "SAM2.1 Tiny"),
    (EDGE_TAM_VARIANT_KEY, "EdgeTAM compiled"),
)
DEFAULT_CAMERA_IDS: tuple[int, ...] = (0, 1, 2)


def sorted_frame_tokens(case_dir: str | Path, *, camera_idx: int, frames: int | None = None) -> list[str]:
    paths = sorted(
        (Path(case_dir) / "color" / str(int(camera_idx))).glob("*.png"),
        key=lambda path: int(path.stem),
    )
    if frames is not None:
        paths = paths[: int(frames)]
    if not paths:
        raise FileNotFoundError(f"No color frames found for cam{camera_idx}: {case_dir}")
    return [path.stem for path in paths]


def compare_masks(reference_mask: np.ndarray, candidate_mask: np.ndarray) -> dict[str, Any]:
    reference = np.asarray(reference_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if reference.shape != candidate.shape:
        raise ValueError(f"mask shapes differ: {reference.shape} vs {candidate.shape}")
    intersection = reference & candidate
    union = reference | candidate
    reference_count = int(np.count_nonzero(reference))
    candidate_count = int(np.count_nonzero(candidate))
    union_count = int(np.count_nonzero(union))
    intersection_count = int(np.count_nonzero(intersection))
    return {
        "reference_pixel_count": reference_count,
        "candidate_pixel_count": candidate_count,
        "intersection_pixel_count": intersection_count,
        "union_pixel_count": union_count,
        "reference_only_pixel_count": int(np.count_nonzero(reference & ~candidate)),
        "candidate_only_pixel_count": int(np.count_nonzero(candidate & ~reference)),
        "iou": float(intersection_count / union_count) if union_count else 1.0,
        "area_ratio_candidate_over_reference": float(candidate_count / reference_count)
        if reference_count
        else 0.0,
    }


def render_mask_difference_overlay(
    color_bgr: np.ndarray,
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    *,
    alpha: float = 0.72,
    background_mode: str = "dim_full",
    color_overlap: bool = True,
) -> np.ndarray:
    image = np.asarray(color_bgr, dtype=np.uint8)
    reference = np.asarray(reference_mask, dtype=bool)
    candidate = np.asarray(candidate_mask, dtype=bool)
    if image.shape[:2] != reference.shape or reference.shape != candidate.shape:
        raise ValueError(
            f"image/reference/candidate shapes differ: {image.shape[:2]}, {reference.shape}, {candidate.shape}"
        )

    union_mask = reference | candidate
    if background_mode == "black_union_rgb":
        canvas = np.zeros_like(image, dtype=np.uint8)
        canvas[union_mask] = image[union_mask]
    elif background_mode == "dim_full":
        canvas = np.clip(image.astype(np.float32) * 0.58 + 18.0, 0.0, 255.0).astype(np.uint8)
    else:
        raise ValueError(f"Unsupported mask overlay background_mode: {background_mode!r}")

    categories = [
        (reference & ~candidate, np.asarray((45, 45, 255), dtype=np.uint8)),
        (candidate & ~reference, np.asarray((255, 230, 35), dtype=np.uint8)),
    ]
    if color_overlap:
        categories.insert(0, (reference & candidate, np.asarray((50, 220, 80), dtype=np.uint8)))
    for mask, color in categories:
        if not np.any(mask):
            continue
        color_image = np.empty_like(canvas)
        color_image[...] = color
        blended = np.clip(
            canvas.astype(np.float32) * (1.0 - float(alpha))
            + color_image.astype(np.float32) * float(alpha),
            0.0,
            255.0,
        ).astype(np.uint8)
        canvas[mask] = blended[mask]

    ref_contours, _ = cv2.findContours(reference.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cand_contours, _ = cv2.findContours(candidate.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(canvas, ref_contours, -1, (35, 35, 255), 2, cv2.LINE_AA)
    cv2.drawContours(canvas, cand_contours, -1, (255, 230, 35), 2, cv2.LINE_AA)
    return canvas


def build_overlay_tile(
    *,
    color_bgr: np.ndarray,
    reference_mask: np.ndarray,
    candidate_mask: np.ndarray,
    variant_label: str,
    tile_width: int,
    tile_height: int,
    background_mode: str = "dim_full",
    color_overlap: bool = True,
) -> tuple[np.ndarray, dict[str, Any]]:
    stats = compare_masks(reference_mask, candidate_mask)
    label = (
        f"{variant_label} | IoU {float(stats['iou']):.3f} | "
        f"area {float(stats['area_ratio_candidate_over_reference']):.2f}x"
    )
    overlay = render_mask_difference_overlay(
        color_bgr,
        reference_mask,
        candidate_mask,
        background_mode=str(background_mode),
        color_overlap=bool(color_overlap),
    )
    return label_tile(overlay, label=label, tile_width=int(tile_width), tile_height=int(tile_height)), stats


def _variant_roots(output_dir: Path, *, case_key: str) -> dict[str, Path]:
    return {
        "small": output_dir / "masks" / case_key / "small",
        "tiny": output_dir / "masks" / case_key / "tiny",
        EDGE_TAM_VARIANT_KEY: output_dir / "masks" / case_key / EDGE_TAM_VARIANT_KEY,
    }


def render_mask_overlay_3x3_gif(
    *,
    root: str | Path,
    case_dir: str | Path,
    output_dir: str | Path,
    case_key: str = DEFAULT_CASE_KEY,
    case_label: str = DEFAULT_CASE_LABEL,
    text_prompt: str = "sloth",
    frames: int | None = None,
    gif_fps: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
    row_label_width: int = 92,
    variants: Sequence[tuple[str, str]] = DEFAULT_VARIANTS,
    camera_ids: Sequence[int] = DEFAULT_CAMERA_IDS,
    sam31_mask_root: str | Path | None = None,
    variant_roots: Mapping[str, str | Path] | None = None,
    output_name: str | None = None,
    background_mode: str = "dim_full",
    color_overlap: bool = True,
) -> dict[str, Any]:
    root = Path(root).resolve()
    case_dir = (root / case_dir).resolve() if not Path(case_dir).is_absolute() else Path(case_dir).resolve()
    output_dir = (root / output_dir).resolve() if not Path(output_dir).is_absolute() else Path(output_dir).resolve()
    default_variant_roots = _variant_roots(output_dir, case_key=str(case_key))
    if variant_roots:
        default_variant_roots.update({str(key): Path(value).resolve() for key, value in variant_roots.items()})
    variant_roots = default_variant_roots
    sam31_root = Path(sam31_mask_root).resolve() if sam31_mask_root is not None else case_dir / "sam31_masks"
    first_tokens = sorted_frame_tokens(case_dir, camera_idx=int(camera_ids[0]), frames=frames)
    frame_count = len(first_tokens)

    gif_dir = output_dir / "gifs"
    first_dir = output_dir / "first_frames"
    gif_dir.mkdir(parents=True, exist_ok=True)
    first_dir.mkdir(parents=True, exist_ok=True)
    output_stem = str(output_name or f"{case_key}_mask_overlay_3x3_small_tiny_edgetam_compiled")
    gif_path = gif_dir / f"{output_stem}.gif"
    first_frame_path = first_dir / f"{output_stem}_first.png"

    per_frame: list[dict[str, Any]] = []
    aggregate: dict[str, dict[str, list[float]]] = {
        str(variant_key): {str(camera_idx): [] for camera_idx in camera_ids}
        for variant_key, _label in variants
    }
    with imageio.get_writer(str(gif_path), mode="I", fps=max(1, int(gif_fps)), loop=0) as writer:
        for frame_idx, frame_token in enumerate(first_tokens):
            image_rows: list[list[np.ndarray]] = []
            frame_stats: list[dict[str, Any]] = []
            for camera_idx in [int(item) for item in camera_ids]:
                color_path = case_dir / "color" / str(camera_idx) / f"{frame_token}.png"
                color = cv2.imread(str(color_path), cv2.IMREAD_COLOR)
                if color is None:
                    raise FileNotFoundError(f"Missing RGB frame: {color_path}")
                reference = load_union_mask(
                    mask_root=sam31_root,
                    case_dir=case_dir,
                    camera_idx=camera_idx,
                    frame_token=str(frame_token),
                    text_prompt=str(text_prompt),
                )
                row_tiles: list[np.ndarray] = []
                for variant_key, variant_label in variants:
                    candidate = load_union_mask(
                        mask_root=variant_roots[str(variant_key)],
                        case_dir=case_dir,
                        camera_idx=camera_idx,
                        frame_token=str(frame_token),
                        text_prompt=str(text_prompt),
                    )
                    tile, stats = build_overlay_tile(
                        color_bgr=color,
                        reference_mask=reference,
                        candidate_mask=candidate,
                        variant_label=str(variant_label),
                        tile_width=int(tile_width),
                        tile_height=int(tile_height),
                        background_mode=str(background_mode),
                        color_overlap=bool(color_overlap),
                    )
                    aggregate[str(variant_key)][str(camera_idx)].append(float(stats["iou"]))
                    frame_stats.append(
                        {
                            "frame_idx": int(frame_idx),
                            "frame_token": str(frame_token),
                            "camera_idx": int(camera_idx),
                            "variant_key": str(variant_key),
                            **stats,
                        }
                    )
                    row_tiles.append(tile)
                image_rows.append(row_tiles)
            board = compose_panel(
                title_lines=[
                    f"{case_label} | mask overlay vs SAM3.1 | frame {frame_idx + 1}/{frame_count}",
                    (
                        "black outside SAM3.1 union candidate | red=SAM3.1 only | cyan=candidate only"
                        if background_mode == "black_union_rgb" and not color_overlap
                        else "green=overlap | red=SAM3.1 only | cyan=candidate only"
                    ),
                ],
                row_headers=[f"cam{int(camera_idx)}" for camera_idx in camera_ids],
                column_headers=[label for _key, label in variants],
                image_rows=image_rows,
                row_label_width=int(row_label_width),
                expected_rows=len(tuple(camera_ids)),
            )
            if frame_idx == 0:
                write_image(first_frame_path, board)
            writer.append_data(cv2.cvtColor(board, cv2.COLOR_BGR2RGB))
            per_frame.append({"frame_idx": int(frame_idx), "frame_token": str(frame_token), "cells": frame_stats})
            if frame_idx == 0 or frame_idx + 1 == frame_count or (frame_idx + 1) % 10 == 0:
                print(f"[mask-overlay] rendered {case_key} frame {frame_idx + 1}/{frame_count}", flush=True)

    aggregate_summary: dict[str, dict[str, Any]] = {}
    for variant_key, by_camera in aggregate.items():
        aggregate_summary[variant_key] = {}
        all_values: list[float] = []
        for camera_idx, values in by_camera.items():
            all_values.extend(values)
            aggregate_summary[variant_key][camera_idx] = {
                "mean_iou": float(np.mean(values)) if values else 0.0,
                "min_iou": float(np.min(values)) if values else 0.0,
                "frame_count": int(len(values)),
            }
        aggregate_summary[variant_key]["all_cameras"] = {
            "mean_iou": float(np.mean(all_values)) if all_values else 0.0,
            "min_iou": float(np.min(all_values)) if all_values else 0.0,
            "sample_count": int(len(all_values)),
        }

    return {
        "case_key": str(case_key),
        "case_label": str(case_label),
        "case_dir": str(case_dir),
        "text_prompt": str(text_prompt),
        "frames": int(frame_count),
        "gif_path": str(gif_path),
        "first_frame_path": str(first_frame_path),
        "sam31_mask_root": str(sam31_root),
        "variant_roots": {key: str(path) for key, path in variant_roots.items()},
        "variants": [{"key": key, "label": label} for key, label in variants],
        "overlay_legend": {
            "background": (
                "black outside SAM3.1 union candidate, original RGB inside the union"
                if background_mode == "black_union_rgb"
                else "dim full RGB frame"
            ),
            "green": "candidate and SAM3.1 overlap" if color_overlap else "overlap keeps original RGB",
            "red": "SAM3.1 only",
            "cyan": "candidate only",
        },
        "background_mode": str(background_mode),
        "color_overlap": bool(color_overlap),
        "aggregate_iou": aggregate_summary,
        "frames_detail": per_frame,
    }


def write_overlay_report(markdown_path: str | Path, summary: Mapping[str, Any]) -> None:
    lines = [
        "# SAM2.1 / EdgeTAM Mask Overlay 3x3",
        "",
        "## Output",
        "",
        f"- GIF: `{summary.get('gif_path')}`",
        f"- first frame: `{summary.get('first_frame_path')}`",
        f"- case: `{summary.get('case_key')}`",
        f"- frames: `{summary.get('frames')}`",
        "",
        "## Overlay Legend",
        "",
        f"- background: {summary.get('overlay_legend', {}).get('background')}",
        f"- overlap: {summary.get('overlay_legend', {}).get('green')}",
        "- red: SAM3.1 only",
        "- cyan: candidate only",
        "",
        "## IoU Summary",
        "",
        "| variant | cam | mean IoU | min IoU |",
        "| --- | ---: | ---: | ---: |",
    ]
    for variant_key, by_camera in summary.get("aggregate_iou", {}).items():
        for camera_key, stats in by_camera.items():
            if camera_key == "all_cameras":
                continue
            lines.append(
                f"| {variant_key} | {camera_key} | "
                f"{float(stats['mean_iou']):.4f} | {float(stats['min_iou']):.4f} |"
            )
        all_stats = by_camera.get("all_cameras", {})
        if all_stats:
            lines.append(
                f"| **{variant_key}** | **all** | "
                f"**{float(all_stats['mean_iou']):.4f}** | **{float(all_stats['min_iou']):.4f}** |"
            )
    timing_records = list(summary.get("sam21_timing_records", [])) + list(summary.get("edgetam_timing_records", []))
    if timing_records:
        lines.extend(
            [
                "",
                "## Worker Timing",
                "",
                "| model | cam | ms/frame | FPS | frames |",
                "| --- | ---: | ---: | ---: | ---: |",
            ]
        )
        for record in timing_records:
            lines.append(
                f"| {record.get('checkpoint_key')} | {int(record['camera_idx'])} | "
                f"{float(record['inference_ms_per_frame']):.2f} | "
                f"{float(record['fps']):.2f} | {int(record['timed_frame_count'])} |"
            )
    Path(markdown_path).parent.mkdir(parents=True, exist_ok=True)
    Path(markdown_path).write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_dynamics_round1_overlay_workflow(
    *,
    root: str | Path,
    output_dir: str | Path = DEFAULT_OUTPUT_DIR,
    case_dir: str | Path = DEFAULT_CASE_DIR,
    frames: int | None = None,
    gif_fps: int = 6,
    tile_width: int = 320,
    tile_height: int = 180,
    row_label_width: int = 92,
    doc_md: str | Path = DEFAULT_DOC_MD,
    doc_json: str | Path = DEFAULT_DOC_JSON,
) -> dict[str, Any]:
    root = Path(root).resolve()
    summary = render_mask_overlay_3x3_gif(
        root=root,
        case_dir=case_dir,
        output_dir=output_dir,
        frames=frames,
        gif_fps=int(gif_fps),
        tile_width=int(tile_width),
        tile_height=int(tile_height),
        row_label_width=int(row_label_width),
    )
    json_path = root / doc_json if not Path(doc_json).is_absolute() else Path(doc_json)
    md_path = root / doc_md if not Path(doc_md).is_absolute() else Path(doc_md)
    write_json(json_path, summary)
    write_overlay_report(md_path, summary)
    summary["docs"] = {"benchmark_md": str(md_path), "benchmark_json": str(json_path)}
    return summary
