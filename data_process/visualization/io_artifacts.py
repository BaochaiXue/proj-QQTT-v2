from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .types import DebugArtifactSet, ProductArtifactSet


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_image(path: Path, image: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), np.asarray(image, dtype=np.uint8))


def write_ply_ascii(path: Path, points: np.ndarray, colors: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        handle.write("ply\nformat ascii 1.0\n")
        handle.write(f"element vertex {len(points)}\n")
        handle.write("property float x\nproperty float y\nproperty float z\n")
        handle.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        handle.write("end_header\n")
        for point, color in zip(points, colors, strict=False):
            handle.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[2])} {int(color[1])} {int(color[0])}\n"
            )


def write_video(video_path: Path, frame_paths: list[Path], fps: int) -> None:
    if not frame_paths:
        return
    first = cv2.imread(str(frame_paths[0]), cv2.IMREAD_COLOR)
    if first is None:
        return
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        float(fps),
        (first.shape[1], first.shape[0]),
    )
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        writer.write(image)
    writer.release()


def write_gif(gif_path: Path, frame_paths: list[Path], fps: int) -> None:
    if not frame_paths:
        return
    frames: list[Image.Image] = []
    for path in frame_paths:
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        frames.append(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
    if not frames:
        return
    duration_ms = max(20, int(round(1000.0 / max(1, int(fps)))))
    frames[0].save(
        str(gif_path),
        save_all=True,
        append_images=frames[1:],
        duration=duration_ms,
        loop=0,
        optimize=False,
    )


def build_artifact_sets(
    *,
    output_dir: Path,
    product_paths: dict[str, str | Path | None],
    summary_paths: dict[str, str | Path | None] | None = None,
    debug_enabled: bool = False,
    debug_dir: Path | None = None,
    debug_paths: dict[str, str | Path | None] | None = None,
) -> tuple[ProductArtifactSet, DebugArtifactSet]:
    product_set = ProductArtifactSet(
        output_dir=Path(output_dir).resolve(),
        top_level_paths={
            str(key): str(Path(value).resolve())
            for key, value in product_paths.items()
            if value is not None
        },
        summary_paths={
            str(key): str(Path(value).resolve())
            for key, value in (summary_paths or {}).items()
            if value is not None
        },
    )
    debug_set = DebugArtifactSet(
        enabled=bool(debug_enabled),
        debug_dir=None if debug_dir is None else Path(debug_dir).resolve(),
        paths={
            str(key): str(Path(value).resolve())
            for key, value in (debug_paths or {}).items()
            if value is not None
        },
    )
    return product_set, debug_set
