from __future__ import annotations

import argparse
import json
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import open3d as o3d


DEFAULT_OUTPUTS_ROOT = Path("outputs_v6_1")
DEFAULT_CASE_NAME = "shape_prior_frame0"

COLOR_OBJECT = (0.95, 0.95, 0.95)
COLOR_CONTROLLER = (1.0, 0.18, 0.12)
COLOR_SURFACE = (0.0, 0.85, 0.95)
COLOR_INTERIOR = (0.1, 0.28, 1.0)


@dataclass(frozen=True)
class MaskedPointStats:
    name: str
    count: int
    bbox_min: np.ndarray
    bbox_max: np.ndarray
    mean: np.ndarray
    std: np.ndarray


@dataclass(frozen=True)
class PublishedDataStats:
    object_shape: tuple[int, ...]
    controller_shape: tuple[int, ...]
    surface_count: int
    interior_count: int
    semantic_label_counts: dict[int, int]
    object_visibility_ratio: float
    object_motion_valid_ratio: float
    controller_proxied_ratio: float
    track_process_status: str


@dataclass(frozen=True)
class ShapePriorInspection:
    outputs_root: Path
    case_dir: Path
    warmup_final_data_path: Path
    published_final_data_path: Path
    metadata: dict[str, Any]
    mask_info: dict[str, str]
    depth_valid_stats: MaskedPointStats
    object_mask_stats: MaskedPointStats
    controller_mask_stats: MaskedPointStats
    mask_overlap_pixels: int
    warmup_object_points: np.ndarray
    warmup_controller_points: np.ndarray
    surface_points: np.ndarray
    interior_points: np.ndarray
    published_stats: PublishedDataStats | None


def build_parser() -> argparse.ArgumentParser:
    """Build the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Inspect Demo v6.1 shape-prior outputs and open an Open3D scene."
        )
    )
    parser.add_argument(
        "--outputs-root",
        type=Path,
        default=DEFAULT_OUTPUTS_ROOT,
        help="Demo v6.1 outputs root. Defaults to ./outputs_v6_1.",
    )
    parser.add_argument(
        "--case-name",
        type=str,
        default=DEFAULT_CASE_NAME,
        help="Shape-prior warmup case name under shape_prior_case/.",
    )
    parser.add_argument(
        "--no-view",
        action="store_true",
        help="Print stats only; do not open the Open3D window.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=4.0,
        help="Open3D point size for object/controller/supplement points.",
    )
    parser.add_argument(
        "--window-width",
        type=int,
        default=1400,
        help="Open3D window width.",
    )
    parser.add_argument(
        "--window-height",
        type=int,
        default=950,
        help="Open3D window height.",
    )
    parser.add_argument(
        "--write-report",
        type=Path,
        default=None,
        help="Optional Markdown report path.",
    )
    return parser


def _require_file(path: Path) -> Path:
    """Return validated file."""
    if not path.is_file():
        raise FileNotFoundError(f"required file not found: {path}")
    return path


def _load_json(path: Path) -> dict[str, Any]:
    """Load JSON."""
    with _require_file(path).open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object at {path}")
    return value


def _load_pickle(path: Path) -> Any:
    """Load pickle."""
    with _require_file(path).open("rb") as handle:
        return pickle.load(handle)


def _as_points(value: Any, *, name: str) -> np.ndarray:
    """Coerce the input into points."""
    points = np.asarray(value, dtype=np.float64).reshape(-1, 3)
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite points")
    return np.ascontiguousarray(points)


def _frame_points(value: Any, *, name: str) -> np.ndarray:
    """Return the frame points."""
    points = np.asarray(value, dtype=np.float64)
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"{name} must have shape (frames, points, 3)")
    if points.shape[0] < 1:
        raise ValueError(f"{name} has no frames")
    if not np.isfinite(points).all():
        raise ValueError(f"{name} contains non-finite points")
    return np.ascontiguousarray(points[0])


def _point_cloud(points: np.ndarray, color: tuple[float, float, float]):
    """Return the point cloud."""
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    cloud.paint_uniform_color(color)
    return cloud


def _stats_for_mask(
    name: str,
    points_world: np.ndarray,
    mask: np.ndarray,
) -> MaskedPointStats:
    """Return the stats for mask."""
    selected = points_world[np.asarray(mask, dtype=bool)]
    selected = _as_points(selected, name=name)
    if selected.shape[0] == 0:
        raise ValueError(f"{name} mask selected no points")
    return MaskedPointStats(
        name=name,
        count=int(selected.shape[0]),
        bbox_min=selected.min(axis=0),
        bbox_max=selected.max(axis=0),
        mean=selected.mean(axis=0),
        std=selected.std(axis=0),
    )


def _ratio(mask: np.ndarray) -> float:
    """Return the ratio."""
    values = np.asarray(mask, dtype=bool)
    if values.size == 0:
        return 0.0
    return float(np.count_nonzero(values)) / float(values.size)


def _semantic_label_counts(labels: np.ndarray) -> dict[int, int]:
    """Return the semantic label counts."""
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    unique, counts = np.unique(labels, return_counts=True)
    return {
        int(label): int(count)
        for label, count in zip(unique.tolist(), counts.tolist())
    }


def _published_stats(path: Path) -> PublishedDataStats | None:
    """Return the published stats."""
    if not path.is_file():
        return None
    data = dict(_load_pickle(path))
    object_points = np.asarray(data["object_points"], dtype=np.float64)
    controller_points = np.asarray(data["controller_points"], dtype=np.float64)
    surface_points = _as_points(data["surface_points"], name="published surface")
    interior_points = _as_points(data["interior_points"], name="published interior")
    return PublishedDataStats(
        object_shape=tuple(int(dim) for dim in object_points.shape),
        controller_shape=tuple(int(dim) for dim in controller_points.shape),
        surface_count=int(surface_points.shape[0]),
        interior_count=int(interior_points.shape[0]),
        semantic_label_counts=_semantic_label_counts(data["query_semantic_labels"]),
        object_visibility_ratio=_ratio(data["object_visibilities"]),
        object_motion_valid_ratio=_ratio(data["object_motions_valid"]),
        controller_proxied_ratio=_ratio(data["controller_proxied"]),
        track_process_status=str(data.get("track_process_status", "")),
    )


def load_inspection(outputs_root: Path, case_name: str) -> ShapePriorInspection:
    """Load inspection."""
    outputs_root = Path(outputs_root)
    case_dir = outputs_root / "shape_prior_case" / str(case_name)
    metadata = _load_json(case_dir / "metadata.json")
    mask_info = {
        str(key): str(value)
        for key, value in _load_json(case_dir / "mask" / "mask_info_0.json").items()
    }

    pcd_path = _require_file(case_dir / "pcd" / "0.npz")
    pcd_data = np.load(pcd_path)
    points_world = np.asarray(pcd_data["points"], dtype=np.float64)
    depth_valid = np.asarray(pcd_data["masks"], dtype=bool)
    if points_world.ndim != 4 or points_world.shape[0] != 1:
        raise ValueError(f"expected one-camera pcd grid at {pcd_path}")
    if depth_valid.shape != points_world.shape[:3]:
        raise ValueError("pcd masks must match pcd points camera/image shape")
    points_world = points_world[0]
    depth_valid = depth_valid[0]

    processed_masks = _load_pickle(case_dir / "mask" / "processed_masks.pkl")
    masks = processed_masks[0][0]
    object_mask = np.asarray(masks["object"], dtype=bool)
    controller_mask = np.asarray(masks["controller"], dtype=bool)
    if object_mask.shape != points_world.shape[:2]:
        raise ValueError("object mask shape does not match pcd image shape")
    if controller_mask.shape != points_world.shape[:2]:
        raise ValueError("controller mask shape does not match pcd image shape")

    final_data_path = _require_file(case_dir / "final_data.pkl")
    final_data = dict(_load_pickle(final_data_path))
    warmup_object_points = _frame_points(
        final_data["object_points"],
        name="warmup object_points",
    )
    warmup_controller_points = _frame_points(
        final_data["controller_points"],
        name="warmup controller_points",
    )
    surface_points = _as_points(final_data["surface_points"], name="surface_points")
    interior_points = _as_points(final_data["interior_points"], name="interior_points")

    return ShapePriorInspection(
        outputs_root=outputs_root,
        case_dir=case_dir,
        warmup_final_data_path=final_data_path,
        published_final_data_path=outputs_root / "data" / "final_data.pkl",
        metadata=metadata,
        mask_info=mask_info,
        depth_valid_stats=_stats_for_mask(
            "depth-valid frame",
            points_world,
            depth_valid,
        ),
        object_mask_stats=_stats_for_mask(
            "processed object mask",
            points_world,
            object_mask,
        ),
        controller_mask_stats=_stats_for_mask(
            "processed controller mask",
            points_world,
            controller_mask,
        ),
        mask_overlap_pixels=int(np.count_nonzero(object_mask & controller_mask)),
        warmup_object_points=warmup_object_points,
        warmup_controller_points=warmup_controller_points,
        surface_points=surface_points,
        interior_points=interior_points,
        published_stats=_published_stats(outputs_root / "data" / "final_data.pkl"),
    )


def _fmt_vec(values: np.ndarray) -> str:
    """Format vec for display."""
    return "[" + ", ".join(f"{float(value):.4f}" for value in values) + "]"


def _fmt_pct(value: float) -> str:
    """Format pct for display."""
    return f"{100.0 * float(value):.2f}%"


def _stats_lines(stats: MaskedPointStats) -> list[str]:
    """Return the stats lines."""
    return [
        f"{stats.name}:",
        f"  count: {stats.count:,}",
        f"  bbox min: {_fmt_vec(stats.bbox_min)}",
        f"  bbox max: {_fmt_vec(stats.bbox_max)}",
        f"  mean: {_fmt_vec(stats.mean)}",
        f"  std: {_fmt_vec(stats.std)}",
    ]


def build_report(inspection: ShapePriorInspection) -> str:
    """Build report."""
    metadata = inspection.metadata
    lines = [
        "# Demo v6.1 Shape Prior Outputs Inspection",
        "",
        "## Paths",
        "",
        f"- case: `{inspection.case_dir}`",
        f"- warmup final data: `{inspection.warmup_final_data_path}`",
        f"- published final data: `{inspection.published_final_data_path}`",
        "",
        "## Warmup Case",
        "",
        f"- input source: `{metadata.get('input_source')}`",
        f"- depth backend: `{metadata.get('depth_backend')}`",
        f"- depth source internal: `{metadata.get('depth_source_internal')}`",
        f"- object label: `{inspection.mask_info.get('0')}`",
        f"- controller label: `{inspection.mask_info.get('1')}`",
        "",
        "## Masked PCD Stats",
        "",
        *_stats_lines(inspection.object_mask_stats),
        "",
        *_stats_lines(inspection.controller_mask_stats),
        "",
        f"object/controller overlap pixels: {inspection.mask_overlap_pixels:,}",
        "",
        "The full depth-valid frame includes background and unrelated pixels:",
        "",
        *_stats_lines(inspection.depth_valid_stats),
        "",
        "## Shape Prior Supplement",
        "",
        f"- warmup object points: {inspection.warmup_object_points.shape[0]:,}",
        f"- warmup controller points: {inspection.warmup_controller_points.shape[0]:,}",
        f"- surface supplement points: {inspection.surface_points.shape[0]:,}",
        f"- interior supplement points: {inspection.interior_points.shape[0]:,}",
        (
            "- total supplement points: "
            f"{inspection.surface_points.shape[0] + inspection.interior_points.shape[0]:,}"
        ),
        "",
        "The supplement is stored separately as `surface_points` and",
        "`interior_points`; it is not appended to `object_points`.",
    ]
    if inspection.published_stats is not None:
        published = inspection.published_stats
        lines.extend(
            [
                "",
                "## Published Final Data",
                "",
                f"- object points shape: `{published.object_shape}`",
                f"- controller points shape: `{published.controller_shape}`",
                f"- surface points: {published.surface_count:,}",
                f"- interior points: {published.interior_count:,}",
                f"- semantic label counts: `{published.semantic_label_counts}`",
                f"- object visibility ratio: {_fmt_pct(published.object_visibility_ratio)}",
                (
                    "- object motion-valid ratio: "
                    f"{_fmt_pct(published.object_motion_valid_ratio)}"
                ),
                f"- controller proxied ratio: {_fmt_pct(published.controller_proxied_ratio)}",
                f"- track process status: `{published.track_process_status}`",
            ]
        )
    return "\n".join(lines) + "\n"


def print_report(inspection: ShapePriorInspection) -> None:
    """Print report."""
    print(build_report(inspection), end="")


def open_viewer(
    inspection: ShapePriorInspection,
    *,
    point_size: float,
    window_width: int,
    window_height: int,
) -> None:
    """Open viewer."""
    geometries = [
        _point_cloud(inspection.warmup_object_points, COLOR_OBJECT),
        _point_cloud(inspection.warmup_controller_points, COLOR_CONTROLLER),
        _point_cloud(inspection.surface_points, COLOR_SURFACE),
        _point_cloud(inspection.interior_points, COLOR_INTERIOR),
    ]

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(
        window_name="Demo v6.1 shape prior PCD outputs",
        width=int(window_width),
        height=int(window_height),
    )
    for geometry in geometries:
        viewer.add_geometry(geometry)
    render_options = viewer.get_render_option()
    render_options.point_size = float(point_size)
    viewer.run()
    viewer.destroy_window()


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    inspection = load_inspection(args.outputs_root, args.case_name)
    report = build_report(inspection)
    print(report, end="")
    if args.write_report is not None:
        args.write_report.parent.mkdir(parents=True, exist_ok=True)
        args.write_report.write_text(report, encoding="utf-8")
        print(f"\nwrote report: {args.write_report}")
    if not bool(args.no_view):
        open_viewer(
            inspection,
            point_size=float(args.point_size),
            window_width=int(args.window_width),
            window_height=int(args.window_height),
        )


if __name__ == "__main__":
    main()
