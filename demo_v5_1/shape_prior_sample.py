"""Sample Demo v5.1 shape-prior points into final_data.pkl."""
from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import pickle

import numpy as np
import trimesh

from demo_v5_1.utils.align_util import as_mesh


DEFAULT_SURFACE_POINTS = 1024
INTERIOR_CANDIDATE_POINTS = 10000
DEFAULT_VOLUME_SAMPLE_SIZE_M = 0.005


def build_parser() -> ArgumentParser:
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument("--shape_prior", action="store_true", default=False)
    parser.add_argument("--num_surface_points", type=int, default=DEFAULT_SURFACE_POINTS)
    parser.add_argument("--volume_sample_size", type=float, default=DEFAULT_VOLUME_SAMPLE_SIZE_M)
    return parser


def _grid_key(
    point: np.ndarray,
    *,
    min_bound: np.ndarray,
    voxel_size: float,
) -> tuple[int, int, int]:
    """Voxel index of ``point`` on the grid anchored at ``min_bound``."""
    return tuple(np.floor((point - min_bound) / float(voxel_size)).astype(int))


def _append_new_voxels(
    points: np.ndarray,
    *,
    min_bound: np.ndarray,
    voxel_size: float,
    occupied: set[tuple[int, int, int]],
) -> np.ndarray:
    """Keep only points that land in a not-yet-occupied voxel (mutates the set)."""
    kept: list[np.ndarray] = []
    for point in points:
        key = _grid_key(point, min_bound=min_bound, voxel_size=voxel_size)
        if key in occupied:
            continue
        occupied.add(key)
        kept.append(point)
    if not kept:
        return np.empty((0, 3), dtype=np.float64)
    return np.ascontiguousarray(np.asarray(kept, dtype=np.float64).reshape(-1, 3))


def process_shape_prior_case(
    base_path: Path,
    case_name: str,
    *,
    use_shape_prior: bool,
    surface_count: int,
    volume_sample_size: float,
) -> dict[str, np.ndarray]:
    """Rewrite of data_process_origin/data_process_sample.py process_unique_points
    without the turntable-video rendering."""
    case_dir = Path(base_path) / str(case_name)
    with (case_dir / "track_process_data.pkl").open("rb") as handle:
        track_data = dict(pickle.load(handle))

    object_points = np.asarray(track_data["object_points"], dtype=np.float64)
    object_colors = np.asarray(track_data["object_colors"], dtype=np.float64)
    object_visibilities = np.asarray(track_data["object_visibilities"], dtype=bool)
    object_motions_valid = np.asarray(track_data["object_motions_valid"], dtype=bool)

    # Drop duplicate frame-0 points. np.unique returns first-occurrence
    # indices ordered by point value; sorting restores capture order.
    unique_idx = np.unique(object_points[0], axis=0, return_index=True)[1]
    unique_idx = np.sort(unique_idx)
    object_points = np.ascontiguousarray(object_points[:, unique_idx, :])
    object_colors = np.ascontiguousarray(object_colors[:, unique_idx, :])
    object_visibilities = np.ascontiguousarray(object_visibilities[:, unique_idx])
    object_motions_valid = np.ascontiguousarray(object_motions_valid[:, unique_idx])

    # Above-table clamp: the table plane is z == 0 and above-table is
    # negative z, so any z > 0 point is pushed back onto the table.
    object_points[object_points[..., 2] > 0, 2] = 0

    if use_shape_prior:
        # Sample the aligned prior mesh: ``surface_count`` surface points plus
        # a fixed pool of interior candidates (origin counts: 1024 / 10000).
        mesh_path = case_dir / "shape" / "matching" / "final_mesh.glb"
        if not mesh_path.is_file():
            raise FileNotFoundError(f"aligned shape-prior mesh not found: {mesh_path}")
        mesh = as_mesh(trimesh.load(mesh_path, force="mesh"))
        surface_points, _ = trimesh.sample.sample_surface(mesh, int(surface_count))
        interior_points = trimesh.sample.volume_mesh(mesh, INTERIOR_CANDIDATE_POINTS)
        surface_points = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
        interior_points = np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)
        all_points = np.concatenate([surface_points, interior_points, object_points[0]], axis=0)
    else:
        surface_points = np.empty((0, 3), dtype=np.float64)
        interior_points = np.empty((0, 3), dtype=np.float64)
        all_points = object_points[0]

    # Voxel-dedup with the origin priority: observed object points claim
    # voxels first, then surface samples, then interior samples.
    min_bound = np.min(all_points, axis=0)
    occupied: set[tuple[int, int, int]] = set()
    object_indices: list[int] = []
    for index, point in enumerate(object_points[0]):
        key = _grid_key(point, min_bound=min_bound, voxel_size=float(volume_sample_size))
        if key in occupied:
            continue
        occupied.add(key)
        object_indices.append(index)

    if use_shape_prior:
        surface_points = _append_new_voxels(
            surface_points,
            min_bound=min_bound,
            voxel_size=float(volume_sample_size),
            occupied=occupied,
        )
        interior_points = _append_new_voxels(
            interior_points,
            min_bound=min_bound,
            voxel_size=float(volume_sample_size),
            occupied=occupied,
        )

    final_data = dict(track_data)
    final_data["object_points"] = np.ascontiguousarray(object_points[:, object_indices, :])
    final_data["object_colors"] = np.ascontiguousarray(object_colors[:, object_indices, :])
    final_data["object_visibilities"] = np.ascontiguousarray(
        object_visibilities[:, object_indices]
    )
    final_data["object_motions_valid"] = np.ascontiguousarray(
        object_motions_valid[:, object_indices]
    )
    final_data["surface_points"] = np.ascontiguousarray(surface_points.reshape(-1, 3))
    final_data["interior_points"] = np.ascontiguousarray(interior_points.reshape(-1, 3))
    return final_data


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    final_data = process_shape_prior_case(
        args.base_path,
        args.case_name,
        use_shape_prior=bool(args.shape_prior),
        surface_count=int(args.num_surface_points),
        volume_sample_size=float(args.volume_sample_size),
    )
    case_dir = Path(args.base_path) / str(args.case_name)
    with (case_dir / "final_data.pkl").open("wb") as handle:
        pickle.dump(final_data, handle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    main()
