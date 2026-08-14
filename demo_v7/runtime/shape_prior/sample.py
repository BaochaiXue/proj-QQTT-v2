"""Sample RAW shape-prior structure candidates from the aligned mesh.

This stage no longer makes any final downsampling decision: it samples the
origin candidate pools (1024 surface + 10000 interior points, matching
data_process_origin/data_process_sample.py:70-74) from the aligned prior mesh
and writes them to ``shape/candidates.npz``. The FINAL origin-parity voxel
selection — final tracked object first, then surface, then interior, one
shared occupied set — happens exactly once, at chunk-0 identity freeze
(``demo_v7.runtime.tracking.sample_origin_unified_structure``).
"""

from __future__ import annotations

from argparse import ArgumentParser
from pathlib import Path
import time

_MODULE_IMPORT_STARTED_S = time.perf_counter()

import numpy as np  # noqa: E402
import trimesh  # noqa: E402

from demo_v7.runtime.shape_prior.timing import (  # noqa: E402
    StageProfileRun,
    elapsed_ms,
)
from demo_v7.runtime.utils.mesh_utils import as_mesh  # noqa: E402

_MODULE_IMPORT_MS = elapsed_ms(_MODULE_IMPORT_STARTED_S)


DEFAULT_SURFACE_POINTS = 1024
INTERIOR_CANDIDATE_POINTS = 10000
CANDIDATES_FILENAME = "candidates.npz"
CANDIDATES_SCHEMA_VERSION = "shape_prior_candidates_v1"


def build_parser() -> ArgumentParser:
    """Build the command-line argument parser."""
    parser = ArgumentParser()
    parser.add_argument("--base_path", type=Path, required=True)
    parser.add_argument("--case_name", type=str, required=True)
    parser.add_argument(
        "--num_surface_points", type=int, default=DEFAULT_SURFACE_POINTS
    )
    parser.add_argument(
        "--profile-json",
        type=Path,
        default=None,
        help="Optional JSON path for detailed sample-stage timing.",
    )
    parser.add_argument(
        "--wait-signal",
        dest="wait_signal",
        action="store_true",
        help="Pay module imports up front, then block on stdin for GO.",
    )
    return parser


def sample_shape_prior_candidates(
    mesh: trimesh.Trimesh,
    *,
    surface_count: int,
    interior_count: int = INTERIOR_CANDIDATE_POINTS,
    timing_ms: dict[str, float] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample the origin raw candidate pools from the aligned mesh."""
    timings = timing_ms if timing_ms is not None else {}
    surface_sample_started_s = time.perf_counter()
    surface_points, _ = trimesh.sample.sample_surface(mesh, int(surface_count))
    timings["surface_sample_ms"] = elapsed_ms(surface_sample_started_s)
    volume_sample_started_s = time.perf_counter()
    interior_points = trimesh.sample.volume_mesh(mesh, int(interior_count))
    timings["volume_sample_ms"] = elapsed_ms(volume_sample_started_s)
    return (
        np.ascontiguousarray(np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)),
        np.ascontiguousarray(np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)),
    )


def write_shape_prior_candidates(
    path: Path,
    *,
    raw_surface_points: np.ndarray,
    raw_interior_points: np.ndarray,
) -> None:
    """Atomically write the raw candidate pools."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp.npz")
    np.savez(
        tmp_path,
        raw_surface_points=np.ascontiguousarray(
            np.asarray(raw_surface_points, dtype=np.float64).reshape(-1, 3)
        ),
        raw_interior_points=np.ascontiguousarray(
            np.asarray(raw_interior_points, dtype=np.float64).reshape(-1, 3)
        ),
        schema_version=np.asarray(CANDIDATES_SCHEMA_VERSION),
    )
    tmp_path.replace(path)


def load_shape_prior_candidates(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the raw candidate pools."""
    payload = np.load(Path(path), allow_pickle=False)
    schema = str(payload["schema_version"])
    if schema != CANDIDATES_SCHEMA_VERSION:
        raise ValueError(
            f"unsupported shape-prior candidates schema {schema!r} in {path}"
        )
    surface = np.asarray(payload["raw_surface_points"], dtype=np.float64).reshape(-1, 3)
    interior = np.asarray(payload["raw_interior_points"], dtype=np.float64).reshape(
        -1, 3
    )
    if not (np.isfinite(surface).all() and np.isfinite(interior).all()):
        raise ValueError(f"shape-prior candidates contain non-finite points: {path}")
    return np.ascontiguousarray(surface), np.ascontiguousarray(interior)


def main(argv: list[str] | None = None) -> None:
    """Run the command-line entry point."""
    args = build_parser().parse_args(argv)
    run = StageProfileRun(
        stage="sample",
        profile_json=args.profile_json,
        wait_signal=args.wait_signal,
        timing_ms={
            "module_import_ms": _MODULE_IMPORT_MS,
            "mesh_load_ms": 0.0,
            "surface_sample_ms": 0.0,
            "volume_sample_ms": 0.0,
            "output_write_ms": 0.0,
            "go_wait_ms": 0.0,
            "total_ms": 0.0,
            "process_lifetime_ms": 0.0,
        },
        active_fields=(
            "module_import_ms",
            "mesh_load_ms",
            "surface_sample_ms",
            "volume_sample_ms",
            "output_write_ms",
        ),
        process_started_s=_MODULE_IMPORT_STARTED_S,
    )
    timing_ms = run.timing_ms
    if args.wait_signal:
        # No models to load: only numpy/trimesh imports (as_mesh comes from
        # the light mesh_utils), already paid above.
        run.write_waiting()
        if not run.wait_for_go():
            return
    case_dir = Path(args.base_path) / str(args.case_name)
    mesh_path = case_dir / "shape" / "matching" / "final_mesh.glb"
    if not mesh_path.is_file():
        raise FileNotFoundError(f"aligned shape-prior mesh not found: {mesh_path}")
    mesh_load_started_s = time.perf_counter()
    mesh = as_mesh(trimesh.load(mesh_path, force="mesh"))
    timing_ms["mesh_load_ms"] = elapsed_ms(mesh_load_started_s)
    raw_surface, raw_interior = sample_shape_prior_candidates(
        mesh,
        surface_count=int(args.num_surface_points),
        timing_ms=timing_ms,
    )
    output_write_started_s = time.perf_counter()
    write_shape_prior_candidates(
        case_dir / "shape" / CANDIDATES_FILENAME,
        raw_surface_points=raw_surface,
        raw_interior_points=raw_interior,
    )
    timing_ms["output_write_ms"] = elapsed_ms(output_write_started_s)
    run.write_completed()


if __name__ == "__main__":
    main()
