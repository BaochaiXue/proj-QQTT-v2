"""Demo v6.1 fixed output layout under ``--base-path``."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

from demo_v6_2.main_config import (
    CAPTURE_DIR_NAME,
    DATA_DIR_NAME,
    ONLINE_DATA_DIR_NAME,
    RUN_SUMMARY_NAME,
    SHAPE_PRIOR_CASE_DIR_NAME,
    SHAPE_PRIOR_DIR_NAME,
)


# ---------------------------------------------------------------------------
# Fixed output layout under --base-path
# ---------------------------------------------------------------------------


def resolve_online_dir(args: argparse.Namespace) -> Path:
    """Return the fixed online_data directory."""
    return Path(args.base_path) / ONLINE_DATA_DIR_NAME


def resolve_static_data_dir(args: argparse.Namespace) -> Path:
    """Return the fixed aggregate data directory."""
    return Path(args.base_path) / DATA_DIR_NAME


def resolve_static_data_path(args: argparse.Namespace) -> Path:
    """Return the aggregate final_data.pkl path."""
    return resolve_static_data_dir(args) / "final_data.pkl"


def resolve_shape_prior_case_root(args: argparse.Namespace) -> Path:
    """Return the fixed shape-prior case root."""
    return Path(args.base_path) / SHAPE_PRIOR_CASE_DIR_NAME


def resolve_shape_prior_points_npz(args: argparse.Namespace) -> Path:
    """Return the fixed shape-prior points export path."""
    return Path(args.base_path) / SHAPE_PRIOR_DIR_NAME / "points.npz"


def resolve_run_summary_path(base_path: str | Path) -> Path:
    """Return the fixed run summary path."""
    return Path(base_path) / RUN_SUMMARY_NAME


def _remove_generated_path(path: Path) -> bool:
    """Delete a generated file or directory; return True when it existed."""
    if path.is_dir():
        shutil.rmtree(path)
        return True
    if path.exists():
        path.unlink()
        return True
    return False


def prepare_realtime_output_for_new_run(
    base_path: str | Path,
    *,
    legacy_case_prefix: str,
) -> dict[str, object]:
    """Remove stale generated outputs before writing fixed Demo v6.1 paths."""
    base = Path(base_path)
    cleanup_paths = {
        "capture": base / CAPTURE_DIR_NAME,
        "shape_prior_case": base / SHAPE_PRIOR_CASE_DIR_NAME,
        "shape_prior": base / SHAPE_PRIOR_DIR_NAME,
        "data": base / DATA_DIR_NAME,
        "online_data": base / ONLINE_DATA_DIR_NAME,
        "run_summary": resolve_run_summary_path(base),
        "legacy_chunks_manifest": base / f"{legacy_case_prefix}_chunks_manifest.json",
    }
    return {
        f"removed_{name}": bool(_remove_generated_path(path))
        for name, path in cleanup_paths.items()
    }
