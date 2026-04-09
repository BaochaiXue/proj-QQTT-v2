from __future__ import annotations

from pathlib import Path
from typing import Any

from ..reprojection_compare import run_reprojection_compare_workflow as _run_reprojection_compare_workflow


def run_reprojection_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_reprojection_compare_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
