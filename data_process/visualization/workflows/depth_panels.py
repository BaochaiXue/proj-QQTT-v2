from __future__ import annotations

from pathlib import Path
from typing import Any

from ..panel_compare import run_depth_panel_workflow as _run_depth_panel_workflow


def run_depth_panels_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_depth_panel_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
