from __future__ import annotations

from pathlib import Path
from typing import Any

from ..triplet_video_compare import run_triplet_video_compare_workflow as _run_triplet_video_compare_workflow


def run_triplet_video_compare_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_triplet_video_compare_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
