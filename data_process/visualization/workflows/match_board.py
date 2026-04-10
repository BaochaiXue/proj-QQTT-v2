from __future__ import annotations

from pathlib import Path
from typing import Any

from ..match_board import run_match_board_workflow as _run_match_board_workflow


def run_match_board_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_match_board_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
