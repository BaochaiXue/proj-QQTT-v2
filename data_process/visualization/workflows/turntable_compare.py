from __future__ import annotations

from pathlib import Path
from typing import Any

from ..turntable_compare import run_turntable_compare_workflow as _run_turntable_compare_workflow


def run_turntable_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_turntable_compare_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
