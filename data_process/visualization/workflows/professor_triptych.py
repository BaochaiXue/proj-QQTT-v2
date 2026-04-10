from __future__ import annotations

from pathlib import Path
from typing import Any

from ..professor_triptych import run_professor_triptych_workflow as _run_professor_triptych_workflow


def run_professor_triptych_workflow(*, aligned_root: Path, output_dir: Path, **kwargs: Any) -> dict[str, Any]:
    return _run_professor_triptych_workflow(aligned_root=aligned_root, output_dir=output_dir, **kwargs)
