"""Shape-prior generation backend vocabulary (import-light, stdlib only).

Shared by the GUI source-select dialog, the orchestrator session, and the
camera service; the heavy client subclass lives in
``demo_v7.service.shape_prior_backends`` (imports the v6.2 warmup stack).

Backends:

- ``sam3d``: the unchanged v6.2 generate stage (SAM3D subprocess).
- ``trellis2``: microsoft/TRELLIS.2 via the ``trellis2`` conda env
  (``demo_v7/service/trellis2_generate.py``); align/sample stay v6.2.
- ``none``: no shape prior at all — the v6.2 ``--no-shape-prior-warmup``
  skip path (observed-only tracking structure, ASAP off, PhysTwin off).
"""

from __future__ import annotations

import os
from pathlib import Path

BACKEND_SAM3D = "sam3d"
BACKEND_TRELLIS2 = "trellis2"
BACKEND_NONE = "none"
SHAPE_PRIOR_BACKENDS: tuple[str, ...] = (
    BACKEND_SAM3D,
    BACKEND_TRELLIS2,
    BACKEND_NONE,
)
DEFAULT_SHAPE_PRIOR_BACKEND = BACKEND_SAM3D

# Machine-local TRELLIS.2 install (memory: trellis2-integration); the env
# python MUST be an absolute path — `python` after conda activate can be
# shadowed by the project .venv.
TRELLIS2_PYTHON = Path(
    os.environ.get(
        "DEMO_V7_TRELLIS2_PYTHON",
        "/home/xinjie/miniforge3/envs/trellis2/bin/python",
    )
)
TRELLIS2_REPO = Path(
    os.environ.get("DEMO_V7_TRELLIS2_REPO", "/home/xinjie/TRELLIS.2")
)


def normalize_backend(value: str | None) -> str:
    """Return a validated backend id (None -> the sam3d default)."""
    backend = str(value).strip().lower() if value is not None else ""
    if not backend:
        return DEFAULT_SHAPE_PRIOR_BACKEND
    if backend not in SHAPE_PRIOR_BACKENDS:
        raise ValueError(
            f"unknown shape-prior backend {value!r}; expected one of "
            f"{SHAPE_PRIOR_BACKENDS}"
        )
    return backend


def ensure_trellis2_available() -> None:
    """Fail fast when the trellis2 backend's local install is missing."""
    problems = []
    if not TRELLIS2_PYTHON.is_file():
        problems.append(
            f"trellis2 env python not found: {TRELLIS2_PYTHON} "
            "(build the `trellis2` conda env; see memory trellis2-integration)"
        )
    if not (TRELLIS2_REPO / "trellis2" / "__init__.py").is_file():
        problems.append(f"TRELLIS.2 checkout not found: {TRELLIS2_REPO}")
    if problems:
        raise FileNotFoundError("; ".join(problems))
