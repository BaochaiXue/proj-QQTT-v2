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
# Default per the 2026-08-07 same-frame quality comparison (sloth fake-live,
# seed 42): TRELLIS.2 beats SAM3D on aligned silhouette IoU (0.905 vs
# 0.852), candidate-to-observation distance (median 5.3 vs 12.0mm), texture
# (2048^2 vs 1024^2) and ships a guarded zero-collapsed-face final mesh;
# warmup time is comparable after the trellis2 runner speedups.
DEFAULT_SHAPE_PRIOR_BACKEND = BACKEND_TRELLIS2

# Upscale (SD x4) stage toggle. On is the unchanged v6.2 chain; off swaps the
# stage for demo_v7/service/upscale_passthrough.py (mask-bbox crop only) —
# faster warmup, generation conditions on the original-resolution crop.
DEFAULT_SHAPE_PRIOR_UPSCALE = True
_UPSCALE_TRUE = ("on", "true", "1", "yes")
_UPSCALE_FALSE = ("off", "false", "0", "no")

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
# Pipeline id; must already be snapshot-complete in the local HF cache (the
# runner loads with HF_HUB_OFFLINE so a boot never hits the network).
TRELLIS2_MODEL_ID = "microsoft/TRELLIS.2-4B"


def _hf_hub_cache_dir() -> Path:
    """The huggingface hub cache root, honoring the standard env overrides."""
    if os.environ.get("HF_HUB_CACHE"):
        return Path(os.environ["HF_HUB_CACHE"])
    if os.environ.get("HF_HOME"):
        return Path(os.environ["HF_HOME"]) / "hub"
    return Path.home() / ".cache" / "huggingface" / "hub"


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


def normalize_upscale(value: str | bool | None) -> bool:
    """Return the upscale toggle as a bool (None -> the on default)."""
    if value is None:
        return DEFAULT_SHAPE_PRIOR_UPSCALE
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if not text:
        return DEFAULT_SHAPE_PRIOR_UPSCALE
    if text in _UPSCALE_TRUE:
        return True
    if text in _UPSCALE_FALSE:
        return False
    raise ValueError(
        f"unknown shape-prior upscale toggle {value!r}; expected one of "
        f"{_UPSCALE_TRUE + _UPSCALE_FALSE}"
    )


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
    snapshot_root = (
        _hf_hub_cache_dir()
        / f"models--{TRELLIS2_MODEL_ID.replace('/', '--')}"
        / "snapshots"
    )
    if not any(snapshot_root.glob("*")):
        problems.append(
            f"HF checkpoint {TRELLIS2_MODEL_ID} not in the local cache "
            f"({snapshot_root}); the runner loads offline and would die "
            "opaquely in the prewarm pool"
        )
    if problems:
        raise FileNotFoundError("; ".join(problems))
