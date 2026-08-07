"""TripoSplat gaussian-splats feature options (import-light, stdlib only)."""

from __future__ import annotations

import os
from pathlib import Path

# Machine-local TripoSplat checkout (weights under <repo>/ckpts). Pure-torch
# model — it runs in the service's own conda env, no extra interpreter.
TRIPOSPLAT_REPO = Path(
    os.environ.get("DEMO_V7_TRIPOSPLAT_REPO", "/home/xinjie/TripoSplat")
)
DEFAULT_GAUSSIAN_SEED = 42
DEFAULT_NUM_GAUSSIANS = 131072
DEFAULT_GAUSSIAN_STEPS = 20

# Gaussian generator vocabulary (GUI selector on the source-select dialog).
# One real model today, but the choice is a first-class run option like the
# shape-prior backend — "none" turns the display-only feature off entirely.
GAUSSIAN_TRIPOSPLAT = "triposplat"
GAUSSIAN_NONE = "none"
GAUSSIAN_BACKENDS: tuple[str, ...] = (GAUSSIAN_TRIPOSPLAT, GAUSSIAN_NONE)
DEFAULT_GAUSSIAN_BACKEND = GAUSSIAN_TRIPOSPLAT


def normalize_gaussian_backend(value: str | None) -> str:
    """Return a validated gaussian backend id (None -> the default)."""
    backend = str(value).strip().lower() if value is not None else ""
    if not backend:
        return DEFAULT_GAUSSIAN_BACKEND
    if backend not in GAUSSIAN_BACKENDS:
        raise ValueError(
            f"unknown gaussian backend {value!r}; expected one of "
            f"{GAUSSIAN_BACKENDS}"
        )
    return backend


def ensure_triposplat_available() -> None:
    """Fail fast (with an actionable message) when the install is missing."""
    problems = []
    if not (TRIPOSPLAT_REPO / "triposplat.py").is_file():
        problems.append(f"TripoSplat checkout not found: {TRIPOSPLAT_REPO}")
    ckpts = TRIPOSPLAT_REPO / "ckpts"
    for relative in (
        "diffusion_models/triposplat_fp16.safetensors",
        "vae/triposplat_vae_decoder_fp16.safetensors",
        "clip_vision/dino_v3_vit_h.safetensors",
        "vae/flux2-vae.safetensors",
        "background_removal/birefnet.safetensors",
    ):
        if not (ckpts / relative).is_file():
            problems.append(f"TripoSplat checkpoint missing: {ckpts / relative}")
    if problems:
        raise FileNotFoundError("; ".join(problems))
