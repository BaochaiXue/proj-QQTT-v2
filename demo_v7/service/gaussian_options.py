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
# - triposplat: independent generative model (its own free geometry, then
#   registration/self-align onto the world).
# - mesh_surface: no second model — splats are DERIVED from the aligned
#   TRELLIS.2 world mesh (face_id + barycentric hard binding; owner trial
#   2026-08-14). Only meaningful when the mesh backend is trellis2.
# - none: turns the display-only feature off entirely.
GAUSSIAN_TRIPOSPLAT = "triposplat"
GAUSSIAN_MESH_SURFACE = "mesh_surface"
GAUSSIAN_NONE = "none"
GAUSSIAN_BACKENDS: tuple[str, ...] = (
    GAUSSIAN_TRIPOSPLAT,
    GAUSSIAN_MESH_SURFACE,
    GAUSSIAN_NONE,
)
DEFAULT_GAUSSIAN_BACKEND = GAUSSIAN_TRIPOSPLAT
# Mesh backends whose align chain produces the world-frame final_mesh.glb
# that mesh_surface derives its splats from (owner instruction: offer the
# option when TRELLIS.2 is the mesh backend).
MESH_SURFACE_REQUIRED_SHAPE_BACKENDS: tuple[str, ...] = ("trellis2",)


def mesh_surface_allowed(shape_prior_backend: str | None) -> bool:
    """True when ``mesh_surface`` is usable with this mesh backend."""
    return str(shape_prior_backend) in MESH_SURFACE_REQUIRED_SHAPE_BACKENDS


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
