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
