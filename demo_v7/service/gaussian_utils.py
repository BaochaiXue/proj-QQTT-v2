"""3DGS ply IO, similarity transforms, and gsplat offscreen rendering.

Shared by the TripoSplat worker (turntable previews), the alignment step
(world-frame export + overlay stills), and the FORMAL realtime gaussian
channel. Conventions verified against the real artifacts (see memory /
exec plan): standard INRIA ply layout, SH degree 0 — stored opacity is a
logit, scales are log(sigma), quats are wxyz and NOT necessarily unit-norm;
display rgb = 0.5 + 0.2820948 * f_dc.
"""

from __future__ import annotations

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SH_C0 = 0.28209479177387814

# v7-private gsplat JIT cache. The default shared torch_extensions dir is
# rebuilt by whichever environment touched it last (another session's shell
# resolves a different nvcc -> different build hash -> full ~137s recompile
# INSIDE a live run; measured twice). Pinning both the cache dir and
# CUDA_HOME during the one import that builds the extension makes the build
# hash identical for every v7 caller (service process, worker subprocess,
# ad-hoc smokes), so the compile happens once ever.
_V7_TORCH_EXTENSIONS_DIR = Path.home() / ".cache" / "demo_v7_torch_extensions"
_PINNED_CUDA_HOME = "/usr/local/cuda"


def _import_gsplat_rasterization():
    """Import gsplat with its CUDA backend loaded under the pinned env.

    The JIT build fires at the first ``gsplat.cuda._backend`` import (NOT at
    ``import gsplat`` — the backend is lazy), so that import must happen
    inside the pinned scope or the env pin is a no-op.
    """
    if "gsplat.cuda._backend" in sys.modules:
        from gsplat import rasterization

        return rasterization
    saved = {
        name: os.environ.get(name)
        for name in ("TORCH_EXTENSIONS_DIR", "CUDA_HOME")
    }
    os.environ["TORCH_EXTENSIONS_DIR"] = str(_V7_TORCH_EXTENSIONS_DIR)
    if Path(_PINNED_CUDA_HOME, "bin", "nvcc").is_file():
        os.environ["CUDA_HOME"] = _PINNED_CUDA_HOME
    try:
        from gsplat import rasterization

        import gsplat.cuda._backend  # noqa: F401  (forces the JIT load now)
    finally:
        for name, value in saved.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value
    return rasterization


@dataclass
class GaussianSplats:
    """Decoded, render-ready splats (activations applied)."""

    means: np.ndarray  # (N,3) float32
    quats: np.ndarray  # (N,4) float32 wxyz, unit-norm
    scales: np.ndarray  # (N,3) float32 sigma (linear)
    opacities: np.ndarray  # (N,) float32 in [0,1]
    colors: np.ndarray  # (N,3) float32 in [0,1] (DC-decoded rgb)

    def __len__(self) -> int:
        return int(self.means.shape[0])


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    positive = x >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-x[positive]))
    exp_x = np.exp(x[~positive])
    out[~positive] = exp_x / (1.0 + exp_x)
    return out


def load_gaussian_ply(path: str | Path) -> GaussianSplats:
    """Read a standard 3DGS ply (SH degree 0) into activated arrays."""
    from plyfile import PlyData

    vertex = PlyData.read(str(path))["vertex"].data
    means = np.column_stack([vertex["x"], vertex["y"], vertex["z"]]).astype(np.float32)
    scale_logs = np.column_stack(
        [vertex["scale_0"], vertex["scale_1"], vertex["scale_2"]]
    ).astype(np.float32)
    quats = np.column_stack(
        [vertex["rot_0"], vertex["rot_1"], vertex["rot_2"], vertex["rot_3"]]
    ).astype(np.float32)
    opacity_logits = np.asarray(vertex["opacity"], dtype=np.float32)
    f_dc = np.column_stack(
        [vertex["f_dc_0"], vertex["f_dc_1"], vertex["f_dc_2"]]
    ).astype(np.float32)

    finite = (
        np.isfinite(means).all(axis=1)
        & np.isfinite(scale_logs).all(axis=1)
        & np.isfinite(quats).all(axis=1)
        & np.isfinite(opacity_logits)
        & np.isfinite(f_dc).all(axis=1)
    )
    means, scale_logs = means[finite], scale_logs[finite]
    quats, opacity_logits, f_dc = quats[finite], opacity_logits[finite], f_dc[finite]

    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    norms[norms < 1e-12] = 1.0
    quats = quats / norms
    return GaussianSplats(
        means=means,
        quats=quats,
        scales=np.exp(scale_logs),
        opacities=_sigmoid(opacity_logits),
        colors=np.clip(0.5 + SH_C0 * f_dc, 0.0, 1.0),
    )


def save_gaussian_ply(path: str | Path, splats: GaussianSplats) -> None:
    """Write activated splats back to the standard layout (re-encoded)."""
    from plyfile import PlyData, PlyElement

    n = len(splats)
    fields = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
        ("f_dc_0", "f4"), ("f_dc_1", "f4"), ("f_dc_2", "f4"),
        ("opacity", "f4"),
        ("scale_0", "f4"), ("scale_1", "f4"), ("scale_2", "f4"),
        ("rot_0", "f4"), ("rot_1", "f4"), ("rot_2", "f4"), ("rot_3", "f4"),
    ]
    data = np.zeros(n, dtype=fields)
    data["x"], data["y"], data["z"] = splats.means.T
    f_dc = (splats.colors - 0.5) / SH_C0
    data["f_dc_0"], data["f_dc_1"], data["f_dc_2"] = f_dc.T
    opacity = np.clip(splats.opacities, 1e-6, 1.0 - 1e-6)
    data["opacity"] = np.log(opacity / (1.0 - opacity))
    scales = np.clip(splats.scales, 1e-9, None)
    data["scale_0"], data["scale_1"], data["scale_2"] = np.log(scales).T
    data["rot_0"], data["rot_1"], data["rot_2"], data["rot_3"] = splats.quats.T
    PlyData([PlyElement.describe(data, "vertex")]).write(str(path))


def _quat_of_matrix(rotation: np.ndarray) -> np.ndarray:
    """One (3,3) rotation -> wxyz quaternion (scipy convention adapter)."""
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.array([xyzw[3], xyzw[0], xyzw[1], xyzw[2]], dtype=np.float64)


def _quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """(4,) wxyz x (N,4) wxyz Hamilton product."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]
    return np.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        axis=1,
    )


def transform_gaussians(
    splats: GaussianSplats, transform: np.ndarray
) -> GaussianSplats:
    """Apply a 4x4 similarity (uniform scale s * rotation + translation).

    Means map through the full matrix; splat orientations rotate by the
    rotation part; anisotropic sigmas multiply by s (similarity only — a
    non-uniform linear part would need full covariance surgery and is
    rejected loudly).
    """
    transform = np.asarray(transform, dtype=np.float64)
    linear = transform[:3, :3]
    scale = float(np.cbrt(abs(np.linalg.det(linear))))
    if scale <= 0.0:
        raise ValueError("transform has non-positive determinant")
    rotation = linear / scale
    if not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-4):
        raise ValueError(
            "transform_gaussians requires a similarity (uniform scale x "
            "rotation); got a general linear part"
        )
    means = splats.means.astype(np.float64) @ linear.T + transform[:3, 3]
    quats = _quat_multiply(_quat_of_matrix(rotation), splats.quats.astype(np.float64))
    quats = quats / np.linalg.norm(quats, axis=1, keepdims=True)
    return GaussianSplats(
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=(splats.scales * scale).astype(np.float32),
        opacities=splats.opacities.copy(),
        colors=splats.colors.copy(),
    )


def render_gaussians(
    splats_or_tensors,
    *,
    viewmat: np.ndarray,
    intrinsics: np.ndarray,
    width: int,
    height: int,
    background: tuple[float, float, float] = (1.0, 1.0, 1.0),
    device: str = "cuda",
):
    """gsplat offscreen render -> (rgb uint8 (H,W,3), alpha float (H,W)).

    ``splats_or_tensors`` is a GaussianSplats (numpy, converted per call) or
    a dict of pre-staged torch tensors {means, quats, scales, opacities,
    colors} for the realtime path (avoids per-frame host->device copies).
    """
    import torch

    rasterization = _import_gsplat_rasterization()

    if isinstance(splats_or_tensors, GaussianSplats):
        tensors = splats_to_tensors(splats_or_tensors, device=device)
    else:
        tensors = splats_or_tensors
    viewmats = torch.as_tensor(
        np.asarray(viewmat, dtype=np.float32), device=device
    ).reshape(1, 4, 4)
    Ks = torch.as_tensor(
        np.asarray(intrinsics, dtype=np.float32), device=device
    ).reshape(1, 3, 3)
    backgrounds = torch.tensor([background], device=device, dtype=torch.float32)
    with torch.no_grad():
        colors, alphas, _meta = rasterization(
            tensors["means"],
            tensors["quats"],
            tensors["scales"],
            tensors["opacities"],
            tensors["colors"],
            viewmats,
            Ks,
            width,
            height,
            sh_degree=None,
            backgrounds=backgrounds,
        )
    rgb = (colors[0].clamp(0, 1) * 255).to(torch.uint8).cpu().numpy()
    alpha = alphas[0, ..., 0].cpu().numpy()
    return rgb, alpha


def splats_to_tensors(splats: GaussianSplats, *, device: str = "cuda") -> dict:
    """Stage a GaussianSplats onto the GPU once (realtime path)."""
    import torch

    return {
        "means": torch.as_tensor(splats.means, device=device),
        "quats": torch.as_tensor(splats.quats, device=device),
        "scales": torch.as_tensor(splats.scales, device=device),
        "opacities": torch.as_tensor(splats.opacities, device=device),
        "colors": torch.as_tensor(splats.colors, device=device),
    }
