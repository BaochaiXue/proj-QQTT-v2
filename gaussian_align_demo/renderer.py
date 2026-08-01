"""Minimal differentiable Gaussian renderer on top of ``gsplat.rasterization``.

Conventions (used across gaussian_align_demo):
- OpenCV pinhole camera: x right, y down, z forward; ``w2c`` maps world ->
  camera for column vectors; K is the usual 3x3 intrinsics.
- Colors are view-independent RGB in [0, 1] (TripoSplat is SH DC only).
- Depth is gsplat's expected depth along the camera z axis (meters in an
  aligned/metric scene), 0 where nothing was rendered.

The torch entry point keeps the autograd graph intact so the same renderer
serves both visualization and the differentiable Sim(3) refinement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gsplat import rasterization

from gaussian_align_demo.gs_ply import GaussianCloud


@dataclass
class RenderOutput:
    """(H, W, 3) rgb in [0,1], (H, W) alpha in [0,1], (H, W) expected depth."""

    rgb: torch.Tensor
    alpha: torch.Tensor
    depth: torch.Tensor

    def numpy(self) -> "RenderOutput":
        return RenderOutput(
            rgb=self.rgb.detach().cpu().numpy(),
            alpha=self.alpha.detach().cpu().numpy(),
            depth=self.depth.detach().cpu().numpy(),
        )

    def rgb_u8(self, background: np.ndarray | None = None) -> np.ndarray:
        rgb = self.rgb.detach().cpu().numpy()
        if background is not None:
            alpha = self.alpha.detach().cpu().numpy()[..., None]
            rgb = rgb * alpha + np.asarray(background, dtype=np.float32) / 255.0 * (1.0 - alpha)
        return np.clip(rgb * 255.0, 0.0, 255.0).astype(np.uint8)


def render_gaussians_torch(
    *,
    means: torch.Tensor,  # (N, 3)
    quats_wxyz: torch.Tensor,  # (N, 4)
    scales: torch.Tensor,  # (N, 3) linear
    opacities: torch.Tensor,  # (N,)
    colors_rgb: torch.Tensor,  # (N, 3) in [0, 1]
    K: torch.Tensor,  # (3, 3)
    w2c: torch.Tensor,  # (4, 4)
    width: int,
    height: int,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> RenderOutput:
    device = means.device
    background = torch.tensor([background_rgb], dtype=torch.float32, device=device)
    render, alphas, _ = rasterization(
        means=means,
        quats=quats_wxyz,
        scales=scales,
        opacities=opacities,
        colors=colors_rgb,
        viewmats=w2c.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=int(width),
        height=int(height),
        render_mode="RGB+ED",
        backgrounds=background,
        near_plane=float(near_plane),
        far_plane=float(far_plane),
        # gsplat's packed=True default trips a backgrounds shape assert; the
        # unpacked path is what FuturePhysTwin's render_gsplat uses too.
        packed=False,
    )
    rgb = render[0, :, :, :3].clamp(0.0, 1.0)
    depth = render[0, :, :, 3]
    alpha = alphas[0, :, :, 0].clamp(0.0, 1.0)
    return RenderOutput(rgb=rgb, alpha=alpha, depth=depth)


def cloud_to_torch(
    cloud: GaussianCloud, device: str | torch.device = "cuda"
) -> dict[str, torch.Tensor]:
    """Move a GaussianCloud into renderer-ready torch tensors (activated)."""
    to = lambda arr: torch.from_numpy(np.ascontiguousarray(arr)).float().to(device)
    return {
        "means": to(cloud.means),
        "quats_wxyz": to(cloud.quats_wxyz),
        "scales": to(cloud.scales),
        "opacities": to(cloud.opacities),
        "colors_rgb": to(cloud.colors_rgb),
    }


def render_cloud(
    cloud_tensors: dict[str, torch.Tensor],
    *,
    K: np.ndarray,
    w2c: np.ndarray,
    width: int,
    height: int,
    background_rgb: tuple[float, float, float] = (1.0, 1.0, 1.0),
    near_plane: float = 0.01,
    far_plane: float = 100.0,
) -> RenderOutput:
    device = cloud_tensors["means"].device
    with torch.no_grad():
        return render_gaussians_torch(
            **cloud_tensors,
            K=torch.as_tensor(K, dtype=torch.float32, device=device),
            w2c=torch.as_tensor(w2c, dtype=torch.float32, device=device),
            width=width,
            height=height,
            background_rgb=background_rgb,
            near_plane=near_plane,
            far_plane=far_plane,
        )
