"""Align-stage wrapper: binned candidate rasterization, then the unchanged
v6.2 align stage (same CLI, same interpreter, same WAITING/GO protocol).

Why: align renders 192 candidate views with PyTorch3D's naive rasterizer
(``bin_size=0`` — every pixel tests every face; ~14s of the ~23.5s post-GO
critical path). Binned rasterization is the same math behind an
acceleration structure; ``max_faces_per_bin=len(faces)`` makes bin
overflow (silent face dropping) impossible.

Quality proof (owner red line, decision-level A/B on a real case with
shape/matching cleared so the full render+match path ran):

- final_mesh.glb: bitwise identical (sha256).
- best_match.pkl: winner pose 4x4, ALL SuperGlue keypoints/matches/
  confidences, depth map, intrinsics, crop box — array_equal; the only
  delta is 3 pixels (max 1/255) of Phong shading float jitter in the
  stored winner IMAGE, whose only consumer past this point is display.
- Wall: 23.4s -> 12.0s on the case that produced those numbers.

The patch targets ``align_util._render_loaded_mesh`` (module-global lookup
at call time), so both callers — ``render_multi_images`` (the 192-view
pose search) and ``render_image`` — get the accelerated path with the
identical function contract. ``DEMO_V7_ALIGN_FAST=0`` disables the patch
(the wrapper then runs stock align verbatim).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))


def _patch_binned_rasterizer() -> None:
    import numpy as np
    import torch

    from demo_v6_2.utils import align_util

    stock_render_loaded_mesh = align_util._render_loaded_mesh

    def _binned_render_loaded_mesh(
        mesh, camera_poses, width=640, height=480, fov=1, device="cpu"
    ):
        """Binned rasterization with a fail-soft fallback to the stock path."""
        try:
            return _binned_render_impl(
                mesh, camera_poses, width=width, height=height, fov=fov,
                device=device,
            )
        except Exception as exc:
            # Robustness beats the speedup: any failure (OOM under foreign
            # GPU load, pytorch3d drift, ...) reverts THIS call to the
            # naive stock renderer — identical output, just slower.
            print(
                f"[align-fast-safe] binned raster failed ({exc}); "
                "falling back to the stock naive rasterizer",
                flush=True,
            )
            return stock_render_loaded_mesh(
                mesh, camera_poses, width=width, height=height, fov=fov,
                device=device,
            )

    def _binned_render_impl(
        mesh, camera_poses, width=640, height=480, fov=1, device="cpu"
    ):
        """align_util._render_loaded_mesh, binned rasterization only."""
        from pytorch3d.renderer import (
            AmbientLights,
            BlendParams,
            MeshRasterizer,
            PerspectiveCameras,
            RasterizationSettings,
            SoftPhongShader,
        )

        camera_poses = align_util._camera_poses_tensor(camera_poses, device)
        R = camera_poses[:, :3, :3]
        T = camera_poses[:, 3, :3]
        num_poses = camera_poses.shape[0]
        cameras = PerspectiveCameras(
            R=R,
            T=T,
            device=device,
            focal_length=torch.ones(num_poses, 1)
            * 0.5
            * width
            / np.tan(fov / 2),
            principal_point=torch.tensor((width / 2, height / 2))
            .repeat(num_poses)
            .reshape(-1, 2),
            image_size=torch.tensor((height, width))
            .repeat(num_poses)
            .reshape(-1, 2),
            in_ndc=False,
        )
        lights = AmbientLights(device=device)
        faces_count = int(mesh.faces_packed().shape[0])
        raster_settings = RasterizationSettings(
            image_size=(height, width),
            blur_radius=0.0,
            faces_per_pixel=1,
            bin_size=None,  # heuristic binning instead of naive (bin_size=0)
            max_faces_per_bin=faces_count,  # overflow impossible
        )
        rasterizer = MeshRasterizer(
            cameras=cameras, raster_settings=raster_settings
        )
        shader = SoftPhongShader(
            device=device,
            blend_params=BlendParams(background_color=(0, 0, 0)),
            cameras=cameras,
            lights=lights,
        )
        mesh_batch = mesh.extend(num_poses).to(device)
        fragments = rasterizer(mesh_batch)
        depth = fragments.zbuf.squeeze().cpu().numpy()
        rendered_images = shader(fragments, mesh_batch)
        color = (rendered_images[..., :3].cpu().numpy() * 255).astype(np.uint8)
        return color, depth

    align_util._render_loaded_mesh = _binned_render_loaded_mesh


def main(argv: list[str] | None = None) -> None:
    if os.environ.get("DEMO_V7_ALIGN_FAST", "1") != "0":
        try:
            _patch_binned_rasterizer()
        except Exception as exc:
            print(
                f"[align-fast-safe] binned patch skipped ({exc}); "
                "running the stock renderer",
                flush=True,
            )
    # Robustness (NOT speed, so not behind DEMO_V7_ALIGN_FAST): a fatal
    # factorize at small world scale becomes a scale-equivariant retry.
    from demo_v7.service.arap_rescue import patch_arap_factorize_rescue

    patch_arap_factorize_rescue()

    from demo_v6_2.shape_prior import align

    if os.environ.get("DEMO_V7_ALIGN_FAST", "1") != "0":
        _patch_extended_prewarm(align)

    align.main(list(sys.argv[1:] if argv is None else argv))


def _patch_extended_prewarm(align) -> None:
    """Extend align's pre-GO prewarm with dummy forwards (post-GO -1s+).

    Stock ``_prewarm_models`` only creates the CUDA context and uploads the
    SuperPoint+SuperGlue weights — the first rasterizer kernel launches,
    pytorch3d's lazy GLB-loader imports, and the first SuperPoint/SuperGlue
    forward all land in post-GO time. All dummy outputs are discarded;
    fail-soft (a skipped prewarm just means stock post-GO cost).
    ``align.main`` looks `_prewarm_models` up as a module global, so the
    rebind is picked up at call time.
    """
    stock_prewarm = align._prewarm_models

    def _prewarm_models_extended():
        stock_prewarm()
        try:
            _extended_prewarm_dummy_work()
        except Exception as exc:
            print(
                f"[align-fast-safe] extended prewarm skipped "
                f"({type(exc).__name__}: {exc})",
                flush=True,
            )

    align._prewarm_models = _prewarm_models_extended


def _extended_prewarm_dummy_work(width: int = 848, height: int = 480) -> None:
    import torch

    if not torch.cuda.is_available():
        return
    device = "cuda"
    # Warm the lazy GLB-loader imports (align_util loads them post-GO).
    from pytorch3d.io import IO  # noqa: F401
    from pytorch3d.io.experimental_gltf_io import MeshGlbFormat  # noqa: F401
    from pytorch3d.renderer import TexturesVertex
    from pytorch3d.structures import Meshes

    from demo_v6_2.shape_prior import match_pairs
    from demo_v6_2.utils import align_util

    # Tiny render THROUGH align_util._render_loaded_mesh (module-global
    # lookup, so this exercises the exact — binned-patched — GO path).
    verts = torch.tensor(
        [[-0.1, -0.1, 2.0], [0.1, -0.1, 2.0], [0.0, 0.1, 2.0], [0.0, 0.0, 2.1]],
        dtype=torch.float32,
        device=device,
    )
    faces = torch.tensor(
        [[0, 1, 2], [0, 2, 3], [0, 3, 1], [1, 2, 3]],
        dtype=torch.int64,
        device=device,
    )
    textures = TexturesVertex(torch.ones(1, 4, 3, device=device))
    mesh = Meshes(verts=[verts], faces=[faces], textures=textures)
    poses = align_util.sample_camera_poses(3.0, 2, 1, device)
    align_util._render_loaded_mesh(
        mesh, poses, width=width, height=height, fov=1.566, device=device
    )
    # One SuperPoint+SuperGlue forward at the real frame size (cached model
    # instance — same object the GO path uses).
    matching = match_pairs.get_matching_model()
    dummy = torch.zeros(1, 1, height, width, device=device)
    with torch.no_grad():
        matching({"image0": dummy, "image1": dummy})
    torch.cuda.synchronize()


if __name__ == "__main__":
    main()
