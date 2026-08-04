"""Build a shape-prior mesh-cache entry from the deepAlign frame.

    conda run -n demo_2_max python deepAlign/build_cache.py [--force]

Purpose: the deepAlign frame shows the SAME sloth plush in a DIFFERENT
articulated pose (spread-eagle) than the demo's frame-0. Generating the shape
prior from it mimics the general failure mode where the generated mesh does
NOT share the frame-0 object's part layout / approximate pose — i.e. the
align stage's core assumption (one global Sim(3) + mild local non-rigid
deformation) is violated.

Mirrors the real demo_v6_2 generation chain (read-only reuse):
  1. shape_prior.upscale  — mask-bbox crop x1.2 + SD x4 upscale (real stage);
  2. segment substitute   — the deepAlign mask IS the ground-truth
     segmentation, so the SAM3.1 step is replaced by cropping/resizing that
     mask into the alpha channel of masked_image.png (same RGBA contract);
  3. shape_prior.generate — SAM3D (real stage, seed 42);
  4. mesh_cache.publish   — cache entry ``sloth_deepalign``.
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

DATA_DIR = REPO_ROOT / "deepAlign" / "data"
BUILD_DIR = REPO_ROOT / "deepAlign" / "outputs" / "cache_build"
OBJECT_ID = "sloth_deepalign"
OBJECT_PROMPT = "sloth stuffed animal"
GENERATE_SEED = 42
CACHE_ROOT = Path.home() / "qqtt_shape_prior_cache"
# SAM3D env landmines (see memory/exec plans): its inference.py overwrites
# CUDA_HOME with CONDA_PREFIX, and nvdiffrast 0.3.3 + torch 2.11 needs the
# prebuilt plugin dir on PYTHONPATH.
SAM3D_ENV = {
    "CONDA_PREFIX": "/usr/local/cuda",
    "CUDA_VISIBLE_DEVICES": "1",
}
NVDIFFRAST_PLUGIN = Path.home() / ".cache/torch_extensions/py312_cu130/nvdiffrast_plugin"


def stage_env() -> dict[str, str]:
    env = os.environ.copy()
    env.update(SAM3D_ENV)
    python_path = [str(REPO_ROOT), str(REPO_ROOT / "demo_v6_2"), str(NVDIFFRAST_PLUGIN)]
    if env.get("PYTHONPATH"):
        python_path.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(python_path)
    return env


def run_stage(command: list[str]) -> None:
    print(f"[build_cache] $ {' '.join(command)}", flush=True)
    subprocess.run(command, check=True, cwd=REPO_ROOT, env=stage_env())


def crop_bbox_like_upscale(mask: np.ndarray) -> tuple[int, int, int, int]:
    """Reproduce shape_prior.upscale's mask-bbox x1.2 square crop box."""
    points = np.argwhere(mask > 0.8 * 255)
    x0, y0 = points[:, 1].min(), points[:, 0].min()
    x1, y1 = points[:, 1].max(), points[:, 0].max()
    center = ((x0 + x1) / 2, (y0 + y1) / 2)
    size = int(max(x1 - x0, y1 - y0) * 1.2)
    return (
        int(center[0] - size // 2),
        int(center[1] - size // 2),
        int(center[0] + size // 2),
        int(center[1] + size // 2),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true",
                        help="replace an existing cache entry")
    parser.add_argument("--skip-upscale", action="store_true",
                        help="reuse an existing high_resolution.png")
    args = parser.parse_args()

    from demo_v6_2.shape_prior.mesh_cache import ShapePriorMeshCache

    cache = ShapePriorMeshCache(object_id=OBJECT_ID, cache_root=CACHE_ROOT)
    entry_dir = cache.entry_dir
    assert entry_dir is not None
    if entry_dir.exists():
        if not args.force:
            print(f"[build_cache] cache entry already exists: {entry_dir} "
                  "(use --force to rebuild)")
            return 0
        shutil.rmtree(entry_dir)

    BUILD_DIR.mkdir(parents=True, exist_ok=True)
    color_path = DATA_DIR / "color" / "000001.png"
    mask_path = DATA_DIR / "mask" / "000001.png"
    high_res_path = BUILD_DIR / "high_resolution.png"
    masked_image_path = BUILD_DIR / "masked_image.png"

    # 1. Real upscale stage (SD x4 on the mask crop).
    if not (args.skip_upscale and high_res_path.exists()):
        run_stage([
            sys.executable, "-m", "demo_v6_2.shape_prior.upscale",
            "--img_path", str(color_path),
            "--mask_path", str(mask_path),
            "--output_path", str(high_res_path),
            "--category", OBJECT_PROMPT,
        ])

    # 2. Segment substitute: the ground-truth mask becomes the alpha channel,
    #    cropped with the identical bbox math and resized to the upscaled size.
    mask = np.asarray(Image.open(mask_path).convert("L"))
    bbox = crop_bbox_like_upscale(mask)
    mask_crop = Image.fromarray(mask).crop(bbox)
    high_res = Image.open(high_res_path).convert("RGB")
    alpha = mask_crop.resize(high_res.size, Image.NEAREST)
    rgba = high_res.copy()
    rgba.putalpha(alpha)
    rgba.save(masked_image_path)
    alpha_px = int((np.asarray(alpha) > 127).sum())
    print(f"[build_cache] masked_image.png {high_res.size}, alpha px {alpha_px}")

    # 3. Real SAM3D generate stage.
    run_stage([
        sys.executable, "-m", "demo_v6_2.shape_prior.generate",
        "--img_path", str(masked_image_path),
        "--output_dir", str(BUILD_DIR),
        "--seed", str(GENERATE_SEED),
        "--skip-visualization",
    ])
    object_glb = BUILD_DIR / "object.glb"
    if not object_glb.exists():
        raise SystemExit(f"[build_cache] generate did not produce {object_glb}")

    # 4. Publish into the persistent cache.
    manifest = cache.publish(
        source_glb=object_glb,
        object_prompt_at_generation=OBJECT_PROMPT,
        generator_seed=GENERATE_SEED,
    )
    print(f"[build_cache] published {entry_dir} sha={manifest['mesh_sha256'][:12]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
