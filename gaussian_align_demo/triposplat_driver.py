"""Multi-seed TripoSplat generation driver.

Runs INSIDE the ``triposplat`` conda env (see README); deliberately
self-contained — stdlib + numpy + PIL + torch + the TripoSplat repo only, no
imports from gaussian_align_demo.

Reproducibility: ``TripoSplatPipeline.run()`` is NOT seed-reproducible — the
octree decoder draws sub-voxel jitter and resampling noise from the *global*
torch RNG. We therefore drive the decomposed stages ourselves and pin the
global RNG right before every ``decode_latent`` call (same pattern as the
proven runs under TripoSplat/results/*/run_triposplat_candidates.py).

The saved PLYs use TripoSplat's default axis transform (model y-up -> z-up),
coordinates in a normalized unit box — NOT metric. Alignment happens later.

Example:
    /home/xinjie/miniforge3/envs/triposplat/bin/python \
        gaussian_align_demo/triposplat_driver.py \
        --rgba runs/<id>/input/frame0_rgba.png \
        --output-dir runs/<id>/seeds --seeds 0,1,2,3,4,5,6,7,8,9
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.split(",") if part.strip()]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rgba", required=True, help="RGBA input image (alpha = object mask)")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seeds", default="0,1,2,3,4,5,6,7,8,9", type=parse_int_list)
    parser.add_argument("--num-gaussians", default="65536,262144", type=parse_int_list)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--guidance-scale", type=float, default=3.0)
    parser.add_argument("--shift", type=float, default=3.0)
    parser.add_argument("--erode-radius", type=int, default=1)
    parser.add_argument(
        "--decode-seed",
        type=int,
        default=314159,
        help="global torch RNG seed pinned before every decode_latent call",
    )
    parser.add_argument("--triposplat-root", default="/home/xinjie/TripoSplat")
    parser.add_argument("--device", default="cuda")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    root = Path(args.triposplat_root)
    if not (root / "triposplat.py").exists():
        raise SystemExit(f"TripoSplat repo not found at {root}")
    sys.path.insert(0, str(root))

    import torch
    from PIL import Image
    from triposplat import TripoSplatPipeline

    rgba_path = Path(args.rgba)
    rgba = Image.open(rgba_path)
    if rgba.mode != "RGBA":
        raise SystemExit(f"{rgba_path} is {rgba.mode}, expected RGBA (alpha = object mask)")
    alpha_min = min(rgba.getchannel(3).getextrema())
    if alpha_min >= 255:
        raise SystemExit(
            "alpha channel is fully opaque — TripoSplat would fall back to its own "
            "background removal instead of the provided object mask"
        )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.perf_counter()
    ckpts = root / "ckpts"
    pipeline = TripoSplatPipeline(
        ckpt_path=str(ckpts / "diffusion_models" / "triposplat_fp16.safetensors"),
        decoder_path=str(ckpts / "vae" / "triposplat_vae_decoder_fp16.safetensors"),
        dinov3_path=str(ckpts / "clip_vision" / "dino_v3_vit_h.safetensors"),
        flux2_vae_encoder_path=str(ckpts / "vae" / "flux2-vae.safetensors"),
        rmbg_path=str(ckpts / "background_removal" / "birefnet.safetensors"),
        device=args.device,
    )
    load_s = time.perf_counter() - t0
    print(f"[driver] pipeline loaded in {load_s:.1f}s")

    prepared = pipeline.preprocess_image(rgba, erode_radius=args.erode_radius)
    prepared_path = output_dir / "prepared.webp"
    prepared.save(prepared_path)

    manifest: dict = {
        "input_rgba": str(rgba_path),
        "input_rgba_sha256": sha256_of(rgba_path),
        "prepared_image": str(prepared_path),
        "prepared_image_sha256": sha256_of(prepared_path),
        "steps": args.steps,
        "guidance_scale": args.guidance_scale,
        "shift": args.shift,
        "erode_radius": args.erode_radius,
        "decode_seed": args.decode_seed,
        "num_gaussians": args.num_gaussians,
        "pipeline_load_s": round(load_s, 2),
        "seeds": [],
    }

    for seed in args.seeds:
        seed_dir = output_dir / f"seed_{seed:03d}"
        seed_dir.mkdir(parents=True, exist_ok=True)
        entry: dict = {"seed": seed, "plys": {}}

        t1 = time.perf_counter()
        # Mirror pipeline.run(): one generator feeds the stochastic VAE encode
        # and the flow noise, so each seed reproduces run(image, seed=seed).
        generator = torch.Generator(device=pipeline._device).manual_seed(seed)
        cond = pipeline.encode_image(prepared, generator=generator)
        sampled = pipeline.sample_latent(
            cond,
            steps=args.steps,
            guidance_scale=args.guidance_scale,
            shift=args.shift,
            generator=generator,
        )
        entry["sample_s"] = round(time.perf_counter() - t1, 2)

        for count in args.num_gaussians:
            t2 = time.perf_counter()
            torch.manual_seed(args.decode_seed)
            torch.cuda.manual_seed_all(args.decode_seed)
            gaussian = pipeline.decode_latent(sampled["latent"], num_gaussians=count)
            ply_path = seed_dir / f"gaussian_{count:06d}.ply"
            gaussian.save_ply(str(ply_path))  # default transform: y-up -> z-up
            entry["plys"][str(count)] = {
                "path": str(ply_path),
                "sha256": sha256_of(ply_path),
                "decode_save_s": round(time.perf_counter() - t2, 2),
            }
            del gaussian

        (seed_dir / "generation.json").write_text(json.dumps(entry, indent=2))
        manifest["seeds"].append(entry)
        print(f"[driver] seed {seed}: sampled in {entry['sample_s']}s, "
              f"{len(args.num_gaussians)} density level(s) saved")

    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"[driver] done: {len(args.seeds)} seeds -> {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
