"""Stage: render per-seed turntables + a synchronized comparison grid.

    python -m gaussian_align_demo.seed_gallery --run-dir <run>

All seeds share the same canonical-space orbit (TripoSplat outputs are
normalized to a unit box), so differences in pose/completeness/floaters are
directly comparable. Outputs under <run>/seed_gallery/:
    seed_comparison_grid.mp4   all seeds, synchronized 360° orbit, white bg
    turntable_seed_XXX.mp4     one per seed
    seed_scores.csv / .json    automatic stats (ranking aid, NOT the decision)

The human picks the seed: write <run>/selected_seed.json (or pass --select N
here to write it for you) before running alignment.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np

from gaussian_align_demo.alignment import robust_center_extent
from gaussian_align_demo.cameras import intrinsics_for_fov, look_at_w2c
from gaussian_align_demo.gs_ply import GaussianCloud, load_gaussian_ply
from gaussian_align_demo.renderer import cloud_to_torch, render_cloud

PANEL = 320
N_ORBIT_FRAMES = 120
FPS = 15


def seed_stats(cloud: GaussianCloud) -> dict:
    opacities = cloud.opacities
    center, extent = robust_center_extent(cloud)
    sorted_extent = np.sort(np.maximum(extent, 1e-9))
    solid = cloud.means[opacities > 0.3]
    if len(solid) == 0:
        solid = cloud.means
    inside = np.all(np.abs(solid - center) <= 0.75 * np.maximum(extent, 1e-9), axis=1)
    return {
        "gaussians": int(len(cloud)),
        "opacity_mass": float(opacities.sum()),
        "high_opacity_frac": float((opacities > 0.5).mean()),
        "extent": [float(v) for v in extent],
        "thin_axis_ratio": float(sorted_extent[0] / sorted_extent[2]),
        "outlier_frac": float(1.0 - inside.mean()),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--density", type=int, default=65536, help="which PLY density to preview")
    parser.add_argument("--select", type=int, default=None,
                        help="write selected_seed.json for this seed and exit")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    seed_dirs = sorted((run_dir / "seeds").glob("seed_*"))
    if not seed_dirs:
        raise SystemExit(f"[gallery] no seeds under {run_dir}/seeds")

    if args.select is not None:
        chosen = run_dir / "seeds" / f"seed_{args.select:03d}"
        plys = sorted(chosen.glob("gaussian_*.ply"))
        if not plys:
            raise SystemExit(f"[gallery] {chosen} has no PLYs")
        best_ply = max(plys, key=lambda p: int(p.stem.split("_")[1]))
        (run_dir / "selected_seed.json").write_text(json.dumps({
            "selected_seed": args.select,
            "selected_ply": str(best_ply),
            "manual_reason": "selected via seed_gallery --select",
        }, indent=2))
        print(f"[gallery] selected seed {args.select}: {best_ply}")
        return 0

    gallery_dir = run_dir / "seed_gallery"
    gallery_dir.mkdir(parents=True, exist_ok=True)

    clouds: list[tuple[str, GaussianCloud]] = []
    scores: list[dict] = []
    for seed_dir in seed_dirs:
        ply = seed_dir / f"gaussian_{args.density:06d}.ply"
        if not ply.exists():
            print(f"[gallery] skip {seed_dir.name}: no {ply.name}")
            continue
        cloud = load_gaussian_ply(ply)
        stats = {"seed": seed_dir.name, **seed_stats(cloud)}
        scores.append(stats)
        clouds.append((seed_dir.name, cloud))
        print(f"[gallery] {seed_dir.name}: {stats['gaussians']} gaussians, "
              f"opacity mass {stats['opacity_mass']:.0f}, thin ratio {stats['thin_axis_ratio']:.2f}, "
              f"outliers {stats['outlier_frac'] * 100:.1f}%")

    with open(gallery_dir / "seed_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(scores[0].keys()))
        writer.writeheader()
        for row in scores:
            writer.writerow({k: (json.dumps(v) if isinstance(v, list) else v) for k, v in row.items()})
    (gallery_dir / "seed_scores.json").write_text(json.dumps(scores, indent=2))

    # Shared canonical orbit: all seeds live in the same normalized unit box.
    max_extent = float(max(np.linalg.norm(s["extent"]) for s in scores))
    radius = (max_extent / 2.0) / (0.7 * math.tan(math.radians(45.0) / 2.0))
    K = intrinsics_for_fov(width=PANEL, height=PANEL, fov_x_deg=45.0)

    tensor_sets = [(name, cloud_to_torch(c, device=args.device)) for name, c in clouds]
    per_seed_writers = {
        name: imageio.get_writer(gallery_dir / f"turntable_{name}.mp4", fps=FPS, macro_block_size=1)
        for name, _ in tensor_sets
    }
    columns = math.ceil(len(tensor_sets) / 2) if len(tensor_sets) > 1 else 1
    rows = 2 if len(tensor_sets) > columns else 1
    grid_writer = imageio.get_writer(
        gallery_dir / "seed_comparison_grid.mp4", fps=FPS, macro_block_size=1
    )

    for frame_idx in range(N_ORBIT_FRAMES):
        azimuth = 360.0 * frame_idx / N_ORBIT_FRAMES
        elevation = 20.0
        eye = np.array([
            radius * math.cos(math.radians(azimuth)) * math.cos(math.radians(elevation)),
            radius * math.sin(math.radians(azimuth)) * math.cos(math.radians(elevation)),
            radius * math.sin(math.radians(elevation)),
        ])
        w2c = look_at_w2c(eye, np.zeros(3))
        panels = []
        for name, tensors in tensor_sets:
            out = render_cloud(tensors, K=K, w2c=w2c, width=PANEL, height=PANEL,
                               background_rgb=(1.0, 1.0, 1.0))
            panel = out.rgb_u8()
            cv2.putText(panel, name, (8, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (30, 30, 30), 1,
                        cv2.LINE_AA)
            per_seed_writers[name].append_data(panel)
            panels.append(panel)
        while len(panels) < rows * columns:
            panels.append(np.full((PANEL, PANEL, 3), 255, dtype=np.uint8))
        grid_rows = [np.concatenate(panels[r * columns:(r + 1) * columns], axis=1)
                     for r in range(rows)]
        grid_writer.append_data(np.concatenate(grid_rows, axis=0))

    for writer in per_seed_writers.values():
        writer.close()
    grid_writer.close()
    print(f"[gallery] wrote {gallery_dir}/seed_comparison_grid.mp4 (+{len(tensor_sets)} turntables)")
    print("[gallery] pick a seed, then: python -m gaussian_align_demo.seed_gallery "
          f"--run-dir {run_dir} --select <N>")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
