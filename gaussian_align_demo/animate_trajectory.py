"""Stage: drive the aligned gaussian along the recorded trajectory and render.

    python -m gaussian_align_demo.animate_trajectory \
        --run-dir <run> --case-dir outputs/shape_prior_case/shape_prior_frame0 \
        --final-data outputs/data/final_data.pkl

Uses <run>/alignment/refined_aligned.ply when present, else coarse_aligned.ply
(--ply overrides). Bones = final_data object_points (T, B, 3), same world
frame as the case (demo_v6_2 writes identical calibrate.pkl for both — gated
here anyway). Incremental rollout at sub-frame granularity: bone positions are
linearly interpolated fps-in -> fps-out and each sub-step applies neighborhood
Procrustes bone rotations + inverse-distance LBS (see dynamic_utils).

Outputs under <run>/motion/:
    binding.npz               frozen bindings + frame-0 bones
    trajectory_fixed_camera.mp4   [recorded RGB | render white | blend+bones]
    trajectory_orbit.mp4      free slow-orbit view, white background
    metrics.json              gates + per-run stats
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import imageio.v2 as imageio
import numpy as np
import torch

from gaussian_align_demo.cameras import look_at_w2c, project_points
from gaussian_align_demo.case_loader import load_frame0_case, load_trajectory
from gaussian_align_demo.dynamic_utils import (
    apply_bone_transforms,
    bind_gaussians,
    build_bone_relations,
    compute_bone_transforms,
)
from gaussian_align_demo.gs_ply import load_gaussian_ply
from gaussian_align_demo.renderer import cloud_to_torch, render_cloud


def frame0_consistency_gate(bones0: np.ndarray, case) -> dict:
    """Trajectory frame 0 must sit on the case's object observation."""
    from scipy.spatial import cKDTree

    observed = case.object_points_world
    nn_dist, _ = cKDTree(observed).query(bones0)
    centroid_dist = float(np.linalg.norm(observed.mean(0) - bones0.mean(0)))
    report = {
        "bones": int(len(bones0)),
        "nn_median_m": float(np.median(nn_dist)),
        "nn_p95_m": float(np.percentile(nn_dist, 95)),
        "centroid_dist_m": float(centroid_dist),
    }
    if report["nn_median_m"] > 0.02 or centroid_dist > 0.05:
        raise SystemExit(
            f"[animate] FRAME-0 GATE FAILED: trajectory and case are not in the same "
            f"world frame ({report}). Refusing to silently re-register."
        )
    return report


def resolve_ply(run_dir: Path, override: str | None) -> Path:
    if override:
        return Path(override)
    for name in ("refined_aligned.ply", "coarse_aligned.ply"):
        path = run_dir / "alignment" / name
        if path.exists():
            return path
    raise SystemExit(f"[animate] no aligned PLY under {run_dir}/alignment; run alignment first")


def find_recorded_frame(color_dir: Path | None, index: int) -> np.ndarray | None:
    if color_dir is None:
        return None
    path = color_dir / f"{index}.png"
    if not path.exists():
        return None
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB) if bgr is not None else None


def draw_bones(panel: np.ndarray, bones: np.ndarray, K: np.ndarray, w2c: np.ndarray) -> None:
    pixels, depth = project_points(bones, K, w2c)
    for (u, v), z in zip(pixels, depth):
        if z <= 0:
            continue
        x, y = int(round(u)), int(round(v))
        if 0 <= x < panel.shape[1] and 0 <= y < panel.shape[0]:
            cv2.circle(panel, (x, y), 3, (255, 60, 60), -1, cv2.LINE_AA)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--final-data", required=True)
    parser.add_argument("--ply", default=None, help="override the aligned PLY")
    parser.add_argument("--online-color-dir", default=None,
                        help="recorded frames dir (default: <final-data>/../../online_data/color/0)")
    parser.add_argument("--fps-in", type=float, default=5.0)
    parser.add_argument("--fps-out", type=float, default=30.0)
    parser.add_argument("--k-bind", type=int, default=16)
    parser.add_argument("--k-relations", type=int, default=16)
    parser.add_argument("--max-frames", type=int, default=0, help="trajectory frames cap (0 = all)")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    run_dir = Path(args.run_dir)
    motion_dir = run_dir / "motion"
    motion_dir.mkdir(parents=True, exist_ok=True)

    case = load_frame0_case(args.case_dir)
    trajectory = load_trajectory(args.final_data)
    ply_path = resolve_ply(run_dir, args.ply)
    cloud = load_gaussian_ply(ply_path)
    print(f"[animate] {len(cloud)} gaussians from {ply_path.name}; "
          f"trajectory {trajectory.frame_count} frames x {trajectory.bone_count} bones")

    bones_all = trajectory.object_points
    if args.max_frames > 0:
        bones_all = bones_all[: args.max_frames]
    gate = frame0_consistency_gate(bones_all[0], case)
    print(f"[animate] frame-0 gate OK: nn median {gate['nn_median_m'] * 1000:.1f} mm, "
          f"centroid {gate['centroid_dist_m'] * 1000:.1f} mm")

    device = torch.device(args.device)
    tensors = cloud_to_torch(cloud, device=device)
    bones_torch = torch.from_numpy(np.ascontiguousarray(bones_all)).float().to(device)
    relations = build_bone_relations(bones_torch[0], args.k_relations)
    bind_indices = bind_gaussians(bones_torch[0], tensors["means"], args.k_bind)
    bind_dist = (tensors["means"][:, None, :] - bones_torch[0][bind_indices]).norm(dim=-1)
    np.savez_compressed(
        motion_dir / "binding.npz",
        ply=str(ply_path),
        final_data=str(trajectory.path),
        bones0=bones_all[0],
        relations=relations.cpu().numpy(),
        bind_indices=bind_indices.cpu().numpy(),
        bind_distances=bind_dist.cpu().numpy(),
    )
    print(f"[animate] bound: median nearest-bone distance "
          f"{bind_dist[:, 0].median().item() * 1000:.1f} mm")

    color_dir = (
        Path(args.online_color_dir)
        if args.online_color_dir
        else Path(args.final_data).resolve().parent.parent / "online_data" / "color" / "0"
    )
    if not color_dir.is_dir():
        print(f"[animate] recorded frames not found at {color_dir}; panel will be blank")
        color_dir = None

    substeps = max(1, int(round(args.fps_out / args.fps_in)))
    total_frames = (bones_all.shape[0] - 1) * substeps + 1
    print(f"[animate] {total_frames} video frames ({substeps} substeps per trajectory step)")

    scene_center = bones_all.reshape(-1, 3).mean(axis=0)
    scene_radius = float(np.linalg.norm(bones_all.reshape(-1, 3) - scene_center, axis=1).max())
    orbit_radius = max(0.6, 3.0 * scene_radius)

    fixed_writer = imageio.get_writer(
        motion_dir / "trajectory_fixed_camera.mp4", fps=args.fps_out, macro_block_size=1
    )
    orbit_writer = imageio.get_writer(
        motion_dir / "trajectory_orbit.mp4", fps=args.fps_out, macro_block_size=1
    )

    means = tensors["means"].clone()
    quats = tensors["quats_wxyz"].clone()
    max_disp = 0.0
    render_kwargs = dict(K=case.K, w2c=case.w2c, width=case.width, height=case.height)

    def interp_bones(step: float) -> torch.Tensor:
        i = min(int(step), bones_torch.shape[0] - 1)
        a = step - i
        if a <= 0.0 or i + 1 >= bones_torch.shape[0]:
            return bones_torch[i]
        return (1.0 - a) * bones_torch[i] + a * bones_torch[i + 1]

    for j in range(total_frames):
        step = j / substeps
        if j > 0:
            prev = interp_bones((j - 1) / substeps)
            curr = interp_bones(step)
            rotations, translations = compute_bone_transforms(prev, curr, relations)
            means, quats = apply_bone_transforms(
                means, quats, prev, rotations, translations, bind_indices
            )
        state = {**tensors, "means": means, "quats_wxyz": quats}

        out = render_cloud(state, background_rgb=(1.0, 1.0, 1.0), **render_kwargs).numpy()
        render_u8 = np.clip(out.rgb * 255.0, 0, 255).astype(np.uint8)
        recorded = find_recorded_frame(color_dir, int(round(step)))
        if recorded is None:
            recorded = np.zeros_like(render_u8)
        alpha = out.alpha[..., None]
        blend = np.clip(
            out.rgb * alpha * 255.0 + recorded.astype(np.float32) * (1.0 - alpha), 0, 255
        ).astype(np.uint8)
        bones_now = interp_bones(step).cpu().numpy()
        draw_bones(blend, bones_now, case.K, case.w2c)
        fixed_writer.append_data(np.concatenate([recorded, render_u8, blend], axis=1))

        azimuth = 360.0 * j / max(1, total_frames)
        eye = scene_center + orbit_radius * np.array(
            [
                np.cos(np.deg2rad(azimuth)) * np.cos(np.deg2rad(25.0)),
                np.sin(np.deg2rad(azimuth)) * np.cos(np.deg2rad(25.0)),
                np.sin(np.deg2rad(25.0)),
            ]
        )
        orbit_w2c = look_at_w2c(eye, scene_center)
        orbit_out = render_cloud(
            state, K=case.K, w2c=orbit_w2c, width=case.width, height=case.height,
            background_rgb=(1.0, 1.0, 1.0),
        )
        orbit_writer.append_data(orbit_out.rgb_u8())

        if j % 200 == 0 or j == total_frames - 1:
            disp = (means - tensors["means"]).norm(dim=1)
            max_disp = max(max_disp, disp.max().item())
            norms = quats.norm(dim=1)
            if not torch.isfinite(means).all() or not torch.isfinite(quats).all():
                raise SystemExit(f"[animate] non-finite state at video frame {j}")
            print(f"[animate] frame {j}/{total_frames - 1}: max disp {disp.max().item():.3f} m, "
                  f"quat norm [{norms.min().item():.4f}, {norms.max().item():.4f}]")

    fixed_writer.close()
    orbit_writer.close()

    metrics = {
        "ply": str(ply_path),
        "final_data": str(trajectory.path),
        "frames_in": int(bones_all.shape[0]),
        "video_frames": int(total_frames),
        "fps_in": args.fps_in,
        "fps_out": args.fps_out,
        "gaussians": len(cloud),
        "bones": int(bones_all.shape[1]),
        "frame0_gate": gate,
        "bind_nearest_median_m": float(bind_dist[:, 0].median().item()),
        "max_gaussian_displacement_m": float(max_disp),
    }
    (motion_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(f"[animate] wrote {motion_dir}/trajectory_fixed_camera.mp4, trajectory_orbit.mp4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
