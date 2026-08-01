"""Stage: coarse-align a TripoSplat gaussian PLY to the case world frame.

Usage (demo_2_max env, repo root):
    python -m gaussian_align_demo.align_gaussian \
        --case-dir outputs/shape_prior_case/shape_prior_frame0 \
        --ply <run>/seeds/seed_000/gaussian_065536.ply \
        --run-dir <run>

Outputs under <run>/alignment/:
    sim3_coarse.json      winning Sim(3) + per-candidate diagnostics
    coarse_aligned.ply    gaussian transformed into the metric world frame
    coarse_overlay.png    [real | aligned render | alpha blend] from the real camera
    winner_matches.png    SuperGlue correspondences of the winning candidate
    candidates/           grayscale candidate renders + reference crop
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from gaussian_align_demo.alignment import (
    RENDER_SIZE,
    build_candidate_poses,
    build_correspondences,
    build_reference_crop,
    match_all_candidates,
    ransac_umeyama,
    render_candidate_grays,
    reprojection_error_px,
)
from gaussian_align_demo.case_loader import load_frame0_case
from gaussian_align_demo.gs_ply import apply_sim3, load_gaussian_ply, save_gaussian_ply
from gaussian_align_demo.renderer import cloud_to_torch, render_cloud


def draw_matches(
    candidate_gray: np.ndarray,
    reference_rgb: np.ndarray,
    kpts_candidate: np.ndarray,
    kpts_reference: np.ndarray,
    inlier_mask: np.ndarray | None,
) -> np.ndarray:
    left = cv2.cvtColor(candidate_gray, cv2.COLOR_GRAY2BGR)
    right = cv2.cvtColor(reference_rgb, cv2.COLOR_RGB2BGR)
    height = max(left.shape[0], right.shape[0])
    canvas = np.zeros((height, left.shape[1] + right.shape[1], 3), dtype=np.uint8)
    canvas[: left.shape[0], : left.shape[1]] = left
    canvas[: right.shape[0], left.shape[1] :] = right
    offset = left.shape[1]
    for i, (uv0, uv1) in enumerate(zip(kpts_candidate, kpts_reference)):
        inlier = bool(inlier_mask[i]) if inlier_mask is not None and i < len(inlier_mask) else True
        color = (0, 220, 0) if inlier else (60, 60, 230)
        p0 = (int(round(uv0[0])), int(round(uv0[1])))
        p1 = (int(round(uv1[0])) + offset, int(round(uv1[1])))
        cv2.line(canvas, p0, p1, color, 1, cv2.LINE_AA)
        cv2.circle(canvas, p0, 2, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, p1, 2, color, -1, cv2.LINE_AA)
    return canvas


def render_real_camera_overlay(cloud_tensors, case) -> np.ndarray:
    out = render_cloud(
        cloud_tensors,
        K=case.K,
        w2c=case.w2c,
        width=case.width,
        height=case.height,
        background_rgb=(1.0, 1.0, 1.0),
    ).numpy()
    render_u8 = np.clip(out.rgb * 255.0, 0, 255).astype(np.uint8)
    alpha = out.alpha[..., None]
    blend = np.clip(
        out.rgb * alpha * 255.0 + case.rgb_u8.astype(np.float32) * (1.0 - alpha), 0, 255
    ).astype(np.uint8)
    return np.concatenate([case.rgb_u8, render_u8, blend], axis=1)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--ply", required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--ransac-threshold-m", type=float, default=0.015)
    parser.add_argument("--min-inliers", type=int, default=6)
    parser.add_argument("--n-azimuth", type=int, default=12)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    case = load_frame0_case(args.case_dir)
    cloud = load_gaussian_ply(args.ply)
    align_dir = Path(args.run_dir) / "alignment"
    candidates_dir = align_dir / "candidates"
    align_dir.mkdir(parents=True, exist_ok=True)
    print(f"[align] case {case.case_dir.name}: {case.width}x{case.height}, "
          f"object '{case.object_name}' ({int(case.object_mask.sum())} px); "
          f"cloud {len(cloud)} gaussians from {args.ply}")

    reference = build_reference_crop(case.rgb_u8, case.object_mask)
    reference_path = candidates_dir / "reference.png"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(reference_path), reference.image_gray)
    cv2.imwrite(str(candidates_dir / "reference_rgb.png"),
                cv2.cvtColor(reference.image_rgb, cv2.COLOR_RGB2BGR))

    poses, render_K, orbit_info = build_candidate_poses(cloud, n_azimuth=args.n_azimuth)
    print(f"[align] rendering {len(poses)} candidate views "
          f"(radius {orbit_info['radius']:.2f}, canonical extent {np.round(orbit_info['extent'], 2)})")
    tensors = cloud_to_torch(cloud, device=args.device)
    candidate_set = render_candidate_grays(tensors, poses, render_K, candidates_dir)

    matches = match_all_candidates(candidate_set.gray_paths, reference_path, device=args.device)
    counts = np.array([m.num_matches for m in matches])
    order = np.argsort(-counts)[: args.top_k]
    print(f"[align] match counts: top {args.top_k} = "
          f"{[(int(i), int(counts[i])) for i in order]}")

    case_valid = case.object_mask & case.depth_valid
    candidates_report = []
    best = None  # (result, match, correspondences, candidate_index)
    for cand_idx in order:
        match = matches[cand_idx]
        if match.num_matches < 4:
            continue
        out = render_cloud(
            tensors,
            K=render_K,
            w2c=poses[cand_idx],
            width=RENDER_SIZE,
            height=RENDER_SIZE,
            background_rgb=(0.0, 0.0, 0.0),
        ).numpy()
        corr = build_correspondences(
            match,
            candidate_depth=out.depth,
            candidate_alpha=out.alpha,
            candidate_K=render_K,
            candidate_w2c=poses[cand_idx],
            reference=reference,
            case_points_world=case.points_world,
            case_valid=case_valid,
        )
        result = (
            ransac_umeyama(
                corr.points_canonical,
                corr.points_world,
                threshold_m=args.ransac_threshold_m,
                min_inliers=args.min_inliers,
            )
            if len(corr.points_canonical) >= 3
            else None
        )
        entry = {
            "candidate": int(cand_idx),
            "matches": match.num_matches,
            "correspondences": int(len(corr.points_canonical)),
            "dropped": corr.dropped,
        }
        if result is not None:
            entry.update(
                inliers=result.num_inliers,
                inlier_rms_m=round(result.inlier_rms_m, 5),
                scale=round(result.sim3.scale, 5),
                reproj_median_px=round(
                    reprojection_error_px(
                        result.sim3, corr.points_canonical, corr.pixels_full_image,
                        case.K, case.w2c,
                    ),
                    2,
                ),
            )
            if best is None or (result.num_inliers, -result.inlier_rms_m) > (
                best[0].num_inliers, -best[0].inlier_rms_m
            ):
                best = (result, match, corr, int(cand_idx))
        candidates_report.append(entry)
        print(f"[align]   candidate {cand_idx}: {entry}")

    if best is None:
        (align_dir / "sim3_coarse.json").write_text(
            json.dumps({"status": "failed", "candidates": candidates_report}, indent=2)
        )
        raise SystemExit("[align] FAILED: no candidate produced a valid Sim(3)")

    result, match, corr, winner_idx = best
    sim3 = result.sim3
    print(f"[align] winner: candidate {winner_idx}, {result.num_inliers} inliers, "
          f"rms {result.inlier_rms_m * 1000:.1f} mm, scale {sim3.scale:.4f}")

    aligned = apply_sim3(
        cloud, rotation=sim3.rotation, translation=sim3.translation, scale=sim3.scale
    )
    save_gaussian_ply(aligned, align_dir / "coarse_aligned.ply")

    payload = {
        "status": "ok",
        "ply": str(args.ply),
        "case_dir": str(case.case_dir),
        "sim3": sim3.to_dict(),
        "winner_candidate": winner_idx,
        "winner_inliers": result.num_inliers,
        "winner_inlier_rms_m": result.inlier_rms_m,
        "orbit": orbit_info,
        "candidates": candidates_report,
    }
    (align_dir / "sim3_coarse.json").write_text(json.dumps(payload, indent=2))

    winner_gray = cv2.imread(str(candidate_set.gray_paths[winner_idx]), cv2.IMREAD_GRAYSCALE)
    cv2.imwrite(
        str(align_dir / "winner_matches.png"),
        draw_matches(
            winner_gray, reference.image_rgb, match.kpts_candidate, match.kpts_reference, None
        ),
    )
    aligned_tensors = cloud_to_torch(aligned, device=args.device)
    overlay = render_real_camera_overlay(aligned_tensors, case)
    cv2.imwrite(str(align_dir / "coarse_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    print(f"[align] wrote {align_dir}/coarse_aligned.ply, coarse_overlay.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
