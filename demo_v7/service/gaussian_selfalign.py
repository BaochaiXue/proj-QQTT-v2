"""Background self-alignment upgrade for the world gaussian (two-phase).

Phase 1 (gaussian_manager, unchanged): the fast chamfer chain publishes
artifacts immediately — REVIEW keeps its zero-wait entry.

Phase 2 (this module, background thread + subprocess): the visual-metric
self-alignment (gaussian_align_demo: candidate-view gsplat renders +
SuperGlue + RANSAC-Umeyama, then depth-anchored refinement) produces a
Sim(3) DIRECTLY against the capture frame-0 observation. Benchmarked on
two real cases (exec plan 2026-08-07): it beat the mesh-chained rigid pose
decisively (IoU 0.881 vs 0.753 and 0.923 vs 0.650) and exposed 9-19%
scale error in the mesh chain's rigid stage. Candidates:
  B  = raw canonical splats @ self-align Sim(3)
  C2 = B + PURE-articulation ARAP residual (the mesh ARAP field with its
       similarity component Umeyama-stripped — transplanting the raw field
       double-corrects; benchmarked) + floater pruning
The incumbent and both candidates are scored against the OBSERVATION ONLY
(real object mask IoU + observed-cloud distance tail; the mesh is not
ground truth and never appears in scoring). The best candidate replaces
the published world ply/overlay/provenance only if it beats the incumbent.

Isolation: the heavy pipeline runs in a SUBPROCESS (this module's CLI) so
a crash can never take down the camera service; every parent-side step is
fail-soft. ``DEMO_V7_GAUSSIAN_SELF_ALIGN=0`` disables phase 2 entirely.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Gates on the self-align pipeline's own confidence (below these the
# Sim(3) is not trusted and the upgrade is skipped):
_MIN_INLIERS = 12
_MIN_REFINED_IOU = 0.5
# Combined observation score: silhouette IoU traded against the coverage
# tail at 3 IoU points per cm (logged per candidate for later tuning).
_TAIL_WEIGHT_PER_CM = 0.03
# Self-align is the DEFAULT alignment (owner decision 2026-08-07): once its
# gates pass, its best candidate replaces the chamfer-chain result unless
# it is clearly WORSE than the incumbent by more than this tolerance (the
# drive21/22 first-generation regressions — 0.34 vs 0.56 combined — stay
# protected; near-ties go to self-align).
_KEEP_INCUMBENT_MARGIN = 0.01
# Among self-align candidates the plain Sim(3) (B) is the default; the
# articulation variant must clearly beat it to be chosen (its benefit was
# case-dependent in the benchmark).
_ART_PREFERENCE_MARGIN = 0.01
_SUBPROCESS_TIMEOUT_S = 240.0


def combined_score(metrics: dict) -> float:
    return float(metrics["iou"]) - _TAIL_WEIGHT_PER_CM * float(
        metrics["c2g_p90_cm"]
    )


def pick_candidate(scored: list) -> tuple:
    """(name, metrics) — B by default; C2 only when it clearly wins."""
    ranked = {name: metrics for name, metrics in scored}
    base = ("self_align", ranked["self_align"])
    art = ranked.get("self_align_art")
    if art is not None and combined_score(art) > combined_score(
        base[1]
    ) + _ART_PREFERENCE_MARGIN:
        return ("self_align_art", art)
    return base


def should_swap(candidate_metrics: dict, incumbent_metrics: dict) -> bool:
    """Self-align by default: keep the incumbent only on a clear loss."""
    return combined_score(candidate_metrics) >= combined_score(
        incumbent_metrics
    ) - _KEEP_INCUMBENT_MARGIN


def load_case_observation(case_dir: Path) -> dict:
    """Frame-0 ground observation: mask, world cloud, camera. Mesh-free."""
    import pickle

    import cv2
    import numpy as np

    case_dir = Path(case_dir)
    with open(case_dir / "calibrate.pkl", "rb") as handle:
        c2w = np.asarray(pickle.load(handle)[0], dtype=np.float64)
    intrinsics = np.asarray(
        json.loads((case_dir / "metadata.json").read_text())["intrinsics"]
    )[0]
    color = cv2.imread(str(case_dir / "color" / "0" / "0.png"))
    height, width = color.shape[:2]
    mask = cv2.imread(str(case_dir / "mask" / "0" / "0" / "0.png"), 0) > 0
    npz = np.load(case_dir / "pcd" / "0.npz")
    points = npz["points"][0]
    valid = npz["masks"][0] if "masks" in npz else np.isfinite(points).all(-1)
    cloud = points[mask & valid]
    cloud = cloud[np.isfinite(cloud).all(axis=1)].astype(np.float64)
    return dict(
        viewmat=np.linalg.inv(c2w),
        intrinsics=intrinsics,
        width=width,
        height=height,
        mask=mask,
        cloud=cloud,
    )


def score_alignment(splats, observation: dict) -> dict:
    """Observation-only quality: silhouette IoU + cloud distance tails."""
    import numpy as np
    from scipy.spatial import cKDTree

    from demo_v7.service.gaussian_utils import render_gaussians

    _rgb, alpha = render_gaussians(
        splats,
        viewmat=observation["viewmat"],
        intrinsics=observation["intrinsics"],
        width=observation["width"],
        height=observation["height"],
        background=(1.0, 1.0, 1.0),
    )
    silhouette = alpha > 0.5
    mask = observation["mask"]
    intersection = float((silhouette & mask).sum())
    iou = intersection / max(float((silhouette | mask).sum()), 1.0)
    solid = splats.means[splats.opacities > 0.3].astype(np.float64)
    g2c = cKDTree(observation["cloud"]).query(solid, k=1, workers=-1)[0] * 100
    c2g = cKDTree(solid).query(observation["cloud"], k=1, workers=-1)[0] * 100
    return {
        "iou": round(iou, 4),
        "g2c_p90_cm": round(float(np.percentile(g2c, 90)), 3),
        "c2g_p90_cm": round(float(np.percentile(c2g, 90)), 3),
    }


def pure_articulation_field(case_dir: Path, mesh2world) -> tuple:
    """(anchors, displacement, final_verts): the mesh ARAP field with its
    global similarity component stripped (Umeyama), leaving articulation
    only — safe to anchor onto an independently-registered gaussian."""
    import numpy as np

    from demo_v7.service.gaussian_align import arap_residual_field

    canonical_verts, displacement = arap_residual_field(
        Path(case_dir), np.asarray(mesh2world)
    )
    mesh2world = np.asarray(mesh2world)
    rigid = canonical_verts @ mesh2world[:3, :3].T + mesh2world[:3, 3]
    final = rigid + displacement
    mu_r, mu_f = rigid.mean(axis=0), final.mean(axis=0)
    centered_r, centered_f = rigid - mu_r, final - mu_f
    covariance = centered_f.T @ centered_r / len(rigid)
    U, S, Vt = np.linalg.svd(covariance)
    D = np.diag([1.0, 1.0, float(np.sign(np.linalg.det(U @ Vt)))])
    rotation = U @ D @ Vt
    scale = float(np.trace(np.diag(S) @ D) / (centered_r**2).sum() * len(rigid))
    translation = mu_f - scale * rotation @ mu_r
    anchors = scale * rigid @ rotation.T + translation
    return anchors, final - anchors, final


def build_candidates(raw_splats, transform, case_dir: Path, mesh2world):
    """[(name, splats)] for B (rigid self-align) and C2 (+articulation)."""
    import numpy as np
    from scipy.spatial import cKDTree

    from demo_v7.service.gaussian_align import _floater_keep_mask
    from demo_v7.service.gaussian_utils import (
        GaussianSplats,
        transform_gaussians,
    )

    b_splats = transform_gaussians(raw_splats, np.asarray(transform))
    candidates = [("self_align", b_splats)]
    try:
        anchors, articulation, final_verts = pure_articulation_field(
            case_dir, mesh2world
        )
        _dist, nearest = cKDTree(anchors).query(
            b_splats.means.astype(np.float64), k=1, workers=-1
        )
        c2 = GaussianSplats(
            means=(b_splats.means.astype(np.float64) + articulation[nearest])
            .astype(np.float32),
            quats=b_splats.quats,
            scales=b_splats.scales,
            opacities=b_splats.opacities,
            colors=b_splats.colors,
        )
        keep = _floater_keep_mask(c2, final_verts)
        if float((~keep).mean()) <= 0.15 and not keep.all():
            c2 = GaussianSplats(
                means=c2.means[keep],
                quats=c2.quats[keep],
                scales=c2.scales[keep],
                opacities=c2.opacities[keep],
                colors=c2.colors[keep],
            )
        candidates.append(("self_align_art", c2))
    except Exception as exc:
        print(
            f"[gaussian-selfalign] articulation candidate skipped: {exc}",
            flush=True,
        )
    return candidates


def run_self_align_subprocess(
    case_dir: Path, raw_ply: Path, work_dir: Path
) -> tuple | None:
    """Run this module's CLI in a child; (transform 4x4, gates) or None."""
    import numpy as np

    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    result_path = work_dir / "self_align_result.json"
    try:
        completed = subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--case-dir",
                str(case_dir),
                "--raw-ply",
                str(raw_ply),
                "--work-dir",
                str(work_dir),
                "--result-json",
                str(result_path),
            ],
            timeout=_SUBPROCESS_TIMEOUT_S,
            capture_output=True,
            text=True,
        )
    except subprocess.TimeoutExpired:
        print("[gaussian-selfalign] pipeline timed out; keeping the "
              "published alignment", flush=True)
        return None
    if not result_path.is_file():
        tail = (completed.stderr or completed.stdout or "")[-400:]
        print(
            f"[gaussian-selfalign] pipeline produced no result "
            f"(rc={completed.returncode}): {tail}",
            flush=True,
        )
        return None
    result = json.loads(result_path.read_text())
    if "error" in result:
        print(f"[gaussian-selfalign] pipeline error: {result['error']}",
              flush=True)
        return None
    gates = result["gates"]
    if (
        int(gates["inliers"]) < _MIN_INLIERS
        or str(gates["refine_status"]) != "ok"
        or float(gates["refined_iou"]) < _MIN_REFINED_IOU
    ):
        print(f"[gaussian-selfalign] gates rejected the Sim(3): {gates}",
              flush=True)
        return None
    return np.asarray(result["transform"], dtype=np.float64), gates


def _cli(argv: list[str]) -> int:
    """Child entry: run align+refine, write {transform, gates} json."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--case-dir", required=True, type=Path)
    parser.add_argument("--raw-ply", required=True, type=Path)
    parser.add_argument("--work-dir", required=True, type=Path)
    parser.add_argument("--result-json", required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        import numpy as np

        from demo_v7.service.gaussian_utils import (
            load_gaussian_ply,
            save_gaussian_ply,
        )
        from gaussian_align_demo import align_gaussian, refine_alignment

        run_dir = args.work_dir / "run"
        (run_dir / "input").mkdir(parents=True, exist_ok=True)
        finite_ply = run_dir / "input" / "gaussian_finite.ply"
        # The demo loader keeps non-finite rows; filter through ours.
        save_gaussian_ply(finite_ply, load_gaussian_ply(args.raw_ply))
        rc = align_gaussian.main(
            [
                "--case-dir",
                str(args.case_dir),
                "--ply",
                str(finite_ply),
                "--run-dir",
                str(run_dir),
            ]
        )
        if rc != 0:
            raise RuntimeError(f"align_gaussian rc={rc}")
        rc = refine_alignment.main(
            ["--run-dir", str(run_dir), "--case-dir", str(args.case_dir)]
        )
        if rc != 0:
            raise RuntimeError(f"refine_alignment rc={rc}")
        coarse = json.loads(
            (run_dir / "alignment" / "sim3_coarse.json").read_text()
        )
        refined = json.loads(
            (run_dir / "alignment" / "sim3_refined.json").read_text()
        )

        def to44(entry: dict) -> np.ndarray:
            matrix = np.eye(4)
            matrix[:3, :3] = np.asarray(entry["rotation"]) * float(entry["scale"])
            matrix[:3, 3] = np.asarray(entry["translation"])
            return matrix

        transform = to44(refined["delta"]) @ to44(coarse["sim3"])
        payload = {
            "transform": transform.tolist(),
            "gates": {
                "inliers": int(coarse["winner_inliers"]),
                "inlier_rms_m": float(coarse["winner_inlier_rms_m"]),
                "refined_iou": float(
                    refined["metrics_refined_full_res"]["mask_iou"]
                ),
                "refine_status": str(refined["status"]),
            },
        }
    except Exception as exc:  # child never crashes the parent's parse
        payload = {"error": f"{type(exc).__name__}: {exc}"}
    args.result_json.write_text(json.dumps(payload, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(_cli(sys.argv[1:]))
