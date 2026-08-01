"""Stage: Sim(3) refinement of the coarse alignment (derivative-free).

    python -m gaussian_align_demo.refine_alignment \
        --run-dir <run> --case-dir outputs/shape_prior_case/shape_prior_frame0

Refines a 7-DoF delta (axis-angle rotation, translation, log-scale) on top of
alignment/coarse_aligned.ply against the real frame-0 observation, rendering
through gsplat at half then full resolution.

Why Nelder-Mead and not autograd: gsplat's forward here is deterministic and
its analytic gradient is numerically correct, but the rasterized-silhouette
loss surface is rough at the sub-0.1 mm scale — the true infinitesimal
gradient points against the macroscopic slope (verified: a descent step only
reduces the loss for eta ~ 1e-5, and Adam climbed steadily). A simplex whose
edges live at the macroscopic scale (mm / degree) ignores that micro-structure
and monotonically keeps the best pose. 7 DoF x a few hundred deterministic
renders is seconds of work.

Losses (real camera):
- alpha vs object mask (BCE + soft IoU), controller/hand pixels excluded —
  the hand occludes the object, so those pixels are "unknown", not "empty";
- Huber on expected depth vs observed metric depth (object ∩ valid, weighted
  by rendered alpha);
- L1 RGB on the confidently-rendered object region;
- small quadratic prior anchoring the delta at identity.

Keeps whichever of {coarse, refined} scores the better full-res mask IoU.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from scipy.optimize import minimize

from gaussian_align_demo.case_loader import load_frame0_case
from gaussian_align_demo.gs_ply import apply_sim3, load_gaussian_ply, save_gaussian_ply
from gaussian_align_demo.renderer import cloud_to_torch, render_gaussians_torch

# Parameter block: [omega(3) rad, t(3) m, log_scale] in NATURAL units, scaled
# by these sigmas so the Nelder-Mead simplex is well conditioned.
PARAM_SIGMAS = np.array([0.06, 0.06, 0.06, 0.01, 0.01, 0.01, 0.04])


def axis_angle_to_matrix_np(omega: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(omega))
    if angle < 1e-12:
        return np.eye(3)
    axis = omega / angle
    x, y, z = axis
    k = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    return np.eye(3) + np.sin(angle) * k + (1.0 - np.cos(angle)) * (k @ k)


def quat_multiply_torch(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack([
        w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
        w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
        w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
    ], dim=-1)


def scaled_intrinsics(K: np.ndarray, factor: float) -> np.ndarray:
    scaled = np.asarray(K, dtype=np.float64).copy()
    scaled[:2, :] *= factor
    return scaled


def build_targets(case, factor: float, device) -> dict:
    width = int(round(case.width * factor))
    height = int(round(case.height * factor))
    size = (width, height)
    object_mask = cv2.resize(case.object_mask.astype(np.uint8), size,
                             interpolation=cv2.INTER_NEAREST).astype(bool)
    controller = cv2.resize(case.controller_mask.astype(np.uint8), size,
                            interpolation=cv2.INTER_NEAREST).astype(bool)
    depth = cv2.resize(case.depth_m, size, interpolation=cv2.INTER_NEAREST)
    rgb = cv2.resize(case.rgb_u8, size, interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    to = lambda arr, dtype: torch.from_numpy(np.ascontiguousarray(arr)).to(device=device, dtype=dtype)
    object_mask_t = to(object_mask, torch.bool)
    controller_t = to(controller, torch.bool)
    depth_t = to(depth, torch.float32)
    return {
        "width": width,
        "height": height,
        "K": scaled_intrinsics(case.K, factor),
        "object_mask": object_mask_t,
        "known": ~controller_t,
        "object_float": object_mask_t.float(),
        "depth": depth_t,
        "depth_region": object_mask_t & ~controller_t & (depth_t > 0),
        "rgb": to(rgb, torch.float32),
    }


class DeltaSim3Objective:
    """Deterministic render-and-score of a Sim(3) delta on the coarse cloud."""

    def __init__(self, tensors, case, device):
        self.tensors = tensors
        self.device = device
        self.w2c = torch.as_tensor(case.w2c, dtype=torch.float32, device=device)
        self.pivot = tensors["means"].mean(dim=0)
        self.pivot_np = self.pivot.cpu().numpy().astype(np.float64)
        self.targets: dict | None = None
        self.evals = 0

    def set_stage(self, case, factor: float) -> None:
        self.targets = build_targets(case, factor, self.device)

    def render(self, params_natural: np.ndarray):
        omega, translation = params_natural[:3], params_natural[3:6]
        scale = float(np.exp(params_natural[6]))
        rotation = axis_angle_to_matrix_np(omega)
        rotation_t = torch.as_tensor(rotation, dtype=torch.float32, device=self.device)
        translation_t = torch.as_tensor(translation, dtype=torch.float32, device=self.device)
        means = ((self.tensors["means"] - self.pivot) @ rotation_t.T) * scale \
            + self.pivot + translation_t
        from gaussian_align_demo.gs_ply import rotation_matrix_to_quat_wxyz

        delta_quat = torch.as_tensor(rotation_matrix_to_quat_wxyz(rotation),
                                     dtype=torch.float32, device=self.device)
        quats = quat_multiply_torch(delta_quat.expand_as(self.tensors["quats_wxyz"]),
                                    self.tensors["quats_wxyz"])
        targets = self.targets
        return render_gaussians_torch(
            means=means,
            quats_wxyz=quats,
            scales=self.tensors["scales"] * scale,
            opacities=self.tensors["opacities"],
            colors_rgb=self.tensors["colors_rgb"],
            K=torch.as_tensor(targets["K"], dtype=torch.float32, device=self.device),
            w2c=self.w2c,
            width=targets["width"],
            height=targets["height"],
            background_rgb=(0.0, 0.0, 0.0),
        )

    def loss(self, params_scaled: np.ndarray) -> float:
        params = params_scaled * PARAM_SIGMAS
        with torch.no_grad():
            out = self.render(params)
            targets = self.targets
            alpha = out.alpha
            known = targets["known"]
            bce = torch.nn.functional.binary_cross_entropy(
                alpha.clamp(1e-5, 1 - 1e-5)[known], targets["object_float"][known])
            intersection = (alpha * targets["object_float"] * known).sum()
            union = ((alpha + targets["object_float"]) * known).sum() - intersection
            soft_iou = 1.0 - intersection / union.clamp_min(1.0)
            depth_sel = targets["depth_region"]
            alpha_w = alpha[depth_sel]
            depth_huber = (torch.nn.functional.huber_loss(
                out.depth[depth_sel], targets["depth"][depth_sel], delta=0.02,
                reduction="none") * alpha_w).sum() / alpha_w.sum().clamp_min(1.0)
            rgb_sel = targets["object_mask"] & known & (alpha > 0.5)
            rgb_l1 = ((out.rgb - targets["rgb"]).abs().mean(dim=-1)[rgb_sel].mean()
                      if rgb_sel.any() else torch.zeros((), device=self.device))
            # Depth is the metric anchor: with a weak depth term the optimizer
            # happily tilts the object off the table to fill the 2D silhouette
            # (observed: IoU +0.11 while depth error went 9 mm -> 51 mm).
            total = 0.5 * (bce + soft_iou) + 50.0 * depth_huber + 0.2 * rgb_l1
        prior = float((params_scaled**2).sum())
        self.evals += 1
        return float(total.item() + 0.02 * prior)

    def metrics(self, params_natural: np.ndarray) -> dict:
        with torch.no_grad():
            out = self.render(params_natural)
            targets = self.targets
            alpha, depth, rgb = out.alpha, out.depth, out.rgb
            known = targets["known"]
            pred = (alpha > 0.5) & known
            gt = targets["object_mask"] & known
            union = (pred | gt).sum().item()
            iou = (pred & gt).sum().item() / union if union else 0.0
            depth_sel = targets["depth_region"] & (alpha > 0.5)
            if depth_sel.any():
                err = (depth[depth_sel] - targets["depth"][depth_sel]).abs()
                depth_median_mm = err.median().item() * 1000.0
                depth_p90_mm = err.quantile(0.9).item() * 1000.0
            else:
                depth_median_mm = depth_p90_mm = float("nan")
            rgb_sel = gt & (alpha > 0.5)
            rgb_l1 = ((rgb[rgb_sel] - targets["rgb"][rgb_sel]).abs().mean().item()
                      if rgb_sel.any() else float("nan"))
        return {"mask_iou": round(iou, 4), "depth_median_mm": round(depth_median_mm, 2),
                "depth_p90_mm": round(depth_p90_mm, 2), "rgb_l1": round(rgb_l1, 4)}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--case-dir", required=True)
    parser.add_argument("--maxfev-half", type=int, default=400)
    parser.add_argument("--maxfev-full", type=int, default=200)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)

    device = torch.device(args.device)
    align_dir = Path(args.run_dir) / "alignment"
    coarse_path = align_dir / "coarse_aligned.ply"
    cloud = load_gaussian_ply(coarse_path)
    case = load_frame0_case(args.case_dir)
    objective = DeltaSim3Objective(cloud_to_torch(cloud, device=device), case, device)

    history_path = align_dir / "refinement_history.jsonl"
    history = open(history_path, "w")
    x = np.zeros(7)
    for stage_name, factor, maxfev in (
        ("half", 0.5, args.maxfev_half), ("full", 1.0, args.maxfev_full)
    ):
        objective.set_stage(case, factor)
        if stage_name == "half":
            metrics_before = objective.metrics(np.zeros(7))
            loss_before = objective.loss(np.zeros(7))
            print(f"[refine] coarse @half-res: loss {loss_before:.4f}, {metrics_before}")
        # scipy's default simplex around x0=0 is microscopic (2.5e-4), which
        # sits inside the rasterization micro-roughness; seed a macroscopic
        # one instead (0.5 sigma per vertex ~ 3 deg / 5 mm / 2% scale).
        simplex = np.zeros((8, 7))
        simplex[0] = x
        step = 0.5 if stage_name == "half" else 0.15
        for i in range(7):
            simplex[i + 1] = x
            simplex[i + 1, i] += step
        result = minimize(
            objective.loss, x, method="Nelder-Mead",
            options={"maxfev": maxfev, "xatol": 0.02, "fatol": 1e-5,
                     "initial_simplex": simplex},
        )
        x = result.x
        params = x * PARAM_SIGMAS
        record = {
            "stage": stage_name, "loss": float(result.fun), "evals": objective.evals,
            "rot_deg": round(float(np.linalg.norm(params[:3])) * 57.2958, 3),
            "translation_mm": [round(float(v) * 1000, 2) for v in params[3:6]],
            "scale": round(float(np.exp(params[6])), 5),
            "metrics": objective.metrics(params),
        }
        history.write(json.dumps(record) + "\n")
        print(f"[refine] {record}")
    history.close()

    params = x * PARAM_SIGMAS
    objective.set_stage(case, 1.0)
    metrics_before_full = objective.metrics(np.zeros(7))
    metrics_after_full = objective.metrics(params)
    print(f"[refine] full-res: coarse {metrics_before_full} -> refined {metrics_after_full}")

    rotation = axis_angle_to_matrix_np(params[:3])
    scale = float(np.exp(params[6]))
    pivot = objective.pivot_np
    effective_translation = pivot + params[3:6] - scale * rotation @ pivot

    # Accept only a genuine polish: silhouette must not regress AND the metric
    # depth must stay honest AND the delta must stay inside the trust region.
    depth_budget_mm = max(metrics_before_full["depth_median_mm"] * 1.15,
                          metrics_before_full["depth_median_mm"] + 2.0)
    improved = (
        metrics_after_full["mask_iou"] >= metrics_before_full["mask_iou"] - 0.005
        and metrics_after_full["depth_median_mm"] <= depth_budget_mm
        and float(np.linalg.norm(x)) <= 3.0
    )
    payload = {
        "status": "ok" if improved else "kept_coarse",
        "delta": {
            "rotation": rotation.tolist(),
            "translation": effective_translation.tolist(),
            "scale": scale,
            "rot_deg": float(np.linalg.norm(params[:3])) * 57.2958,
        },
        "metrics_coarse_full_res": metrics_before_full,
        "metrics_refined_full_res": metrics_after_full,
    }
    (align_dir / "sim3_refined.json").write_text(json.dumps(payload, indent=2))

    if improved:
        refined = apply_sim3(cloud, rotation=rotation, translation=effective_translation,
                             scale=scale)
        save_gaussian_ply(refined, align_dir / "refined_aligned.ply")
        source = refined
        print(f"[refine] wrote refined_aligned.ply (IoU {metrics_before_full['mask_iou']:.3f}"
              f" -> {metrics_after_full['mask_iou']:.3f}, depth median "
              f"{metrics_after_full['depth_median_mm']:.1f} mm)")
    else:
        source = cloud
        print("[refine] no improvement over coarse; keeping coarse_aligned.ply")

    from gaussian_align_demo.align_gaussian import render_real_camera_overlay
    overlay = render_real_camera_overlay(cloud_to_torch(source, device=device), case)
    cv2.imwrite(str(align_dir / "refined_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
