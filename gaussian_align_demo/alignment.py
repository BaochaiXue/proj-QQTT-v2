"""Coarse alignment core: candidate views + SuperGlue + RANSAC-Umeyama Sim(3).

Pipeline (align_gaussian.py drives this):

1. Render the canonical TripoSplat gaussian from an orbit of candidate poses
   (azimuth x elevation x in-plane roll) on black, grayscale for matching.
2. SuperGlue-match every candidate against the masked/cropped real frame-0
   image, reusing demo_v6_2's vendored matcher (read-only import).
3. For the top-K candidates: back-project matched candidate keypoints through
   the rendered expected-depth into canonical gaussian 3D, look up the real
   keypoints' metric world 3D in the case PCD, and estimate a Sim(3) with
   RANSAC + Umeyama. Both sides of the correspondence are 3D, so unlike
   demo_v6_2's PnP+scale two-step this is a single closed-form similarity fit;
   PnP-style reprojection error is only reported as a diagnostic.

Caution: importing demo_v6_2.shape_prior.match_pairs disables autograd
globally (module-level ``torch.set_grad_enabled(False)``). Alignment never
needs gradients; the differentiable refinement stage must NOT import this
module (and re-enables grad explicitly anyway).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np
import torch

from gaussian_align_demo.cameras import (
    intrinsics_for_fov,
    project_points,
    sample_orbit_w2c,
    unproject_pixels,
)
from gaussian_align_demo.gs_ply import GaussianCloud
from gaussian_align_demo.renderer import render_cloud

RENDER_SIZE = 512
RENDER_FOV_X_DEG = 45.0


# ---------------------------------------------------------------------------
# Canonical-cloud framing
# ---------------------------------------------------------------------------


def robust_center_extent(
    cloud: GaussianCloud, *, opacity_threshold: float = 0.3, percentile: float = 1.0
) -> tuple[np.ndarray, np.ndarray]:
    """Opacity-gated percentile bbox — floaters must not blow up the orbit."""
    solid = cloud.means[cloud.opacities > opacity_threshold]
    if len(solid) < 100:
        solid = cloud.means
    low = np.percentile(solid, percentile, axis=0)
    high = np.percentile(solid, 100.0 - percentile, axis=0)
    return (low + high) / 2.0, high - low


def orbit_radius_for(extent: np.ndarray, *, fov_x_deg: float = RENDER_FOV_X_DEG) -> float:
    """Radius at which the object's diagonal fills ~70% of the image width."""
    diagonal = float(np.linalg.norm(extent))
    return (diagonal / 2.0) / (0.7 * np.tan(np.deg2rad(fov_x_deg) / 2.0))


# ---------------------------------------------------------------------------
# Real-image reference crop
# ---------------------------------------------------------------------------


@dataclass
class ReferenceCrop:
    """Masked real image cropped square around the object, resized to size px.

    ``to_full_image(uv_crop)`` maps crop pixels back to full-image pixels.
    """

    image_gray: np.ndarray  # (size, size) uint8
    image_rgb: np.ndarray  # (size, size, 3) uint8 (for viz)
    origin_xy: np.ndarray  # (2,) crop top-left in full-image pixels
    crop_side: float  # side length in full-image pixels
    size: int

    def to_full_image(self, uv_crop: np.ndarray) -> np.ndarray:
        return np.asarray(uv_crop, dtype=np.float64) * (self.crop_side / self.size) + self.origin_xy


def build_reference_crop(
    rgb_u8: np.ndarray, object_mask: np.ndarray, *, size: int = RENDER_SIZE, margin: float = 1.3
) -> ReferenceCrop:
    ys, xs = np.nonzero(object_mask)
    if len(xs) == 0:
        raise ValueError("object mask is empty")
    cx, cy = (xs.min() + xs.max()) / 2.0, (ys.min() + ys.max()) / 2.0
    side = max(xs.max() - xs.min(), ys.max() - ys.min()) * margin
    origin = np.array([cx - side / 2.0, cy - side / 2.0])

    masked = np.where(object_mask[..., None], rgb_u8, 0).astype(np.uint8)
    # Sample the crop with a single affine warp (handles out-of-bounds as black).
    scale = size / side
    warp = np.array(
        [[scale, 0.0, -origin[0] * scale], [0.0, scale, -origin[1] * scale]], dtype=np.float64
    )
    crop_rgb = cv2.warpAffine(masked, warp, (size, size), flags=cv2.INTER_LINEAR)
    return ReferenceCrop(
        image_gray=cv2.cvtColor(crop_rgb, cv2.COLOR_RGB2GRAY),
        image_rgb=crop_rgb,
        origin_xy=origin,
        crop_side=side,
        size=size,
    )


# ---------------------------------------------------------------------------
# Candidate rendering
# ---------------------------------------------------------------------------


@dataclass
class CandidateSet:
    poses_w2c: list[np.ndarray]
    K: np.ndarray
    size: int
    gray_paths: list[Path]


def build_candidate_poses(
    cloud: GaussianCloud,
    *,
    n_azimuth: int = 12,
    elevations_deg: tuple[float, ...] = (-60.0, -30.0, 0.0, 30.0, 60.0),
    roll_angles_deg: tuple[float, ...] = (0.0, 90.0, 180.0, 270.0),
) -> tuple[list[np.ndarray], np.ndarray, dict]:
    center, extent = robust_center_extent(cloud)
    radius = orbit_radius_for(extent)
    poses = sample_orbit_w2c(
        center=center,
        radius=radius,
        n_azimuth=n_azimuth,
        elevations_deg=elevations_deg,
        roll_angles_deg=roll_angles_deg,
    )
    K = intrinsics_for_fov(width=RENDER_SIZE, height=RENDER_SIZE, fov_x_deg=RENDER_FOV_X_DEG)
    info = {
        "center": center.tolist(),
        "extent": extent.tolist(),
        "radius": float(radius),
        "n_candidates": len(poses),
        "n_azimuth": n_azimuth,
        "elevations_deg": list(elevations_deg),
        "roll_angles_deg": list(roll_angles_deg),
    }
    return poses, K, info


def render_candidate_grays(
    cloud_tensors: dict[str, torch.Tensor],
    poses_w2c: list[np.ndarray],
    K: np.ndarray,
    output_dir: Path,
    *,
    size: int = RENDER_SIZE,
) -> CandidateSet:
    output_dir.mkdir(parents=True, exist_ok=True)
    gray_paths: list[Path] = []
    for idx, w2c in enumerate(poses_w2c):
        out = render_cloud(
            cloud_tensors, K=K, w2c=w2c, width=size, height=size, background_rgb=(0.0, 0.0, 0.0)
        )
        gray = cv2.cvtColor(out.rgb_u8(), cv2.COLOR_RGB2GRAY)
        path = output_dir / f"candidate_{idx:04d}.png"
        cv2.imwrite(str(path), gray)
        gray_paths.append(path)
    return CandidateSet(poses_w2c=list(poses_w2c), K=K, size=size, gray_paths=gray_paths)


# ---------------------------------------------------------------------------
# SuperGlue matching (demo_v6_2 vendored matcher, all-candidate scores)
# ---------------------------------------------------------------------------


@dataclass
class CandidateMatch:
    index: int
    kpts_candidate: np.ndarray  # (M, 2) pixels in the candidate render
    kpts_reference: np.ndarray  # (M, 2) pixels in the reference crop
    confidence: np.ndarray  # (M,)

    @property
    def num_matches(self) -> int:
        return int(len(self.kpts_candidate))


def match_all_candidates(
    candidate_paths: list[Path],
    reference_path: Path,
    *,
    device: str = "cuda",
    match_confidence_min: float = 0.3,
) -> list[CandidateMatch]:
    """SuperGlue every candidate against the reference; keep per-candidate data.

    demo_v6_2's image_pair_matching only returns the single best pair, so this
    reimplements its loop on top of the same cached model + feature helpers to
    expose all candidates (needed for top-K Sim(3) fitting).
    """
    from demo_v6_2.models.utils import read_image
    from demo_v6_2.shape_prior.match_pairs import (
        extract_superpoint_features,
        get_matching_model,
    )

    matching = get_matching_model(
        nms_radius=4,
        keypoint_threshold=0.005,
        max_keypoints=1024,
        superglue="indoor",
        sinkhorn_iterations=20,
        match_threshold=0.2,
        device=device,
    )

    def load_gray(path: Path) -> np.ndarray:
        # demo_v6_2's read_image takes an already-loaded grayscale array.
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"failed to read image {path}")
        return gray

    _, ref_tensor, _ = read_image(load_gray(reference_path), device, [-1], 0, False)
    ref_features = extract_superpoint_features(matching, ref_tensor)
    ref_kpts = ref_features["keypoints"][0].cpu().numpy()

    results: list[CandidateMatch] = []
    for idx, path in enumerate(candidate_paths):
        _, cand_tensor, _ = read_image(load_gray(path), device, [-1], 0, False)
        cand_features = extract_superpoint_features(matching, cand_tensor)
        data = {"image0": cand_tensor, "image1": ref_tensor}
        data.update({k + "0": v for k, v in cand_features.items()})
        data.update({k + "1": v for k, v in ref_features.items()})
        pred = matching(data)
        matches = pred["matches0"][0].cpu().numpy()
        confidence = pred["matching_scores0"][0].cpu().numpy()
        kpts_cand = cand_features["keypoints"][0].cpu().numpy()
        valid = (matches > -1) & (confidence >= match_confidence_min)
        results.append(
            CandidateMatch(
                index=idx,
                kpts_candidate=kpts_cand[valid],
                kpts_reference=ref_kpts[matches[valid]],
                confidence=confidence[valid],
            )
        )
    return results


# ---------------------------------------------------------------------------
# 3D-3D correspondences
# ---------------------------------------------------------------------------


@dataclass
class Correspondences:
    points_canonical: np.ndarray  # (M, 3) gaussian canonical frame
    points_world: np.ndarray  # (M, 3) metric world frame
    pixels_full_image: np.ndarray  # (M, 2) real-image pixels (diagnostics)
    dropped: dict = field(default_factory=dict)


def _snap_to_valid(
    uv: np.ndarray, valid_mask: np.ndarray, *, max_radius_px: int = 6
) -> tuple[int, int] | None:
    """Nearest valid pixel within a small window (align_util.select_point,
    but bounded so a keypoint can't snap to a far-away surface)."""
    height, width = valid_mask.shape
    x, y = int(round(uv[0])), int(round(uv[1]))
    if not (0 <= x < width and 0 <= y < height):
        return None
    if valid_mask[y, x]:
        return x, y
    r = max_radius_px
    y0, y1 = max(0, y - r), min(height, y + r + 1)
    x0, x1 = max(0, x - r), min(width, x + r + 1)
    window = valid_mask[y0:y1, x0:x1]
    if not window.any():
        return None
    wy, wx = np.nonzero(window)
    d2 = (wy + y0 - y) ** 2 + (wx + x0 - x) ** 2
    best = int(np.argmin(d2))
    return int(wx[best] + x0), int(wy[best] + y0)


def build_correspondences(
    match: CandidateMatch,
    *,
    candidate_depth: np.ndarray,  # (size, size) expected depth of the render
    candidate_alpha: np.ndarray,  # (size, size)
    candidate_K: np.ndarray,
    candidate_w2c: np.ndarray,
    reference: ReferenceCrop,
    case_points_world: np.ndarray,  # (H, W, 3)
    case_valid: np.ndarray,  # (H, W) bool: object mask ∩ depth valid
    alpha_min: float = 0.6,
) -> Correspondences:
    canonical: list[np.ndarray] = []
    world: list[np.ndarray] = []
    pixels: list[np.ndarray] = []
    dropped = {"alpha_or_depth": 0, "no_valid_world": 0, "out_of_bounds": 0}

    size = candidate_depth.shape[0]
    for uv_cand, uv_ref in zip(match.kpts_candidate, match.kpts_reference):
        x, y = int(round(uv_cand[0])), int(round(uv_cand[1]))
        if not (0 <= x < size and 0 <= y < size):
            dropped["out_of_bounds"] += 1
            continue
        depth = float(candidate_depth[y, x])
        if candidate_alpha[y, x] < alpha_min or not np.isfinite(depth) or depth <= 0.0:
            dropped["alpha_or_depth"] += 1
            continue
        p_canonical = unproject_pixels(
            np.array([[uv_cand[0], uv_cand[1]]]), np.array([depth]), candidate_K, candidate_w2c
        )[0]

        uv_full = reference.to_full_image(uv_ref)
        snapped = _snap_to_valid(uv_full, case_valid)
        if snapped is None:
            dropped["no_valid_world"] += 1
            continue
        canonical.append(p_canonical)
        world.append(case_points_world[snapped[1], snapped[0]])
        pixels.append(uv_full)

    return Correspondences(
        points_canonical=np.asarray(canonical, dtype=np.float64).reshape(-1, 3),
        points_world=np.asarray(world, dtype=np.float64).reshape(-1, 3),
        pixels_full_image=np.asarray(pixels, dtype=np.float64).reshape(-1, 2),
        dropped=dropped,
    )


# ---------------------------------------------------------------------------
# Sim(3) estimation
# ---------------------------------------------------------------------------


@dataclass
class Sim3:
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    scale: float

    def apply(self, points: np.ndarray) -> np.ndarray:
        return (np.asarray(points) @ self.rotation.T) * self.scale + self.translation

    def to_dict(self) -> dict:
        return {
            "rotation": self.rotation.tolist(),
            "translation": self.translation.tolist(),
            "scale": float(self.scale),
        }


def umeyama_sim3(src: np.ndarray, dst: np.ndarray) -> Sim3:
    """Closed-form similarity aligning src -> dst (Umeyama 1991, with scale)."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if src.shape != dst.shape or src.shape[0] < 3:
        raise ValueError(f"need >= 3 paired points, got {src.shape} vs {dst.shape}")
    mu_src, mu_dst = src.mean(axis=0), dst.mean(axis=0)
    x, y = src - mu_src, dst - mu_dst
    covariance = y.T @ x / src.shape[0]
    u, singular, vt = np.linalg.svd(covariance)
    sign = np.eye(3)
    if np.linalg.det(u) * np.linalg.det(vt) < 0:
        sign[2, 2] = -1.0
    rotation = u @ sign @ vt
    var_src = float((x**2).sum() / src.shape[0])
    if var_src < 1e-18:
        raise ValueError("degenerate source points (zero variance)")
    scale = float(np.trace(np.diag(singular) @ sign) / var_src)
    if scale <= 0:
        raise ValueError(f"non-positive similarity scale {scale}")
    translation = mu_dst - scale * rotation @ mu_src
    return Sim3(rotation=rotation, translation=translation, scale=scale)


@dataclass
class RansacResult:
    sim3: Sim3
    inlier_mask: np.ndarray
    inlier_rms_m: float

    @property
    def num_inliers(self) -> int:
        return int(self.inlier_mask.sum())


def ransac_umeyama(
    src: np.ndarray,
    dst: np.ndarray,
    *,
    threshold_m: float = 0.015,
    iterations: int = 2000,
    seed: int = 0,
    min_inliers: int = 6,
) -> RansacResult | None:
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    n = src.shape[0]
    if n < max(3, min_inliers):
        return None
    rng = np.random.default_rng(seed)
    best_mask: np.ndarray | None = None
    for _ in range(iterations):
        sample = rng.choice(n, size=3, replace=False)
        try:
            model = umeyama_sim3(src[sample], dst[sample])
        except (ValueError, np.linalg.LinAlgError):
            continue
        residual = np.linalg.norm(model.apply(src) - dst, axis=1)
        mask = residual < threshold_m
        if best_mask is None or mask.sum() > best_mask.sum():
            best_mask = mask
    if best_mask is None or best_mask.sum() < min_inliers:
        return None
    # Two refit rounds on inliers (standard local optimization).
    mask = best_mask
    model = None
    for _ in range(2):
        try:
            model = umeyama_sim3(src[mask], dst[mask])
        except (ValueError, np.linalg.LinAlgError):
            return None
        residual = np.linalg.norm(model.apply(src) - dst, axis=1)
        new_mask = residual < threshold_m
        if new_mask.sum() < min_inliers:
            break
        mask = new_mask
    if model is None:
        return None
    inlier_res = np.linalg.norm(model.apply(src[mask]) - dst[mask], axis=1)
    return RansacResult(
        sim3=model, inlier_mask=mask, inlier_rms_m=float(np.sqrt((inlier_res**2).mean()))
    )


def reprojection_error_px(
    sim3: Sim3,
    points_canonical: np.ndarray,
    pixels_full_image: np.ndarray,
    K: np.ndarray,
    w2c: np.ndarray,
) -> float:
    """Median pixel error of aligned canonical points vs their real-image pixels."""
    aligned = sim3.apply(points_canonical)
    projected, depth = project_points(aligned, K, w2c)
    valid = depth > 0
    if not valid.any():
        return float("inf")
    err = np.linalg.norm(projected[valid] - pixels_full_image[valid], axis=1)
    return float(np.median(err))
