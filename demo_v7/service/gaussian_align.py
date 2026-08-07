"""Align a TripoSplat gaussian into the run's world frame.

Chain (all verified empirically on real artifacts, see exec plan):

1. canonical registration — TripoSplat's canonical frame (a [0,1]^3-ish
   unit box, axis conventions unknown by contract; even SAM3D's own ply is
   rotated 90 deg against its own glb) -> the mesh backend's canonical
   frame (object.glb, [-0.5,0.5]^3-ish). Coarse: centroid/RMS normalize +
   best of the 24 axis-aligned rotations by symmetric chamfer; refine:
   scaled ICP (open3d, with_scaling=True).
2. mesh2world — align.py's canonical->world similarity (PnP pose + uniform
   scale + c2w). Never persisted, but exactly recomputable from persisted
   case files; we reuse the v6.2 visualization replay (read-only import).
3. ARAP residual — final_mesh.glb is vertex-index-aligned with object.glb,
   so the per-vertex world displacement (final - mesh2world @ canonical)
   transfers to each gaussian from its nearest canonical mesh vertex.

Output: world-frame splats + a provenance dict with every matrix and the
registration quality numbers (the GUI shows the overlay still; the numbers
land in the artifacts json so a bad registration is visible, not silent).
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demo_v7.service.gaussian_utils import (
    GaussianSplats,
    transform_gaussians,
)

# Coarse-stage sizes: chamfer on subsampled clouds is plenty to rank 24
# axis-aligned candidates; ICP then works on the full subsample.
_COARSE_POINTS = 4000
_ICP_POINTS = 20000
_OPACITY_KEEP = 0.3
# The coarse chamfer is a weak ranker for near-symmetric objects (measured
# on the sloth: the true rotation lost the coarse ranking to a 172.5-deg
# flip by 0.7%, and refining only the winner locked the misregistration
# in). Refine every candidate within this margin of the coarse winner
# (capped) and keep the best REFINED symmetric chamfer.
_REFINE_MARGIN = 1.3
_REFINE_MAX_CANDIDATES = 5
# Refined symmetric chamfer above this (mesh-canonical units, object spans
# ~1.0) means the registration is likely wrong (good runs measure ~0.02,
# the known-bad flip measured 0.066): flag it loudly instead of silently
# shipping ghost limbs.
_CHAMFER_SUSPECT = 0.045


@dataclass
class GaussianAlignment:
    canonical_reg: np.ndarray  # 4x4 ply-canonical -> mesh-canonical (similarity)
    mesh2world: np.ndarray  # 4x4 mesh-canonical -> world (similarity)
    composed: np.ndarray  # 4x4 ply-canonical -> world (pre-ARAP)
    chamfer_after_m: float  # symmetric chamfer in mesh-canonical units
    arap_residual_mean_m: float
    registration_suspect: bool = False

    def provenance(self) -> dict:
        return {
            "canonical_reg": self.canonical_reg.tolist(),
            "mesh2world": self.mesh2world.tolist(),
            "composed": self.composed.tolist(),
            "chamfer_after_m": self.chamfer_after_m,
            "arap_residual_mean_m": self.arap_residual_mean_m,
            "registration_suspect": self.registration_suspect,
        }


def _axis_rotations() -> list[np.ndarray]:
    """The 24 proper axis-aligned rotation matrices."""
    rotations = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1.0, -1.0), repeat=3):
            mat = np.zeros((3, 3))
            for row, (col, sign) in enumerate(zip(perm, signs)):
                mat[row, col] = sign
            if np.isclose(np.linalg.det(mat), 1.0):
                rotations.append(mat)
    return rotations


def _subsample(points: np.ndarray, count: int, seed: int = 0) -> np.ndarray:
    if len(points) <= count:
        return points
    rng = np.random.default_rng(seed)
    return points[rng.choice(len(points), size=count, replace=False)]


def _chamfer(a: np.ndarray, b: np.ndarray) -> float:
    from scipy.spatial import cKDTree

    d_ab, _ = cKDTree(b).query(a, k=1, workers=-1)
    d_ba, _ = cKDTree(a).query(b, k=1, workers=-1)
    return float(d_ab.mean() + d_ba.mean()) / 2.0


def register_canonical(
    gaussian_means: np.ndarray,
    gaussian_opacities: np.ndarray,
    mesh_surface_points: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Similarity transform ply-canonical -> mesh-canonical.

    Returns (4x4, symmetric chamfer after refinement). Both inputs are
    complete single objects generated from the same image, so a global
    similarity is well-posed; opacity-filtering drops floater splats that
    would bias the correspondence.
    """
    import open3d as o3d

    solid = gaussian_means[gaussian_opacities > _OPACITY_KEEP]
    if len(solid) < 100:
        solid = gaussian_means
    src_center = solid.mean(axis=0)
    dst_center = mesh_surface_points.mean(axis=0)
    # Robust size ratio: percentile extents are stable against floater
    # splats and interior mass (an RMS ratio underestimates the scale for a
    # volume-filling gaussian vs a surface sample — measured: point-to-point
    # scaled ICP then locks into a shrunken local minimum).
    src_extent = np.percentile(solid, 97, axis=0) - np.percentile(solid, 3, axis=0)
    dst_extent = np.percentile(mesh_surface_points, 97, axis=0) - np.percentile(
        mesh_surface_points, 3, axis=0
    )
    src_size = float(np.linalg.norm(src_extent))
    dst_size = float(np.linalg.norm(dst_extent))
    base_scale = dst_size / max(src_size, 1e-12)

    src_coarse = _subsample(solid, _COARSE_POINTS)
    dst_coarse = _subsample(mesh_surface_points, _COARSE_POINTS)
    ranked: list[tuple[float, np.ndarray]] = []
    for rotation in _axis_rotations():
        candidate = (src_coarse - src_center) @ rotation.T * base_scale + dst_center
        ranked.append((_chamfer(candidate, dst_coarse), rotation))
    ranked.sort(key=lambda item: item[0])
    # Near-symmetric objects rank near-flips within a hair of the truth at
    # the coarse stage; only the REFINED chamfer separates them.
    rotation_candidates = [
        rotation
        for cost, rotation in ranked[:_REFINE_MAX_CANDIDATES]
        if cost <= ranked[0][0] * _REFINE_MARGIN
    ]

    src_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(_subsample(solid, _ICP_POINTS).astype(np.float64))
    )
    dst_cloud = o3d.geometry.PointCloud(
        o3d.utility.Vector3dVector(
            _subsample(mesh_surface_points, _ICP_POINTS).astype(np.float64)
        )
    )
    extent = float(np.linalg.norm(np.ptp(mesh_surface_points, axis=0)))
    estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=True
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)

    def _run_icp(init: np.ndarray) -> np.ndarray:
        result = init
        for threshold in (0.10 * extent, 0.03 * extent):
            icp = o3d.pipelines.registration.registration_icp(
                src_cloud, dst_cloud, threshold, result, estimation, criteria
            )
            result = np.asarray(icp.transformation)
        return result

    def _symmetric_chamfer_of(transform: np.ndarray) -> float:
        moved = src_coarse @ transform[:3, :3].T + transform[:3, 3]
        return _chamfer(moved, dst_coarse)

    # Scale POLICY: pin to the robust extent ratio. Both objects come from
    # the same image, so their true relative scale is the size ratio; a
    # scaled ICP instead shrinks the similarity to average out genuine
    # generation-pose differences (measured on the sloth: arms posed
    # differently -> scaled ICP drifts to 0.80x and the world overlay is a
    # miniature). Rigid ICP refines R,t on a pre-scaled source; a scaled-ICP
    # candidate is accepted only when it stays within 10% of the pinned
    # scale AND beats the rigid result's symmetric chamfer.
    prescale = np.eye(4)
    prescale[:3, :3] *= base_scale
    scaled_src = o3d.geometry.PointCloud(src_cloud)
    scaled_src.scale(base_scale, center=(0.0, 0.0, 0.0))
    rigid = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=False
    )
    best_transform, best_metric = None, np.inf
    for rotation in rotation_candidates:
        rigid_init = np.eye(4)
        rigid_init[:3, :3] = rotation
        rigid_init[:3, 3] = dst_center - rotation @ (src_center * base_scale)
        result = rigid_init
        for threshold in (0.10 * extent, 0.03 * extent):
            icp = o3d.pipelines.registration.registration_icp(
                scaled_src, dst_cloud, threshold, result, rigid, criteria
            )
            result = np.asarray(icp.transformation)
        transform = result @ prescale
        metric = _symmetric_chamfer_of(transform)
        if metric < best_metric:
            best_metric, best_transform = metric, transform

    # Scaled-ICP acceptance pass, seeded from the refined rigid pose (it
    # is already a similarity: rigid ∘ prescale) rather than the coarse
    # basin it descended from.
    candidate = _run_icp(best_transform)
    icp_scale = float(np.cbrt(abs(np.linalg.det(candidate[:3, :3]))))
    if 0.9 * base_scale <= icp_scale <= 1.1 * base_scale:
        metric = _symmetric_chamfer_of(candidate)
        if metric < best_metric:
            best_metric, best_transform = metric, candidate

    return best_transform, best_metric


def recompute_mesh2world(case_dir: Path, controller_name: str) -> np.ndarray:
    """Replay align.py's PnP + scale from persisted case files (v6.2 code)."""
    from demo_v6_2.visualization.visualize_shape_prior_matches import (
        load_case,
        replay_match_stages,
    )

    case = load_case(Path(case_dir), controller_name)
    return np.asarray(replay_match_stages(case).mesh2world, dtype=np.float64)


def arap_residual_field(
    case_dir: Path, mesh2world: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """(canonical object.glb verts, per-vertex world ARAP displacement).

    final_mesh.glb shares object.glb's vertex order (align.py writes the
    deformed positions back through trimesh_indices), so the displacement
    field is a direct subtraction.
    """
    import trimesh

    shape_dir = Path(case_dir) / "shape"
    canonical = trimesh.load(shape_dir / "object.glb", force="mesh", process=False)
    final = trimesh.load(
        shape_dir / "matching" / "final_mesh.glb", force="mesh", process=False
    )
    canonical_verts = np.asarray(canonical.vertices, dtype=np.float64)
    final_verts = np.asarray(final.vertices, dtype=np.float64)
    rigid_world = canonical_verts @ mesh2world[:3, :3].T + mesh2world[:3, 3]
    if len(canonical_verts) == len(final_verts):
        displacement = final_verts - rigid_world
    else:
        # Index alignment broken (e.g. a cleanup rewrote final_mesh):
        # fall back to nearest-neighbor displacement in world space — an
        # approximation good to the ARAP deformation's local scale (~2 cm),
        # fine for a display-only refinement.
        from scipy.spatial import cKDTree

        _, nearest = cKDTree(final_verts).query(rigid_world, k=1, workers=-1)
        displacement = final_verts[nearest] - rigid_world
    return canonical_verts, displacement


def align_gaussian_to_world(
    splats: GaussianSplats,
    case_dir: str | Path,
    controller_name: str,
    *,
    mesh_surface_samples: int = 60000,
) -> tuple[GaussianSplats, GaussianAlignment]:
    """Full chain: canonical registration -> mesh2world -> ARAP residual."""
    import trimesh

    case_dir = Path(case_dir)
    mesh = trimesh.load(
        case_dir / "shape" / "object.glb", force="mesh", process=False
    )
    surface_points, _ = trimesh.sample.sample_surface(mesh, mesh_surface_samples)
    surface_points = np.asarray(surface_points, dtype=np.float64)

    canonical_reg, chamfer_after = register_canonical(
        splats.means.astype(np.float64), splats.opacities, surface_points
    )
    mesh2world = recompute_mesh2world(case_dir, controller_name)
    composed = mesh2world @ canonical_reg
    world = transform_gaussians(splats, composed)

    canonical_verts, displacement = arap_residual_field(case_dir, mesh2world)
    # Each gaussian inherits the ARAP displacement of its nearest canonical
    # mesh vertex (measured in mesh-canonical frame, applied in world).
    from scipy.spatial import cKDTree

    gaussians_canonical = (
        splats.means.astype(np.float64) @ canonical_reg[:3, :3].T
        + canonical_reg[:3, 3]
    )
    _, nearest = cKDTree(canonical_verts).query(gaussians_canonical, k=1, workers=-1)
    world.means = (world.means.astype(np.float64) + displacement[nearest]).astype(
        np.float32
    )

    suspect = chamfer_after > _CHAMFER_SUSPECT
    if suspect:
        print(
            f"[gaussian-align] WARNING: canonical registration chamfer "
            f"{chamfer_after:.4f} exceeds {_CHAMFER_SUSPECT} (canonical "
            "units) — the world gaussian is likely misrotated; check "
            "gaussian_world_overlay.png",
            flush=True,
        )
    alignment = GaussianAlignment(
        canonical_reg=canonical_reg,
        mesh2world=mesh2world,
        composed=composed,
        chamfer_after_m=chamfer_after,
        arap_residual_mean_m=float(np.linalg.norm(displacement, axis=1).mean()),
        registration_suspect=suspect,
    )
    return world, alignment


def rigid_world_catchup(
    means: np.ndarray,
    opacities: np.ndarray,
    target_points: np.ndarray,
    *,
    max_translation_m: float = 0.30,
    max_rotation_deg: float = 60.0,
) -> tuple[np.ndarray | None, dict]:
    """Rigid ICP snapping world splats onto the FORMAL frame-0 object cloud.

    The world gaussian is registered to the CAPTURE frame-0 pose, but the
    tracker's bones live on the object's FORMAL frame-0 pose — anything the
    object did in between (the whole warmup, ~60s+) is invisible to the
    tracker. This closes the rigid part of that gap. Articulated pose
    changes across the gap are not recoverable (no correspondence exists);
    REPOSITION bounds them in real-live runs.

    Returns ``(4x4 world->world transform | None, info)``; None means the
    correction was rejected (ICP made things worse or moved implausibly
    far — likely a degenerate cloud) and the caller keeps the identity.
    """
    import open3d as o3d
    from scipy.spatial import cKDTree

    info: dict = {}
    solid = means[opacities > _OPACITY_KEEP]
    if len(solid) < 100:
        solid = means
    src = _subsample(solid.astype(np.float64), _ICP_POINTS)
    dst = np.asarray(target_points, dtype=np.float64)
    dst = dst[np.isfinite(dst).all(axis=1)]
    if len(dst) < 100:
        info["rejected"] = "target cloud too small"
        return None, info
    dst = _subsample(dst, _ICP_POINTS)
    tree = cKDTree(dst)

    def _fit_cm(points: np.ndarray) -> float:
        distance, _ = tree.query(_subsample(points, _COARSE_POINTS), k=1, workers=-1)
        return float(distance.mean() * 100.0)

    before_cm = _fit_cm(src)
    extent = float(np.linalg.norm(np.ptp(dst, axis=0)))
    src_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(src))
    dst_cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(dst))
    rigid = o3d.pipelines.registration.TransformationEstimationPointToPoint(
        with_scaling=False
    )
    criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=60)
    # Two inits, keep the better fit. The ICP threshold (fractions of a
    # ~0.4m object) is far below a large warmup slide, and identity-init
    # ICP then locks a confidently-wrong partial pull (measured: a 25cm
    # table slide "improves" the fit while leaving a 10cm+ residual). A
    # centroid-difference init closes the translation gap first; identity
    # stays as a candidate because the partial single-view target biases
    # the centroid toward the camera-facing shell.
    centroid_init = np.eye(4)
    centroid_init[:3, 3] = dst.mean(axis=0) - src.mean(axis=0)
    result, after_cm = None, np.inf
    for init in (centroid_init, np.eye(4)):
        candidate = init
        for threshold in (0.10 * extent, 0.03 * extent):
            icp = o3d.pipelines.registration.registration_icp(
                src_cloud, dst_cloud, threshold, candidate, rigid, criteria
            )
            candidate = np.asarray(icp.transformation)
        candidate_cm = _fit_cm(src @ candidate[:3, :3].T + candidate[:3, 3])
        if candidate_cm < after_cm:
            result, after_cm = candidate, candidate_cm

    # Object displacement, not the raw matrix translation (a rotation about
    # the world origin inflates the latter for an object sitting off-origin).
    centroid = src.mean(axis=0)
    translation_m = float(
        np.linalg.norm(result[:3, :3] @ centroid + result[:3, 3] - centroid)
    )
    rotation_deg = float(
        np.degrees(np.arccos(np.clip((np.trace(result[:3, :3]) - 1.0) / 2.0, -1, 1)))
    )
    info.update(
        before_cm=round(before_cm, 2),
        after_cm=round(after_cm, 2),
        translation_m=round(translation_m, 4),
        rotation_deg=round(rotation_deg, 2),
    )
    if after_cm >= before_cm:
        info["rejected"] = "no improvement"
        return None, info
    if translation_m > max_translation_m or rotation_deg > max_rotation_deg:
        info["rejected"] = "implausibly large correction"
        return None, info
    return result, info
