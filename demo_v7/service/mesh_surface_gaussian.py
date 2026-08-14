"""Deterministic surface gaussianization of the aligned shape-prior mesh.

``mesh_surface`` gaussian backend (owner direction 2026-08-14): the
TRELLIS.2 mesh is the single geometry truth — instead of generating a
SECOND free geometry with TripoSplat and reconciling it (24-rotation
registration, ICP, ARAP residual transfer, floater pruning, self-align),
splats are DERIVED from the aligned world mesh itself
(``shape/matching/final_mesh.glb``: the ARAP-refined mesh already in the
camera world frame, textures intact — verified on drive27: vertex delta vs
``object.glb @ mesh2world`` reproduces the provenance ``arap_residual_mean_m``
exactly).

Hard binding contract: every splat stores ``face_index`` + barycentric
coordinates on the source mesh, so its center lies ON the mesh surface by
construction (numerically zero center-to-mesh distance) and can be replayed
from any deformed copy of the SAME topology. The anchors file embeds the
rest vertices/faces it was built against plus a topology hash — a consumer
holding a differently-cleaned mesh fails loudly instead of drifting
(the align chain and ASAP each re-clean topology; anchors must never be
mixed across cleanings).

Sampling is per-face stratified (area-proportional largest-remainder
allocation, >=1 sample per non-degenerate face) with a seeded generator —
same mesh + same seed => bit-identical splats. Tangential sigmas come from
the per-face sample spacing so coverage has no holes; the normal sigma is a
small fraction of that (surfel-like, center stays exactly on the surface).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from demo_v7.service.gaussian_utils import GaussianSplats

DEFAULT_TARGET_SPLATS = 45000
# sqrt(area/n) is the mean sample spacing on a face; sigma at 0.55x spacing
# makes neighboring splats overlap at ~1 sigma (no visible holes) without
# washing out texture detail (checked on the drive27 final_mesh render).
_TANGENT_SIGMA_FACTOR = 0.55
# Surfel thickness: fraction of the tangential sigma. Small enough that the
# shell reads as the mesh surface, large enough to stay numerically far from
# gsplat's degenerate-covariance regime (unlike Mesh2Splat's 1e-7 absolute).
_NORMAL_SIGMA_FRACTION = 0.1
_SPLAT_OPACITY = 0.95
# Faces below this area (m^2) are zero-extent junk — never sampled.
_MIN_FACE_AREA = 1e-14


@dataclass
class MeshAnchors:
    """Splat->mesh binding plus the exact rest topology it binds to."""

    face_index: np.ndarray  # (N,) int32 into ``faces``
    barycentric: np.ndarray  # (N,3) float32, rows sum to 1, >=0
    rest_vertices: np.ndarray  # (V,3) float32 world-frame rest vertices
    faces: np.ndarray  # (F,3) int32
    topology_sha256: str

    def __len__(self) -> int:
        return int(self.face_index.shape[0])


def topology_hash(vertices: np.ndarray, faces: np.ndarray) -> str:
    """Order-sensitive hash of the exact vertex/face arrays (float32/int32)."""
    digest = hashlib.sha256()
    digest.update(np.ascontiguousarray(vertices, dtype=np.float32).tobytes())
    digest.update(np.ascontiguousarray(faces, dtype=np.int32).tobytes())
    return digest.hexdigest()


def face_frames(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Per-face orthonormal frames (F,3,3): columns = (t1, t2, normal).

    t1 follows the first edge, the normal is the face normal, t2 completes
    the right-handed basis. Degenerate faces (zero-length edge or zero
    area) get the identity frame — their samples are excluded at
    generation time, and at replay time a transiently-degenerate face just
    keeps a neutral orientation for one frame instead of emitting NaNs.
    """
    corners = vertices[faces]  # (F,3,3)
    e1 = corners[:, 1] - corners[:, 0]
    normal = np.cross(e1, corners[:, 2] - corners[:, 0])
    e1_len = np.linalg.norm(e1, axis=1)
    n_len = np.linalg.norm(normal, axis=1)
    ok = (e1_len > 1e-12) & (n_len > 1e-12)
    t1 = np.where(ok[:, None], e1 / np.maximum(e1_len, 1e-12)[:, None], 0.0)
    n_hat = np.where(ok[:, None], normal / np.maximum(n_len, 1e-12)[:, None], 0.0)
    t2 = np.cross(n_hat, t1)
    frames = np.stack([t1, t2, n_hat], axis=2)  # columns
    frames[~ok] = np.eye(3)
    return frames


def _frames_to_wxyz(frames: np.ndarray) -> np.ndarray:
    """Batch (F,3,3) rotation matrices -> (F,4) wxyz unit quaternions."""
    from scipy.spatial.transform import Rotation

    xyzw = Rotation.from_matrix(frames).as_quat()
    return np.concatenate([xyzw[:, 3:4], xyzw[:, :3]], axis=1)


def _allocate_samples(areas: np.ndarray, target: int) -> np.ndarray:
    """Area-proportional largest-remainder counts, >=1 per live face."""
    live = areas > _MIN_FACE_AREA
    counts = np.zeros(len(areas), dtype=np.int64)
    if not live.any():
        raise ValueError("mesh has no faces with positive area")
    quota = target * areas[live] / areas[live].sum()
    base = np.floor(quota).astype(np.int64)
    deficit = int(target - base.sum())
    if deficit > 0:
        remainder = quota - base
        base[np.argsort(-remainder)[:deficit]] += 1
    counts[live] = np.maximum(base, 1)
    return counts


def _sample_barycentric(counts: np.ndarray, rng: np.random.Generator):
    """(face_index, barycentric) for ``counts[f]`` samples on each face f."""
    face_index = np.repeat(
        np.arange(len(counts), dtype=np.int64), counts
    )
    uv = rng.random((len(face_index), 2))
    fold = uv.sum(axis=1) > 1.0  # reflect into the lower triangle
    uv[fold] = 1.0 - uv[fold]
    barycentric = np.column_stack([1.0 - uv.sum(axis=1), uv[:, 0], uv[:, 1]])
    return face_index, barycentric


def _sample_colors(mesh, face_index, barycentric) -> np.ndarray:
    """Per-sample display rgb in [0,1] from the mesh texture (or fallback).

    GLB textures are authored/read as display-sRGB and every consumer in
    this repo (GUI mesh view, overlay renders) shows them raw, so the texel
    values are used as-is — matching what the user sees of the same mesh.
    """
    visual = getattr(mesh, "visual", None)
    uv = getattr(visual, "uv", None)
    material = getattr(visual, "material", None)
    image = getattr(material, "baseColorTexture", None) or getattr(
        material, "image", None
    )
    if uv is not None and image is not None:
        sample_uv = np.einsum(
            "sc,scu->su", barycentric, np.asarray(uv)[mesh.faces[face_index]]
        )
        from trimesh.visual.color import uv_to_interpolated_color

        rgba = uv_to_interpolated_color(sample_uv, image)
        rgb = np.asarray(rgba, dtype=np.float32)[:, :3] / 255.0
        factor = getattr(material, "baseColorFactor", None)
        if factor is not None:
            rgb = rgb * (np.asarray(factor, dtype=np.float32)[:3] / 255.0)
        return np.clip(rgb, 0.0, 1.0)
    colors = None
    try:
        colors = np.asarray(visual.to_color().vertex_colors, dtype=np.float32)
    except Exception:  # noqa: BLE001 — untextured/colorless GLB
        pass
    if colors is not None and len(colors) == len(mesh.vertices):
        corner = colors[mesh.faces[face_index], :3] / 255.0
        return np.clip(np.einsum("sc,scu->su", barycentric, corner), 0.0, 1.0)
    return np.full((len(face_index), 3), 0.6, dtype=np.float32)


def replay_splat_means(
    vertices: np.ndarray, faces: np.ndarray, face_index, barycentric
) -> np.ndarray:
    """Barycentric replay of splat centers from (possibly deformed) verts."""
    corners = vertices[faces[face_index]]  # (N,3,3)
    return np.einsum("sc,scx->sx", barycentric, corners)


def gaussianize_mesh(
    mesh_path: str | Path,
    *,
    target_splats: int = DEFAULT_TARGET_SPLATS,
    seed: int = 42,
) -> tuple[GaussianSplats, MeshAnchors]:
    """World-frame surface splats + anchors from an aligned textured GLB."""
    import trimesh

    mesh = trimesh.load(str(mesh_path), force="mesh")
    vertices = np.asarray(mesh.vertices, dtype=np.float64)
    faces = np.asarray(mesh.faces, dtype=np.int64)
    if len(faces) == 0:
        raise ValueError(f"mesh has no faces: {mesh_path}")
    areas = np.asarray(mesh.area_faces, dtype=np.float64)

    counts = _allocate_samples(areas, int(target_splats))
    rng = np.random.default_rng(int(seed))
    face_index, barycentric = _sample_barycentric(counts, rng)
    means = replay_splat_means(vertices, faces, face_index, barycentric)

    spacing = np.sqrt(areas[face_index] / np.maximum(counts[face_index], 1))
    sigma_t = _TANGENT_SIGMA_FACTOR * spacing
    scales = np.column_stack(
        [sigma_t, sigma_t, _NORMAL_SIGMA_FRACTION * sigma_t]
    )
    quats = _frames_to_wxyz(face_frames(vertices, faces))[face_index]
    colors = _sample_colors(mesh, face_index, barycentric)

    splats = GaussianSplats(
        means=means.astype(np.float32),
        quats=quats.astype(np.float32),
        scales=scales.astype(np.float32),
        opacities=np.full(len(face_index), _SPLAT_OPACITY, dtype=np.float32),
        colors=colors.astype(np.float32),
    )
    anchors = MeshAnchors(
        face_index=face_index.astype(np.int32),
        barycentric=barycentric.astype(np.float32),
        rest_vertices=vertices.astype(np.float32),
        faces=faces.astype(np.int32),
        topology_sha256=topology_hash(vertices, faces),
    )
    return splats, anchors


def save_anchors(path: str | Path, anchors: MeshAnchors) -> None:
    """Persist the binding + its exact rest topology as one npz."""
    np.savez_compressed(
        str(path),
        face_index=anchors.face_index,
        barycentric=anchors.barycentric,
        rest_vertices=anchors.rest_vertices,
        faces=anchors.faces,
        topology_sha256=np.array(anchors.topology_sha256),
        schema_version=np.array("mesh_surface_anchors_v1"),
    )


def load_anchors(path: str | Path) -> MeshAnchors:
    """Load + self-verify an anchors npz (hash must match its own arrays)."""
    data = np.load(str(path))
    anchors = MeshAnchors(
        face_index=np.asarray(data["face_index"], dtype=np.int32),
        barycentric=np.asarray(data["barycentric"], dtype=np.float32),
        rest_vertices=np.asarray(data["rest_vertices"], dtype=np.float32),
        faces=np.asarray(data["faces"], dtype=np.int32),
        topology_sha256=str(data["topology_sha256"]),
    )
    recomputed = topology_hash(anchors.rest_vertices, anchors.faces)
    if recomputed != anchors.topology_sha256:
        raise ValueError(
            f"anchors topology hash mismatch in {path}: stored "
            f"{anchors.topology_sha256[:12]}.., arrays hash {recomputed[:12]}.. "
            "— file corrupt or written against a different mesh cleaning"
        )
    if int(anchors.face_index.max()) >= len(anchors.faces):
        raise ValueError(f"anchors face_index out of range in {path}")
    return anchors
