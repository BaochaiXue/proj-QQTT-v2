"""ASAP augmentation for Demo v6 online chunks (see demo_v6/design_spec_v6.md).

Adapted from the downstream-provided offline postprocessor
(``july2_chunk_vis.py::write_asap_online_chunks``). Per published frame the
aligned shape-prior mesh (``final_mesh.glb``) is ARAP-deformed toward the
currently valid tracked object points; the mesh motion then carries

- temporarily invalid object points (visibility or motion invalid),
- shape-prior surface points,
- shape-prior interior points

to estimated positions. The augmented window keeps ``object_points`` at its
tracking width — invalid entries are filled in place — and publishes the
deformed shape-prior trajectories as dedicated per-frame keys
``asap_surface_points`` / ``asap_interior_points``. Estimated entries keep
the original ``object_visibilities`` / ``object_motions_valid`` values
(False), so downstream losses that gate on those masks never treat an
estimate as a measurement, while consumers of raw ``object_points`` see a
complete, temporally coherent particle set.

The offline original recomputed everything after the capture finished; this
module runs live at chunk-materialization time. The heavy per-query rigid
fits are vectorized (batched 3x3 SVD) but keep the original math.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

# Notebook-parity defaults (vis.ipynb cell 12: write_asap_online_chunks).
DEFAULT_ARAP_ITER = 20
DEFAULT_EMBED_K = 8
DEFAULT_MAX_CONSTRAINT_DIST_M = 0.03
DEFAULT_MAX_CONSTRAINTS = 1500
DEFAULT_MIN_CONSTRAINTS = 30

MESH_RELATIVE_PATH = Path("shape") / "matching" / "final_mesh.glb"


class AsapMeshError(RuntimeError):
    """The aligned shape-prior mesh required by ASAP is missing or unusable."""


def resolve_final_mesh_path(
    metadata: Mapping[str, Any],
    *,
    override: str | Path | None = None,
) -> Path:
    """Locate final_mesh.glb; ASAP fails fast when the mesh is unavailable."""
    if override is not None:
        path = Path(override)
        if not path.is_file():
            raise AsapMeshError(f"ASAP mesh override does not exist: {path}")
        return path
    case_dir = metadata.get("shape_prior_case_dir")
    if not case_dir:
        raise AsapMeshError(
            "ASAP augmentation requires the aligned shape-prior mesh, but the "
            "capture metadata has no shape_prior_case_dir (shape-prior warmup "
            "did not run or did not finish)"
        )
    path = Path(str(case_dir)) / MESH_RELATIVE_PATH
    if not path.is_file():
        raise AsapMeshError(f"ASAP mesh not found: {path}")
    return path


def _load_clean_mesh(path: Path):
    """Load final_mesh.glb and apply the notebook's cleanup passes."""
    import open3d as o3d  # noqa: PLC0415

    mesh = o3d.io.read_triangle_mesh(str(path))
    if np.asarray(mesh.vertices).shape[0] == 0:
        # Open3D's glTF reader rejects some valid glb variants; trimesh loads
        # them (the production mesh is trimesh-exported).
        import trimesh  # noqa: PLC0415

        loaded = trimesh.load(str(path), force="mesh")
        mesh = o3d.geometry.TriangleMesh(
            o3d.utility.Vector3dVector(np.asarray(loaded.vertices, dtype=np.float64)),
            o3d.utility.Vector3iVector(np.asarray(loaded.faces, dtype=np.int32)),
        )
    mesh.remove_duplicated_vertices()
    mesh.remove_duplicated_triangles()
    mesh.remove_degenerate_triangles()
    mesh.remove_non_manifold_edges()
    mesh.remove_unreferenced_vertices()
    vertices = np.asarray(mesh.vertices)
    if vertices.shape[0] == 0:
        raise AsapMeshError(f"ASAP mesh has no vertices after cleanup: {path}")
    if not np.isfinite(vertices).all():
        raise AsapMeshError(f"ASAP mesh has non-finite vertices: {path}")
    mesh.compute_vertex_normals()
    return mesh


def fit_weighted_rigid_batch(
    src: np.ndarray,
    dst: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Batched weighted rigid fits (original fit_weighted_rigid, vectorized).

    ``src``/``dst`` are ``(M, k, 3)`` neighbor sets, ``weights`` is ``(M, k)``.
    Returns rotations ``(M, 3, 3)`` and translations ``(M, 3)``.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / (w.sum(axis=1, keepdims=True) + 1e-12)
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    center_src = np.sum(src * w[:, :, None], axis=1)
    center_dst = np.sum(dst * w[:, :, None], axis=1)
    src0 = src - center_src[:, None, :]
    dst0 = dst - center_dst[:, None, :]
    covariance = np.einsum("mki,mkj->mij", src0 * w[:, :, None], dst0)
    u, _s, vt = np.linalg.svd(covariance)
    rotation = np.einsum("mji,mkj->mik", vt, u)
    flip = np.linalg.det(rotation) < 0.0
    if np.any(flip):
        vt = vt.copy()
        vt[flip, -1, :] *= -1.0
        rotation = np.einsum("mji,mkj->mik", vt, u)
    translation = center_dst - np.einsum("mij,mj->mi", rotation, center_src)
    return rotation, translation


class LocalRigidEmbedding:
    """Static query points embedded in mesh motion via local rigid fits.

    The neighbor sets and weights are precomputed once against the reference
    mesh vertices (the queries never change), so per-frame transfer is one
    batched rigid fit — the live-optimized form of the original
    ``transfer_points_by_local_rigid``.
    """

    def __init__(
        self,
        query_points: np.ndarray,
        mesh_vertices_ref: np.ndarray,
        *,
        k: int = DEFAULT_EMBED_K,
        eps: float = 1e-8,
    ) -> None:
        """Initialize LocalRigidEmbedding."""
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.query_points = np.asarray(query_points, dtype=np.float64).reshape(-1, 3)
        vertices = np.asarray(mesh_vertices_ref, dtype=np.float64).reshape(-1, 3)
        count = self.query_points.shape[0]
        self.k = int(min(max(1, int(k)), vertices.shape[0]))
        if count:
            tree = cKDTree(vertices)
            dist, idx = tree.query(self.query_points, k=self.k, workers=-1)
            dist = np.atleast_2d(dist).reshape(count, self.k)
            idx = np.atleast_2d(idx).reshape(count, self.k)
            weights = 1.0 / (dist + float(eps))
            self.neighbor_idx = idx.astype(np.int64)
            self.weights = weights / (weights.sum(axis=1, keepdims=True) + 1e-12)
            self.neighbor_ref = vertices[self.neighbor_idx]
        else:
            self.neighbor_idx = np.zeros((0, self.k), dtype=np.int64)
            self.weights = np.zeros((0, self.k), dtype=np.float64)
            self.neighbor_ref = np.zeros((0, self.k, 3), dtype=np.float64)
        self._last = self.query_points.astype(np.float32)

    def transfer_frame(self, mesh_vertices_frame: np.ndarray) -> np.ndarray:
        """Carry the queries through one frame of mesh motion."""
        if self.query_points.shape[0] == 0:
            return np.zeros((0, 3), dtype=np.float32)
        frame_vertices = np.asarray(mesh_vertices_frame, dtype=np.float64)
        neighbor_now = frame_vertices[self.neighbor_idx]
        # Original guard: a query whose neighborhood went non-finite keeps its
        # previous-frame estimate. The batched SVD would raise on ANY
        # non-finite matrix, so bad neighborhoods are replaced with their
        # reference positions (identity fit) before the fit and the result is
        # overwritten afterwards.
        bad = ~np.isfinite(neighbor_now).all(axis=(1, 2))
        if np.any(bad):
            neighbor_now = neighbor_now.copy()
            neighbor_now[bad] = self.neighbor_ref[bad]
        rotation, translation = fit_weighted_rigid_batch(
            self.neighbor_ref, neighbor_now, self.weights
        )
        moved = (
            np.einsum("mij,mj->mi", rotation, self.query_points) + translation
        ).astype(np.float32)
        if np.any(bad):
            moved[bad] = self._last[bad]
        self._last = moved
        return moved


class AsapRuntime:
    """Session-lived live ASAP augmenter for Demo v6 chunk windows.

    Initialization is lazy: the aligned mesh only exists once shape-prior
    warmup finished, so the first materialized window resolves and loads it
    (and fails fast when it is missing).
    """

    def __init__(
        self,
        *,
        mesh_path_override: str | Path | None = None,
        arap_iter: int = DEFAULT_ARAP_ITER,
        embed_k: int = DEFAULT_EMBED_K,
        max_constraint_dist_m: float = DEFAULT_MAX_CONSTRAINT_DIST_M,
        max_constraints: int = DEFAULT_MAX_CONSTRAINTS,
        min_constraints: int = DEFAULT_MIN_CONSTRAINTS,
    ) -> None:
        """Initialize AsapRuntime."""
        self.mesh_path_override = mesh_path_override
        self.arap_iter = int(arap_iter)
        self.embed_k = int(embed_k)
        self.max_constraint_dist_m = float(max_constraint_dist_m)
        self.max_constraints = int(max_constraints)
        self.min_constraints = int(min_constraints)
        self.mesh_path: Path | None = None
        self._base_mesh = None
        self._vertices_ref: np.ndarray | None = None
        self._ref_valid: np.ndarray | None = None
        self._ref_vertex_idx: np.ndarray | None = None
        self._ref_vertex_dist: np.ndarray | None = None
        self._object_embedding: LocalRigidEmbedding | None = None
        self._surface_embedding: LocalRigidEmbedding | None = None
        self._interior_embedding: LocalRigidEmbedding | None = None
        self._fallback_vertices: np.ndarray | None = None
        self._first_frame_is_reference = True

    @property
    def initialized(self) -> bool:
        """Return the initialized."""
        return self._vertices_ref is not None

    def _initialize(
        self,
        metadata: Mapping[str, Any],
        reference_object_points: np.ndarray,
        reference_valid: np.ndarray,
        surface_points: np.ndarray,
        interior_points: np.ndarray,
    ) -> None:
        """Load the mesh and freeze embeddings from chunk-0 frame 0."""
        from scipy.spatial import cKDTree  # noqa: PLC0415

        self.mesh_path = resolve_final_mesh_path(
            metadata, override=self.mesh_path_override
        )
        mesh = _load_clean_mesh(self.mesh_path)
        import open3d as o3d  # noqa: PLC0415

        self._base_mesh = o3d.geometry.TriangleMesh(mesh)
        self._vertices_ref = np.asarray(mesh.vertices, dtype=np.float64)
        reference = np.asarray(reference_object_points, dtype=np.float64).reshape(-1, 3)
        # Reference-frame gate, matching the original build_asap_constraints:
        # a column may act as a constraint handle only when its reference
        # (frame-0) entry was a usable measurement — visibility, motion
        # validity, finite, nonzero — evaluated once and frozen.
        self._ref_valid = (
            np.asarray(reference_valid, dtype=bool).reshape(-1)
            & np.isfinite(reference).all(axis=1)
            & (np.linalg.norm(reference, axis=1) > 1e-9)
        )
        # The reference columns are frozen for the session, so their nearest
        # mesh vertices (constraint handles) are precomputed once.
        tree = cKDTree(self._vertices_ref)
        if reference.shape[0]:
            safe_reference = np.where(self._ref_valid[:, None], reference, 0.0)
            dist, idx = tree.query(safe_reference, k=1, workers=-1)
            self._ref_vertex_idx = np.asarray(idx, dtype=np.int64).reshape(-1)
            self._ref_vertex_dist = np.asarray(dist, dtype=np.float64).reshape(-1)
        else:
            self._ref_vertex_idx = np.zeros((0,), dtype=np.int64)
            self._ref_vertex_dist = np.zeros((0,), dtype=np.float64)
        self._object_embedding = LocalRigidEmbedding(
            reference, self._vertices_ref, k=self.embed_k
        )
        surface = np.asarray(surface_points, dtype=np.float64).reshape(-1, 3)
        interior = np.asarray(interior_points, dtype=np.float64).reshape(-1, 3)
        self._surface_embedding = LocalRigidEmbedding(
            surface, self._vertices_ref, k=self.embed_k
        )
        self._interior_embedding = LocalRigidEmbedding(
            interior, self._vertices_ref, k=self.embed_k
        )
        self._fallback_vertices = self._vertices_ref.copy()
        self._first_frame_is_reference = True

    def _frame_constraints(
        self,
        target_points: np.ndarray,
        valid_columns: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Constraints for one frame: frozen handles -> current valid targets."""
        assert self._ref_vertex_idx is not None
        assert self._ref_vertex_dist is not None
        keep = np.asarray(valid_columns, dtype=bool) & self._ref_valid
        if self.max_constraint_dist_m > 0:
            keep = keep & (self._ref_vertex_dist <= self.max_constraint_dist_m)
        columns = np.flatnonzero(keep)
        if columns.shape[0] == 0:
            return (
                np.zeros((0,), dtype=np.int32),
                np.zeros((0, 3), dtype=np.float64),
            )
        vertex_idx = self._ref_vertex_idx[columns]
        targets = np.asarray(target_points, dtype=np.float64)[columns]
        # Merge duplicate mesh-vertex constraints by averaging their targets
        # (original semantics; np.unique ordering replaces dict insertion
        # order, which only changes which entries a max_constraints cap keeps).
        unique_vertices, inverse = np.unique(vertex_idx, return_inverse=True)
        sums = np.zeros((unique_vertices.shape[0], 3), dtype=np.float64)
        np.add.at(sums, inverse, targets)
        counts = np.bincount(inverse, minlength=unique_vertices.shape[0])
        averaged = sums / counts[:, None]
        if 0 < self.max_constraints < unique_vertices.shape[0]:
            sel = np.linspace(
                0, unique_vertices.shape[0] - 1, self.max_constraints, dtype=np.int64
            )
            unique_vertices = unique_vertices[sel]
            averaged = averaged[sel]
        return unique_vertices.astype(np.int32), averaged

    def _deform_frame(
        self,
        constraint_idx: np.ndarray,
        constraint_target: np.ndarray,
    ) -> tuple[np.ndarray, bool]:
        """ARAP-deform the base mesh toward one frame's constraints."""
        import open3d as o3d  # noqa: PLC0415

        assert self._base_mesh is not None
        assert self._fallback_vertices is not None
        if constraint_idx.shape[0] < self.min_constraints:
            # Downstream-provided experimental behavior: when constraints are
            # too thin, reuse the PREVIOUS frame's mesh vertices instead of
            # failing. We keep it for contract parity with the offline
            # postprocessor, but this silently freezes geometry during long
            # occlusions — revisit with the downstream owners.
            return self._fallback_vertices.copy(), False
        deformed = self._base_mesh.deform_as_rigid_as_possible(
            o3d.utility.IntVector(np.asarray(constraint_idx, dtype=np.int32).tolist()),
            o3d.utility.Vector3dVector(np.asarray(constraint_target, dtype=np.float64)),
            max_iter=int(self.arap_iter),
        )
        vertices = np.asarray(deformed.vertices, dtype=np.float64)
        if not np.isfinite(vertices).all():
            # Same downstream-provided fallback as above — revisit.
            return self._fallback_vertices.copy(), False
        return vertices, True

    def augment_window(
        self,
        track_process: Mapping[str, np.ndarray],
        *,
        metadata: Mapping[str, Any],
        surface_points: np.ndarray,
        interior_points: np.ndarray,
    ) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
        """Augment one published window; returns (track_process, summary)."""
        start_s = time.perf_counter()
        result = dict(track_process)
        object_points = np.asarray(result["object_points"], dtype=np.float32)
        visibilities = np.asarray(result["object_visibilities"], dtype=bool)
        motions_valid = np.asarray(result["object_motions_valid"], dtype=bool)
        frame_count = int(object_points.shape[0])

        if not self.initialized:
            if frame_count == 0 or object_points.shape[1] == 0:
                raise AsapMeshError(
                    "ASAP augmentation cannot initialize from an empty window"
                )
            self._initialize(
                metadata,
                object_points[0],
                visibilities[0] & motions_valid[0],
                surface_points,
                interior_points,
            )

        assert self._vertices_ref is not None
        assert self._object_embedding is not None
        assert self._surface_embedding is not None
        assert self._interior_embedding is not None

        # A usable direct measurement needs visibility, motion validity, and a
        # real (finite, nonzero) coordinate; everything else is estimated.
        finite = np.isfinite(object_points).all(axis=2)
        nonzero = np.linalg.norm(object_points, axis=2) > 1e-9
        valid_now = visibilities & motions_valid & finite & nonzero

        surface_count = int(self._surface_embedding.query_points.shape[0])
        interior_count = int(self._interior_embedding.query_points.shape[0])
        filled = np.empty_like(object_points)
        surface_frames = np.empty((frame_count, surface_count, 3), dtype=np.float32)
        interior_frames = np.empty((frame_count, interior_count, 3), dtype=np.float32)
        constraint_counts = np.zeros((frame_count,), dtype=np.int64)
        deform_ok = np.zeros((frame_count,), dtype=bool)
        for frame_idx in range(frame_count):
            if self._first_frame_is_reference:
                # The session reference frame is the mesh's own alignment
                # frame: identity deformation by construction.
                vertices = self._vertices_ref.copy()
                ok = True
                constraint_counts[frame_idx] = int(self._vertices_ref.shape[0])
                self._first_frame_is_reference = False
            else:
                constraint_idx, constraint_target = self._frame_constraints(
                    object_points[frame_idx], valid_now[frame_idx]
                )
                constraint_counts[frame_idx] = int(constraint_idx.shape[0])
                vertices, ok = self._deform_frame(constraint_idx, constraint_target)
            deform_ok[frame_idx] = bool(ok)
            self._fallback_vertices = vertices
            filled[frame_idx] = self._object_embedding.transfer_frame(vertices)
            surface_frames[frame_idx] = self._surface_embedding.transfer_frame(vertices)
            interior_frames[frame_idx] = self._interior_embedding.transfer_frame(
                vertices
            )
        # Direct measurements always win over mesh estimates.
        filled[valid_now] = object_points[valid_now]

        # Publish contract (design_spec_v6.md): object arrays keep their
        # tracking width — invalid entries are filled in place and keep their
        # original visibility/motion masks (False), so downstream losses that
        # gate on those masks never treat an estimate as a measurement. The
        # deformed shape-prior trajectories are published as dedicated
        # per-frame keys instead of widening object_points.
        result["object_points"] = np.ascontiguousarray(filled, dtype=np.float32)
        result["asap_surface_points"] = np.ascontiguousarray(
            surface_frames, dtype=np.float32
        )
        result["asap_interior_points"] = np.ascontiguousarray(
            interior_frames, dtype=np.float32
        )

        summary = {
            "asap_augmented": True,
            "asap_mesh_path": str(self.mesh_path),
            "asap_transfer_method": "local_rigid",
            "asap_embed_k": int(self.embed_k),
            "asap_arap_iter": int(self.arap_iter),
            "asap_object_column_count": int(object_points.shape[1]),
            "asap_surface_column_count": surface_count,
            "asap_interior_column_count": interior_count,
            "asap_estimated_entry_count": int(np.count_nonzero(~valid_now)),
            "asap_constraint_min": int(constraint_counts.min()) if frame_count else 0,
            "asap_constraint_max": int(constraint_counts.max()) if frame_count else 0,
            "asap_fallback_frame_count": int(np.count_nonzero(~deform_ok)),
            "asap_ms": float((time.perf_counter() - start_s) * 1000.0),
        }
        return result, summary
