"""Factorize rescue for Open3D's ARAP deformation (import-light).

Failure shape (real GUI run 2026-08-07, trellis2 + upscale-on frame-0):
``DeformAsRigidAsPossible ... Failed to build solver (factorize)`` — o3d's
sparse factorization goes singular at small ABSOLUTE mesh scale (the world
mesh after a ~0.4x rescale). Measured on the failing case: every constraint
set fails at x1 (even zero-displacement targets, even different ids), and
every one succeeds at x10 — pure floating-point conditioning, not mesh
topology (the same mesh's canonical-scale twin factorizes).

ARAP with cotangent weights commutes EXACTLY with uniform scaling, so
solving the scaled-up system and dividing the result is the SAME
deformation (verified numerically: max |d| = 2.7e-15 on a case where both
scales factorize). The rescue therefore has zero quality cost: runs that
factorize at x1 are untouched; runs that would have died fatal now produce
the mathematically identical solution.

Applied by the v7 stage wrappers (align_fast_safe, sample_asap_safe) inside
their own subprocesses only.
"""

from __future__ import annotations

_RESCUE_SCALE = 10.0


def patch_arap_factorize_rescue() -> None:
    """Wrap TriangleMesh.deform_as_rigid_as_possible with the x10 retry."""
    import numpy as np
    import open3d as o3d

    original = o3d.geometry.TriangleMesh.deform_as_rigid_as_possible
    if getattr(original, "_v7_arap_rescue", False):
        return

    def deform_with_rescue(self, constraint_ids, constraint_pos, *args, **kwargs):
        try:
            return original(self, constraint_ids, constraint_pos, *args, **kwargs)
        except RuntimeError as exc:
            if "factorize" not in str(exc):
                raise
            print(
                "[arap-rescue] factorize failed at x1; retrying at "
                f"x{_RESCUE_SCALE:g} (scale-equivariant, same solution)",
                flush=True,
            )
            scaled = o3d.geometry.TriangleMesh(
                o3d.utility.Vector3dVector(
                    np.asarray(self.vertices) * _RESCUE_SCALE
                ),
                self.triangles,
            )
            targets = o3d.utility.Vector3dVector(
                np.asarray(constraint_pos, dtype=np.float64) * _RESCUE_SCALE
            )
            out = original(scaled, constraint_ids, targets, *args, **kwargs)
            out.vertices = o3d.utility.Vector3dVector(
                np.asarray(out.vertices) / _RESCUE_SCALE
            )
            return out

    deform_with_rescue._v7_arap_rescue = True
    o3d.geometry.TriangleMesh.deform_as_rigid_as_possible = deform_with_rescue


# A post-cleanup triangle component below this fraction of the mesh is
# debris, not object geometry.
_ISLAND_FRACTION = 0.01


def patch_asap_island_cleanup() -> None:
    """Drop tiny disconnected components from the chunk-stream ASAP mesh.

    Failure shape (drive21, trellis2): ``_load_clean_mesh``'s own
    ``remove_non_manifold_edges`` pass CUTS a handful of 1-31-triangle
    islands loose; a free-floating island that receives no constraint
    handle makes the ARAP system singular and factorize throws at ANY
    scale (the x10 rescue cannot help — verified). Whether a run survives
    is numerical luck: a mesh with 100 islands passed E2E, one with 80
    died. The islands are cleanup debris with no valid deformation of
    their own; removing them AFTER the stock cleanup (module-global
    rebind, align_fast_safe precedent) makes factorization deterministic.
    """
    from demo_v6_2.streaming import asap

    if getattr(asap._load_clean_mesh, "_v7_island_cleanup", False):
        return
    stock_load_clean_mesh = asap._load_clean_mesh

    def load_clean_mesh_no_islands(path):
        import numpy as np

        mesh = stock_load_clean_mesh(path)
        try:
            labels, counts, _areas = mesh.cluster_connected_triangles()
            counts = np.asarray(counts)
            if len(counts) > 1:
                faces_total = int(np.asarray(mesh.triangles).shape[0])
                tiny = counts < max(2, int(faces_total * _ISLAND_FRACTION))
                if bool(tiny.any()) and not bool(tiny.all()):
                    remove = tiny[np.asarray(labels)]
                    mesh.remove_triangles_by_mask(remove.tolist())
                    mesh.remove_unreferenced_vertices()
                    print(
                        f"[arap-rescue] dropped {int(remove.sum())} island "
                        f"face(s) across {int(tiny.sum())} tiny component(s) "
                        "from the ASAP mesh (unconstrained islands make the "
                        "ARAP factorization singular)",
                        flush=True,
                    )
        except Exception as exc:
            # Robustness patch must never become its own failure mode.
            print(f"[arap-rescue] island cleanup skipped: {exc}", flush=True)
        return mesh

    load_clean_mesh_no_islands._v7_island_cleanup = True
    asap._load_clean_mesh = load_clean_mesh_no_islands
