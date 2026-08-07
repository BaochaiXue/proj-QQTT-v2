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
