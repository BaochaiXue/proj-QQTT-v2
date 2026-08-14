"""Light trimesh-only mesh helpers.

Split out of ``align_util`` so consumers that only need ``as_mesh``
(the sample stage, mesh-cache validation — both on warm-up/startup paths)
do not pay align_util's torch / PyTorch3D / matplotlib import chain.
"""

from __future__ import annotations

import trimesh


def as_mesh(scene_or_mesh):
    """Convert a possible scene to a mesh.

    If conversion occurs, the returned mesh has only vertex and face data.
    """
    if isinstance(scene_or_mesh, trimesh.Scene):
        meshes = [
            geometry
            for geometry in scene_or_mesh.geometry.values()
            if isinstance(geometry, trimesh.Trimesh)
        ]
        if len(meshes) > 1:
            return trimesh.util.concatenate(meshes)
        if len(meshes) == 1:
            return meshes[0]
        raise ValueError("No valid meshes found in the GLB file")
    assert isinstance(scene_or_mesh, trimesh.Trimesh)
    return scene_or_mesh


__all__ = ["as_mesh"]
