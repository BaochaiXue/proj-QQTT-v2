"""Sample-stage wrapper: zero-extent-face cleanup of final_mesh.glb, then
the unchanged v6.2 sample stage.

Why (TRELLIS.2 backend only): align's final ARAP pins many constraint
handles to shared / z-clamped targets, which lands previously-distinct
vertices on bitwise-identical positions — the exported final_mesh.glb then
carries a few hundred faces whose corners collapse under the exact-weld
dedup that BOTH later ARAP consumers perform (align did too, but ASAP's
chunk-0 deformation is the one that dies on the TRELLIS.2 topology:
"Failed to build solver (factorize)"). SAM3D meshes carry the same class
of collapsed faces yet happen to factorize; dropping the zero-extent faces
makes the mesh strictly cleaner than that known-good baseline.

Quality guarantee (owner red line): ONLY faces with exactly zero extent are
removed — corners bitwise-coincident after weld, or area exactly 0.0. Such
faces render to zero pixels from every viewpoint, carry zero surface area
(zero sampling probability), and exist in the ARAP system purely as
nan/inf sources. Near-zero sliver faces are untouched.

This wrapper runs in place of ``python -m demo_v6_2.shape_prior.sample``
with the identical CLI (the prewarm pool may append ``--wait-signal``).
The sample stage is strictly ordered after align (the parent gates on the
align profile), so final_mesh.glb is complete when the cleanup runs:
- cold: clean before calling the real ``sample.main``;
- prewarmed: ``stage_prewarm.wait_for_go`` is wrapped so the cleanup runs
  exactly between GO and the sample compute. Timing caveat: the cleanup
  cost (~100s of ms) is therefore booked into the profile's ``go_wait_ms``
  (excluded from ``total_ms`` by the schema's accounting rule) — the
  parent-side critical-path wall time still covers it; accepted drift.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# Self-limiting by construction (exact zero extent only), but refuse
# pathological meshes loudly rather than rewriting them wholesale.
_MAX_DROP_FRACTION = 0.05


def _zero_extent_face_mask(vertices, faces):
    """True = keep. Drops ONLY exact-zero-extent faces (see module doc)."""
    import numpy as np

    verts = np.asarray(vertices, dtype=np.float64)
    tris = np.asarray(faces, dtype=np.int64)
    _, weld = np.unique(verts, axis=0, return_inverse=True)
    welded = weld[tris]
    distinct = (
        (welded[:, 0] != welded[:, 1])
        & (welded[:, 1] != welded[:, 2])
        & (welded[:, 0] != welded[:, 2])
    )
    corners = verts[tris]
    areas = 0.5 * np.linalg.norm(
        np.cross(corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]),
        axis=1,
    )
    return distinct & (areas > 0.0)


def clean_final_mesh(mesh_path: Path) -> None:
    """Drop zero-extent faces from ``mesh_path`` in place (atomic replace)."""
    import trimesh

    loaded = trimesh.load(str(mesh_path), process=False)
    if isinstance(loaded, trimesh.Scene):
        geoms = list(loaded.geometry.values())
        if len(geoms) != 1:
            print(
                f"[sample-asap-safe] {len(geoms)} geometries; skipping cleanup",
                flush=True,
            )
            return
        geom = geoms[0]
    else:
        geom = loaded
    keep = _zero_extent_face_mask(geom.vertices, geom.faces)
    dropped = int((~keep).sum())
    if dropped == 0:
        print("[sample-asap-safe] final_mesh already clean", flush=True)
        return
    if dropped > max(2, int(len(keep) * _MAX_DROP_FRACTION)):
        raise ValueError(
            f"final_mesh cleanup wants to drop {dropped}/{len(keep)} faces — "
            "beyond zero-extent junk; refusing (inspect the align output)"
        )
    geom.update_faces(keep)
    # Deliberately KEEP now-unreferenced vertices: final_mesh.glb shares
    # object.glb's vertex order (align's index-aligned export), and the
    # gaussian ARAP-residual transfer depends on that contract. The orphaned
    # rows are exact seam-duplicates, so ASAP's exact-weld folds them into
    # referenced vertices (verified: 0 isolated post-weld, ARAP factorizes).
    tmp_path = mesh_path.with_name(mesh_path.name + ".cleaning.tmp.glb")
    try:
        geom.export(str(tmp_path))
        os.replace(tmp_path, mesh_path)
    finally:
        tmp_path.unlink(missing_ok=True)
    print(
        f"[sample-asap-safe] dropped {dropped} zero-extent face(s) of "
        f"{len(keep)} from {mesh_path.name} (render-invisible; ARAP-safe "
        "for the downstream ASAP deformation)",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)

    def _value_of(flag: str) -> str:
        return argv[argv.index(flag) + 1]

    mesh_path = (
        Path(_value_of("--base_path"))
        / _value_of("--case_name")
        / "shape"
        / "matching"
        / "final_mesh.glb"
    )

    if "--wait-signal" in argv:
        from demo_v6_2.utils import stage_prewarm

        real_wait_for_go = stage_prewarm.wait_for_go

        def wait_for_go_then_clean(stage, *, on_directive=None):
            should_run = real_wait_for_go(stage, on_directive=on_directive)
            if should_run:
                clean_final_mesh(mesh_path)
            return should_run

        stage_prewarm.wait_for_go = wait_for_go_then_clean
    else:
        clean_final_mesh(mesh_path)

    from demo_v6_2.shape_prior import sample

    sample.main(argv)


if __name__ == "__main__":
    main()
