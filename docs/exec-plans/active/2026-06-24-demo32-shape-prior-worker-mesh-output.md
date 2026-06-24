# Demo 3.2 Shape Prior Worker Mesh Output Fix

## Goal

Make the Demo 3.2 SAM3D shape-prior worker accept the mesh/scene output shape observed during the native RealSense side-by-side run, so image upscale and SAM3D results can proceed into single-view alignment instead of failing after inference.

## Steps

1. Add a small helper in `services/shape_prior_remote/server.py` that extracts vertices from direct mesh objects, `trimesh.Scene`-style geometry dictionaries, or list/tuple mesh outputs.
2. Prefer usable mesh vertices over `glb` containers that only wrap scene geometry.
3. Include worker error text in `--debug` logs.
4. Add focused unit coverage for scene-style SAM3D outputs.
5. Re-run the focused shape-prior tests, then retry Demo 3.2 native RealSense side-by-side with the real worker.
