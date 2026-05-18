# Demo 2.3 Fusion Mismatch Diagnostics

## Goal

Add opt-in diagnostics for the current `stuffed animal` / `towel` Demo 2.3 fused point-cloud mismatch without changing FFS, EdgeTAM, filtering, renderer behavior, or demo defaults.

## Plan

- Add debug flags for per-camera color rendering, per-camera PLY export, mask overlays, identity/inverted c2w experiments, and single-camera isolation.
- Write one run-scoped debug directory under `docs/generated/debug_fusion/<timestamp>/` with calibration diagnostics and first saved group artifacts.
- Add profile fields for per-camera point counts, bounds, centroids, and mask pixel counts.
- Cover CLI translation and the diagnostic hot path with deterministic smoke tests.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo_v2_3_dual_gpu_smoke`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
