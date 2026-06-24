# Demo 3.2 Robust Shape Alignment Exec Plan

## Goal

Replace the Demo 3.2 single-view shape-prior alignment path that treats equal-length
canonical and observation point arrays as point-wise correspondences. The fixed
path must handle unordered point clouds, preserve the realtime Demo 3.2/4
contract, and be verified through focused shape-prior and Demo v4/PhysTwin tests.

## Current Evidence

- Branch confirmed: `single-camera`.
- `git pull --ff-only origin main` completed with "already up to date".
- Current alignment bug is in `qqtt/demo/single_view_shape_align.py`: equal point
  counts trigger Umeyama even though SAM3D vertices and RGB-D raster points do
  not share ordering.
- Original `data_process_sam3d/align.py` avoids this failure mode by deriving
  feature correspondences before PnP/scale/ARAP. Demo 3.2 warmup cannot call
  that script directly in the realtime process because it is offline-case and
  multi-view oriented, but the single-view worker can use the same evidence-first
  principle.

## Design

1. Add a failing unit test where observation points are a rotated, scaled,
   translated, and shuffled version of canonical points. The old equal-length
   Umeyama path must fail this test.
2. Replace point-count dispatch with a no-correspondence alignment strategy:
   - center both point sets;
   - estimate scale from RMS radius;
   - build PCA frame hypotheses with determinant-safe sign flips;
   - include identity/centroid-scale hypotheses for backwards compatibility;
   - score candidates by symmetric nearest-neighbor Chamfer, coverage p95, and
     centroid drift;
   - refine the best candidate with a few nearest-neighbor ICP iterations;
   - report these metrics in the existing validation payload.
3. Keep the worker API unchanged so Demo 3.2/4 and SAM3D remote protocol remain
   compatible.
4. Run focused unit tests, then Demo v4/PhysTwin chunk tests. If the real SAM3D
   worker, D455, or external optimization runtime is unavailable, record exactly
   which command could not complete and what passed locally.

## Validation Commands

```bash
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup.SingleViewShapeAlignmentTest
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo32_shape_prior_warmup
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v4_futurephystwin_chunks
conda run -n demo_2_max --no-capture-output python -m unittest tests.test_phystwin_strict_product
conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke
```

## Validation Results

- RED confirmed: the unordered equal-size point-cloud test failed on the old
  point-count/Umeyama path before implementation.
- PASS: `tests.test_demo32_shape_prior_warmup.SingleViewShapeAlignmentTest`
  after implementation.
- PASS: `tests.test_demo32_shape_prior_warmup`.
- PASS: `tests.test_demo_v4_futurephystwin_chunks`.
- PASS: `tests.test_phystwin_strict_product`.
- PASS: `python -m py_compile qqtt/demo/single_view_shape_align.py
  tests/test_demo32_shape_prior_warmup.py services/shape_prior_remote/server.py
  demo_v4/realtime_futurephystwin_chunks.py
  demo_v4/futurephystwin_chunk_writer.py`.
- PASS: `git diff --check`.
- PASS: `scripts/harness/validation/run.py --profile smoke`.
- PASS: Demo v4 source-headless chunk materialization using
  `result/demo_v4/single_gpu_shape_bootstrap_20260624/capture` and explicit
  700/1000 shape-prior NPYs wrote five chunk cases under
  `result/demo_v4/robust_shape_alignment_source_chunks_20260624`.
- PASS: selector chose `robust_shape_align_chunk_0004` and
  `robust_shape_align_chunk_0001` as second-last and fifth-last validation
  cases; both passed `validate_futurephystwin_case(require_ready=True)` with 25
  frames, 30 controller points, 700 surface points, and 1000 interior points.

## Notes

- This plan intentionally does not move SAM3D/SuperGlue/PyTorch3D into the Demo
  3.2 realtime process.
- A future heavier worker path can adapt `align.py` or `align_mvsam3d.py`, but
  the immediate fix must remove the mathematically invalid equal-length
  correspondence assumption.
