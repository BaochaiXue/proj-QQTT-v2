# SAM3D Sampling Exact-Equivalent Optimization

## Goal

Reduce SAM3D shape-prior sampling time while preserving candidate schedules,
seed/RNG call order, batch priority, voxel priority, distance thresholds, and
target counts for single-camera Demo v4 and offline `data_process_sam3d`.

## Plan

1. Add focused tests for the optimized per-batch candidate processing helper:
   same-voxel nearest selection, earlier-batch priority, disabled distance
   filtering, and equivalence to the legacy vstack/sort/dedupe baseline.
2. Add a shared internal helper in `qqtt/demo/single_view_shape_prior_sampling.py`
   that builds one reference KD-tree, queries each candidate batch once, uses
   the same distances for filtering and sorting, vectorizes first-per-voxel
   selection within a batch, and tracks occupied voxels incrementally across
   batches.
3. Update Demo/worker sampling to use the helper without changing public
   metadata keys, target counts, sampling schedules, or RNG call order.
4. Correct the distance helper to be backend-aware: canonical single-view and
   legacy sampling keep the configured max distance, while only MV-SAM3D caps
   positive values at 0.035 m.
5. Update Demo v4 metadata so the default single-view route records configured
   0.05 m, effective 0.05 m, canonical single-view policy, and offline parity.
6. Validate with focused unit tests and smoke harness; record benchmark evidence
   or blocker details under `docs/generated/`.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_shape_prior_sampling_optimization tests.test_demo_v4_futurephystwin_chunks tests.test_demo32_shape_prior_warmup`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
