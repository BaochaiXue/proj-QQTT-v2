# Demo 3.2 FFS LiteTracker

## Goal

Create Demo 3.2 as a copied/organized Demo 3.1 lineage that uses FFS TensorRT
builderOptimizationLevel=5 batch=3 depth asynchronously before EdgeTAM and a
LiteTracker serial backend for tracking.

## Requirements

- Add a Demo 3.2 entrypoint and docs rather than mutating Demo 3.1 semantics.
- Reuse Demo 3.1 dual-GPU point-tracker process/lift/render behavior.
- Reuse the Demo 2.3 FFS batch=3 opt=5 depth contract.
- Default tracker backend to `litetracker` and serial execution.
- Expose a dry-run contract that makes the pipeline order explicit:
  capture -> FFS -> EdgeTAM -> tracker -> render/diagnostics.

## Plan

1. Inspect Demo 3.1 bridge defaults and Demo 2.3 FFS batch=3 preset wiring.
2. Add Demo 3.2 mode support without breaking Demo 3.1 defaults.
3. Add a `demo_v3_2/` entrypoint and README.
4. Add contract tests for FFS batch=3 opt=5 and LiteTracker serial defaults.
5. Run targeted deterministic tests.

## Results

Pending.
