# Demo 3.2 Shape Prior Table Direction Fix

## Goal

Fix SAM3D shape-prior single-view alignment validation so it uses the Demo 3.2 table-world Z convention. The observed native RealSense run failed with `ground_z_fraction=0.7655` because the worker validated with the default positive-above-table direction while the runtime metadata uses negative-above-table.

## Steps

1. Add failing tests proving snapshots/protocol payloads carry table Z metadata and the worker uses negative-above-table when requested.
2. Extend `ShapePriorSnapshot` with `table_z_m` and `table_z_above_direction` metadata.
3. Forward those fields from Demo 3.2 runtime snapshot construction through the ZeroMQ request metadata.
4. Build `ShapeAlignmentConfig` in the worker from request metadata.
5. Run focused shape-prior tests, smoke validation, and retry native RealSense side-by-side warmup.
