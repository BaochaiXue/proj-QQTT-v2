# PhysTwin Volume Object Filter

## Goal

Replace Demo 2.3 / Demo 3 live object render filtering's fixed point-count
control with FuturePhysTwin-style world-volume voxel sampling. Keep fixed caps
available only as ablation/safety, and add a parameter for keeping more than
one representative point per occupied voxel when needed.

## Plan

- Add a reusable `phystwin_volume_filter` helper with deterministic
  world-space voxel sampling.
- Add CLI/config fields for object point control, 5mm voxel size, voxel origin
  policy, adaptive voxel size, emergency cap, and points-per-voxel.
- Wire the new object filter path into the shared three-view runtime while
  preserving existing fixed-cap behavior as `--object-point-control fixed-cap`.
- Make Demo 2.3 and Demo 3 / 3.1 default to `phystwin-volume`.
- Add profile metadata for occupied voxels, output points, voxel size, timing,
  safety cap, and points-per-voxel.
- Add tests for voxel sampling semantics and adapter pass-through.

## Validation

- Focused unit tests for the volume sampler and Demo runtime contracts.
- `python -m py_compile` for touched runtime modules.
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`.

## Outcome

- Added a reusable FuturePhysTwin-style object volume sampler with
  `points_per_voxel` support.
- Made Demo 2.3, Demo 3, and Demo 3.1 expose/pass through
  `--object-volume-points-per-voxel`.
- Demo 2.3 / Demo 3 / Demo 3.1 default rendered object PCD filtering to
  `phystwin-volume` at 5mm with one representative per voxel.
- Added profile/contract fields for object volume timing, occupied voxels,
  output points, adaptive voxel size, safety cap, and per-voxel representative
  count.
- Added focused sampler and runtime contract tests.

## Validation Result

- `python -m py_compile qqtt/demo/phystwin_volume_filter.py qqtt/demo/three_view_masked_fused_pcd_runtime.py qqtt/demo/demo23_runtime.py qqtt/demo/demo3_runtime.py qqtt/demo/demo31_runtime.py demo_v2_3/realtime_three_view_dual_gpu_async_filtered_fused_pcd.py`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_phystwin_volume_filter tests.test_demo_v2_3_dual_gpu_smoke tests.test_demo3_contract tests.test_demo31_dual_gpu_contract`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
