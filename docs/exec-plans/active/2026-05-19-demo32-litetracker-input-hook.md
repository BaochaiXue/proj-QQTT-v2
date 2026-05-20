# 2026-05-19 Demo 3.2 LiteTracker Input Hook

## Goal

Make Demo 3.2 its own FFS + LiteTracker live runtime path and fix it so the async raw-fusion path publishes live RGB plus object/controller/union masks to the LiteTracker input slot and publishes matching surface-anchor snapshots before render waits for tracking markers.

## Non-Goals

- no change to Demo 3.1 RealSense-depth semantics
- no change to LiteTracker tracking quality or model weights
- no change to the requirement that Demo 3.2 waits for LiteTracker result plus anchored markers before first render
- no migration of the shared fusion/filter services beyond the minimum hook fix

## Files To Touch

- `qqtt/demo/demo32_runtime.py`
- `qqtt/demo/demo31_runtime.py` common marker/input helper hook
- `demo_v3_2/realtime_three_view_litetracker_ffs_dual4090.py`
- `demo_v3_2/README.md`
- focused Demo 3.1/3.2 contract tests
- this exec plan

## Implementation Plan

1. Add `qqtt.demo.demo32_runtime` as Demo 3.2's public runtime owner.
2. Point the Demo 3.2 entrypoint at `demo32_runtime`, not `demo31_runtime.main`.
3. Extract tracker-input publication into a helper that accepts a depth group, masks, and an optional hook name.
4. Call the helper from both `_build_fused_packet()` and `_build_raw_fused_packet()` so Demo 3.2 async fusion/filter feeds LiteTracker.
5. Publish the matching surface-anchor snapshot from the same helper so tracker results can become anchored red 3D markers.
6. Make Demo 3.2 LiteTracker warmup tracker-native: default lazy query initialization, ready-to-receive-inputs status, and tracker-neutral profile aliases while preserving old `cotracker_*` fields for compatibility.
7. Add focused tests for the independent Demo 3.2 contract and the async raw-fusion hook: input packet published, object/controller masks preserved, surface-anchor cache populated, and hook profile recorded.
8. Run targeted tests and deterministic harness checks.

## Validation Plan

- `conda run -n demo_2_max --no-capture-output python -m unittest -v tests.test_demo31_dual_gpu_contract`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/check_all.py`
