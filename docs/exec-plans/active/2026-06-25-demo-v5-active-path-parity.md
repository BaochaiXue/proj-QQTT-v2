# Demo v5 Active Path Parity Plan

Goal: make Demo v5's runtime path, not unused helper files, carry the
`data_process_sam3d`-aligned quality contract while preserving continuous online
topology for `realtime_phystwin`.

## Scope

- Keep Demo v5 on the `single-camera` branch.
- Keep `realtime_phystwin` wire topology compatible with
  `demo_v4_session_topology_v1`.
- Keep `data_process_sam3d` quality semantics in the active realtime path:
  first-frame object/controller labels, per-frame mask gating, depth-valid
  gating, radius-outlier cleanup, motion consistency, fixed sample ids, and
  finite output tensors.
- Do not move formal recording/alignment contracts.
- Do not make the 5 FPS proof gate part of this patch; it is tracked
  separately.

## Implementation Tasks

1. [x] Add failing tests showing that `demo_v5` has no shadow product modules and
   that its runtime contract resolves through the active writer/topology path.
2. [x] Add failing tests for bounded anchor revive: a lost controller/object anchor
   should keep the same sample id and be revived from nearby stable anchors when
   the motion estimate is plausible; otherwise it should fall back to the last
   finite point.
3. [x] Implement bounded KNN motion revive in
   `qqtt/demo/phystwin_strict_product.py` for the streaming object/controller
   selectors.
4. [x] Remove or fold shadow Demo v5 helpers that are not active runtime
   dependencies.
5. [x] Keep writer metadata/tests proving the `data_process_sam3d` reference and
   `realtime_phystwin` topology contract.
6. [x] Run targeted tests, dry-run, smoke validation if feasible, then commit and
   push to `origin single-camera`.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin tests.test_phystwin_strict_product`
- `conda run -n demo_2_max --no-capture-output python -m unittest realtime_phystwin.tests.test_online_topology_contract`
- `conda run -n demo_2_max --no-capture-output python demo_v5/realtime_futurephystwin_chunks.py --dry-run`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`

All validation commands passed on 2026-06-25 before commit.
