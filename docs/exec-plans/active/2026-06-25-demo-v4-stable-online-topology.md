# Demo v4 Stable Online Topology

## Goal

Ensure Demo v4 online/static aggregation can concatenate real chunks whose raw
object/controller candidate counts vary across windows by fixing topology once
per streaming session.

## Root Cause

Per-chunk object voxel sampling can choose a different number of object columns
for each `final_data.pkl`. The online writer and aggregate writer concatenate
time arrays on axis 0 and require identical tail shapes, so variable object
counts break real fake-live runs. Existing streaming anchor selectors keep tail
shapes stable, but their revive/fallback behavior may replace a missing anchor
with another query, which violates stable physical-column identity.

## Design

- Treat the first valid chunk as topology initialization.
- Store fixed object and controller anchor query ids for the session.
- For each later chunk, output the same columns in the same order.
- If a fixed query is absent or invalid in a later chunk, keep the column,
  preserve finite placeholder coordinates from the previous anchor position,
  and set visibility/motion-valid values to false.
- Do not re-sample object columns per chunk under the same topology.
- Keep generated metadata explicit: anchor query ids are stable and missing
  anchors are reported as `missing`, not silently revived as other identities.

## Touch Points

- `qqtt/demo/phystwin_strict_product.py`
  - Update `StreamingObjectAnchorSelector`.
  - Update `StreamingControllerAnchorSelector`.
- `demo_v4/headless_chunk_bridge.py`
  - Keep the existing session selector wiring.
  - Update manifest status summaries if needed.
- `tests/test_phystwin_strict_product.py`
  - Add/adjust unit tests for missing fixed anchors.
- `tests/test_demo_v4_futurephystwin_chunks.py`
  - Add/adjust integration coverage for changing raw candidates across chunks.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_phystwin_strict_product.py tests/test_demo_v4_futurephystwin_chunks.py -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
