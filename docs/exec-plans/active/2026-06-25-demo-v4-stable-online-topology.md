# Demo v4 Stable Online Topology

## Goal

Ensure Demo v4 online/static aggregation and `realtime_phystwin` online
consumption share one explicit session topology. Object/controller point
columns remain stable across chunks, and `realtime_phystwin` rejects any online
chunk whose topology differs from the initialized topology.

## Root Cause

Per-chunk `build_track_process_input()` was deriving `query_is_object` and
`query_is_controller` from each chunk's local first frame. Demo v4 therefore
could output the same object/controller counts while changing which physical
query occupied a column. That violates `realtime_phystwin`, whose trainer builds
object/controller spring topology once from the initial frame.

## Design

- Treat the first valid chunk as topology initialization.
- Derive `query_semantic_labels` once from the first session frame and pass the
  same labels into later `build_track_process_input()` calls.
- Store fixed object and controller anchor query ids for the session.
- Write these topology fields into chunk/static/online payloads:
  - `query_ids`
  - `query_semantic_labels` (`0=none`, `1=object`, `2=controller`)
  - `object_sample_query_ids`
  - `controller_sample_query_ids`
  - `topology_version = "demo_v4_session_topology_v1"`
  - `topology_hash`
- For each later chunk, output the same columns in the same order.
- If a fixed query is absent or invalid in a later chunk, keep the column,
  preserve finite placeholder coordinates from the previous anchor position,
  and set visibility/motion-valid values to false.
- Do not re-sample object columns per chunk under the same topology.
- Keep generated metadata explicit: anchor query ids are stable and missing
  anchors are reported as `missing`, not silently revived as other identities.
- `realtime_phystwin/qqtt/data/online_stream.py` loads topology from
  `static_data_path`; every online chunk must include matching topology fields.
  Hash, full query ids/labels, object sample ids, controller sample ids, and
  sample-id lengths are checked before frames append.
- `realtime_phystwin/scripts/fake_online_tracker.py` preserves topology fields
  from modern `final_data.pkl` and synthesizes a deterministic legacy topology
  when replaying older data without those fields.

## Touch Points

- `qqtt/demo/phystwin_strict_product.py`
  - Update `StreamingObjectAnchorSelector`.
  - Update `StreamingControllerAnchorSelector`.
- `demo_v4/headless_chunk_bridge.py`
  - Keep the existing session selector wiring.
  - Add session-level query topology state.
- `demo_v4/futurephystwin_chunk_writer.py`
  - Write the six topology fields into `final_data.pkl`,
    `track_process_data.pkl`, and per-chunk manifests.
- `demo_v4/online_chunk_output.py`
  - Write topology fields into `chunks/chunk_*.pkl`, aggregate static
    `final_data.pkl`, and online manifests.
- `demo_v4/online_case_aggregate.py`
  - Reject cross-chunk topology mismatches while aggregating.
- `realtime_phystwin/qqtt/data/online_stream.py`
  - Reject online chunks with missing or mismatched topology.
- `realtime_phystwin/scripts/fake_online_tracker.py`
  - Preserve or synthesize topology when replaying `final_data.pkl`.
- `tests/test_phystwin_strict_product.py`
  - Add/adjust unit tests for missing fixed anchors.
- `tests/test_demo_v4_futurephystwin_chunks.py`
  - Add/adjust integration coverage for changing raw candidates, relabeling
    masks, topology fields, and aggregate mismatch rejection.
- `realtime_phystwin/tests/test_online_topology_contract.py`
  - Add consumer-side acceptance/rejection tests.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m pytest tests/test_phystwin_strict_product.py tests/test_demo_v4_futurephystwin_chunks.py -q`
- `conda run -n demo_2_max --no-capture-output python -m pytest realtime_phystwin/tests -q`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
- Demo v4 headless capture replay with two 25-frame chunks:
  `result/demo_v4/stable_anchor_online_verify_20260624_v6/cases`.
- Local `realtime_phystwin/train_online_zero_then_first.py` with
  `WANDB_MODE=offline CUDA_VISIBLE_DEVICES=0`, one zero-order iteration and one
  first-order iteration.

## Status

- Focused topology red tests were added and observed failing before
  implementation.
- Focused Demo v4/strict product/realtime online topology tests pass after
  implementation:
  `70 passed`.
- Demo v4 generated two online chunks and an aggregate static case with stable
  `topology_hash=3aee47f209b927f0e088ac3cc89c3949136556b2e5a5ed143e2e8644b936b16a`.
- Aggregate `final_data.pkl` has `object_points=(50, 2005, 3)`,
  `controller_points=(50, 30, 3)`, no `controller_mask`, and no `numpy._core`
  pickle references.
- Local `realtime_phystwin` online optimization completed:
  zero-order wrote `optimal_params.pkl`; first-order consumed 50 online frames,
  latest chunk 1, saved `best_0.pth`/`iter_0.pth`, loss
  `0.0002519197987567168`.
- Smoke validation passes:
  `302 tests OK`, `smoke checks passed`.
