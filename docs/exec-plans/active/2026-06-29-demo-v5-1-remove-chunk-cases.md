# Demo v5.1 Remove Data-Process Chunk Cases

## Requirement

Problem:
Demo v5.1 currently materializes every realtime window as a full
data_process-compatible case named `demo_v5_1_chunk_XXXX`. That artifact was not
approved and is not part of the needed realtime flow.

Required final behavior:
Realtime chunking must publish only the online chunk stream and the growing
`data/<case>/final_data.pkl` view. It must not create `demo_v5_1_chunk_XXXX`
case directories, `track_process_data.pkl`, `mask/processed_masks.pkl`,
`tracking/0.npz`, or per-window case metadata for each chunk.

Inputs:
Existing headless capture frames, prepared per-frame payloads, masks, tracks,
shape-prior points, and CLI chunk settings.

Outputs:
`online_data/<case>/chunks/chunk_*.pkl`, `online_data/<case>/manifest.json`,
and `data/<case>/final_data.pkl`.

State changes:
Remove the runtime dependency on case-directory chunk publishing and aggregate
from in-memory final_data/track_process payloads.

Invalid cases:
Keep existing fail-fast validation for malformed chunk tensors, changed query
schema, invalid/degraded track status, and missing required shape prior.

Constraints:
Do not add a compatibility switch. Do not preserve the old case-directory path.
Do not revert unrelated in-progress local changes.

Unknowns:
None affecting correctness.

## Plan

- [ ] Replace per-window case writing with an in-memory final_data builder.
- [ ] Make online output commit final_data plus diagnostics directly.
- [ ] Remove aggregate-from-chunk-case runtime wiring and stale CLI text.
- [ ] Update tests/docs to assert no chunk-case writer path remains.
- [ ] Run focused tests and syntax checks.
