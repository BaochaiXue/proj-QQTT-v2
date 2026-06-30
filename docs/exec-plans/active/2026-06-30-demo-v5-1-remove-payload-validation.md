# Demo v5.1 Remove Payload-Layer Validation

## Requirement

Problem:
`chunk_data_payload.py` currently describes and implements schema,
shape, and query-id validation. The payload layer should not be a defensive
validation gate.

Required final behavior:
The payload layer only converts strict upstream tracking output into online
`final_data` payloads and manifest fields. It does not run explicit
`_validate_*` schema/shape/query-id checks.

Inputs:
`ChunkDataWindow.track_process_data`, shape-prior points, and chunk metadata.

Outputs:
`final_data`, track diagnostics, and manifest fields for online chunk writing.

State changes:
Remove explicit payload validation helpers and their tests.

Invalid cases:
Malformed inputs are not normalized or silently repaired in this layer; normal
array/key operations fail naturally if required data is absent.

Constraints:
No compatibility path. No old chunk-case writer behavior.

## Plan

- [x] Remove explicit payload validation helpers and calls.
- [x] Rename frame-count helper so it does not imply validation.
- [x] Update tests that expected validation rejection.
- [x] Run focused tests and smoke validation.
