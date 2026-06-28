# Demo v5 Drop Legacy Query Schema Plan

Goal: remove Demo v5 and Demo v5.1 compatibility with the legacy
`demo_v4_session_topology_v1` query schema spelling.

## Scope

- Keep work on the `single-camera` branch.
- Remove the `LEGACY_QUERY_SCHEMA_VERSION` constant and validation allowlist from
  `demo_v5/data_process_chunk_writer.py` and `demo_v5_1/data_process_chunk_writer.py`.
- Keep the canonical `data_process_sam3d_realtime_query_schema_v1` schema as the
  only accepted query schema version for Demo v5 and Demo v5.1 artifacts.
- Update Demo v5 and Demo v5.1 README contract text so it no longer documents
  legacy v4 compatibility.
- Do not alter Demo v4 or `realtime_phystwin` historical topology tests.

## Implementation Tasks

1. [x] Add targeted tests proving Demo v5 and Demo v5.1 reject
   `demo_v4_session_topology_v1` query schema payloads and no longer export the
   legacy constant.
2. [x] Run the targeted tests and confirm they fail before the implementation.
3. [x] Remove the legacy query schema constant, validation allowlist, and README
   compatibility text.
4. [x] Run the targeted tests and deterministic smoke validation.
5. [x] Summarize validation and note any pre-existing unrelated workspace
   changes.

## Validation

- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_realtime_phystwin`
- `conda run -n demo_2_max --no-capture-output python -m unittest tests.test_demo_v5_1_live_camera`
- `conda run -n demo_2_max --no-capture-output python scripts/harness/validation/run.py --profile smoke`
