# Demo v5 data_process_sam3d Contract Rename

## Goal

Rename Demo v5's public modules, classes, functions, CLI surface, and payload
keys so the chunked realtime path speaks the same `data_process_sam3d`,
`track_process_data.pkl`, and `final_data.pkl` vocabulary as the original
offline pipeline.

## Scope

- Add canonical Demo v5 modules for data-process chunk writing, chunked
  final-data output, aggregation, visualization, dense realtime tracking, and
  the realtime runner.
- Keep old `futurephystwin`/`online` module names as thin wrappers for one
  compatibility window.
- Emit canonical payload keys from new writers.
- Accept legacy keys in loaders and validators by normalizing them to canonical
  keys.
- Preserve tracker, selector, recovery, and quality-gate behavior.

## Non-Goals

- No TAPNext++ logic changes.
- No controller recovery redesign.
- No quality-gate relaxation.
- No second tracker or TAPNext++ recurrent-state mutation.
- No changes to formal recording/alignment products.

## Validation

- Run Demo v5 realtime unit tests.
- Run the repo smoke validation profile.
- Confirm old wrapper imports still resolve while canonical modules are the
  implementation source.
