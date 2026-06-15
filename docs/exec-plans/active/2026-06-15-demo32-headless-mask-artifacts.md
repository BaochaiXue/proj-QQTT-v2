# Demo 3.2 Headless Mask Artifacts

## Goal

Extend Demo 3.2 headless capture so every saved enhanced-pt PCD frame also
writes the corresponding controller/object binary masks for offline diagnosis.

## Implementation Notes

- Add a `masks/{seq}.npz` artifact alongside `pcd/`, `ffs_depth/`, and
  `query_trajectory/`.
- Save both the raw EdgeTAM controller/object masks and the post stride/erosion
  masks actually used for masked PCD generation.
- Add `mask_path` to each `frames.jsonl` row and record mask source metadata in
  `metadata.json`.
- Keep artifact cadence tied to completed filtered PCD frames; do not fill in
  unsaved fake-live camera frames.

## Validation

- Update the headless writer unit test to assert mask artifacts and schema.
- Run focused TAPNext++ overlay tests and the quick harness checks.
