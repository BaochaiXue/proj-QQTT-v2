# Demo v5.1 Origin Table-Z Convention

## Goal

Keep Demo v5.1 table-world semantics aligned with `data_process_origin`:

- `z = 0` is the table plane.
- Negative z is above the table, inside the workspace.
- Positive z is below the table and is clamped/filtered as invalid.

## Changes

- Remove the runtime `--table-z-above-direction` override.
- Use `table_z - point_z` as the only table clearance definition.
- Keep metadata field `table_z_above_direction = "negative"` as a recorded
  convention, not a configurable mode.

## Validation

- Compile `demo_v5_1`.
- Run the focused Demo v5.1 unit tests.
