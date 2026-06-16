# Single-Camera Table Z0 Calibration Active Exec Plan

Canonical plan:

- `docs/superpowers/plans/2026-06-16-single-camera-table-z0-calibration.md`

Design spec:

- `docs/superpowers/specs/2026-06-16-single-camera-table-z0-calibration-design.md`

Scope:

- add strict one-shot `cameras_calibrate_table.py`
- write separate `table_calibrate.pkl` and `table_calibrate_metadata.json`
- preserve existing `calibrate.pkl` behavior
- add explicit table calibration handling for recording, alignment, and Demo 3.x contract visibility

Execution rule:

- follow the canonical Superpowers plan task by task
- keep unrelated dirty demo/test files out of table calibration commits
