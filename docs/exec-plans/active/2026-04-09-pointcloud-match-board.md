# 2026-04-09 Pointcloud Match Board

## Goal

Add one thin professor-facing match-board output that writes only:

- `01_pointcloud_match_board.png`
- `match_board_summary.json`

by default.

## Non-Goals

- another broad compare stack
- restoring many top-level artifacts
- changing calibration or compare engine semantics
- replacing the existing turntable/reprojection internals

## Files To Modify

- `data_process/visualization/match_board.py`
- `data_process/visualization/workflows/match_board.py`
- `scripts/harness/visual_make_match_board.py`
- `scripts/harness/check_all.py`
- `scripts/harness/check_visual_architecture.py`
- `AGENTS.md`
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`
- match-board tests

## Validation

Before:
- current professor-facing package defaults to multiple figures
- too much of the output is organized around hero/truth semantics rather than direct 3-view point-cloud match quality

After:
- one command writes one 2x3 match board plus one compact summary file
- the chosen angle is object-aware and match-oriented
- debug output stays gated under `debug/`
- `python scripts/harness/check_all.py` passes

## Acceptance Criteria

- default top-level output is exactly one figure plus one summary file
- board layout is 2 rows x 3 columns
- rows = Native / FFS
- columns = Source attribution / Support count / Mismatch residual
- existing source/support/mismatch render paths are reused
