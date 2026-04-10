# 2026-04-09 Professor Three-Figure Pack

## Goal

Add one thin professor-facing wrapper that reuses the current compare engines but writes only:

- `01_hero_compare.png`
- `02_merge_evidence.png`
- `03_truth_board.png`
- `summary.json`

by default.

## Non-Goals

- new render modes
- another large visualization stack
- changing turntable/reprojection engine semantics
- default debug/video/keyframe dumps

## Files To Modify

- `scripts/harness/visual_make_professor_triptych.py`
- `data_process/visualization/professor_triptych.py`
- `data_process/visualization/workflows/professor_triptych.py`
- `data_process/visualization/layouts.py`
- `README.md`
- `docs/WORKFLOWS.md`
- `docs/generated/depth_visualization_validation.md`
- tests for angle selection, output contract, debug gating, and summary generation

## Validation

Before:
- professor-facing evidence is scattered across many hero/debug/video artifacts
- default output is too debug-heavy

After:
- default run writes only the three figures plus `summary.json`
- hero angle is chosen deterministically from support/mismatch evidence
- debug/video/keyframe outputs stay gated behind explicit flags
- `python scripts/harness/check_all.py` passes

## Acceptance Criteria

- one command produces the three-figure pack
- figure 1 is a clean 1x2 Native-vs-FFS hero compare
- figure 2 is a clean merge-evidence board
- figure 3 is a clean truth board from one camera pair
- debug clutter is off by default
