# 2026-04-22 Static Replay PPT Summary + PCD Only

## Goal

Adjust the static replay experiment deliverable so the PPT deck contains only:

1. one summary slide per experiment
2. one frame-10 masked FFS-only PCD board slide per experiment

The RGB slide is removed.

## Scope

- update the static replay PPT export contract
- regenerate the PPT from the existing full-run results without rerunning the 24 experiments
- update docs/tests that mention the old 3-slide-per-experiment deck

## Non-Goals

- no rerun of the full matrix unless a regeneration bug forces it
- no change to benchmark results, PCD renders, or artifact-generation logic

## Validation Plan

- `python -m unittest -v tests.test_ffs_static_replay_matrix_smoke`
- regenerate the PPT from `result/_archived_obsolete/ffs_static_replay_matrix_20260422_sequential_obsolete_fullrun`
- verify final slide count is `48`
