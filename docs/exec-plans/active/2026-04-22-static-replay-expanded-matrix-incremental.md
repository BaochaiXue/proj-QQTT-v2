# 2026-04-22 Static Replay Expanded Matrix Incremental Run

## Goal

Expand the static replay TRT matrix from:

- `scale ∈ {1.0, 0.5}`
- `valid_iters ∈ {4, 2}`

to:

- `scale ∈ {1.0, 0.75, 0.5}`
- `valid_iters ∈ {8, 4, 2}`

while reusing the already completed 24 experiments and running only the 30 newly added configs.

## Scope

- extend the matrix constants and TRT size policy
- reuse existing experiment `summary.json` / `pcd_board.png` outputs when present
- change PPT export to one slide per experiment:
  - compact summary text at the top
  - frame-10 masked FFS-only PCD `3x3` panel as the main image
- regenerate the full ranked deck for all 54 experiments

## Non-Goals

- no rerun of the already completed 24 configs
- no change to old FPS results
- no change to aligned-case inputs

## Validation Plan

- `python -m unittest -v tests.test_ffs_static_replay_matrix_smoke`
- incremental rerun against the existing full-run output root
- verify final `results.csv` row count is `54`
- verify final PPT slide count is `54`
